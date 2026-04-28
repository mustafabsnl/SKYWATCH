"""
SKYWATCH — Pipeline (Ana Orkestratör)
Tüm modülleri doğru sırada çağırarak frame → sonuç akışını yönetir.
Tek giriş noktası: process_frame()

KURALLAR:
  1. Her track embedding aldığında → session cache + DB kontrolü yapılır.
  2. Re-ID: Kişi çıkıp tekrar girerse session cache'den tanınır.
  3. Bir track'ın kontrolü Pipeline tarafında yönetilir (tracker'a bağımlı değil).

Geliştirmeler:
  ┌────────────────────────────────────────────────────────────────┐
  │  1. Session Cache → OrderedDict (LRU, O(1) erişim, max 500)  │
  │  2. Dinamik Re-ID eşiği (aktif track sayısına göre)           │
  │  3. GMC modülü entegrasyonu                                    │
  │  4. _searched_tracks — sınırlı boyutlu set (max 2000)          │
  │  5. Velocity Consistency → zayıf eşleşmede DB aramasını geciktir│
  └────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

from core.face_analyzer import FaceAnalyzer
from core.tracker import Tracker
from core.movement import MovementAnalyzer
from core.gmc import GMCModule
from core.models import Track, DecisionResult, MatchResult
from database.db import Database
from engine.decision import DecisionEngine
from utils.config import AppConfig
from utils.logger import EventLogger, EventType


class Pipeline:
    """Tüm modülleri birleştiren ana işlem hattı."""

    def __init__(self, config: AppConfig, logger: EventLogger):
        self.config = config
        self.logger = logger

        # Modülleri başlat
        self.face_analyzer = FaceAnalyzer(config)
        self.tracker       = Tracker(config.tracking)
        self.movement      = MovementAnalyzer(config.movement)
        self.gmc           = GMCModule(config.tracking)   # Sabit kamera → varsayılan kapalı
        self.db            = Database(config, logger)
        self.decision      = DecisionEngine()

        # DB embedding cache (bellekte)
        self._cached_embeddings: list[tuple[int, np.ndarray]] = []
        self._refresh_cache()

        # Screenshot ayarları
        self._save_screenshots = config.logging.get("save_detection_screenshots", True)
        self._screenshot_dir   = config.get_screenshot_dir()

        # Frame Skip
        self._detect_every_n                  = 4          # Her 4 frame'de 1 algılama
        self._frame_counter: dict[str, int]   = {}
        self._last_faces: dict[str, list]     = {}

        # GMC için önceki frame
        self._prev_frames: dict[str, np.ndarray] = {}

        # ═══ PIPELINE-LEVEL TRACK YÖNETİMİ ═══
        # Tracker'ın is_new'ına güvenmiyoruz. Pipeline kendi takibini yapıyor.
        # Max 2000 eleman sınırı: sonsuz büyüme engeli
        self._searched_tracks: set[int] = set()
        self._searched_tracks_max = 2000

        # ═══ RE-ID: Session-Level Embedding Cache (LRU OrderedDict) ═══
        # Yapı: person_id → (embedding, MatchResult|None, criminal_info|None)
        # OrderedDict kullanımı: O(1) erişim + LRU temizlik
        self._session_cache: OrderedDict[int, tuple[np.ndarray, MatchResult | None, dict | None]] = OrderedDict()
        self._session_cache_max = 500  # Maksimum kişi sayısı

        # Re-ID eşiği — baz değer (dinamik olarak ayarlanır)
        self._base_reid_threshold = 0.50
        # Yanlış pozitifleri azaltmak için daha sıkı eşik/ayrım
        self._criminal_reid_min_threshold = 0.72
        self._db_match_min_threshold = 0.78
        self._db_match_min_margin = 0.04

        # ═══ Person ID: Sabit Kişi Numarası ═══
        # DeepSORT track_id değişir ama person_id aynı kalır
        self._next_person_id             = 1
        self._track_to_person: dict[int, int] = {}   # track_id → person_id

        # İstatistikler
        self.stats = {
            "total_faces_scanned": 0,
            "total_matches": 0,
            "active_tracks": 0,
            "reid_hits": 0,
            "velocity_rejected": 0,   # OC-SORT velocity consistency reddedilen
        }

        # Periyodik cache yenileme (30sn × 5fps ≈ 150 frame)
        self._cache_refresh_interval = 150
        self._frame_total            = 0

    # ──────────────────────────────────────────────────────────────────────
    def _refresh_cache(self):
        """DB'den tüm embedding'leri belleğe çeker."""
        self._cached_embeddings = self.db.get_all_embeddings()
        self.logger.debug(f"Pipeline: {len(self._cached_embeddings)} embedding cache'e alındı")

    # ──────────────────────────────────────────────────────────────────────
    def _dynamic_reid_threshold(self) -> float:
        """
        Aktif track sayısına göre Re-ID eşiğini dinamik olarak ayarlar.

        Kalabalık sahnelerde eşiği yükselt (yanlış pozitif önlemi).
        Sakin sahnelerde düşür (yanlış negatif önlemi).
        """
        active = self.stats["active_tracks"]
        if active > 10:
            return 0.62
        if active > 5:
            return 0.56
        return self._base_reid_threshold  # 0.50

    # ──────────────────────────────────────────────────────────────────────
    def _session_cache_add(self, person_id: int, embedding: np.ndarray,
                           match: MatchResult | None, info: dict | None):
        """
        Session cache'e yeni giriş ekler.
        LRU mantığı: Doluysa en eski girişi sil.
        """
        if person_id in self._session_cache:
            # Güncelle ve en sona taşı (LRU)
            self._session_cache.move_to_end(person_id)
            self._session_cache[person_id] = (embedding, match, info)
            return

        if len(self._session_cache) >= self._session_cache_max:
            # En eski girişi sil (FIFO / LRU)
            self._session_cache.popitem(last=False)

        self._session_cache[person_id] = (embedding, match, info)

    # ──────────────────────────────────────────────────────────────────────
    def process_frame(self, camera_id: str, frame: np.ndarray) -> list[DecisionResult]:
        """Tek bir frame'i uçtan uca işler."""

        # Periyodik cache yenileme
        self._frame_total += 1
        if self._frame_total % self._cache_refresh_interval == 0:
            self._refresh_cache()

        # ═══ AŞAMA 1: YÜZ ALGILAMA (FRAME SKIP) ═══
        if camera_id not in self._frame_counter:
            self._frame_counter[camera_id] = 0
            self._last_faces[camera_id]    = []

        self._frame_counter[camera_id] += 1

        if self._frame_counter[camera_id] >= self._detect_every_n:
            faces = self.face_analyzer.detect_faces(frame)
            self._last_faces[camera_id]    = faces
            self._frame_counter[camera_id] = 0
            self.stats["total_faces_scanned"] += len(faces)
        else:
            faces = self._last_faces.get(camera_id, [])

        # ═══ AŞAMA 2: GMC — Kamera Kayma Tahmini ═══
        # Sabit kamera + gmc_enabled=False → gmc_delta = (0, 0) → hiç etkisi yok
        gmc_delta = self.gmc.estimate(frame)

        # ═══ AŞAMA 3: TRACKING (DeepSORT + Geliştirmeler) ═══
        all_tracks = self.tracker.update(camera_id, faces, frame, gmc_delta)

        # Hayalet kutu önlemi: time_since_update > 3 → yüzü zaten görmüyoruz
        tracks = [t for t in all_tracks if t.time_since_update <= 3]
        self.stats["active_tracks"] = len(tracks)

        # ═══ AŞAMA 4-7: HER TRACK İÇİN İŞLE ═══
        results: list[DecisionResult] = []

        for track in tracks:
            criminal_info = None

            # ─── AŞAMA 4: EMBEDDING KONTROLÜ ───────────────────────────
            if (track.face_embedding is not None
                    and track.track_id not in self._searched_tracks):

                # OC-SORT Velocity Consistency: Hız tutarsızsa DB'yi geciktir
                # (yanlış eşleşme sonucu gereksiz DB aramasını engelle)
                if not track.velocity_ok:
                    self.stats["velocity_rejected"] += 1
                    # Sonraki frame'de tekrar denenecek (searched_tracks'e eklemiyoruz)
                else:
                    self._searched_tracks.add(track.track_id)

                    # _searched_tracks sınır kontrolü
                    if len(self._searched_tracks) > self._searched_tracks_max:
                        # En eski track ID'leri sil (yaklaşık FIFO)
                        overflow = len(self._searched_tracks) - self._searched_tracks_max
                        to_remove = list(self._searched_tracks)[:overflow]
                        self._searched_tracks -= set(to_remove)

                    # Dinamik Re-ID eşiği
                    reid_threshold = self._dynamic_reid_threshold()

                    # ── 4a: Session Cache (Re-ID) ──────────────────────
                    reid_result = self._check_session_cache(
                        track.face_embedding, reid_threshold
                    )

                    if reid_result is not None:
                        match, criminal_info, person_id = reid_result
                        self.stats["reid_hits"] += 1

                        # Aynı Person ID'yi ata
                        self._track_to_person[track.track_id] = person_id

                        if match is not None:
                            track.criminal_match = match
                            self.tracker.set_criminal_match(track.track_id, match)
                            name = criminal_info.get("name", "?") if criminal_info else "?"
                            self.logger.info(
                                f"Re-ID: {name} tekrar göründü (PID:{person_id})"
                            )

                    else:
                        # ── 4b: DB Cache'de ara (ilk kez) ─────────────
                        pid = self._next_person_id
                        self._next_person_id += 1
                        self._track_to_person[track.track_id] = pid

                        match = self._search_in_db_cache(track.face_embedding)

                        if match is not None:
                            track.criminal_match = match
                            self.tracker.set_criminal_match(track.track_id, match)
                            criminal_info = self.db.get_criminal_info(match.criminal_id)
                            self.stats["total_matches"] += 1

                            # Session cache'e ekle (LRU)
                            self._session_cache_add(
                                pid, track.face_embedding.copy(), match, criminal_info
                            )

                            status = criminal_info.get("status", "") if criminal_info else ""
                            name   = criminal_info.get("name", "?")   if criminal_info else "?"

                            if status == "WANTED":
                                self.logger.log(
                                    EventType.WANTED_FOUND,
                                    f"ARANAN KISI TESPIT: {name}",
                                    camera_id=camera_id,
                                    confidence=f"{match.confidence:.2f}",
                                    track_id=track.track_id
                                )
                            else:
                                self.logger.log(
                                    EventType.CRIMINAL_DETECTED,
                                    f"Sabıkalı tespit: {name}",
                                    camera_id=camera_id,
                                    confidence=f"{match.confidence:.2f}",
                                    track_id=track.track_id
                                )

                            if self._save_screenshots:
                                self._save_detection_screenshot(frame, track, camera_id)

                            self.db.log_detection(
                                criminal_id=match.criminal_id,
                                camera_id=camera_id,
                                screenshot_path="",
                                confidence=match.confidence
                            )
                        else:
                            # Temiz → session cache'e ekle
                            self._session_cache_add(
                                pid, track.face_embedding.copy(), None, None
                            )

            # ─── Person ID'yi track'e ata ───────────────────────────────
            if track.track_id in self._track_to_person:
                track.global_id = f"P{self._track_to_person[track.track_id]}"

            # ─── Daha önce eşleşmiş track (criminal_info çek) ──────────
            if track.criminal_match is not None and criminal_info is None:
                criminal_info = self.db.get_criminal_info(track.criminal_match.criminal_id)

            # ─── AŞAMA 5: HAREKET ANALİZİ ──────────────────────────────
            movement_report = self.movement.analyze(track)
            track.movement  = movement_report

            # ─── AŞAMA 6: KARAR ────────────────────────────────────────
            decision = self.decision.evaluate(track, criminal_info)
            results.append(decision)

        # Ölü track'leri temizle
        active_ids = {t.track_id for t in tracks}
        self.movement.cleanup(active_ids)

        # _searched_tracks: Sadece aktif olmayanları temizle
        dead_ids = self._searched_tracks - active_ids
        # Ama hepsini silmiyoruz: kısmen korunsun (Re-ID yanılgısı önlemi)
        # Sadece çok büyüyen bölümü kırp (bu zaten yukarıdaki max kontrolünde yapılıyor)
        self._searched_tracks -= dead_ids

        return results

    # ──────────────────────────────────────────────────────────────────────
    def _check_session_cache(
        self, embedding: np.ndarray, threshold: float
    ) -> tuple | None:
        """
        Session cache'de bu embedding'i daha önce gördük mü?

        OrderedDict üzerinde lineer tarama (N küçükse kabul edilebilir).
        N büyüdüğünde FAISS veya ball-tree ile değiştirilebilir.

        Returns:
            (MatchResult|None, criminal_info|None, person_id) veya None
        """
        best_score  = 0.0
        best_result = None

        for person_id, (seen_emb, seen_match, seen_info) in self._session_cache.items():
            score = self.face_analyzer.compare(embedding, seen_emb)

            # Suçlu eşleşmesini session cache'den geri kullanırken daha sıkı eşik uygula.
            req_threshold = threshold
            if seen_match is not None:
                req_threshold = max(req_threshold, self._criminal_reid_min_threshold)

            if score >= req_threshold and score > best_score:
                best_score  = score
                best_result = (seen_match, seen_info, person_id)

        return best_result

    # ──────────────────────────────────────────────────────────────────────
    def _search_in_db_cache(self, embedding: np.ndarray) -> MatchResult | None:
        """DB cache üzerinden embedding karşılaştırma."""
        best_cid = None
        best_score = 0.0
        second_score = 0.0

        for cid, db_emb in self._cached_embeddings:
            score = self.face_analyzer.compare(embedding, db_emb)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_cid = cid
            elif score > second_score:
                second_score = score

        # Genel threshold + ek güvenlik eşiği + top1/top2 ayrımı.
        min_thr = max(self.face_analyzer.threshold, self._db_match_min_threshold)
        if best_cid is None or best_score < min_thr:
            return None

        if (best_score - second_score) < self._db_match_min_margin:
            return None

        return MatchResult(criminal_id=best_cid, confidence=best_score)

    # ──────────────────────────────────────────────────────────────────────
    def _save_detection_screenshot(
        self, frame: np.ndarray, track: Track, camera_id: str
    ):
        try:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{camera_id}_track{track.track_id}_{timestamp}.jpg"
            cv2.imwrite(str(self._screenshot_dir / filename), frame)
        except Exception as e:
            self.logger.error(f"Screenshot kaydedilemedi: {e}")

    # ──────────────────────────────────────────────────────────────────────
    def on_criminal_added(self):
        """Yeni sabıkalı eklendiğinde cache'i yenile."""
        self._refresh_cache()

    # ──────────────────────────────────────────────────────────────────────
    def get_session_cache_size(self) -> int:
        """Bellekteki session cache boyutunu döndürür (debug için)."""
        return len(self._session_cache)
