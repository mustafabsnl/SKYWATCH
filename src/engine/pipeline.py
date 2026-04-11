"""
SKYWATCH — Pipeline (Ana Orkestratör)
Tüm modülleri doğru sırada çağırarak frame → sonuç akışını yönetir.
Tek giriş noktası: process_frame()

KURALLAR:
  1. Her track embedding aldığında → session cache + DB kontrolü yapılır.
  2. Re-ID: Kişi çıkıp tekrar girerse session cache'den tanınır.
  3. Bir track'ın kontrolü Pipeline tarafında yönetilir (tracker'a bağımlı değil).
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from core.face_analyzer import FaceAnalyzer
from core.tracker import Tracker
from core.movement import MovementAnalyzer
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
        self.tracker = Tracker(config.tracking)
        self.movement = MovementAnalyzer(config.movement)
        self.db = Database(config, logger)
        self.decision = DecisionEngine()

        # DB embedding cache (bellekte)
        self._cached_embeddings: list[tuple[int, np.ndarray]] = []
        self._refresh_cache()

        # Screenshot ayarları
        self._save_screenshots = config.logging.get("save_detection_screenshots", True)
        self._screenshot_dir = config.get_screenshot_dir()

        # Frame Skip
        self._detect_every_n = 4          # Her 4 frame'de 1 algılama
        self._frame_counter: dict[str, int] = {}
        self._last_faces: dict[str, list] = {}

        # ═══ PIPELINE-LEVEL TRACK YÖNETİMİ ═══
        # Tracker'ın is_new'ına güvenmiyoruz. Pipeline kendi takibini yapıyor.
        self._searched_tracks: set[int] = set()

        # ═══ RE-ID: Session-Level Embedding Cache ═══
        # Yapı: [(embedding, MatchResult|None, criminal_info|None, person_id), ...]
        self._session_seen: list[tuple[np.ndarray, MatchResult | None, dict | None, int]] = []
        self._reid_threshold = 0.50

        # ═══ Person ID: Sabit Kişi Numarası ═══
        # DeepSORT track_id değişir ama person_id aynı kalır
        self._next_person_id = 1
        self._track_to_person: dict[int, int] = {}   # track_id → person_id

        # İstatistikler
        self.stats = {
            "total_faces_scanned": 0,
            "total_matches": 0,
            "active_tracks": 0,
            "reid_hits": 0
        }

        # Periyodik cache yenileme
        self._cache_refresh_interval = 30 * 5
        self._frame_total = 0

    def _refresh_cache(self):
        """DB'den tüm embedding'leri belleğe çeker."""
        self._cached_embeddings = self.db.get_all_embeddings()
        self.logger.debug(f"Pipeline: {len(self._cached_embeddings)} embedding cache'e alındı")

    def process_frame(self, camera_id: str, frame: np.ndarray) -> list[DecisionResult]:
        """Tek bir frame'i uçtan uca işler."""

        # Periyodik cache yenileme
        self._frame_total += 1
        if self._frame_total % self._cache_refresh_interval == 0:
            self._refresh_cache()

        # ═══ AŞAMA 1: YÜZ ALGILAMA (FRAME SKIP) ═══
        if camera_id not in self._frame_counter:
            self._frame_counter[camera_id] = 0
            self._last_faces[camera_id] = []

        self._frame_counter[camera_id] += 1

        if self._frame_counter[camera_id] >= self._detect_every_n:
            faces = self.face_analyzer.detect_faces(frame)
            self._last_faces[camera_id] = faces
            self._frame_counter[camera_id] = 0
            self.stats["total_faces_scanned"] += len(faces)
        else:
            faces = self._last_faces.get(camera_id, [])

        # ═══ AŞAMA 2: TRACKING (DeepSORT) ═══
        all_tracks = self.tracker.update(camera_id, faces, frame)

        # Sadece güncel track'leri işle (hayalet kutu önlemi)
        # time_since_update > 3 → bu track son 3 frame'dir eşleşme almadı → gösterme
        tracks = [t for t in all_tracks if t.time_since_update <= 3]
        self.stats["active_tracks"] = len(tracks)

        # ═══ AŞAMA 3-6: HER TRACK İÇİN İŞLE ═══
        results: list[DecisionResult] = []

        for track in tracks:
            criminal_info = None

            # ─── AŞAMA 3: EMBEDDING KONTROLÜ ───
            if (track.face_embedding is not None
                    and track.track_id not in self._searched_tracks):

                self._searched_tracks.add(track.track_id)

                # ── 3a: Session Cache (Re-ID) ──
                reid_result = self._check_session_cache(track.face_embedding)

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
                            f"Re-ID: {name} tekrar gorundu (PID:{person_id})"
                        )

                else:
                    # ── 3b: DB Cache'de ara (ilk kez) ──
                    # Yeni kişi → yeni Person ID ata
                    pid = self._next_person_id
                    self._next_person_id += 1
                    self._track_to_person[track.track_id] = pid

                    match = self._search_in_db_cache(track.face_embedding)

                    if match is not None:
                        track.criminal_match = match
                        self.tracker.set_criminal_match(track.track_id, match)
                        criminal_info = self.db.get_criminal_info(match.criminal_id)
                        self.stats["total_matches"] += 1

                        # Session cache'e ekle (person_id ile)
                        self._session_seen.append(
                            (track.face_embedding.copy(), match, criminal_info, pid)
                        )

                        status = criminal_info.get("status", "") if criminal_info else ""
                        name = criminal_info.get("name", "?") if criminal_info else "?"

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
                                f"Sabikali tespit: {name}",
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
                        self._session_seen.append(
                            (track.face_embedding.copy(), None, None, pid)
                        )

            # ─── Person ID'yi track'e ata (global_id olarak) ───
            if track.track_id in self._track_to_person:
                track.global_id = f"P{self._track_to_person[track.track_id]}"

            # ─── Daha önce eşleşmiş track (criminal_info çek) ───
            if track.criminal_match is not None and criminal_info is None:
                criminal_info = self.db.get_criminal_info(track.criminal_match.criminal_id)

            # ─── AŞAMA 4: HAREKET ANALİZİ ───
            movement_report = self.movement.analyze(track)
            track.movement = movement_report


            # ─── AŞAMA 5: KARAR ───
            decision = self.decision.evaluate(track, criminal_info)
            results.append(decision)

        # Ölü track'leri temizle
        active_ids = {t.track_id for t in tracks}
        self.movement.cleanup(active_ids)

        # Sadece ölen track'leri _searched_tracks'tan sil
        # (aktif track'ler tekrar aranmasın diye set'te kalır)
        dead_ids = self._searched_tracks - active_ids
        self._searched_tracks -= dead_ids

        return results

    def _check_session_cache(self, embedding: np.ndarray) -> tuple | None:
        """Session cache'de bu embedding'i daha önce gördük mü?

        Returns:
            (MatchResult|None, criminal_info|None, person_id) veya None
        """
        for seen_emb, seen_match, seen_info, person_id in self._session_seen:
            score = self.face_analyzer.compare(embedding, seen_emb)
            if score >= self._reid_threshold:
                return (seen_match, seen_info, person_id)
        return None

    def _search_in_db_cache(self, embedding: np.ndarray) -> MatchResult | None:
        """DB cache üzerinden embedding karşılaştırma."""
        best_match = None
        best_score = 0.0

        for cid, db_emb in self._cached_embeddings:
            score = self.face_analyzer.compare(embedding, db_emb)
            if score > self.face_analyzer.threshold and score > best_score:
                best_score = score
                best_match = MatchResult(criminal_id=cid, confidence=score)

        return best_match

    def _save_detection_screenshot(self, frame: np.ndarray, track: Track, camera_id: str):
        try:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_id}_track{track.track_id}_{timestamp}.jpg"
            cv2.imwrite(str(self._screenshot_dir / filename), frame)
        except Exception as e:
            self.logger.error(f"Screenshot kaydedilemedi: {e}")

    def on_criminal_added(self):
        """Yeni sabıkalı eklendiğinde cache'i yenile."""
        self._refresh_cache()
