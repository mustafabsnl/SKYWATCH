"""
SKYWATCH — Pipeline (Ana Orkestratör)
Tüm modülleri doğru sırada çağırarak frame → sonuç akışını yönetir.
Tek giriş noktası: process_frame()

PERFORMANS KURALI:
  DB sorgusu SADECE yeni track oluştuğunda yapılır (is_new==True).
  30 FPS'te saniyede 30 yerine 1 sorgu → %97 kazanç.
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

        # Tüm DB embedding'lerini bellekte tut (hız için)
        self._cached_embeddings: list[tuple[int, np.ndarray]] = []
        self._refresh_cache()

        # Screenshot ayarları
        self._save_screenshots = config.logging.get("save_detection_screenshots", True)
        self._screenshot_dir = config.get_screenshot_dir()

        # ═══ PERFORMANS: Frame Skip ═══
        # Yüz algılama her frame'de değil, her N frame'de bir çalışır
        self._detect_every_n = 5          # Her 5 frame'de 1 algılama
        self._frame_counter: dict[str, int] = {}  # kamera bazlı sayaç
        self._last_faces: dict[str, list] = {}     # son algılanan yüzler

        # İstatistikler
        self.stats = {
            "total_faces_scanned": 0,
            "total_matches": 0,
            "active_tracks": 0
        }

    def _refresh_cache(self):
        """DB'den tüm embedding'leri belleğe çeker."""
        self._cached_embeddings = self.db.get_all_embeddings()
        self.logger.debug(f"Pipeline: {len(self._cached_embeddings)} embedding cache'e alındı")

    def process_frame(self, camera_id: str, frame: np.ndarray) -> list[DecisionResult]:
        """
        Tek bir frame'i uçtan uca işler.
        
        Frame → Yüz Algılama → Tracking → DB Sorgusu (1 kez) → Hareket → Karar
        
        Args:
            camera_id: Kamera ID'si
            frame: BGR formatında OpenCV frame
            
        Returns:
            list[DecisionResult]: Her kişi için karar sonucu (overlay bilgisi)
        """
        # ═══════════════════════════════════════════
        # AŞAMA 1: YÜZ ALGILAMA (FRAME SKIP)
        # ═══════════════════════════════════════════
        # Her frame'de algılama yapMA — her N frame'de bir yap
        if camera_id not in self._frame_counter:
            self._frame_counter[camera_id] = 0
            self._last_faces[camera_id] = []

        self._frame_counter[camera_id] += 1

        if self._frame_counter[camera_id] >= self._detect_every_n:
            # Algılama frame'i — InsightFace çalıştır
            faces = self.face_analyzer.detect_faces(frame)
            self._last_faces[camera_id] = faces
            self._frame_counter[camera_id] = 0
            self.stats["total_faces_scanned"] += len(faces)
        else:
            # Ara frame — son algılamayı kullan (tracking devam eder)
            faces = self._last_faces.get(camera_id, [])

        # ═══════════════════════════════════════════
        # AŞAMA 2: TRACKING (DeepSORT)
        # ═══════════════════════════════════════════
        tracks = self.tracker.update(camera_id, faces, frame)
        self.stats["active_tracks"] = len(tracks)

        # ═══════════════════════════════════════════
        # AŞAMA 3-6: HER TRACK İÇİN İŞLE
        # ═══════════════════════════════════════════
        results: list[DecisionResult] = []

        for track in tracks:
            criminal_info = None

            # --- AŞAMA 3: DB SORGUSU (SADECE 1 KEZ - is_new) ---
            if track.is_new and track.face_embedding is not None:
                match = self._search_in_cache(track.face_embedding)

                if match is not None:
                    track.criminal_match = match
                    criminal_info = self.db.get_criminal_info(match.criminal_id)
                    self.stats["total_matches"] += 1

                    # Alarm ve log
                    status = criminal_info.get("status", "") if criminal_info else ""
                    name = criminal_info.get("name", "?") if criminal_info else "?"

                    if status == "WANTED":
                        self.logger.log(
                            EventType.WANTED_FOUND,
                            f"ARANAN KİŞİ TESPİT: {name}",
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

                    # Ekran görüntüsü kaydet
                    if self._save_screenshots:
                        self._save_detection_screenshot(frame, track, camera_id)

                    # Tespit kaydı
                    self.db.log_detection(
                        criminal_id=match.criminal_id,
                        camera_id=camera_id,
                        screenshot_path="",
                        confidence=match.confidence
                    )

            # Daha önce eşleşmiş ama bu frame'de is_new değilse info'yu çek
            elif track.criminal_match is not None and criminal_info is None:
                criminal_info = self.db.get_criminal_info(track.criminal_match.criminal_id)

            # --- AŞAMA 4: HAREKET ANALİZİ ---
            movement_report = self.movement.analyze(track)
            track.movement = movement_report

            # --- AŞAMA 5: KARAR ---
            decision = self.decision.evaluate(track, criminal_info)
            results.append(decision)

        # Ölü track'lerin hareket geçmişini temizle
        active_ids = {t.track_id for t in tracks}
        self.movement.cleanup(active_ids)

        return results

    def _search_in_cache(self, embedding: np.ndarray) -> MatchResult | None:
        """Bellekteki cache üzerinden embedding karşılaştırma yapar."""
        best_match = None
        best_score = 0.0

        for cid, db_emb in self._cached_embeddings:
            score = self.face_analyzer.compare(embedding, db_emb)

            if score > self.face_analyzer.threshold and score > best_score:
                best_score = score
                best_match = MatchResult(criminal_id=cid, confidence=score)

        return best_match

    def _save_detection_screenshot(self, frame: np.ndarray, track: Track, camera_id: str):
        """Tespit anının ekran görüntüsünü kaydeder."""
        try:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera_id}_track{track.track_id}_{timestamp}.jpg"
            cv2.imwrite(str(self._screenshot_dir / filename), frame)
        except Exception as e:
            self.logger.error(f"Screenshot kaydedilemedi: {e}")

    def on_criminal_added(self):
        """Yeni sabıkalı eklendiğinde cache'i yenile. GUI'den çağrılır."""
        self._refresh_cache()
