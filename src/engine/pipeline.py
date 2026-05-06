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
  │  4. Vote-Based Criminal Matching (3 oy gerekli)                │
  │  5. Velocity Consistency → zayıf eşleşmede DB aramasını geciktir│
  └────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import cv2
import time
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
from engine.track_registry import TrackRegistry
from utils.config import AppConfig
from utils.logger import EventLogger, EventType


class Pipeline:
    """Tüm modülleri birleştiren ana işlem hattı."""

    def __init__(self, config: AppConfig, logger: EventLogger):
        self.config = config
        self.logger = logger

        # Modülleri başlat
        self.face_analyzer = FaceAnalyzer(config)
        self.logger.info(
            "FACE_ANALYZER_DEVICE "
            f"face_analyzer_device={self.face_analyzer.device} "
            f"torch_cuda_available={self.face_analyzer.torch_cuda_available} "
            f"torch_version={self.face_analyzer.torch_version} "
            f"torch_version_cuda={self.face_analyzer.torch_version_cuda} "
            f"gpu_name={self.face_analyzer.gpu_name} "
            f"yolo_model_path={self.face_analyzer.yolo_model_path} "
            f"yolo_device={self.face_analyzer.yolo_device}"
        )
        self.logger.info(
            "INSIGHTFACE_PROVIDER "
            f"insightface_providers_requested={self.face_analyzer.insightface_providers_requested} "
            f"insightface_ctx_id={self.face_analyzer.insightface_ctx_id} "
            f"insightface_device_mode={self.face_analyzer.insightface_device_mode}"
        )
        if self.face_analyzer.torch_cuda_available and str(self.face_analyzer.yolo_device).startswith("cpu"):
            self.logger.warning("CUDA is available but YOLO appears to be running on CPU.")
        self.tracker       = Tracker(config.tracking)
        self.movement      = MovementAnalyzer(config.movement)
        self.gmc           = GMCModule(config.tracking)   # Sabit kamera → varsayılan kapalı
        self.db            = Database(config, logger)
        self.decision      = DecisionEngine()
        self.track_registry = TrackRegistry()
        self._active_camera_ids = [c.get("id", "") for c in config.get_active_cameras() if c.get("id")]
        if self._active_camera_ids:
            self.logger.info(
                f"Pipeline active cameras ({len(self._active_camera_ids)}): {', '.join(self._active_camera_ids)}"
            )

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
        # Kamera bazlı track key: (camera_id, track_id) — çoklu kamerada çakışma engeli

        # ═══ RE-ID: Session-Level Embedding Cache (LRU OrderedDict) ═══
        self._session_cache: OrderedDict[int, tuple[np.ndarray, MatchResult | None, dict | None]] = OrderedDict()
        self._session_cache_max = 500

        self._base_reid_threshold = 0.75
        self._criminal_reid_min_threshold = 0.88
        self._db_match_min_threshold = 0.88
        self._db_match_min_margin = 0.08

        # ═══ Person ID: Sabit Kişi Numarası ═══
        self._next_person_id             = 1
        self._track_to_person: dict[tuple[str, int], int] = {}  # (cam_id, track_id) → person_id

        # ═══ Vote-Based Criminal Matching (3 ardışık oy gerekli) ═══
        # (cam_id, track_id) → {criminal_id: {count, best_conf, last_frame}}
        self._match_votes: dict[tuple[str, int], dict[int, int]] = {}
        self._vote_threshold = 3

        # ═══ DB Kontrol Aralığı (CPU koruma) ═══
        self._last_db_check_frame: dict[tuple[str, int], int] = {}
        self._db_check_interval = 8  # Her 8 frame'de bir DB kontrolü

        # İstatistikler
        self.stats = {
            "total_faces_scanned": 0,
            "total_matches": 0,
            "active_tracks": 0,
            "reid_hits": 0,
            "velocity_rejected": 0,
        }

        # Periyodik cache yenileme
        self._cache_refresh_interval = 150
        self._frame_total            = 0
        self.last_profile: dict[str, dict] = {}
        self._debug_cfg = config.get("debug", {}) if hasattr(config, "get") else {}
        self._overlay_cfg = config.get("overlay", {}) if hasattr(config, "get") else {}

        # Predicted track limit'i Tracker'a aktar
        try:
            self.tracker.max_predicted_tracks_per_camera = int(self._debug_cfg.get("max_predicted_tracks_per_camera", 2) or 0)
        except Exception:
            self.tracker.max_predicted_tracks_per_camera = 2

        # Dedup ayarları
        self._overlay_dedup_enabled = bool(self._debug_cfg.get("overlay_dedup_enabled", True))
        try:
            self._overlay_dedup_iou = float(self._debug_cfg.get("overlay_dedup_iou", 0.45))
        except Exception:
            self._overlay_dedup_iou = 0.45

        # PREDICTED kutuların çizim/overlay kontrolü
        self._draw_predicted_tracks = bool(self._debug_cfg.get("draw_predicted_tracks", True))
        self._prediction_render_enabled = bool(config.tracking.get("prediction_render_enabled", False)) if hasattr(config, "tracking") else False
        self._overlay_limit_faces_enabled = bool(self._overlay_cfg.get("limit_faces_enabled", False))
        self._overlay_draw_all_unique_faces = bool(self._overlay_cfg.get("draw_all_unique_faces", True))
        self._overlay_one_box_per_identity = bool(self._overlay_cfg.get("one_box_per_identity", True))
        self._overlay_suppress_raw_if_track_exists = bool(self._overlay_cfg.get("suppress_raw_if_track_exists", True))
        self._overlay_suppress_predicted_if_real_detection_exists = bool(self._overlay_cfg.get("suppress_predicted_if_real_detection_exists", True))
        self._overlay_draw_orphan_raw_faces = bool(self._overlay_cfg.get("draw_orphan_raw_faces", True))

    def _bbox_iou(self, a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _bbox_center_distance(self, a: list[int], b: list[int]) -> float:
        acx = (a[0] + a[2]) / 2.0
        acy = (a[1] + a[3]) / 2.0
        bcx = (b[0] + b[2]) / 2.0
        bcy = (b[1] + b[3]) / 2.0
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)

    def _bbox_containment_ratio(self, outer: list[int], inner: list[int]) -> float:
        # inner bbox'in ne kadarı outer içinde?
        xA = max(outer[0], inner[0])
        yA = max(outer[1], inner[1])
        xB = min(outer[2], inner[2])
        yB = min(outer[3], inner[3])
        iw = max(0, xB - xA)
        ih = max(0, yB - yA)
        inter = iw * ih
        inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return inter / float(inner_area)

    def _is_real_status(self, status: str) -> bool:
        s = str(status or "").upper()
        return s in ("WANTED", "CRIMINAL", "ARANIYOR", "CLEAN", "TEMIZ", "TEMİZ", "UNKNOWN", "TRACKING", "TENTATIVE", "FACE")

    def _is_overlay_close(self, a: list[int], b: list[int]) -> bool:
        if self._bbox_iou(a, b) > 0.25:
            return True
        if self._bbox_center_distance(a, b) < 80.0:
            return True
        cont_ab = self._bbox_containment_ratio(a, b)
        cont_ba = self._bbox_containment_ratio(b, a)
        return max(cont_ab, cont_ba) > 0.60

    # ──────────────────────────────────────────────────────────────────────
    # Status-priority NMS / dedup — final decisions çizilmeden önce uygulanır.
    # Aynı kişi için birden çok kutu varsa en yüksek öncelikli olanı bırakır.
    # ──────────────────────────────────────────────────────────────────────
    _STATUS_PRIORITY: dict[str, int] = {
        "WANTED": 100,
        "CRIMINAL": 100,
        "ARANIYOR": 100,
        "CLEAN": 80,
        "TEMIZ": 80,
        "TEMİZ": 80,
        "UNKNOWN": 60,
        "TRACKING": 40,
        "TENTATIVE": 40,
        "FACE": 30,
        "RAW_FALLBACK": 20,
        "PREDICTED": 10,
    }

    def _decision_priority(self, dec) -> tuple[int, float, int]:
        status = str(getattr(dec, "status", "")).upper()
        prio = self._STATUS_PRIORITY.get(status, 0)
        conf = float(getattr(dec, "confidence", 0.0) or 0.0)
        # Daha eski (büyük) track_id daha stabil sayılır → tie-breaker
        tid = int(getattr(dec, "track_id", 0) or 0)
        return (prio, conf, tid)

    def _is_duplicate_candidate(self, a, b) -> tuple[bool, str]:
        abox = getattr(a, "bbox", None)
        bbox = getattr(b, "bbox", None)
        if not abox or not bbox or len(abox) != 4 or len(bbox) != 4:
            return False, ""

        # A) Aynı track/global kimlik
        if getattr(a, "track_id", None) is not None and getattr(a, "track_id", None) == getattr(b, "track_id", None):
            if getattr(a, "track_id", None) not in (None, 0):
                return True, "same_track"
        if getattr(a, "global_id", None) and getattr(a, "global_id", None) == getattr(b, "global_id", None):
            return True, "same_track"

        iou = self._bbox_iou(abox, bbox)
        center_d = self._bbox_center_distance(abox, bbox)
        min_dim = float(min(max(1, abox[2] - abox[0]), max(1, abox[3] - abox[1]), max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])))

        # B) Raw vs track/decision zinciri
        asrc = str(getattr(a, "_overlay_source", "")).lower()
        bsrc = str(getattr(b, "_overlay_source", "")).lower()
        if {"raw_fallback", "decision"} == {asrc, bsrc} and iou >= 0.5:
            return True, "raw"

        # C) Yüksek overlap + yakın merkez
        if iou > 0.55 and center_d < (min_dim * 0.45):
            return True, "iou_center"

        # D) Büyük oranda containment + merkez yakın
        cont_ab = self._bbox_containment_ratio(abox, bbox)
        cont_ba = self._bbox_containment_ratio(bbox, abox)
        if max(cont_ab, cont_ba) > 0.75 and center_d < (min_dim * 0.6):
            return True, "containment"

        return False, ""

    def _deduplicate_decisions(self, decisions: list) -> tuple[list, dict]:
        """
        Aynı kamera frame'i içinde IoU > overlay_dedup_iou olan kutuları
        STATUS_PRIORITY'ye göre tekilleştirir. PREDICTED kutu, gerçek bir
        FACE/UNKNOWN/TRACKING/CLEAN/CRIMINAL/WANTED kutusuyla çakışırsa düşer.

        Returns:
            (kept_list, stats)
        """
        if not decisions:
            return decisions, {
                "duplicate_removed_count": 0,
                "duplicate_removed_raw": 0,
                "duplicate_removed_predicted": 0,
                "duplicate_removed_same_track": 0,
            }
        if not self._overlay_dedup_enabled:
            return decisions, {
                "duplicate_removed_count": 0,
                "duplicate_removed_raw": 0,
                "duplicate_removed_predicted": 0,
                "duplicate_removed_same_track": 0,
            }

        # Önceliği yüksekten düşüğe sırala — yüksek öncelikli olanlar her zaman tutulur
        ordered = sorted(
            decisions,
            key=self._decision_priority,
            reverse=True,
        )
        kept: list = []
        stats = {
            "duplicate_removed_count": 0,
            "duplicate_removed_raw": 0,
            "duplicate_removed_predicted": 0,
            "duplicate_removed_same_track": 0,
        }
        for d in ordered:
            bbox = getattr(d, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            duplicate = False
            for k in kept:
                is_dup, reason = self._is_duplicate_candidate(d, k)
                if is_dup:
                    duplicate = True
                    stats["duplicate_removed_count"] += 1
                    if reason == "same_track":
                        stats["duplicate_removed_same_track"] += 1
                    if str(getattr(d, "_overlay_source", "")).lower() == "raw_fallback":
                        stats["duplicate_removed_raw"] += 1
                    if str(getattr(d, "status", "")).upper() == "PREDICTED":
                        stats["duplicate_removed_predicted"] += 1
                    break
            if not duplicate:
                kept.append(d)
        return kept, stats

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
            return 0.88
        if active > 5:
            return 0.82
        return self._base_reid_threshold  # 0.75

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
        t0 = time.perf_counter()
        cache_refresh_ms = 0.0
        face_detection_ms = 0.0
        gmc_ms = 0.0
        tracker_ms = 0.0
        db_search_ms = 0.0
        movement_ms = 0.0
        decision_ms = 0.0

        # Periyodik cache yenileme
        self._frame_total += 1
        if self._frame_total % self._cache_refresh_interval == 0:
            t_cache = time.perf_counter()
            self._refresh_cache()
            cache_refresh_ms = (time.perf_counter() - t_cache) * 1000.0

        # ═══ AŞAMA 1: YÜZ ALGILAMA (FRAME SKIP) ═══
        if camera_id not in self._frame_counter:
            self._frame_counter[camera_id] = 0
            self._last_faces[camera_id]    = []

        self._frame_counter[camera_id] += 1

        t_detect = time.perf_counter()
        face_detection_ran = False
        used_cached_faces = False
        frame_counter_val = self._frame_counter[camera_id]
        if self._frame_counter[camera_id] >= self._detect_every_n:
            face_detection_ran = True
            faces = self.face_analyzer.detect_faces(frame)
            self._last_faces[camera_id]    = faces
            self._frame_counter[camera_id] = 0
            self.stats["total_faces_scanned"] += len(faces)
            frame_counter_val = 0
        else:
            used_cached_faces = True
            faces = self._last_faces.get(camera_id, [])
            frame_counter_val = self._frame_counter[camera_id]
        face_detection_ms = (time.perf_counter() - t_detect) * 1000.0

        # ═══ AŞAMA 2: GMC — Kamera Kayma Tahmini ═══
        # Sabit kamera + gmc_enabled=False → gmc_delta = (0, 0) → hiç etkisi yok
        t_gmc = time.perf_counter()
        gmc_delta = self.gmc.estimate(frame)
        gmc_ms = (time.perf_counter() - t_gmc) * 1000.0

        # ═══ AŞAMA 3: TRACKING (DeepSORT + Geliştirmeler) ═══
        t_tracker = time.perf_counter()
        all_tracks = self.tracker.update(camera_id, faces, frame, gmc_delta)
        tracker_ms = (time.perf_counter() - t_tracker) * 1000.0
        tracker_debug = dict(self.tracker.last_debug.get(camera_id, {}))

        # Hayalet kutu önlemi: time_since_update > 3 → yüzü zaten görmüyoruz
        tracks = [t for t in all_tracks if t.time_since_update <= 3]
        self.stats["active_tracks"] = len(tracks)
        raw_detection_fallback_used = False
        fallback_decisions_count = 0

        # ═══ AŞAMA 4-7: HER TRACK İÇİN İŞLE ═══
        results: list[DecisionResult] = []

        for track in tracks:
            criminal_info = None
            track_key = (camera_id, track.track_id)
            track.camera_id = camera_id

            # TrackRegistry'ye kaydet → stabil global_id al
            global_id = self.track_registry.register(track, camera_id)
            track.global_id = global_id


            # ─── AŞAMA 4: EMBEDDING KONTROLÜ ───────────────────────
            confirmed_match = self._get_confirmed_match(track_key)

            # Track oturmadan DB arama yapma
            if track.age < 5:
                pass  # Yeni track — henüz güvenilir değil
            elif track.face_embedding is not None and confirmed_match is None:
                if not track.velocity_ok:
                    self.stats["velocity_rejected"] += 1
                else:
                    # Person ID ata (ilk kez görülmüşse)
                    if track_key not in self._track_to_person:
                        reid_threshold = self._dynamic_reid_threshold()
                        reid_result = self._check_session_cache(
                            track.face_embedding, reid_threshold
                        )
                        if reid_result is not None:
                            _, _, person_id = reid_result
                            self.stats["reid_hits"] += 1
                            self._track_to_person[track_key] = person_id
                        else:
                            pid = self._next_person_id
                            self._next_person_id += 1
                            self._track_to_person[track_key] = pid
                            self._session_cache_add(
                                pid, track.face_embedding.copy(), None, None
                            )

                    # Aralıklı DB kontrolü (her _db_check_interval frame'de bir)
                    last_check = self._last_db_check_frame.get(track_key, -9999)
                    if self._frame_total - last_check >= self._db_check_interval:
                        self._last_db_check_frame[track_key] = self._frame_total
                        t_db = time.perf_counter()
                        match = self._search_in_db_cache(track.face_embedding)
                        db_search_ms += (time.perf_counter() - t_db) * 1000.0
                        if match is not None:
                            self._register_vote(track_key, match, camera_id, track, frame)

            # ─── Person ID'yi track'e ata ───────────────────────────────
            if track_key in self._track_to_person:
                track.global_id = f"{camera_id}-T{track.track_id}"

            # ─── Onaylı criminal match ─────────────────────────────
            if confirmed_match is None:
                confirmed_match = self._get_confirmed_match(track_key)
            if confirmed_match is not None:
                track.criminal_match = confirmed_match
                self.tracker.set_criminal_match(camera_id, track.track_id, confirmed_match)
                if criminal_info is None:
                    criminal_info = self.db.get_criminal_info(confirmed_match.criminal_id)

            # ─── AŞAMA 5: HAREKET ANALİZİ ──────────────────────────────
            t_movement = time.perf_counter()
            movement_report = self.movement.analyze(track)
            track.movement  = movement_report
            movement_ms += (time.perf_counter() - t_movement) * 1000.0

            # ─── AŞAMA 6: KARAR ────────────────────────────────────────
            t_decision = time.perf_counter()
            decision = self.decision.evaluate(track, criminal_info)
            # Unique overlay builder için kaynak metadatası
            decision._overlay_source = "decision"
            decision._camera_id = camera_id
            decision._track_source = getattr(track, "source", "")
            if (
                str(decision.status).upper() == "PREDICTED"
                or str(getattr(track, "source", "")).lower() == "predicted"
            ) and (not self._prediction_render_enabled):
                continue
            results.append(decision)
            decision_ms += (time.perf_counter() - t_decision) * 1000.0

        # PREDICTED kutuların çizimi tracker/pipeline config ile kapatılabilir
        if (not self._draw_predicted_tracks) or (not self._prediction_render_enabled):
            results = [
                r for r in results
                if (
                    str(getattr(r, "status", "")).upper() != "PREDICTED"
                    and str(getattr(r, "_track_source", "")).lower() != "predicted"
                )
            ]

        # Raw YOLO / FaceResult kutuları üretim modunda tamamen kapalı:
        # draw_raw_yolo_boxes=false ise ham detector kutuları overlay'e girmez.
        # raw_detection_fallback_enabled tek başına yeterli değildir; debug raw çizimi
        # açık değilse fallback kutuları da eklenmez.
        raw_detection_fallback_enabled = bool(self._debug_cfg.get("raw_detection_fallback_enabled", True))
        draw_raw_yolo_boxes = bool(self._debug_cfg.get("draw_raw_yolo_boxes", False))
        decisions_before_dedup = len(results)
        raw_faces_count = len([f for f in faces if getattr(f, "bbox", None) is not None and len(getattr(f, "bbox", [])) == 4])
        predicted_tracks_total = int(tracker_debug.get("tracker_predicted_tracks", 0))
        predicted_tracks_drawn = int(tracker_debug.get("tracker_predicted_drawn", 0))
        orphan_raw_faces_drawn = 0
        predicted_suppressed_by_real_detection = 0

        if self._overlay_suppress_predicted_if_real_detection_exists:
            real_boxes = [r.bbox for r in results if self._is_real_status(getattr(r, "status", ""))]
            filtered = []
            for r in results:
                if str(getattr(r, "status", "")).upper() != "PREDICTED":
                    filtered.append(r)
                    continue
                if any(self._bbox_iou(r.bbox, rb) >= 0.45 for rb in real_boxes):
                    predicted_suppressed_by_real_detection += 1
                    continue
                filtered.append(r)
            results = filtered
        should_inject_raw = (
            raw_detection_fallback_enabled
            and draw_raw_yolo_boxes
            and len(faces) > 0
        )
        if should_inject_raw:
            used_boxes = [r.bbox for r in results]
            for idx, face in enumerate(faces):
                if face.bbox is None or len(face.bbox) != 4:
                    continue
                face_bbox = [int(v) for v in face.bbox]
                # Raw yüz, mevcut track/decision ile çakışıyorsa production'da bastır
                if self._overlay_suppress_raw_if_track_exists and any(self._bbox_iou(face_bbox, ub) >= 0.45 for ub in used_boxes):
                    continue
                tmp_id = -int((self._frame_total % 100000) * 100 + idx + 1)
                results.append(DecisionResult(
                    track_id=tmp_id,
                    bbox=face_bbox,
                    status="RAW_FALLBACK",
                    danger_level="LOW",
                    color=(0, 220, 220),
                    criminal_id=None,
                    confidence=float(face.det_score),
                    behavior_label="normal",
                    global_id=None,
                ))
                results[-1]._overlay_source = "raw_fallback"
                results[-1]._camera_id = camera_id
                used_boxes.append(face_bbox)
                fallback_decisions_count += 1
                orphan_raw_faces_drawn += 1
            raw_detection_fallback_used = fallback_decisions_count > 0

        # Real detection varsa aynı bölgede PREDICTED/RAW_FALLBACK bastır
        raw_fallback_suppressed_by_track = 0
        real_candidates = []
        for r in results:
            st = str(getattr(r, "status", "")).upper()
            src = str(getattr(r, "_track_source", "")).lower()
            if st == "PREDICTED" or src == "predicted":
                continue
            if st == "RAW_FALLBACK" or src == "raw_fallback":
                continue
            if self._is_real_status(st):
                real_candidates.append(r)
        filtered_results = []
        for r in results:
            status_u = str(getattr(r, "status", "")).upper()
            track_source_u = str(getattr(r, "_track_source", "")).lower()
            is_predicted = status_u == "PREDICTED" or track_source_u == "predicted"
            is_raw_fallback = status_u == "RAW_FALLBACK" or track_source_u == "raw_fallback"
            if not is_predicted and not is_raw_fallback:
                filtered_results.append(r)
                continue
            suppressed = False
            for real in real_candidates:
                same_track = (
                    getattr(r, "track_id", None) is not None
                    and getattr(r, "track_id", None) == getattr(real, "track_id", None)
                    and getattr(r, "track_id", None) not in (None, 0)
                )
                if same_track or self._is_overlay_close(r.bbox, real.bbox):
                    suppressed = True
                    break
            if suppressed:
                if is_predicted:
                    predicted_suppressed_by_real_detection += 1
                else:
                    raw_fallback_suppressed_by_track += 1
                continue
            filtered_results.append(r)
        results = filtered_results

        unique_overlay_before = len(results)
        # ── Overlay-level dedup / NMS ─────────────────────────────────────
        results, dedup_stats = self._deduplicate_decisions(results)
        overlay_dedup_removed = int(dedup_stats.get("duplicate_removed_count", 0))
        unique_overlay_after = len(results)

        # Ölü track'leri temizle
        active_keys = {(camera_id, t.track_id) for t in tracks}
        self.movement.cleanup(active_keys, camera_id=camera_id)

        # Ölü track'lerin vote geçmişini ve DB check cache'ini temizle
        for k in list(self._match_votes.keys()):
            if k[0] == camera_id and k not in active_keys:
                del self._match_votes[k]
        for k in list(self._last_db_check_frame.keys()):
            if k[0] == camera_id and k not in active_keys:
                del self._last_db_check_frame[k]

        total_process_ms = (time.perf_counter() - t0) * 1000.0
        self.last_profile[camera_id] = {
            "camera_id": camera_id,
            "cache_refresh_ms": round(cache_refresh_ms, 3),
            "face_detection_ms": round(face_detection_ms, 3),
            "gmc_ms": round(gmc_ms, 3),
            "tracker_ms": round(tracker_ms, 3),
            "db_search_ms": round(db_search_ms, 3),
            "movement_ms": round(movement_ms, 3),
            "decision_ms": round(decision_ms, 3),
            "total_process_ms": round(total_process_ms, 3),
            "faces_count": len(faces),
            "tracks_count": len(tracks),
            "decisions_count": len(results),
            "decisions_before_dedup": int(decisions_before_dedup),
            "raw_faces_count": int(raw_faces_count),
            "detect_every_n": self._detect_every_n,
            "db_check_interval": self._db_check_interval,
            "face_detection_ran": face_detection_ran,
            "used_cached_faces": used_cached_faces,
            "frame_counter": frame_counter_val,
            "face_analyzer_device": self.face_analyzer.device,
            "torch_cuda_available": self.face_analyzer.torch_cuda_available,
            "yolo_device": self.face_analyzer.yolo_device,
            "tracker_input_faces": tracker_debug.get("tracker_input_faces", 0),
            "tracker_prepared_detections": tracker_debug.get("tracker_prepared_detections", 0),
            "tracker_rejected_no_bbox": tracker_debug.get("tracker_rejected_no_bbox", 0),
            "tracker_rejected_bad_bbox": tracker_debug.get("tracker_rejected_bad_bbox", 0),
            "tracker_rejected_low_conf": tracker_debug.get("tracker_rejected_low_conf", 0),
            "tracker_rejected_no_embedding": tracker_debug.get("tracker_rejected_no_embedding", 0),
            "tracker_no_embed_faces": tracker_debug.get("tracker_no_embed_faces", 0),
            "tracker_deepsort_called": tracker_debug.get("tracker_deepsort_called", False),
            "tracker_deepsort_embed_count": tracker_debug.get("tracker_deepsort_embed_count", 0),
            "tracker_raw_tracks": tracker_debug.get("tracker_raw_tracks", 0),
            "tracker_confirmed_tracks": tracker_debug.get("tracker_confirmed_tracks", 0),
            "tracker_tentative_tracks": tracker_debug.get("tracker_tentative_tracks", 0),
            "tracker_fallback_tracks": tracker_debug.get("tracker_fallback_tracks", 0),
            "tracker_output_tracks": tracker_debug.get("tracker_output_tracks", 0),
            "tracker_predicted_tracks": tracker_debug.get("tracker_predicted_tracks", 0),
            "tracker_predicted_drawn": tracker_debug.get("tracker_predicted_drawn", 0),
            "tracker_predicted_suppressed_by_config": tracker_debug.get("tracker_predicted_suppressed_by_config", 0),
            "tracker_predicted_dropped_by_age": tracker_debug.get("tracker_predicted_dropped_by_age", 0),
            "tracker_predicted_dropped_by_confidence": tracker_debug.get("tracker_predicted_dropped_by_confidence", 0),
            "tracker_predicted_dropped_by_limit": tracker_debug.get("tracker_predicted_dropped_by_limit", 0),
            "predicted_tracks_total": int(predicted_tracks_total),
            "predicted_tracks_drawn": int(predicted_tracks_drawn),
            "predicted_tracks_rendered": int(predicted_tracks_drawn),
            "predicted_tracks_suppressed_by_config": tracker_debug.get("tracker_predicted_suppressed_by_config", 0),
            "predicted_tracks_dropped_by_limit": tracker_debug.get("tracker_predicted_dropped_by_limit", 0),
            "predicted_tracks_dropped_by_dedup": int(dedup_stats.get("duplicate_removed_predicted", 0)),
            "predicted_suppressed_by_real_detection": int(predicted_suppressed_by_real_detection),
            "tracker_motion_state_dropped_by_cleanup": tracker_debug.get("tracker_motion_state_dropped_by_cleanup", 0),
            "tracker_lost_tracks": tracker_debug.get("tracker_lost_tracks", 0),
            "tracker_reacquired_tracks": tracker_debug.get("tracker_reacquired_tracks", 0),
            "tracker_missed_tracks": tracker_debug.get("tracker_missed_tracks", 0),
            "tracker_id_switch_count": tracker_debug.get("tracker_id_switch_count", 0),
            "avg_prediction_age_sec": tracker_debug.get("avg_prediction_age_sec", 0.0),
            "max_prediction_age_sec": tracker_debug.get("max_prediction_age_sec", 0.0),
            "tracker_embedding_cache_hits": tracker_debug.get("tracker_embedding_cache_hits", 0),
            "tracker_embedding_cache_updates": tracker_debug.get("tracker_embedding_cache_updates", 0),
            "tracker_embedding_match_failures": tracker_debug.get("tracker_embedding_match_failures", 0),
            "fallback_id_reused": tracker_debug.get("fallback_id_reused", 0),
            "fallback_id_created": tracker_debug.get("fallback_id_created", 0),
            "fallback_id_dropped": tracker_debug.get("fallback_id_dropped", 0),
            "raw_detection_fallback_used": raw_detection_fallback_used,
            "fallback_decisions_count": fallback_decisions_count,
            "orphan_raw_faces_drawn": int(orphan_raw_faces_drawn),
            "raw_fallback_suppressed_by_track": int(raw_fallback_suppressed_by_track),
            "unique_overlay_before": int(unique_overlay_before),
            "unique_overlay_after": int(unique_overlay_after),
            "overlay_dedup_enabled": self._overlay_dedup_enabled,
            "overlay_dedup_iou": self._overlay_dedup_iou,
            "overlay_dedup_removed": int(overlay_dedup_removed),
            "duplicate_removed_count": int(dedup_stats.get("duplicate_removed_count", 0)),
            "duplicate_removed_raw": int(dedup_stats.get("duplicate_removed_raw", 0)),
            "duplicate_removed_predicted": int(dedup_stats.get("duplicate_removed_predicted", 0)),
            "duplicate_removed_raw_fallback": int(dedup_stats.get("duplicate_removed_raw", 0)),
            "duplicate_removed_same_track": int(dedup_stats.get("duplicate_removed_same_track", 0)),
            "overlay_final_decisions_count": len(results),
            "unique_overlay_count": len(results),
            "real_faces_drawn_count": len([r for r in results if str(getattr(r, "status", "")).upper() in ("CLEAN", "TEMIZ", "TEMİZ", "UNKNOWN", "TRACKING", "TENTATIVE", "FACE", "WANTED", "CRIMINAL", "ARANIYOR")]),
        }
        return results

    def begin_cycle(self):
        """
        Çoklu-kamera döngüsünde her tur başında bir kez çağrılır.
        TrackRegistry stale hesaplarını kamera başına değil, tur başına yapar.
        """
        self.track_registry.begin_frame()

    def end_cycle(self):
        """
        Çoklu-kamera döngüsünde her tur sonunda bir kez çağrılır.
        """
        self.track_registry.end_frame()

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
    def _register_vote(self, track_key: tuple, match: MatchResult,
                       camera_id: str, track, frame: np.ndarray = None):
        """
        Bir track için suçlu eşleşme oyu kaydeder.
        Ardışık frame kontrolü: arada frame atlanmışsa sayı sıfırlanır.
        Vote threshold'a ulaşılırsa loglama, DB kaydı ve session cache yazılır.
        """
        if track_key not in self._match_votes:
            self._match_votes[track_key] = {}

        votes = self._match_votes[track_key]
        cid = match.criminal_id

        current_frame = self._frame_total

        if cid not in votes:
            votes[cid] = {"count": 0, "best_conf": 0.0, "last_frame": 0}

        info = votes[cid]

        # Ardışıklık kontrolü: DB kontrolü her _db_check_interval frame'de bir
        # yapıldığı için, arada (_db_check_interval + 3) frame'den fazla
        # boşluk varsa ardışıklık bozulmuştur.
        max_gap = self._db_check_interval + 3
        if info["last_frame"] > 0 and (current_frame - info["last_frame"]) > max_gap:
            info["count"] = 1  # Ardışıklık bozuldu, yeniden başla
        else:
            info["count"] += 1

        info["best_conf"] = max(info["best_conf"], match.confidence)
        info["last_frame"] = current_frame

        if info["count"] == self._vote_threshold:
            # 3 ardışık frame eşleşmesi → suçlu onaylandı
            self.stats["total_matches"] += 1
            criminal_info = self.db.get_criminal_info(cid)
            status = criminal_info.get("status", "") if criminal_info else ""
            name = criminal_info.get("name", "?") if criminal_info else "?"

            if status == "WANTED":
                self.logger.log(
                    EventType.WANTED_FOUND,
                    f"ARANAN KISI TESPIT: {name} ({self._vote_threshold} ardışık oy)",
                    camera_id=camera_id,
                    confidence=f"{match.confidence:.2f}",
                    track_id=track.track_id
                )
            else:
                self.logger.log(
                    EventType.CRIMINAL_DETECTED,
                    f"Sabıkalı tespit: {name} ({self._vote_threshold} ardışık oy)",
                    camera_id=camera_id,
                    confidence=f"{match.confidence:.2f}",
                    track_id=track.track_id
                )

            if self._save_screenshots and frame is not None:
                self._save_detection_screenshot(frame, track, camera_id)

            self.db.log_detection(
                criminal_id=cid,
                camera_id=camera_id,
                screenshot_path="",
                confidence=match.confidence
            )

            # Onaylanmış suçlu eşleşmesini session cache'e yaz (şimdi güvenli)
            pid = self._track_to_person.get(track_key)
            if pid is not None and track.face_embedding is not None:
                self._session_cache_add(
                    pid, track.face_embedding.copy(),
                    MatchResult(criminal_id=cid, confidence=info["best_conf"]),
                    criminal_info
                )

    def _get_confirmed_match(self, track_key: tuple) -> MatchResult | None:
        """Vote threshold'u aşmış en güçlü criminal match'i gerçek confidence ile döner."""
        votes = self._match_votes.get(track_key)
        if not votes:
            return None

        for cid, info in votes.items():
            if info["count"] >= self._vote_threshold:
                return MatchResult(criminal_id=cid, confidence=info["best_conf"])

        return None

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

    # ──────────────────────────────────────────────────────────────────────
    def reset_camera_state(self, camera_id: str):
        """
        Belirli bir kameranın pipeline state'ini temizler.
        Kamera kaldırıldığında veya seçim değiştiğinde çağrılır.
        """
        # Frame skip state
        self._frame_counter.pop(camera_id, None)
        self._last_faces.pop(camera_id, None)
        self._prev_frames.pop(camera_id, None)

        # Vote ve DB check state
        for k in list(self._match_votes.keys()):
            if k[0] == camera_id:
                del self._match_votes[k]
        for k in list(self._last_db_check_frame.keys()):
            if k[0] == camera_id:
                del self._last_db_check_frame[k]
        for k in list(self._track_to_person.keys()):
            if k[0] == camera_id:
                del self._track_to_person[k]

        # TrackRegistry ve Tracker
        self.track_registry.clear_camera(camera_id)
        self.tracker.reset(camera_id)
