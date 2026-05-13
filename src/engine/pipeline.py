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


def _normalize_person_search_target_ids(options: dict) -> list[int]:
    raw = options.get("target_person_ids")
    ordered: list[int] = []
    seen: set[int] = set()
    if isinstance(raw, (list, tuple)):
        for x in raw:
            try:
                pid = int(x)
            except (TypeError, ValueError):
                continue
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
    if not ordered and options.get("target_person_id") is not None:
        try:
            ordered.append(int(options["target_person_id"]))
        except (TypeError, ValueError):
            pass
    return ordered


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
        self.decision      = DecisionEngine(logger=self.logger)
        self.track_registry = TrackRegistry()
        self._active_camera_ids = [c.get("id", "") for c in config.get_active_cameras() if c.get("id")]
        if self._active_camera_ids:
            self.logger.info(
                f"Pipeline active cameras ({len(self._active_camera_ids)}): {', '.join(self._active_camera_ids)}"
            )

        # DB embeddings: GENERAL için aday listesi (+ legacy tuple cache)
        self._general_candidates: list[dict] = []
        self._person_search_candidates: list[dict] = []
        self._cached_embeddings: list[tuple[int, np.ndarray]] = []
        self._general_hash_to_pids: dict[str, list[int]] = {}
        self._general_pid_to_hash: dict[int, str] = {}
        self._general_match_counts: dict[int, int] = {}
        self._refresh_cache()

        # Screenshot ayarları
        self._save_screenshots = config.logging.get("save_detection_screenshots", True)
        if hasattr(self.logger, "_run_logger") and self.logger._run_logger and hasattr(self.logger._run_logger, "resolve_run_path"):
            self._screenshot_dir = self.logger._run_logger.resolve_run_path("detections")
        else:
            self._screenshot_dir = config.get_screenshot_dir()

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

        _face_cfg = config.get("face", {}) if hasattr(config, "get") else {}
        _g_nested = _face_cfg.get("general") if isinstance(_face_cfg.get("general"), dict) else {}
        _ps_nested = _face_cfg.get("person_search") if isinstance(_face_cfg.get("person_search"), dict) else {}
        self._general_cosine_threshold = float(
            _g_nested.get("cosine_threshold", _face_cfg.get("general_match_threshold", 0.55))
        )
        self._general_ambiguous_margin = float(
            _g_nested.get("match_margin", _face_cfg.get("general_match_margin", 0.05))
        )
        self._person_search_cosine_threshold = float(_ps_nested.get("cosine_threshold", 0.50))

        # ═══ Person ID: Sabit Kişi Numarası ═══
        self._next_person_id             = 1
        self._track_to_person: dict[tuple[str, int], int] = {}  # (cam_id, track_id) → person_id

        # ═══ Vote-Based Criminal Matching ═══
        # (cam_id, track_id) → {criminal_id: {count, best_conf, last_frame}}
        self._match_votes: dict[tuple[str, int], dict[int, dict]] = {}
        self._vote_threshold_general = int(_face_cfg.get("general_vote_threshold", 2))
        self._vote_threshold_ps = 1
        self._vote_threshold = self._vote_threshold_general

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
        self._overlay_draw_orphan_fallback_only = bool(self._overlay_cfg.get("draw_orphan_fallback_only", True))
        self._overlay_fallback_min_hits_to_draw = int(self._overlay_cfg.get("fallback_min_hits_to_draw", 2) or 0)
        self._overlay_near_duplicate_enabled = bool(self._overlay_cfg.get("near_duplicate_enabled", True))
        self._overlay_near_duplicate_center_factor = float(self._overlay_cfg.get("near_duplicate_center_factor", 0.85) or 0.85)
        self._overlay_near_duplicate_min_center_px = float(self._overlay_cfg.get("near_duplicate_min_center_px", 35.0) or 35.0)
        self._overlay_near_duplicate_size_ratio_min = float(self._overlay_cfg.get("near_duplicate_size_ratio_min", 0.50) or 0.50)
        self._overlay_near_duplicate_size_ratio_max = float(self._overlay_cfg.get("near_duplicate_size_ratio_max", 2.00) or 2.00)
        self._overlay_near_duplicate_vertical_factor = float(self._overlay_cfg.get("near_duplicate_vertical_factor", 0.90) or 0.90)
        self._overlay_fallback_suppress_center_factor = float(self._overlay_cfg.get("fallback_suppress_center_factor", 0.90) or 0.90)
        self._overlay_fallback_suppress_min_center_px = float(self._overlay_cfg.get("fallback_suppress_min_center_px", 40.0) or 40.0)
        self._overlay_fallback_suppress_iou = float(self._overlay_cfg.get("fallback_suppress_iou", 0.20) or 0.20)
        self._overlay_fallback_suppress_iou = float(self._overlay_cfg.get("fallback_suppress_iou", 0.20) or 0.20)
        self._fallback_hits: dict[tuple[str, int], int] = {}
        
        self.current_mode = "GENERAL"
        self.target_person_id = None
        self.target_person_ids: list[int] = []
        self.target_embedding = None
        self._person_search_embeddings: dict[int, np.ndarray] = {}

        # GENERAL mode summary tracking
        self._general_summary_interval = 1.0  # seconds
        self._general_summary_ts = 0.0
        self._general_search_enter = 0
        self._general_matches = 0
        self._general_confirmed = 0

    def set_mode(self, mode: str, options: dict = None):
        """Çalışma modunu ve (varsa) hedef kişi bilgisini ayarlar."""
        options = dict(options or {})
        self.logger.person_search_trace("PIPELINE_MODE_SET_BEGIN", requested_mode=mode, options=str(options))
        
        mode = str(mode or "GENERAL").upper()
        # Legacy mode normalization
        if mode in ["DATABASE", "WANTED_TRACKING"]:
            mode = "GENERAL"
        if mode not in ("GENERAL", "PERSON_SEARCH"):
            self.logger.warning(f"[PIPELINE_MODE_SET] unsupported mode={mode}; falling back to GENERAL")
            mode = "GENERAL"
            
        if mode == "PERSON_SEARCH":
            next_targets = _normalize_person_search_target_ids(options)
            next_signature = tuple(next_targets)
        else:
            next_targets = []
            next_signature = tuple()

        prev_signature = tuple(self.target_person_ids) if getattr(self, "target_person_ids", None) else tuple()
        if not prev_signature and self.target_person_id is not None:
            prev_signature = (int(self.target_person_id),)

        mode_changed = mode != self.current_mode or next_signature != prev_signature
        self.current_mode = mode
        self.target_person_ids = []
        self.target_person_id = None
        self.target_embedding = None
        self._person_search_embeddings.clear()
        self._person_search_candidates.clear()

        if mode_changed:
            self._match_votes.clear()
            self._last_db_check_frame.clear()
            if hasattr(self.tracker, "_criminal_matches"):
                self.tracker._criminal_matches.clear()
            
        self.logger.person_search_trace(
            "PIPELINE_MODE_NORMALIZED",
            current_mode=self.current_mode,
            target_person_id=self.target_person_id,
            target_person_ids=list(self.target_person_ids),
        )

        if mode == "PERSON_SEARCH" and len(next_targets) == 0:
            self.current_mode = "GENERAL"
            self.decision.set_mode("GENERAL", {})
            self.logger.warning("[PIPELINE_MODE_SET_FAILED] PERSON_SEARCH missing target person id(s)")
            self.logger.person_search_trace("PERSON_SEARCH_TARGET_LOAD_FAILED", reason="missing_target_person_ids")
            return False

        if mode == "PERSON_SEARCH":
            self.logger.info(f"[PIPELINE_MODE_SET] mode=PERSON_SEARCH target_person_ids={next_targets}")
            self.logger.person_search_trace(
                "PERSON_SEARCH_TARGETS_LOAD_BEGIN", person_ids=list(next_targets)
            )

            self._person_search_candidates.clear()
            for pid in next_targets:
                info = self.db.get_person_embedding_for_search(pid)
                if info is None or info.get("embedding") is None:
                    self.logger.warning(f"[PERSON_SEARCH_TARGET_SKIP_NO_EMBEDDING] person_id={pid}")
                    self.logger.person_search_trace(
                        "PERSON_SEARCH_TARGET_LOAD_FAILED",
                        person_id=pid,
                        reason="embedding_missing",
                    )
                    continue
                emb = info["embedding"]
                self._person_search_embeddings[pid] = emb
                self._person_search_candidates.append(
                    {
                        "person_id": int(pid),
                        "name": info.get("name") or "",
                        "status": info.get("status") or "",
                        "embedding": emb,
                    }
                )
                self.target_person_ids.append(int(pid))
                self.logger.info(
                    f"[PERSON_SEARCH_TARGET_CACHE] person_id={pid} name=\"{info['name']}\" "
                    f"emb_shape={tuple(emb.shape)} norm={np.linalg.norm(emb):.3f}"
                )
                self.logger.person_search_trace(
                    "PERSON_SEARCH_TARGET_LOAD_OK",
                    person_id=pid,
                    name=info["name"],
                    status=info["status"],
                    emb_shape=str(emb.shape),
                    emb_dtype=str(emb.dtype),
                    norm=float(np.linalg.norm(emb)),
                )

            if not self._person_search_embeddings:
                self.current_mode = "GENERAL"
                self.target_person_ids = []
                self.target_person_id = None
                self._person_search_candidates.clear()
                self.decision.set_mode("GENERAL", {})
                self.logger.warning("[PIPELINE_MODE_SET_FAILED] PERSON_SEARCH no embeddings loaded for targets")
                self.logger.person_search_trace(
                    "PERSON_SEARCH_TARGET_LOAD_FAILED", reason="no_embeddings_for_any_target"
                )
                return False

            self.logger.info(
                f"[PERSON_SEARCH_TARGETS_LOADED] count={len(self._person_search_candidates)} "
                f"ids={list(self.target_person_ids)}"
            )

            self.target_person_id = self.target_person_ids[0]
            self.target_embedding = (
                self._person_search_embeddings.get(self.target_person_id)
                if len(self.target_person_ids) == 1
                else None
            )

            dedup_opts = dict(options)
            dedup_opts["target_person_ids"] = list(self.target_person_ids)
            dedup_opts["target_person_id"] = self.target_person_id
            self.decision.set_mode(mode, dedup_opts)
            self.logger.person_search_trace(
                "PIPELINE_DECISION_MODE_SET",
                mode=mode,
                target_person_id=self.target_person_id,
                target_person_ids=list(self.target_person_ids),
            )

        elif mode == "GENERAL":
            self.target_embedding = None
            self.target_person_ids = []
            self._person_search_embeddings.clear()
            self._person_search_candidates.clear()
            self._general_match_counts.clear()
            self._refresh_cache()
            self.decision.set_mode(mode, options)
            _ids = sorted({int(c["person_id"]) for c in self._general_candidates})
            self.logger.info(
                f"[GENERAL_CANDIDATES_LOADED] count={len(self._general_candidates)} "
                f"unique_persons={len(_ids)} ids={_ids}"
            )
            for c in self._general_candidates:
                self.logger.info(
                    f"[GENERAL_CANDIDATE_HASH] person_id={c['person_id']} "
                    f"name=\"{c.get('name', '')}\" status={c.get('status', '')} "
                    f"hash={c.get('embedding_hash', 'N/A')} obj_id={id(c.get('embedding'))}"
                )
            dup_groups = {h: pids for h, pids in self._general_hash_to_pids.items() if len(pids) > 1}
            if dup_groups:
                self.logger.warning(
                    f"[GENERAL_DUPLICATE_EMBEDDINGS_DETECTED] "
                    f"groups={dup_groups} "
                    f"note='These persons share identical embeddings; GENERAL will use status_priority resolution'"
                )
            self.logger.info(
                f"[GENERAL_MODE_READY] current_mode=GENERAL "
                f"target_embedding=false candidates={len(self._general_candidates)} "
                f"vote_threshold={self._vote_threshold_general} "
                f"cosine_threshold={self._general_cosine_threshold:.2f}"
            )

        if self.current_mode == "PERSON_SEARCH":
            self._vote_threshold = self._vote_threshold_ps
        else:
            self._vote_threshold = self._vote_threshold_general

        emb_ready = (
            len(self._person_search_candidates) > 0
            if self.current_mode == "PERSON_SEARCH"
            else False
        )
        self.logger.person_search_trace(
            "PERSON_SEARCH_TARGET_CACHE_STATE",
            has_target_embedding=emb_ready,
            target_person_ids=list(self.target_person_ids),
            target_person_id=self.target_person_id,
        )
        if self.current_mode == "GENERAL":
            _guniq = len({int(c["person_id"]) for c in self._general_candidates})
            extra_state = (
                f"general_candidate_rows={len(self._general_candidates)} "
                f"general_unique_persons={_guniq}"
            )
        else:
            extra_state = f"person_search_candidates={len(self._person_search_candidates)}"

        self.logger.info(
            f"[PIPELINE_MODE_STATE] current_mode={self.current_mode} target_person_ids={self.target_person_ids} "
            f"has_targets_embedded={str(emb_ready).lower()} {extra_state}"
        )
        self.logger.info(f"[PIPELINE_MODE_SET_DONE] mode={self.current_mode} current_mode={self.current_mode}")
        return True

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
        if self._status_norm(status) == "HEDEF BULUNDU":
            return True
        s = str(status or "").upper()
        return s in ("WANTED", "CRIMINAL", "ARANIYOR", "CLEAN", "TEMIZ", "TEMİZ", "UNKNOWN", "TRACKING", "TENTATIVE")

    _STATUS_PRIORITY: dict[str, int] = {
        "HEDEF BULUNDU": 120,
        "TARGET_FOUND": 120,
        "WANTED": 110,
        "CRIMINAL": 105,
        "ARANIYOR": 110,
        "CLEAN": 90,
        "TEMIZ": 90,
        "TEMİZ": 90,
        "SUSPICIOUS": 85,
        "UNKNOWN": 80,
        "TRACKING": 70,
        "TENTATIVE": 60,
        "FACE": 50,
        "RAW_FALLBACK": 30,
        "PREDICTED": 10,
    }

    def _status_norm(self, value: str) -> str:
        s = str(value or "").upper()
        if s == "TARGET_FOUND":
            return "HEDEF BULUNDU"
        if s == "TEMİZ":
            return "TEMIZ"
        return s

    def _is_low_priority_overlay(self, dec) -> bool:
        status = self._status_norm(getattr(dec, "status", ""))
        source = str(getattr(dec, "_overlay_source", "") or getattr(dec, "_track_source", "")).upper()
        return status in ("RAW_FALLBACK", "FACE", "TENTATIVE", "TRACKING", "PREDICTED") or source in ("RAW_FALLBACK", "PREDICTED")

    def _is_real_decision(self, dec) -> bool:
        status = self._status_norm(getattr(dec, "status", ""))
        if status == "HEDEF BULUNDU":
            return True
        return status in ("WANTED", "CRIMINAL", "ARANIYOR", "CLEAN", "TEMIZ", "UNKNOWN", "TRACKING", "TENTATIVE")

    def _bbox_area(self, bbox: list[int]) -> float:
        return float(max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1]))

    def _decision_priority(self, dec) -> tuple[int, float, int]:
        status = self._status_norm(getattr(dec, "status", ""))
        prio = self._STATUS_PRIORITY.get(status, 0)
        conf = float(getattr(dec, "confidence", 0.0) or 0.0)
        updated = int(getattr(dec, "_updated_at", 0) or 0)
        area = self._bbox_area(getattr(dec, "bbox", [0, 0, 1, 1]))
        return (prio, conf, updated, int(area))

    def _is_same_face_candidate(self, a, b) -> tuple[bool, str]:
        abox = getattr(a, "bbox", None)
        bbox = getattr(b, "bbox", None)
        if not abox or not bbox or len(abox) != 4 or len(bbox) != 4:
            return False, ""
        same_camera = str(getattr(a, "_camera_id", "")) == str(getattr(b, "_camera_id", ""))
        if same_camera:
            ta = getattr(a, "track_id", None)
            tb = getattr(b, "track_id", None)
            if ta is not None and tb is not None and ta == tb and ta not in (0, None):
                return True, "same_track"
        if getattr(a, "global_id", None) and getattr(a, "global_id", None) == getattr(b, "global_id", None):
            return True, "same_track"

        aw = float(max(1, abox[2] - abox[0]))
        ah = float(max(1, abox[3] - abox[1]))
        bw = float(max(1, bbox[2] - bbox[0]))
        bh = float(max(1, bbox[3] - bbox[1]))
        avg_box_size = (aw + ah + bw + bh) / 4.0
        avg_height = (ah + bh) / 2.0
        acx = (abox[0] + abox[2]) / 2.0
        acy = (abox[1] + abox[3]) / 2.0
        bcx = (bbox[0] + bbox[2]) / 2.0
        bcy = (bbox[1] + bbox[3]) / 2.0
        center_distance = self._bbox_center_distance(abox, bbox)
        vertical_distance = abs(acy - bcy)
        size_ratio = self._bbox_area(abox) / max(1.0, self._bbox_area(bbox))
        iou = self._bbox_iou(abox, bbox)

        a_low = self._is_low_priority_overlay(a)
        b_low = self._is_low_priority_overlay(b)
        if a_low != b_low:
            center_thr = max(self._overlay_near_duplicate_min_center_px, self._overlay_near_duplicate_center_factor * avg_box_size)
            if (
                center_distance < center_thr
                and self._overlay_near_duplicate_size_ratio_min <= size_ratio <= self._overlay_near_duplicate_size_ratio_max
                and vertical_distance < (self._overlay_near_duplicate_vertical_factor * avg_height)
            ):
                return True, "near_duplicate_low_priority"

        if self._is_real_decision(a) and self._is_real_decision(b) and iou > 0.70:
            return True, "real_overlap"

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
        
        is_ps = (self.current_mode == "PERSON_SEARCH")
        cfg_debug = self.config.get("debug", {}) if hasattr(self.config, "get") else {}
        do_ps_trace = is_ps and bool(cfg_debug.get("person_search_log_renderer_filter", True))
        
        if do_ps_trace:
            self.logger.person_search_trace("PS_DEDUP_BEFORE", count=len(decisions), statuses=str([d.status for d in decisions]))
            
        for d in ordered:
            bbox = getattr(d, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            duplicate = False
            for k in kept:
                is_dup, reason = self._is_same_face_candidate(d, k)
                if is_dup:
                    duplicate = True
                    stats["duplicate_removed_count"] += 1
                    if reason == "same_track":
                        stats["duplicate_removed_same_track"] += 1
                    if self._status_norm(getattr(d, "status", "")) == "RAW_FALLBACK" or str(getattr(d, "_track_source", "")).lower() == "raw_fallback":
                        stats["duplicate_removed_raw"] += 1
                    if self._status_norm(getattr(d, "status", "")) == "PREDICTED":
                        stats["duplicate_removed_predicted"] += 1
                    if do_ps_trace:
                        self.logger.person_search_trace("PS_DEDUP_DROP", dropped_status=d.status, kept_status=k.status, reason=reason, dropped_track=d.track_id, kept_track=k.track_id)
                    break
            if not duplicate:
                kept.append(d)
                
        if do_ps_trace:
            self.logger.person_search_trace("PS_DEDUP_AFTER", count=len(kept), statuses=str([k.status for k in kept]))
            
        return kept, stats

    def _is_fallback_or_predicted(self, dec) -> tuple[bool, str]:
        status = self._status_norm(getattr(dec, "status", ""))
        source = str(getattr(dec, "_track_source", "")).lower()
        if status == "PREDICTED" or source == "predicted":
            return True, "predicted"
        if status in ("RAW_FALLBACK", "FACE") or source == "raw_fallback":
            return True, "raw_fallback"
        return False, ""

    def _is_near_real_for_fallback(self, candidate, real_decision) -> bool:
        cb = getattr(candidate, "bbox", None)
        rb = getattr(real_decision, "bbox", None)
        if not cb or not rb:
            return False
        cw = float(max(1, cb[2] - cb[0]))
        ch = float(max(1, cb[3] - cb[1]))
        rw = float(max(1, rb[2] - rb[0]))
        rh = float(max(1, rb[3] - rb[1]))
        avg_box = (cw + ch + rw + rh) / 4.0
        center_thr = max(self._overlay_fallback_suppress_min_center_px, self._overlay_fallback_suppress_center_factor * avg_box)
        return (
            self._bbox_center_distance(cb, rb) < center_thr
            or self._bbox_iou(cb, rb) > self._overlay_fallback_suppress_iou
        )

    def _suppress_near_duplicate_overlay(self, decisions: list) -> tuple[list, dict]:
        stats = {
            "near_duplicate_removed": 0,
            "fallback_suppressed_near_real": 0,
            "predicted_suppressed_near_real": 0,
            "orphan_fallback_drawn": 0,
            "fallback_hidden_min_hits": 0,
            "same_track_duplicates_removed": 0,
        }
        if not decisions or not self._overlay_near_duplicate_enabled:
            return decisions, stats

        ordered = sorted(decisions, key=self._decision_priority, reverse=True)
        kept: list = []
        
        is_ps = (self.current_mode == "PERSON_SEARCH")
        cfg_debug = self.config.get("debug", {}) if hasattr(self.config, "get") else {}
        do_ps_trace = is_ps and bool(cfg_debug.get("person_search_log_renderer_filter", True))
        
        for dec in ordered:
            duplicate = False
            for existing in kept:
                is_same, reason = self._is_same_face_candidate(dec, existing)
                if is_same:
                    duplicate = True
                    stats["near_duplicate_removed"] += 1
                    if reason == "same_track":
                        stats["same_track_duplicates_removed"] += 1
                    if do_ps_trace:
                        self.logger.person_search_trace("PS_NEAR_SUPPRESS_DROP", dropped_status=dec.status, kept_status=existing.status, reason=reason)
                    break
            if not duplicate:
                kept.append(dec)

        real_decisions = [d for d in kept if self._is_real_decision(d)]
        final: list = []
        for dec in kept:
            is_fallback, fallback_kind = self._is_fallback_or_predicted(dec)
            if not is_fallback:
                final.append(dec)
                continue

            near_real = any(self._is_near_real_for_fallback(dec, rd) for rd in real_decisions)
            if near_real:
                if fallback_kind == "predicted":
                    stats["predicted_suppressed_near_real"] += 1
                else:
                    stats["fallback_suppressed_near_real"] += 1
                continue

            if fallback_kind == "predicted":
                continue

            track_id = int(getattr(dec, "track_id", 0) or 0)
            camera_id = str(getattr(dec, "_camera_id", ""))
            key = (camera_id, track_id)
            self._fallback_hits[key] = self._fallback_hits.get(key, 0) + 1
            is_wanted = self._status_norm(getattr(dec, "status", "")) in ("WANTED", "CRIMINAL", "ARANIYOR")
            if (not is_wanted) and self._overlay_fallback_min_hits_to_draw > 1 and self._fallback_hits[key] < self._overlay_fallback_min_hits_to_draw:
                stats["fallback_hidden_min_hits"] += 1
                continue
            if self._overlay_draw_orphan_fallback_only and near_real:
                stats["fallback_suppressed_near_real"] += 1
                continue
            stats["orphan_fallback_drawn"] += 1
            final.append(dec)

        live_fallback_keys = {
            (str(getattr(d, "_camera_id", "")), int(getattr(d, "track_id", 0) or 0))
            for d in kept
            if self._is_fallback_or_predicted(d)[0]
        }
        for key in list(self._fallback_hits.keys()):
            if key not in live_fallback_keys:
                del self._fallback_hits[key]
        return final, stats

    # ──────────────────────────────────────────────────────────────────────
    def _refresh_cache(self):
        """GENERAL mod aday listesini ve legacy tuple cache'i DB'den yükler."""
        self._general_candidates = self.db.get_all_person_embeddings_for_general()
        self._cached_embeddings = [
            (int(c["person_id"]), c["embedding"]) for c in self._general_candidates
        ]
        self._general_hash_to_pids: dict[str, list[int]] = {}
        self._general_pid_to_hash: dict[int, str] = {}
        for c in self._general_candidates:
            h = c.get("embedding_hash", "")
            pid = int(c["person_id"])
            if h:
                self._general_hash_to_pids.setdefault(h, []).append(pid)
                self._general_pid_to_hash[pid] = h

        count = len(self._general_candidates)
        if count > 0:
            self.logger.info(f"[GENERAL_DB_CACHE_REFRESH] rows={count}")
        else:
            self.logger.warning("[GENERAL_DB_CACHE_EMPTY] reason=no_valid_embeddings")

        dup_groups = {h: pids for h, pids in self._general_hash_to_pids.items() if len(pids) > 1}
        if dup_groups:
            self.logger.warning(
                f"[GENERAL_DUPLICATE_HASH_GROUPS] groups={dup_groups}"
            )

    # ──────────────────────────────────────────────────────────────────────
    _STATUS_PRIORITY_FOR_RESOLVE: dict[str, int] = {
        "WANTED": 100,
        "CRIMINAL": 90,
        "CLEARED": 50,
        "CLEAN": 50,
    }

    def _resolve_duplicate_embedding_group(self, pids: list[int]) -> int:
        """
        Aynı embedding hash'e sahip birden fazla person_id varsa,
        status önceliğine göre (WANTED > CRIMINAL > CLEAN) ve
        en küçük person_id ile deterministik seçim yapar.
        """
        if not pids:
            return -1
        if len(pids) == 1:
            return pids[0]

        best_pid = pids[0]
        best_prio = -1
        for pid in pids:
            cand = next((c for c in self._general_candidates if int(c["person_id"]) == pid), None)
            if cand is None:
                continue
            status_u = str(cand.get("status", "")).upper()
            prio = self._STATUS_PRIORITY_FOR_RESOLVE.get(status_u, 0)
            if prio > best_prio or (prio == best_prio and pid < best_pid):
                best_prio = prio
                best_pid = pid

        self.logger.info(
            f"[GENERAL_DUPLICATE_EMBEDDING_RESOLVE] candidates={pids} "
            f"selected={best_pid} reason=status_priority_then_id"
        )
        return best_pid

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
        
        cfg_debug = self.config.get("debug", {}) if hasattr(self.config, "get") else {}
        is_ps = (self.current_mode == "PERSON_SEARCH")
        log_every = int(cfg_debug.get("person_search_log_every_n_frames", 10))
        do_ps_trace = is_ps and (self._frame_total % log_every == 0)

        if do_ps_trace:
            self.logger.person_search_trace(
                "PS_FRAME_BEGIN", 
                frame_total=self._frame_total, 
                camera_id=camera_id, 
                mode=self.current_mode, 
                target_person_ids=list(self.target_person_ids),
                target_id=self.target_person_id,
                has_target_embedding=(len(self._person_search_candidates) > 0),
            )
            self.logger.person_search_trace(
                "PS_FACE_DETECT", 
                camera_id=camera_id, 
                ran=face_detection_ran, 
                faces=len(faces), 
                used_cached_faces=used_cached_faces, 
                detect_every_n=self._detect_every_n
            )
            for idx, face in enumerate(faces):
                has_emb = face.embedding is not None
                emb_shape = str(face.embedding.shape) if has_emb else "None"
                emb_norm = f"{np.linalg.norm(face.embedding):.3f}" if has_emb else "0.000"
                self.logger.person_search_trace(
                    "PS_FACE_ITEM", 
                    camera_id=camera_id, 
                    face_idx=idx, 
                    bbox=str(face.bbox), 
                    det_score=f"{face.det_score:.2f}", 
                    has_embedding=has_emb, 
                    emb_shape=emb_shape, 
                    emb_norm=emb_norm
                )

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

        # Production overlay: yalnızca bu frame'de update alan track'ler çizilsin.
        # DeepSORT iç state korunur; burada sadece output/overlay akışı filtrelenir.
        allow_stale = bool(getattr(self.tracker, "render_stale_deepsort_tracks", True))
        max_tsu = int(getattr(self.tracker, "max_render_time_since_update", 0) or 0)
        if not allow_stale:
            tracks = [t for t in all_tracks if int(getattr(t, "time_since_update", 0) or 0) <= max_tsu]
        else:
            # Hayalet kutu önlemi: time_since_update > 3 → yüzü zaten görmüyoruz
            tracks = [t for t in all_tracks if int(getattr(t, "time_since_update", 0) or 0) <= 3]
        self.stats["active_tracks"] = len(tracks)
        
        if do_ps_trace:
            self.logger.person_search_trace(
                "PS_TRACKER_OUTPUT", 
                camera_id=camera_id, 
                raw_tracks=tracker_debug.get("tracker_raw_tracks", 0),
                rendered_tracks=len(tracks),
                confirmed=tracker_debug.get("tracker_confirmed_tracks", 0),
                tentative=tracker_debug.get("tracker_tentative_tracks", 0),
                fallback=tracker_debug.get("tracker_fallback_tracks", 0),
                rejected_no_embedding=tracker_debug.get("tracker_rejected_no_embedding", 0),
                tracker_input_faces=tracker_debug.get("tracker_input_faces", 0)
            )
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
            _log_gates = bool(cfg_debug.get("person_search_log_track_gates", True))

            if track.face_embedding is None:
                if _log_gates:
                    if is_ps:
                        self.logger.person_search_trace("PS_TRACK_GATE_SKIP", track_id=track.track_id, reason="no_face_embedding")
            elif confirmed_match is not None:
                if _log_gates:
                    if is_ps:
                        self.logger.person_search_trace("PS_TRACK_GATE_SKIP", track_id=track.track_id, reason="confirmed_match_already_exists", criminal_id=confirmed_match.criminal_id)
            else:
                if not track.velocity_ok:
                    self.stats["velocity_rejected"] += 1
                    if is_ps and _log_gates:
                        self.logger.person_search_trace("PS_TRACK_VELOCITY_NOTE", track_id=track.track_id, velocity_ok=False)

                if track.face_embedding is not None:
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

                    if is_ps and _log_gates:
                        self.logger.person_search_trace(
                            "PS_TRACK_GATE_PASS", track_id=track.track_id,
                            age=track.age, has_embedding=True,
                            velocity_ok=bool(track.velocity_ok),
                        )

                    self._last_db_check_frame[track_key] = self._frame_total
                    t_db = time.perf_counter()
                    match = self._match_track_against_candidates(
                        track.face_embedding,
                        track.track_id,
                        "PERSON_SEARCH" if is_ps else "GENERAL",
                    )
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
            
            if is_ps and do_ps_trace:
                self.logger.person_search_trace(
                    "DECISION_INPUT",
                    mode="PERSON_SEARCH",
                    track_id=track.track_id,
                    criminal_match_id=track.criminal_match.criminal_id if track.criminal_match else None,
                    criminal_info_found=(criminal_info is not None),
                    target_person_ids=list(self.target_person_ids),
                    target_id=self.target_person_id,
                )
                
            decision = self.decision.evaluate(track, criminal_info)
            
            if is_ps and do_ps_trace:
                self.logger.person_search_trace(
                    "DECISION_OUTPUT",
                    track_id=track.track_id,
                    status=decision.status,
                    label=decision.behavior_label,
                    criminal_id=decision.criminal_id,
                    confidence=float(decision.confidence),
                    color=str(decision.color)
                )
                
            # Unique overlay builder için kaynak metadatası
            decision._overlay_source = "decision"
            decision._camera_id = camera_id
            decision._track_source = getattr(track, "source", "")
            decision._updated_at = self._frame_total
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
                    time_since_update=0,
                ))
                results[-1]._overlay_source = "raw_fallback"
                results[-1]._camera_id = camera_id
                results[-1]._track_source = "raw_fallback"
                results[-1]._updated_at = self._frame_total
                used_boxes.append(face_bbox)
                fallback_decisions_count += 1
                orphan_raw_faces_drawn += 1
            raw_detection_fallback_used = fallback_decisions_count > 0

        unique_overlay_before = len(results)
        results, dedup_stats = self._deduplicate_decisions(results)
        overlay_before_near_suppression = len(results)
        results, near_stats = self._suppress_near_duplicate_overlay(results)
        overlay_after_near_suppression = len(results)
        overlay_dedup_removed = int(dedup_stats.get("duplicate_removed_count", 0)) + int(near_stats.get("near_duplicate_removed", 0))
        unique_overlay_after = len(results)
        raw_fallback_suppressed_by_track = int(near_stats.get("fallback_suppressed_near_real", 0))
        predicted_suppressed_by_real_detection += int(near_stats.get("predicted_suppressed_near_real", 0))

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
            "tracker_stale_tracks_seen": int(tracker_debug.get("tracker_stale_tracks_seen", 0)),
            "tracker_stale_tracks_suppressed": int(tracker_debug.get("tracker_stale_tracks_suppressed", 0)),
            "tracker_renderable_tracks": int(tracker_debug.get("tracker_renderable_tracks", 0)),
            "tracker_time_since_update_gt0": int(tracker_debug.get("tracker_time_since_update_gt0", 0)),
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
            "overlay_before_near_suppression": int(overlay_before_near_suppression),
            "overlay_after_near_suppression": int(overlay_after_near_suppression),
            "near_duplicate_removed": int(near_stats.get("near_duplicate_removed", 0)),
            "fallback_suppressed_near_real": int(near_stats.get("fallback_suppressed_near_real", 0)),
            "predicted_suppressed_near_real": int(near_stats.get("predicted_suppressed_near_real", 0)),
            "same_track_duplicates_removed": int(near_stats.get("same_track_duplicates_removed", 0)),
            "orphan_fallback_drawn": int(near_stats.get("orphan_fallback_drawn", 0)),
            "fallback_hidden_min_hits": int(near_stats.get("fallback_hidden_min_hits", 0)),
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
            "real_faces_drawn_count": len([r for r in results if self._status_norm(getattr(r, "status", "")) in ("CLEAN", "TEMIZ", "UNKNOWN", "TRACKING", "TENTATIVE", "FACE", "WANTED", "CRIMINAL", "ARANIYOR", "HEDEF BULUNDU")]),
        }

        # GENERAL mode periodic summary
        if self.current_mode == "GENERAL":
            now = time.perf_counter()
            if (now - self._general_summary_ts) >= self._general_summary_interval:
                statuses = [self._status_norm(getattr(r, "status", "")) for r in results]
                active_cm = {
                    int(t.track_id): int(t.criminal_match.criminal_id)
                    for t in tracks
                    if getattr(t, "criminal_match", None) is not None
                }
                _uids = sorted({int(c["person_id"]) for c in self._general_candidates})
                dup_groups = {h: pids for h, pids in self._general_hash_to_pids.items() if len(pids) > 1}
                self.logger.info(
                    f"[GENERAL_SUMMARY] faces={len(faces)} tracks={len(tracks)} "
                    f"tracks_with_emb={sum(1 for t in tracks if t.face_embedding is not None)} "
                    f"general_candidates={len(self._general_candidates)} unique_db_persons={len(_uids)} "
                    f"decisions={statuses} active_matches={active_cm or {}} "
                    f"matches={self.stats['total_matches']}"
                )
                if self._general_match_counts:
                    self.logger.info(
                        f"[GENERAL_MATCH_DISTRIBUTION] matches_by_person={dict(self._general_match_counts)}"
                    )
                if dup_groups:
                    self.logger.info(
                        f"[GENERAL_DUPLICATE_HASH_GROUPS] groups={dup_groups}"
                    )
                self._general_summary_ts = now

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
        is_ps = (self.current_mode == "PERSON_SEARCH")
        
        if is_ps:
            self.logger.person_search_trace("PS_VOTE_REGISTER_BEGIN", track_id=track.track_id, target_id=cid, score=float(match.confidence))

        if cid not in votes:
            votes[cid] = {"count": 0, "best_conf": 0.0, "last_frame": 0}

        info = votes[cid]

        # GENERAL modda frame gap toleransı daha geniş tutulur çünkü
        # detect_every_n=4 frame'de 1 yüz çıkarılır — ardışıklık kopmamalı.
        max_gap = (self._detect_every_n * (self._vote_threshold + 1)) + 4
        if info["last_frame"] > 0 and (current_frame - info["last_frame"]) > max_gap:
            info["count"] = 1
        else:
            info["count"] += 1

        info["best_conf"] = max(info["best_conf"], match.confidence)
        info["last_frame"] = current_frame

        if is_ps:
            self.logger.person_search_trace(
                "PS_VOTE_STATE",
                track_id=track.track_id,
                criminal_id=cid,
                count=info["count"],
                threshold=self._vote_threshold,
                best_conf=float(info["best_conf"]),
            )
        else:
            self.logger.info(
                f"[GENERAL_VOTE_REGISTER] track_id={track.track_id} person_id={cid} "
                f"score={match.confidence:.4f} count={info['count']}/{self._vote_threshold}"
            )

        if info["count"] >= self._vote_threshold:
            if is_ps:
                self.logger.info(
                    f"[PERSON_SEARCH_DIRECT_MATCH_SET] track_id={track.track_id} target_id={cid} score={float(info['best_conf']):.3f}"
                )
                self.logger.person_search_trace(
                    "PS_VOTE_CONFIRMED",
                    track_id=track.track_id,
                    criminal_id=cid,
                    count=info["count"],
                    best_conf=float(info["best_conf"]),
                )
            else:
                self.logger.info(
                    f"[GENERAL_VOTE_CONFIRMED] track_id={track.track_id} person_id={cid} "
                    f"confidence={float(info['best_conf']):.4f} votes={info['count']}"
                )
                self.logger.info(
                    f"[GENERAL_MATCH_SET] camera_id={camera_id} track_id={track.track_id} "
                    f"person_id={cid} confidence={float(info['best_conf']):.4f}"
                )

            self.stats["total_matches"] += 1
            criminal_info = self.db.get_criminal_info(cid)
            status = criminal_info.get("status", "") if criminal_info else ""
            name = criminal_info.get("name", "?") if criminal_info else "?"

            if not is_ps:
                if criminal_info:
                    self.logger.info(
                        f"[GENERAL_CRIMINAL_INFO_LOAD] criminal_id={cid} found=true "
                        f"name={name} status={status}"
                    )
                else:
                    self.logger.warning(f"[GENERAL_CRIMINAL_INFO_MISSING] criminal_id={cid}")

            if is_ps:
                self.logger.log(
                    EventType.SEARCH_COMPLETED,
                    f"HEDEF KISI BULUNDU: {name}",
                    camera_id=camera_id,
                    confidence=f"{match.confidence:.2f}",
                    track_id=track.track_id
                )
            elif status == "WANTED":
                self.logger.log(
                    EventType.WANTED_FOUND,
                    f"ARANAN KISI TESPIT: {name} ({self._vote_threshold} ardışık oy)",
                    camera_id=camera_id,
                    confidence=f"{match.confidence:.2f}",
                    track_id=track.track_id
                )
            elif status == "CRIMINAL":
                self.logger.log(
                    EventType.CRIMINAL_DETECTED,
                    f"Sabıkalı tespit: {name} ({self._vote_threshold} ardışık oy)",
                    camera_id=camera_id,
                    confidence=f"{match.confidence:.2f}",
                    track_id=track.track_id
                )

            elif status in ("CLEARED", "CLEAN"):
                self.logger.info(f"[DB_MATCH_CLEAN] person_id={cid} name={name} confidence={match.confidence:.2f}")
            else:
                self.logger.info(f"[DB_MATCH_UNKNOWN_STATUS] person_id={cid} name={name} status={status} confidence={match.confidence:.2f}")

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
        is_ps = (self.current_mode == "PERSON_SEARCH")
        track_id = track_key[1] if len(track_key) > 1 else None
        
        if not votes:
            return None

        for cid, info in votes.items():
            if info["count"] >= self._vote_threshold:
                if is_ps:
                    self.logger.person_search_trace("PS_CONFIRMED_MATCH_GET", track_id=track_id, result=True, criminal_id=cid, confidence=float(info["best_conf"]))
                return MatchResult(criminal_id=cid, confidence=info["best_conf"])

        if is_ps:
            # Sadece ID ve count map'i gönderelim
            vote_summary = {str(k): v["count"] for k,v in votes.items()}
            self.logger.person_search_trace("PS_CONFIRMED_MATCH_GET", track_id=track_id, result=False, votes=str(vote_summary))
            
        return None

    def _match_track_against_candidates(
        self,
        embedding: np.ndarray | None,
        track_id: int | None,
        mode: str,
    ) -> MatchResult | None:
        """
        GENERAL ve PERSON_SEARCH için ortak eşleştirici.
        Adaylar embedding sahibi tüm satırlardır; skorlar person_id bazında toplanır (duplicate satırlar birleşir).
        """
        tid = track_id if track_id is not None else -1
        mode_u = str(mode or "GENERAL").upper()

        if embedding is None:
            self.logger.info(
                f"[COMMON_MATCH_MISS] mode={mode_u} track_id={tid} reason=no_embedding"
            )
            if mode_u == "PERSON_SEARCH":
                self.logger.person_search_trace("PS_SEARCH_FAIL", reason="live_embedding_none")
            return None

        if mode_u == "PERSON_SEARCH":
            candidates = self._person_search_candidates
            if not candidates:
                self.logger.person_search_trace("PS_SEARCH_FAIL", reason="targets_embedding_none")
                return None
            threshold = self._person_search_cosine_threshold
            use_ambiguous = False
        else:
            candidates = self._general_candidates
            threshold = self._general_cosine_threshold
            use_ambiguous = True

        n = len(candidates)
        if n == 0:
            self.logger.info(
                f"[COMMON_MATCH_MISS] mode={mode_u} track_id={tid} reason=empty_candidates"
            )
            return None

        self.logger.info(f"[COMMON_MATCH_ENTER] mode={mode_u} track_id={tid} candidates={n}")

        best_by_pid: dict[int, float] = {}
        for c in candidates:
            pid = int(c["person_id"])
            emb = c.get("embedding")
            if emb is None:
                continue
            try:
                score = self.face_analyzer.compare(embedding, emb)
            except Exception as e:
                if mode_u == "PERSON_SEARCH":
                    self.logger.person_search_trace(
                        "PS_SEARCH_FAIL",
                        reason="compare_exception",
                        candidate_id=pid,
                        error=str(e),
                    )
                continue
            self.logger.info(
                f"[COMMON_MATCH_SCORE] mode={mode_u} track_id={tid} person_id={pid} score={score:.4f}"
            )
            prev = best_by_pid.get(pid, -1.0)
            if score > prev:
                best_by_pid[pid] = score

        if not best_by_pid:
            if mode_u == "PERSON_SEARCH":
                self.logger.person_search_trace("PS_SEARCH_FAIL", reason="compare_all_failed")
            else:
                self.logger.info(
                    f"[COMMON_MATCH_MISS] mode={mode_u} track_id={tid} reason=no_scores"
                )
            return None

        ranked = sorted(best_by_pid.items(), key=lambda x: x[1], reverse=True)
        best_id = int(ranked[0][0])
        best_score = float(ranked[0][1])
        second_id = int(ranked[1][0]) if len(ranked) > 1 else None
        second_score = float(ranked[1][1]) if second_id is not None else -1.0

        self.logger.info(
            f"[COMMON_MATCH_BEST] mode={mode_u} track_id={tid} best_id={best_id} "
            f"best_score={best_score:.4f} second_id={second_id} second_score={second_score:.4f} "
            f"threshold={threshold:.4f}"
        )

        if best_score < threshold:
            if mode_u == "PERSON_SEARCH":
                self.logger.info(
                    f"[PERSON_SEARCH_NO_MATCH] track_id={tid} best_target_id={best_id} "
                    f"score={best_score:.4f}"
                )
            self.logger.info(
                f"[COMMON_MATCH_MISS] mode={mode_u} track_id={tid} "
                f"best_score={best_score:.4f} threshold={threshold:.4f}"
            )
            return None

        margin = self._general_ambiguous_margin if use_ambiguous else 0.0
        final_id = best_id

        if use_ambiguous and second_id is not None and margin > 0:
            competitor = second_score >= threshold
            gap_ok = (best_score - second_score) >= margin
            best_hash = self._general_pid_to_hash.get(best_id, "")
            second_hash = self._general_pid_to_hash.get(second_id, "")
            same_hash = bool(best_hash and best_hash == second_hash)

            self.logger.info(
                f"[GENERAL_AMBIGUOUS_CHECK] best_id={best_id} second_id={second_id} "
                f"best_hash={best_hash[:8] if best_hash else 'N/A'} "
                f"second_hash={second_hash[:8] if second_hash else 'N/A'} "
                f"same_hash={str(same_hash).lower()} "
                f"best_score={best_score:.4f} second_score={second_score:.4f}"
            )

            if same_hash and competitor:
                dup_pids = self._general_hash_to_pids.get(best_hash, [best_id])
                final_id = self._resolve_duplicate_embedding_group(dup_pids)
                self.logger.info(
                    f"[GENERAL_AMBIGUOUS_ACCEPT] reason=same_embedding_hash "
                    f"resolved_id={final_id}"
                )
            elif competitor and not gap_ok:
                self.logger.info(
                    f"[GENERAL_AMBIGUOUS_REJECT] reason=different_identity_margin_low "
                    f"best_id={best_id} second_id={second_id} "
                    f"gap={best_score - second_score:.4f} margin={margin:.4f}"
                )
                return None
            elif not competitor:
                self.logger.info(
                    f"[GENERAL_AMBIGUOUS_ACCEPT] reason=second_below_threshold "
                    f"second_id={second_id} second_score={second_score:.4f}"
                )
            elif gap_ok:
                self.logger.info(
                    f"[GENERAL_AMBIGUOUS_ACCEPT] reason=sufficient_gap "
                    f"gap={best_score - second_score:.4f}"
                )
        elif use_ambiguous:
            self.logger.info("[GENERAL_AMBIGUOUS_ACCEPT] reason=no_competitor_or_single_identity")

        if mode_u == "GENERAL":
            self._general_match_counts[final_id] = self._general_match_counts.get(final_id, 0) + 1

        self.logger.info(
            f"[COMMON_MATCH_HIT] mode={mode_u} track_id={tid} person_id={final_id} score={best_score:.4f}"
        )
        if mode_u == "PERSON_SEARCH":
            self.logger.info(
                f"[PERSON_SEARCH_MATCH] track_id={tid} target_id={best_id} score={best_score:.4f}"
            )
        return MatchResult(criminal_id=final_id, confidence=best_score)

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
        for k in list(self._fallback_hits.keys()):
            if k[0] == camera_id:
                del self._fallback_hits[k]

        # TrackRegistry ve Tracker
        self.track_registry.clear_camera(camera_id)
        self.tracker.reset(camera_id)
