"""
SKYWATCH — LocalBBoxTracker
YOLO/tracker inference frame'leri arasında bbox'ları KLT optical flow ile
yüksek display FPS hızında günceller.
"""

from __future__ import annotations

import time
import copy
import math
from dataclasses import is_dataclass, replace
import cv2
import numpy as np


class LocalBBoxTracker:
    def __init__(self, config: dict, logger=None):
        self.logger = logger
        self.enabled = bool(config.get("local_tracker_enabled", True))
        self.method = str(config.get("local_tracker_method", "klt")).lower()
        self.max_points = int(config.get("local_tracker_max_points", 40))
        self.quality = float(config.get("local_tracker_quality", 0.01))
        self.min_distance = float(config.get("local_tracker_min_distance", 4))
        self.min_points = int(config.get("local_tracker_min_points", 6))
        self.max_lost_frames = int(config.get("local_tracker_max_lost_frames", 3))
        self.max_shift_px = float(config.get("local_tracker_max_shift_px", 80.0))
        self.reinit_on_detection = bool(config.get("local_tracker_reinit_on_detection", True))

        # camera_id -> track_id -> state
        self._states: dict[str, dict[int, dict]] = {}
        self._metrics: dict[str, dict] = {}

    def reset_camera(self, camera_id: str):
        self._states.pop(camera_id, None)
        self._metrics.pop(camera_id, None)

    def _gray(self, frame: np.ndarray) -> np.ndarray | None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _clamp_bbox(self, bbox: list[int], w: int, h: int) -> list[int] | None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _init_points(self, gray: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        h, w = gray.shape[:2]
        clamped = self._clamp_bbox(bbox, w, h)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        roi = gray[y1:y2, x1:x2]
        if roi is None or roi.size == 0:
            return None
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.max_points,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            blockSize=3,
        )
        if pts is None or len(pts) == 0:
            return None
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts.astype(np.float32)

    def _ensure_metric(self, camera_id: str):
        if camera_id not in self._metrics:
            self._metrics[camera_id] = {
                "local_tracker_enabled": bool(self.enabled),
                "local_tracker_active_tracks": 0,
                "local_tracker_updated_count": 0,
                "local_tracker_failed_count": 0,
                "local_tracker_avg_shift_px": 0.0,
                "local_tracker_max_shift_px": 0.0,
                "local_tracker_reinit_count": 0,
                "local_tracker_low_points_count": 0,
                "local_tracker_display_failed": False,
                "local_tracker_input_decisions": 0,
                "local_tracker_output_decisions": 0,
                "renderer_input_decisions": 0,
            }

    def _with_bbox(self, decision, bbox: list[int]):
        try:
            nd = copy.copy(decision)
            nd.bbox = bbox
            return nd
        except Exception:
            pass
        try:
            if is_dataclass(decision):
                return replace(decision, bbox=bbox)
        except Exception:
            pass
        return decision

    def update_from_detections(self, camera_id: str, frame, decisions_or_tracks):
        self._ensure_metric(camera_id)
        if not self.enabled or self.method != "klt":
            return self.get_camera_metrics(camera_id)
        gray = self._gray(frame)
        if gray is None:
            return self.get_camera_metrics(camera_id)
        cam_states = self._states.setdefault(camera_id, {})
        seen_ids: set[int] = set()
        reinit_count = 0
        for item in list(decisions_or_tracks or []):
            tid = int(getattr(item, "track_id", 0) or 0)
            bbox = getattr(item, "bbox", None)
            status = str(getattr(item, "status", "")).upper()
            if tid == 0 or bbox is None or len(bbox) != 4:
                continue
            if status == "PREDICTED":
                continue
            points = self._init_points(gray, [int(v) for v in bbox])
            if points is None:
                continue
            cam_states[tid] = {
                "bbox": [int(v) for v in bbox],
                "prev_gray": gray,
                "points": points,
                "last_ts": time.time(),
                "confidence": float(getattr(item, "confidence", 0.0) or 0.0),
                "lost_frames": 0,
            }
            seen_ids.add(tid)
            reinit_count += 1
        # Bu detection turunda görünmeyen çok eski local state'leri düşür
        for tid in list(cam_states.keys()):
            if seen_ids and tid not in seen_ids:
                st = cam_states[tid]
                st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                if st["lost_frames"] > self.max_lost_frames:
                    del cam_states[tid]
        self._metrics[camera_id]["local_tracker_reinit_count"] = reinit_count
        self._metrics[camera_id]["local_tracker_active_tracks"] = len(cam_states)
        return self.get_camera_metrics(camera_id)

    def predict_on_frame(self, camera_id: str, frame, decisions):
        self._ensure_metric(camera_id)
        if not self.enabled or self.method != "klt":
            return list(decisions or [])
        gray = self._gray(frame)
        if gray is None:
            return list(decisions or [])
        cam_states = self._states.setdefault(camera_id, {})
        out = []
        input_count = len(list(decisions or []))
        updated_count = 0
        failed_count = 0
        low_points_count = 0
        shifts: list[float] = []

        for dec in list(decisions or []):
            try:
                nd = copy.copy(dec)
                tid = int(getattr(nd, "track_id", 0) or 0)
                st = cam_states.get(tid)
                if tid == 0 or st is None:
                    out.append(nd)
                    continue
                prev_gray = st.get("prev_gray")
                pts = st.get("points")
                if prev_gray is None or pts is None or len(pts) < self.min_points:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    low_points_count += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue
                new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
                if new_pts is None or status is None:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    failed_count += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue
                good_new = new_pts[status.flatten() == 1]
                good_old = pts[status.flatten() == 1]
                if good_new is None or good_old is None or len(good_new) < self.min_points:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    low_points_count += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue
                delta = good_new - good_old
                dx = float(np.median(delta[:, 0]))
                dy = float(np.median(delta[:, 1]))
                shift = math.hypot(dx, dy)
                if not np.isfinite(dx) or not np.isfinite(dy) or shift > self.max_shift_px:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    failed_count += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue

                h, w = gray.shape[:2]
                bx1, by1, bx2, by2 = [int(v) for v in st.get("bbox", nd.bbox)]
                moved = [int(round(bx1 + dx)), int(round(by1 + dy)), int(round(bx2 + dx)), int(round(by2 + dy))]
                clamped = self._clamp_bbox(moved, w, h)
                if clamped is None:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    failed_count += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue

                nd = self._with_bbox(nd, clamped)
                st["bbox"] = clamped
                st["prev_gray"] = gray
                st["points"] = good_new.reshape(-1, 1, 2).astype(np.float32)
                st["last_ts"] = time.time()
                st["lost_frames"] = 0
                updated_count += 1
                shifts.append(shift)
                out.append(nd)
            except Exception:
                failed_count += 1
                out.append(dec)
                continue

        avg_shift = float(sum(shifts) / len(shifts)) if shifts else 0.0
        max_shift = float(max(shifts)) if shifts else 0.0
        m = self._metrics[camera_id]
        m["local_tracker_enabled"] = bool(self.enabled)
        m["local_tracker_active_tracks"] = len(cam_states)
        m["local_tracker_updated_count"] = updated_count
        m["local_tracker_failed_count"] = failed_count
        m["local_tracker_avg_shift_px"] = round(avg_shift, 3)
        m["local_tracker_max_shift_px"] = round(max_shift, 3)
        m["local_tracker_low_points_count"] = low_points_count
        m["local_tracker_display_failed"] = False
        m["local_tracker_input_decisions"] = input_count
        m["local_tracker_output_decisions"] = len(out)
        m["renderer_input_decisions"] = len(out)
        return out

    def get_camera_metrics(self, camera_id: str) -> dict:
        self._ensure_metric(camera_id)
        return dict(self._metrics.get(camera_id, {}))

