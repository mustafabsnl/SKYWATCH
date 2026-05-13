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
        self.roi_expand_ratio = float(config.get("local_tracker_roi_expand_ratio", 0.0) or 0.0)
        self.reinit_min_iou_change = float(config.get("local_tracker_reinit_min_iou_change", 0.0) or 0.0)
        self.use_grid_fallback = bool(config.get("local_tracker_use_grid_fallback", False))
        self.grid_size = int(config.get("local_tracker_grid_size", 3) or 3)

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
        expand = max(0.0, float(self.roi_expand_ratio))
        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * expand)
        my = int(bh * expand)
        ex1 = max(0, x1 - mx)
        ey1 = max(0, y1 - my)
        ex2 = min(w, x2 + mx)
        ey2 = min(h, y2 + my)
        roi = gray[ey1:ey2, ex1:ex2]
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
            if not self.use_grid_fallback:
                return None
            # Grid fallback: ROI içinde düzenli noktalar üret (kenarlardan uzak).
            gs = max(2, int(self.grid_size))
            margin = 0.18
            xs = np.linspace(ex1 + (ex2 - ex1) * margin, ex2 - (ex2 - ex1) * margin, gs)
            ys = np.linspace(ey1 + (ey2 - ey1) * margin, ey2 - (ey2 - ey1) * margin, gs)
            grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
            if grid.size == 0:
                return None
            if len(grid) > self.max_points:
                grid = grid[: self.max_points]
            return grid.reshape(-1, 1, 2).astype(np.float32)
        # goodFeaturesToTrack ROI koordinatında döner; full-frame'e kaydır
        pts[:, 0, 0] += float(ex1)
        pts[:, 0, 1] += float(ey1)
        return pts.astype(np.float32)

    def _bbox_iou(self, a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _ensure_metric(self, camera_id: str):
        if camera_id not in self._metrics:
            self._metrics[camera_id] = {
                "local_tracker_enabled": bool(self.enabled),
                "local_tracker_active_tracks": 0,
                "local_tracker_updated_count": 0,
                "local_tracker_failed_count": 0,
                "local_tracker_success_count": 0,
                "local_tracker_fail_no_state": 0,
                "local_tracker_fail_no_points": 0,
                "local_tracker_fail_low_points": 0,
                "local_tracker_fail_flow_none": 0,
                "local_tracker_fail_large_shift": 0,
                "local_tracker_fail_exception": 0,
                "local_tracker_avg_shift_px": 0.0,
                "local_tracker_max_shift_px": 0.0,
                "local_tracker_reinit_count": 0,
                "local_tracker_snap_no_reinit_count": 0,
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
        snap_no_reinit_count = 0
        for item in list(decisions_or_tracks or []):
            tid = int(getattr(item, "track_id", 0) or 0)
            bbox = getattr(item, "bbox", None)
            status = str(getattr(item, "status", "")).upper()
            if tid == 0 or bbox is None or len(bbox) != 4:
                continue
            if status == "PREDICTED":
                continue
            ibox = [int(v) for v in bbox]
            existing = cam_states.get(tid)
            should_reinit = bool(self.reinit_on_detection) or existing is None
            if existing is not None and should_reinit and self.reinit_min_iou_change > 0:
                prev_box = existing.get("bbox")
                if prev_box is not None and len(prev_box) == 4:
                    iou = self._bbox_iou(prev_box, ibox)
                    iou_change = 1.0 - float(iou)
                    if iou_change < float(self.reinit_min_iou_change):
                        should_reinit = False
            if not should_reinit and existing is not None:
                # Sadece bbox snap + prev_gray güncelle; pointleri resetleme
                existing["bbox"] = ibox
                existing["last_ts"] = time.time()
                existing["confidence"] = float(getattr(item, "confidence", 0.0) or 0.0)
                existing["lost_frames"] = 0
                if existing.get("prev_gray") is None:
                    existing["prev_gray"] = gray.copy()
                snap_no_reinit_count += 1
                seen_ids.add(tid)
                continue
            points = self._init_points(gray, ibox)
            if points is None:
                continue
            cam_states[tid] = {
                "bbox": ibox,
                "prev_gray": gray.copy(),
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
        self._metrics[camera_id]["local_tracker_snap_no_reinit_count"] = snap_no_reinit_count
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
        input_decisions = list(decisions or [])
        input_count = len(input_decisions)
        updated_count = 0
        failed_count = 0
        low_points_count = 0
        shifts: list[float] = []
        fail_no_state = 0
        fail_no_points = 0
        fail_low_points = 0
        fail_flow_none = 0
        fail_large_shift = 0
        fail_exception = 0
        success_count = 0

        best_by_track: dict[int, object] = {}
        for dec in input_decisions:
            try:
                nd = copy.copy(dec)
                tid = int(getattr(nd, "track_id", 0) or 0)
                st = cam_states.get(tid)
                if tid == 0 or st is None:
                    if tid != 0:
                        fail_no_state += 1
                    out.append(nd)
                    continue
                prev_gray = st.get("prev_gray")
                pts = st.get("points")
                if prev_gray is None or pts is None or len(pts) < self.min_points:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    low_points_count += 1
                    if pts is None or len(pts) == 0:
                        fail_no_points += 1
                    else:
                        fail_low_points += 1
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
                    fail_flow_none += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue
                good_new = new_pts[status.flatten() == 1].reshape(-1, 2)
                good_old = pts[status.flatten() == 1].reshape(-1, 2)
                if len(good_new) < self.min_points or len(good_old) < self.min_points:
                    st["lost_frames"] = int(st.get("lost_frames", 0)) + 1
                    low_points_count += 1
                    fail_low_points += 1
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
                    fail_large_shift += 1
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
                    fail_large_shift += 1
                    if st["lost_frames"] > self.max_lost_frames:
                        cam_states.pop(tid, None)
                    out.append(nd)
                    continue

                nd = self._with_bbox(nd, clamped)
                st["bbox"] = clamped
                st["prev_gray"] = gray.copy()
                st["points"] = good_new.reshape(-1, 1, 2).astype(np.float32)
                st["last_ts"] = time.time()
                st["lost_frames"] = 0
                updated_count += 1
                success_count += 1
                shifts.append(shift)
                out.append(nd)
            except Exception:
                failed_count += 1
                fail_exception += 1
                out.append(dec)
                continue

        deduped_out: list = []
        for item in out:
            tid = int(getattr(item, "track_id", 0) or 0)
            if tid == 0:
                deduped_out.append(item)
                continue
            current = best_by_track.get(tid)
            if current is None:
                best_by_track[tid] = item
                continue
            cur_conf = float(getattr(current, "confidence", 0.0) or 0.0)
            new_conf = float(getattr(item, "confidence", 0.0) or 0.0)
            cur_area = 0
            new_area = 0
            cb = getattr(current, "bbox", None)
            nb = getattr(item, "bbox", None)
            if cb is not None and len(cb) == 4:
                cur_area = max(1, cb[2] - cb[0]) * max(1, cb[3] - cb[1])
            if nb is not None and len(nb) == 4:
                new_area = max(1, nb[2] - nb[0]) * max(1, nb[3] - nb[1])
            if (new_conf, new_area) > (cur_conf, cur_area):
                best_by_track[tid] = item
        deduped_out.extend(best_by_track.values())

        avg_shift = float(sum(shifts) / len(shifts)) if shifts else 0.0
        max_shift = float(max(shifts)) if shifts else 0.0
        m = self._metrics[camera_id]
        m["local_tracker_enabled"] = bool(self.enabled)
        m["local_tracker_active_tracks"] = len(cam_states)
        m["local_tracker_updated_count"] = updated_count
        m["local_tracker_failed_count"] = failed_count
        m["local_tracker_success_count"] = success_count
        m["local_tracker_fail_no_state"] = fail_no_state
        m["local_tracker_fail_no_points"] = fail_no_points
        m["local_tracker_fail_low_points"] = fail_low_points
        m["local_tracker_fail_flow_none"] = fail_flow_none
        m["local_tracker_fail_large_shift"] = fail_large_shift
        m["local_tracker_fail_exception"] = fail_exception
        m["local_tracker_avg_shift_px"] = round(avg_shift, 3)
        m["local_tracker_max_shift_px"] = round(max_shift, 3)
        m["local_tracker_low_points_count"] = low_points_count
        m["local_tracker_display_failed"] = False
        m["local_tracker_input_decisions"] = input_count
        m["local_tracker_output_decisions"] = len(deduped_out)
        m["renderer_input_decisions"] = len(deduped_out)
        return deduped_out

    def get_camera_metrics(self, camera_id: str) -> dict:
        self._ensure_metric(camera_id)
        return dict(self._metrics.get(camera_id, {}))

