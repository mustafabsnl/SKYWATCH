"""
SKYWATCH — OverlayRenderer
Frame uzerine tespit kutulari ve bilgi metinlerini cizer.
"""

import cv2
import math
import time
import numpy as np
from core.models import DecisionResult

# Pipeline ile aynı hayalet eşiği: tsu bunun üzerindeki sonuçlar çizilmez
_MAX_DRAW_TIME_SINCE_UPDATE = 3


# Durum → BGR renk (anahtarlar _color_key ile uyumlu)
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "CLEAN":      (255, 120, 30),    # Güçlü mavi ton (BGR)
    "CRIMINAL":   (0, 140, 255),    # Turuncu
    "WANTED":     (0, 0, 255),       # Kırmızı
    "SUSPICIOUS": (200, 0, 200),     # Mor
    "UNKNOWN":    (145, 145, 145),   # Gri
    "FACE":       (220, 200, 80),
    "TENTATIVE":  (170, 170, 0),
    "TRACKING":   (170, 170, 0),
    "PREDICTED":  (140, 180, 255),
    "RAW_FALLBACK": (220, 200, 80),
    "HEDEF BULUNDU": (0, 255, 255),  # Sarı (BGR)
    "TARGET_FOUND": (0, 255, 255),
}

STATUS_LABELS: dict[str, str] = {
    "CLEAN":      "TEMIZ",
    "CRIMINAL":   "! SABIKALI !",
    "WANTED":     "!! ARANIYOR !!",
    "SUSPICIOUS": "SUPHELI",
    "UNKNOWN":    "?",
    "FACE":       "FACE",
    "TENTATIVE":  "TRACKING",
    "TRACKING":   "TRACKING",
    "PREDICTED":  "TRACKING",   # Production: PREDICTED yerine "TRACKING" göster
    "RAW_FALLBACK": "FACE",
    "HEDEF BULUNDU": "!!! HEDEF BULUNDU !!!",
    "TARGET_FOUND": "!!! HEDEF BULUNDU !!!",
}

# Pipeline._STATUS_PRIORITY ile uyumlu kalmalı.
STATUS_PRIORITY: dict[str, int] = {
    "HEDEF BULUNDU": 120,
    "TARGET_FOUND": 120,
    "WANTED":     110,
    "CRIMINAL":   105,
    "ARANIYOR":   110,
    "CLEAN":       90,
    "TEMIZ":       90,
    "SUSPICIOUS":  85,
    "UNKNOWN":     80,
    "TRACKING":    70,
    "TENTATIVE":   60,
    "FACE":        50,
    "RAW_FALLBACK": 30,
    "PREDICTED":   10,
}

class OverlayRenderer:
    """Frame'e tespit sonuclarini cizer."""

    _LABEL_BG = (24, 24, 26)
    _LABEL_TEXT = (245, 245, 248)

    def __init__(
        self,
        font_scale: float = 0.55,
        thickness: int = 1,
        dedup_iou: float = 0.45,
        draw_predicted_tracks: bool = False,
        fallback_suppress_center_factor: float = 0.9,
        fallback_suppress_min_center_px: float = 40.0,
        fallback_suppress_iou: float = 0.20,
        draw_corner_stats: bool = False,
        trace_logger=None,
        render_diag_interval_sec: float = 1.25,
    ):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self._frame_count = 0
        self._draw_corner_stats = bool(draw_corner_stats)
        self._trace_logger = trace_logger
        self._render_diag_interval_sec = float(render_diag_interval_sec)
        self._last_render_diag_ts = 0.0
        # Pipeline zaten dedup uyguluyor; renderer defansif olarak aynı IoU eşiğini kullanır.
        self._dedup_iou = float(dedup_iou)
        self._draw_predicted_tracks = bool(draw_predicted_tracks)
        self._fallback_suppress_center_factor = float(fallback_suppress_center_factor)
        self._fallback_suppress_min_center_px = float(fallback_suppress_min_center_px)
        self._fallback_suppress_iou = float(fallback_suppress_iou)

    def draw(self, frame: np.ndarray,
             results: list[DecisionResult],
             stats: dict,
             criminal_names: dict | None = None,
             *,
             camera_id: str | None = None,
             trace_logger=None) -> np.ndarray:
        """Kararlari frame uzerine cizer."""
        log = trace_logger if trace_logger is not None else self._trace_logger
        display = frame.copy()
        fh, fw = display.shape[:2]
        self._frame_count += 1
        blink = (self._frame_count // 8) % 2 == 0

        incoming = list(results or [])
        now_m = time.monotonic()
        elapsed = now_m - self._last_render_diag_ts
        verbose_diag = False
        if log is not None:
            has_in = len(incoming) > 0
            if has_in and elapsed >= self._render_diag_interval_sec:
                verbose_diag = True
                self._last_render_diag_ts = now_m
            elif not has_in and elapsed >= max(3.0, self._render_diag_interval_sec * 2):
                verbose_diag = True
                self._last_render_diag_ts = now_m

        if verbose_diag and log is not None:
            log.info(
                f"[RENDER_INPUT] camera_id={camera_id or '?'} "
                f"frame_shape={(fh, fw)} decisions_count={len(incoming)} "
                f"frame_count={self._frame_count}"
            )
            if len(incoming) == 0:
                log.warning(
                    f"[RENDER_INPUT] decisions_count=0 — Renderer'a overlay kararı gelmedi "
                    f"(camera_id={camera_id or '?'})."
                )

        raw_for_diag = list(incoming)
        safe_results = self._apply_renderer_safety(incoming)

        if verbose_diag and log is not None:
            for r in raw_for_diag[:24]:
                st = self._status_norm(getattr(r, "status", ""))
                bx = getattr(r, "bbox", None)
                conf = self._confidence_scalar(r)
                nm = ""
                if criminal_names and getattr(r, "criminal_id", None) is not None:
                    try:
                        nm = str(criminal_names.get(r.criminal_id, "") or "")[:48]
                    except Exception:
                        nm = ""
                log.info(
                    f"[RENDER_DECISION] camera_id={camera_id or '?'} "
                    f"track_id={getattr(r, 'track_id', '')} "
                    f"status={st} behavior_label={getattr(r, 'behavior_label', '')} "
                    f"name_hint={nm!r} bbox={bx} confidence={conf} "
                )
            if len(raw_for_diag) > 24:
                log.info(f"[RENDER_DECISION] ...truncated_total={len(raw_for_diag)}")
            if len(safe_results) != len(incoming):
                log.info(
                    f"[RENDER_SAFETY] camera_id={camera_id or '?'} "
                    f"incoming={len(incoming)} after_safety={len(safe_results)}"
                )

        for r in safe_results:
            ck = self._color_key(getattr(r, "status", ""))
            if ck == "PREDICTED" and not self._draw_predicted_tracks:
                continue

            bx = getattr(r, "bbox", None)
            if bx is None or len(bx) != 4:
                if verbose_diag and log:
                    log.info(
                        f"[RENDER_SKIP_BOX] camera_id={camera_id or '?'} "
                        f"reason=no_bbox track_id={getattr(r, 'track_id', '')}"
                    )
                continue
            try:
                for v in bx:
                    fv = float(v)
                    if not math.isfinite(fv):
                        raise ValueError("non-finite bbox")
            except (TypeError, ValueError):
                if verbose_diag and log:
                    log.info(
                        f"[RENDER_SKIP_BOX] camera_id={camera_id or '?'} "
                        f"reason=invalid_bbox_nan track_id={getattr(r, 'track_id', '')}"
                    )
                continue

            x1, y1, x2, y2 = self._clamp_bbox(bx, fw, fh)

            color = STATUS_COLORS.get(ck, STATUS_COLORS["UNKNOWN"])

            if x2 <= x1 + 1 or y2 <= y1 + 1:
                if verbose_diag and log:
                    log.info(
                        f"[RENDER_SKIP_BOX] camera_id={camera_id or '?'} "
                        f"reason=invalid_bbox_degenerate track_id={getattr(r, 'track_id', '')} "
                        f"bbox_clamped=[{x1},{y1},{x2},{y2}]"
                    )
                continue

            is_threat = ck in ("WANTED", "CRIMINAL", "HEDEF BULUNDU", "TARGET_FOUND")
            is_predicted = ck == "PREDICTED"
            is_raw = ck == "RAW_FALLBACK"

            pct = self._match_confidence_percent(r)

            if is_threat:
                box_thick = max(4, self.thickness + 3)
            elif is_predicted or is_raw:
                box_thick = max(3, self.thickness + 2)
            else:
                box_thick = max(3, self.thickness + 2)

            if verbose_diag and log is not None:
                log.info(
                    f"[RENDER_DRAW_BOX] camera_id={camera_id or '?'} "
                    f"track_id={getattr(r, 'track_id', '')} status={ck} "
                    f"color={color} bbox_clamped=[{x1},{y1},{x2},{y2}] "
                    f"confidence_pct={pct}"
                )

            if ck == "WANTED" and blink:
                cv2.rectangle(display, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 4)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, box_thick)

            name = ""
            if criminal_names and r.criminal_id is not None and r.criminal_id in criminal_names:
                name = str(criminal_names[r.criminal_id] or "").strip()[:24]

            extras: list[str] = []
            if r.behavior_label not in ("normal", "") and ck not in ("PREDICTED", "TRACKING", "RAW_FALLBACK"):
                extras.append(str(r.behavior_label)[:14])

            chip_fs_main = 0.55 if (is_predicted or is_raw) else 0.62
            chip_fs_sub = max(0.40, chip_fs_main - 0.14)
            chip_th_main = max(2, min(3, self.thickness + 2))
            chip_th_sub = max(2, min(3, self.thickness + 1))
            self._put_face_label_chip(
                display,
                x1=x1,
                y1=y1,
                y2=y2,
                frame_w=fw,
                frame_h=fh,
                name=name,
                percent=pct,
                track_display=self._short_track_label(r),
                extras=extras,
                accent_bgr=color,
                fs_main=chip_fs_main,
                fs_sub=chip_fs_sub,
                th_main=chip_th_main,
                th_sub=chip_th_sub,
            )

        if self._draw_corner_stats:
            stat_text = (f"T:{stats.get('active_tracks', 0)} "
                         f"M:{stats.get('total_matches', 0)}")
            cv2.putText(display, stat_text, (10, 22),
                        self.font, 0.45, (140, 140, 140), 1)

        has_alert = any(
            self._color_key(getattr(r, "status", ""))
            in ("WANTED", "CRIMINAL", "HEDEF BULUNDU", "TARGET_FOUND")
            for r in safe_results
        )
        if has_alert and blink:
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
            cv2.putText(display, "!!! ARANAN / HEDEF KISI TESPIT EDILDI !!!",
                        (w // 2 - 270, h - 14),
                        self.font, 0.75, (255, 255, 255), 2)

        return display

    def _status_norm(self, status: str) -> str:
        s = str(status or "").upper()
        if s == "TARGET_FOUND":
            return "HEDEF BULUNDU"
        if s == "TEMİZ":
            return "TEMIZ"
        return s

    def _color_key(self, status: str) -> str:
        """STATUS_COLORS anahtarı: Türkçe / eş anlamlı durumları birleştir."""
        s = self._status_norm(status)
        syn = {
            "TEMIZ": "CLEAN",
            "ARANIYOR": "WANTED",
            "SABIKALI": "CRIMINAL",
            "CLEARED": "CLEAN",
            "BILINMIYOR": "UNKNOWN",
            "BİLİNMİYOR": "UNKNOWN",
        }
        return syn.get(s, s)

    def _confidence_scalar(self, r: DecisionResult) -> float | None:
        """Sırasıyla: confidence, match_confidence, score, similarity, track_confidence."""
        for key in ("confidence", "match_confidence", "score", "similarity", "track_confidence"):
            if not hasattr(r, key):
                continue
            raw = getattr(r, key)
            try:
                c = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(c):
                continue
            if abs(c) < 1e-9:
                continue
            return c
        return None

    def _clamp_bbox(self, bbox: list, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w - 1, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 <= x1:
            x2 = min(frame_w - 1, x1 + 2)
        if y2 <= y1:
            y2 = min(frame_h - 1, y1 + 2)
        return x1, y1, x2, y2

    def _match_confidence_percent(self, r: DecisionResult) -> int | None:
        """0.91→%91; geçersiz/yok ise None."""
        c = self._confidence_scalar(r)
        if c is None or abs(c) < 1e-9:
            return None
        if not math.isfinite(c):
            return None
        if c > 1.0001:
            return int(round(min(100.0, max(0.0, c))))
        return int(round(min(100.0, max(0.0, c * 100.0))))

    def _short_track_label(self, r: DecisionResult) -> str:
        gid = getattr(r, "global_id", None)
        if gid and str(gid).strip():
            g = str(gid).strip()
            return g if len(g) <= 14 else g[:11] + ".."
        tid = int(getattr(r, "track_id", 0) or 0)
        if tid < 0:
            return f"r{-tid}"
        return f"T{tid}"

    def _put_face_label_chip(
        self,
        frame: np.ndarray,
        *,
        x1: int,
        y1: int,
        y2: int,
        frame_w: int,
        frame_h: int,
        name: str,
        percent: int | None,
        track_display: str,
        extras: list[str],
        accent_bgr: tuple[int, int, int],
        fs_main: float,
        fs_sub: float,
        th_main: int,
        th_sub: int,
    ) -> None:
        pad_x = 10
        pad_y = 6
        bar_w = 3
        line_gap = 4
        outer_gap = 6

        if name and percent is not None:
            main_txt = f"{name} %{percent}"
        elif name:
            main_txt = name
        elif percent is not None:
            main_txt = f"%{percent}"
        else:
            main_txt = ""

        text_rows: list[tuple[str, float, int, tuple[int, int, int]]] = []
        if main_txt:
            text_rows.append((main_txt, fs_main, th_main, self._LABEL_TEXT))
        elif track_display:
            text_rows.append((track_display, fs_sub, th_sub, (186, 186, 194)))
        only_pct_line = bool(main_txt.startswith("%") and not name)
        if track_display and main_txt and not only_pct_line:
            text_rows.append((track_display, fs_sub, th_sub, (186, 186, 194)))
        for ex in extras[:1]:
            if ex:
                fs_ex = max(0.34, fs_sub - 0.06)
                text_rows.append((str(ex).upper(), fs_ex, th_sub, (150, 150, 168)))

        if not text_rows:
            return

        line_sizes: list[tuple[int, int]] = []
        tw_max = bar_w + pad_x * 2
        inner_h = 0
        for txt, fs, th, _ in text_rows:
            (tw, h), bl = cv2.getTextSize(txt, self.font, fs, th)
            line_sizes.append((h, bl))
            tw_max = max(tw_max, tw + pad_x + bar_w + pad_x)
            inner_h += h + bl + line_gap
        inner_h -= line_gap if text_rows else 0

        chip_h = inner_h + pad_y * 2
        chip_w = int(min(frame_w - 4, max(64, tw_max)))
        x0 = int(max(0, min(x1, frame_w - chip_w - 2)))

        if y1 - chip_h - outer_gap >= 0:
            y_top = int(y1 - chip_h - outer_gap)
        else:
            y_top = int(min(frame_h - chip_h - 2, max(0, y2 + outer_gap)))

        x1b = min(frame_w - 2, x0 + chip_w)
        y_bt = min(frame_h - 2, y_top + chip_h)

        sub = frame[y_top:y_bt, x0:x1b].copy()
        fill = np.zeros_like(sub)
        fill[:] = self._LABEL_BG
        cv2.addWeighted(fill, 0.94, sub, 0.06, 0, sub)
        frame[y_top:y_bt, x0:x1b] = sub

        cv2.rectangle(frame, (x0, y_top), (x1b, y_bt), accent_bgr, 2)
        cv2.rectangle(frame, (x0, y_top), (x0 + bar_w, y_bt), accent_bgr, -1)

        xt = x0 + bar_w + pad_x
        y_row_top = y_top + pad_y
        for (txt, fs, th, col), (h_txt, bl) in zip(text_rows, line_sizes):
            baseline_y = y_row_top + h_txt
            cv2.putText(frame, txt, (xt, baseline_y), self.font, fs, col, th)
            y_row_top = baseline_y + max(2, bl) + line_gap

    def _decision_priority(self, r: DecisionResult) -> tuple[int, float, int]:
        status = self._status_norm(getattr(r, "status", ""))
        prio = STATUS_PRIORITY.get(status, 0)
        conf = float(self._confidence_scalar(r) or 0.0)
        area = int(max(1, r.bbox[2] - r.bbox[0]) * max(1, r.bbox[3] - r.bbox[1])) if getattr(r, "bbox", None) else 0
        return (prio, conf, area)

    def _is_real(self, r: DecisionResult) -> bool:
        return self._status_norm(getattr(r, "status", "")) in ("WANTED", "CRIMINAL", "ARANIYOR", "CLEAN", "TEMIZ", "UNKNOWN", "TRACKING", "TENTATIVE", "HEDEF BULUNDU")

    def _is_raw_fallback(self, r: DecisionResult) -> bool:
        status = self._status_norm(getattr(r, "status", ""))
        src = str(getattr(r, "_track_source", "")).lower()
        return status in ("RAW_FALLBACK", "FACE") or src == "raw_fallback"

    def _near_real(self, candidate: DecisionResult, real: DecisionResult) -> bool:
        cb = getattr(candidate, "bbox", None)
        rb = getattr(real, "bbox", None)
        if cb is None or rb is None:
            return False
        cw = float(max(1, cb[2] - cb[0]))
        ch = float(max(1, cb[3] - cb[1]))
        rw = float(max(1, rb[2] - rb[0]))
        rh = float(max(1, rb[3] - rb[1]))
        avg_box = (cw + ch + rw + rh) / 4.0
        center_thr = max(self._fallback_suppress_min_center_px, self._fallback_suppress_center_factor * avg_box)
        return self._bbox_center_distance(cb, rb) < center_thr or self._bbox_iou(cb, rb) > self._fallback_suppress_iou

    def _apply_renderer_safety(self, results: list[DecisionResult]) -> list[DecisionResult]:
        if not results:
            return []
        filtered: list[DecisionResult] = []
        for r in results:
            status = self._status_norm(getattr(r, "status", ""))
            if status == "PREDICTED" and not self._draw_predicted_tracks:
                continue
            # Pipeline tsu≤3 olan kutuları tutar; eski kod tsu>0 ile hepsini düşürüyordu.
            tsu = int(getattr(r, "time_since_update", 0) or 0)
            if tsu > _MAX_DRAW_TIME_SINCE_UPDATE:
                continue
            filtered.append(r)

        by_track: dict[int, DecisionResult] = {}
        no_track: list[DecisionResult] = []
        for r in filtered:
            tid = int(getattr(r, "track_id", 0) or 0)
            if tid == 0:
                no_track.append(r)
                continue
            cur = by_track.get(tid)
            if cur is None or self._decision_priority(r) > self._decision_priority(cur):
                by_track[tid] = r
        deduped = list(by_track.values()) + no_track

        real = [r for r in deduped if self._is_real(r)]
        final: list[DecisionResult] = []
        for r in deduped:
            if self._is_raw_fallback(r) and any(self._near_real(r, rr) for rr in real):
                continue
            final.append(r)
        return final

    def _put_label(self, frame, text, x, y, color, fs=None, th=None):
        fs = fs or self.font_scale
        th = th or self.thickness
        (tw, height), _ = cv2.getTextSize(text, self.font, fs, th)
        cv2.rectangle(frame, (x, y - height - 10), (x + tw + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - 5),
                    self.font, fs, (255, 255, 255), th)

    def _deduplicate_results(self, results: list[DecisionResult]) -> list[DecisionResult]:
        """Pipeline-level dedup'tan sonra defansif tekilleştirme."""
        if not results:
            return results
        ordered = sorted(
            results,
            key=lambda r: (STATUS_PRIORITY.get(self._status_norm(getattr(r, "status", "")), 1), float(getattr(r, "confidence", 0.0)), int(getattr(r, "track_id", 0) or 0)),
            reverse=True,
        )
        kept: list[DecisionResult] = []
        for r in ordered:
            if not hasattr(r, "bbox") or r.bbox is None or len(r.bbox) != 4:
                continue
            if any(self._bbox_iou(r.bbox, k.bbox) >= self._dedup_iou for k in kept):
                continue
            kept.append(r)
        return kept

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
