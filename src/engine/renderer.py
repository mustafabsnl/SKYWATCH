"""
SKYWATCH — OverlayRenderer
Frame uzerine tespit kutulari ve bilgi metinlerini cizer.
"""

import cv2
import numpy as np
from core.models import DecisionResult


# Durum → BGR renk
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "CLEAN":      (0, 200, 0),
    "CRIMINAL":   (0, 165, 255),    # Turuncu
    "WANTED":     (0, 0, 255),      # Kirmizi
    "SUSPICIOUS": (200, 0, 200),    # Mor
    "UNKNOWN":    (160, 160, 160),  # Gri
    "FACE":       (0, 220, 220),
    "TENTATIVE":  (180, 180, 0),
    "TRACKING":   (180, 180, 0),
    "PREDICTED":  (120, 180, 255),
    "RAW_FALLBACK": (0, 220, 220),
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
}

# Pipeline._STATUS_PRIORITY ile uyumlu kalmalı.
STATUS_PRIORITY: dict[str, int] = {
    "WANTED":     100,
    "CRIMINAL":    90,
    "SUSPICIOUS":  80,
    "CLEAN":       70,
    "UNKNOWN":     60,
    "TRACKING":    50,
    "TENTATIVE":   45,
    "FACE":        30,
    "PREDICTED":   10,
    "RAW_FALLBACK":   5,
}

class OverlayRenderer:
    """Frame'e tespit sonuclarini cizer."""

    def __init__(self, font_scale: float = 0.55, thickness: int = 1, dedup_iou: float = 0.45, draw_predicted_tracks: bool = False):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self._frame_count = 0
        # Pipeline zaten dedup uyguluyor; renderer defansif olarak aynı IoU eşiğini kullanır.
        self._dedup_iou = float(dedup_iou)
        self._draw_predicted_tracks = bool(draw_predicted_tracks)

    def draw(self, frame: np.ndarray,
             results: list[DecisionResult],
             stats: dict,
             criminal_names: dict = None) -> np.ndarray:
        """Kararlari frame uzerine cizer."""
        display = frame.copy()
        self._frame_count += 1
        blink = (self._frame_count // 8) % 2 == 0  # ~3 Hz yanip sonme
        # Pipeline final unique decision listesini üretir; renderer yalnızca çizer.

        for r in results:
            if str(getattr(r, "status", "")).upper() == "PREDICTED" and not self._draw_predicted_tracks:
                continue
            color = STATUS_COLORS.get(r.status, STATUS_COLORS["UNKNOWN"])
            x1, y1, x2, y2 = r.bbox
            is_threat = r.status in ("WANTED", "CRIMINAL")
            is_predicted = r.status == "PREDICTED"
            is_raw = r.status == "RAW_FALLBACK"

            # Tek kutu cizimi (cift kutu gorunumu kaldirildi)
            # PREDICTED kutular ince + production'da daha sade etiket ile çizilir
            if is_threat:
                box_thick = 4
            elif is_predicted or is_raw:
                box_thick = 1
            else:
                box_thick = 3
            if r.status == "WANTED" and blink:
                cv2.rectangle(display, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 4)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, box_thick)

            # Ust etiket — Person ID varsa onu göster
            display_id = r.global_id if r.global_id else f"ID:{r.track_id}"
            label = f"{display_id} {STATUS_LABELS.get(r.status, r.status)}"
            if is_threat:
                fs = 0.65
                th = 2
            elif is_predicted or is_raw:
                fs = max(0.4, self.font_scale - 0.1)
                th = self.thickness
            else:
                fs = self.font_scale
                th = self.thickness
            self._put_label(display, label, x1, y1, color, fs, th)

            # Alt etiket: isim + guven skoru
            if is_threat and r.confidence > 0:
                name = ""
                if criminal_names and r.criminal_id in criminal_names:
                    name = criminal_names[r.criminal_id]
                bottom = f"{name}  %{r.confidence * 100:.0f}".strip()
                cv2.putText(display, bottom, (x1, y2 + 22),
                            self.font, 0.6, color, 2)

            elif r.behavior_label not in ("normal", ""):
                cv2.putText(display, r.behavior_label.upper(),
                            (x1, y2 + 18), self.font,
                            self.font_scale - 0.05, color, self.thickness)

        # Sol ust istatistik
        stat_text = (f"Aktif: {stats.get('active_tracks', 0)} | "
                     f"Tarama: {stats.get('total_faces_scanned', 0)} | "
                     f"Eslesme: {stats.get('total_matches', 0)} | "
                     f"Re-ID: {stats.get('reid_hits', 0)}")
        cv2.putText(display, stat_text, (10, 25),
                    self.font, self.font_scale, (0, 220, 0), self.thickness)

        # Alt banner: WANTED varsa yanip sonen uyari
        has_wanted = any(r.status == "WANTED" for r in results)
        if has_wanted and blink:
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
            cv2.putText(display, "!!! ARANAN KISI TESPIT EDILDI !!!",
                        (w // 2 - 270, h - 14),
                        self.font, 0.75, (255, 255, 255), 2)

        return display

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
            key=lambda r: (STATUS_PRIORITY.get(r.status, 1), float(getattr(r, "confidence", 0.0)), int(getattr(r, "track_id", 0) or 0)),
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
