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
}

STATUS_LABELS: dict[str, str] = {
    "CLEAN":      "TEMIZ",
    "CRIMINAL":   "! SABIKALI !",
    "WANTED":     "!! ARANIYOR !!",
    "SUSPICIOUS": "SUPHELI",
    "UNKNOWN":    "?",
}


class OverlayRenderer:
    """Frame'e tespit sonuclarini cizer."""

    def __init__(self, font_scale: float = 0.55, thickness: int = 1):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self._frame_count = 0

    def draw(self, frame: np.ndarray,
             results: list[DecisionResult],
             stats: dict,
             criminal_names: dict = None) -> np.ndarray:
        """Kararlari frame uzerine cizer."""
        display = frame.copy()
        self._frame_count += 1
        blink = (self._frame_count // 8) % 2 == 0  # ~3 Hz yanip sonme

        for r in results:
            color = STATUS_COLORS.get(r.status, STATUS_COLORS["UNKNOWN"])
            x1, y1, x2, y2 = r.bbox
            is_threat = r.status in ("WANTED", "CRIMINAL")

            # Kutu
            box_thick = 3 if is_threat else 2
            if r.status == "WANTED" and blink:
                cv2.rectangle(display, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 4)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, box_thick)

            # Ust etiket — Person ID varsa onu göster
            display_id = r.global_id if r.global_id else f"ID:{r.track_id}"
            label = f"{display_id} {STATUS_LABELS.get(r.status, r.status)}"
            fs = 0.65 if is_threat else self.font_scale
            th = 2 if is_threat else self.thickness
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
