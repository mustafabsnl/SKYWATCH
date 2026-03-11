"""
SKYWATCH — OverlayRenderer
Tek Sorumluluk: Frame üzerine tespit kutularını ve bilgi metinlerini çizer.
Pipeline'dan ayrıldı — bağımsız olarak test edilebilir ve değiştirilebilir.
"""

import cv2
import numpy as np
from core.models import DecisionResult


# Durum → BGR renk eşlemesi
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "CLEAN":      (0, 200, 0),       # Yeşil
    "CRIMINAL":   (0, 200, 255),     # Turuncu
    "WANTED":     (0, 0, 220),       # Kırmızı
    "SUSPICIOUS": (200, 0, 200),     # Mor
    "UNKNOWN":    (160, 160, 160),   # Gri
}


class OverlayRenderer:
    """Frame'e tespit sonuçlarını çizer."""

    def __init__(self, font_scale: float = 0.52, thickness: int = 1):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw(self, frame: np.ndarray,
             results: list[DecisionResult],
             stats: dict) -> np.ndarray:
        """
        Kararları frame üzerine çizer ve frame kopyasını döndürür.

        Args:
            frame:   Orijinal BGR frame
            results: Pipeline'dan gelen DecisionResult listesi
            stats:   {'active_tracks', 'total_faces_scanned', 'total_matches'}

        Returns:
            np.ndarray: Üzerinde çizim yapılmış frame kopyası
        """
        display = frame.copy()

        for r in results:
            color = STATUS_COLORS.get(r.status, STATUS_COLORS["UNKNOWN"])
            x1, y1, x2, y2 = r.bbox

            # Kutu
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Üst etiket
            label = self._build_label(r)
            self._put_label(display, label, x1, y1, color)

            # Alt etiket (davranış)
            if r.behavior_label not in ("normal", ""):
                cv2.putText(display, r.behavior_label.upper(),
                            (x1, y2 + 16), self.font,
                            self.font_scale - 0.05, color, self.thickness)

        # Sol üst köşe: sistem istatistikleri
        stat_text = (f"Aktif: {stats.get('active_tracks', 0)} | "
                     f"Tarama: {stats.get('total_faces_scanned', 0)} | "
                     f"Eslesme: {stats.get('total_matches', 0)}")
        cv2.putText(display, stat_text, (10, 25),
                    self.font, self.font_scale, (0, 220, 0), self.thickness)

        return display

    # ──────────────────────────────────────────
    # Yardımcı metodlar
    # ──────────────────────────────────────────

    def _build_label(self, r: DecisionResult) -> str:
        parts = [f"ID:{r.track_id}"]
        if r.status == "WANTED":
            parts.append("ARANAN!")
        elif r.status == "CRIMINAL":
            parts.append("SABIKALI")
        elif r.status == "SUSPICIOUS":
            parts.append("SUPHELI")
        if r.confidence > 0:
            parts.append(f"%{r.confidence * 100:.0f}")
        return " ".join(parts)

    def _put_label(self, frame: np.ndarray, text: str,
                   x: int, y: int, color: tuple):
        (tw, th), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(frame, text, (x + 2, y - 4),
                    self.font, self.font_scale, (255, 255, 255), self.thickness)
