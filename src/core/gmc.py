"""
SKYWATCH — Global Motion Compensation (GMC)
Kaynak: KLT Optical Flow tabanlı kamera ego-motion tahmini.

Makaleler: #1 (Affine GMC), #8 (Phase Correlation), #12 (KLT Point Tracker)

SKYWATCH Kullanımı:
  • Sabit kameraları için genellikle Δ≈(0,0) çıkar → performans maliyeti düşük.
  • PTZ veya titrayan/sarsılan kameralarda açıldığında track kutuları
    kamera hareketinden arındırılır.
  • config.yaml'daki  tracking.gmc_enabled: false  ile tamamen devre dışı bırakılır.
"""

import cv2
import numpy as np


class GMCModule:
    """
    KLT (Kanade-Lucas-Tomasi) tabanlı Global Motion Compensation.

    Her karede arka plandaki stabil köşe noktalarını takip ederek
    kameranın yer değiştirme vektörünü (Δx, Δy) hesaplar.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("gmc_enabled", False)  # Sabit kamera → varsayılan kapalı

        # Şüphe eşiği: Bu değerin altındaki kamera hareketleri gürültü sayılır
        self._min_movement = config.get("gmc_min_movement_px", 1.0)

        # KLT parametreleri
        self._feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self._prev_gray: np.ndarray | None = None
        self._prev_pts:  np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────
    def estimate(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Önceki frame ile mevcut frame arasındaki kamera yer değiştirmesini
        (Δx, Δy) piksel cinsinden döndürür.

        Sabit kamera + gürültüsüz ortam → (0.0, 0.0)
        Kamera sağa 5px kaydıysa → (-5.0, 0.0)  [telafi yönü ters]

        Args:
            frame: Mevcut BGR frame

        Returns:
            (delta_x, delta_y): Kamera kayma vektörü (telafi için eksi işaret)
        """
        if not self.enabled:
            return (0.0, 0.0)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # İlk frame — başla
        if self._prev_gray is None or self._prev_pts is None:
            self._prev_gray = curr_gray
            self._prev_pts  = self._detect_features(curr_gray)
            return (0.0, 0.0)

        if self._prev_pts is None or len(self._prev_pts) < 4:
            # Yeterli nokta yok → yeniden tespit et
            self._prev_gray = curr_gray
            self._prev_pts  = self._detect_features(curr_gray)
            return (0.0, 0.0)

        # KLT ile noktaları bir sonraki frame'e taşı
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            curr_gray,
            self._prev_pts,
            None,
            **self._lk_params
        )

        # Sadece başarıyla takip edilen noktalar
        good_prev = self._prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        delta_x, delta_y = 0.0, 0.0

        if len(good_prev) >= 4:
            # Affine dönüşümü tahmin et (Makale #1 yaklaşımı)
            M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)

            if M is not None:
                tx = float(M[0, 2])
                ty = float(M[1, 2])

                # Gürültü eşiği kontrolü
                if abs(tx) > self._min_movement or abs(ty) > self._min_movement:
                    # Negatif çünkü biz track'i tazmin ediyoruz (ters yönde)
                    delta_x = -tx
                    delta_y = -ty

        # Sonraki iterasyon için güncelle
        self._prev_gray = curr_gray
        self._prev_pts  = self._detect_features(curr_gray)

        return (delta_x, delta_y)

    # ──────────────────────────────────────────────────────────────
    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        """Arka plandaki stabil köşe noktalarını tespit eder."""
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
        if pts is not None:
            return pts.reshape(-1, 1, 2).astype(np.float32)
        return None

    # ──────────────────────────────────────────────────────────────
    def reset(self):
        """Kamera değiştiğinde veya yeniden bağlanıldığında sıfırla."""
        self._prev_gray = None
        self._prev_pts  = None
