"""
Soft-NMS ve Face Aligner Post-Processing

1. SoftNMS — Kalabalık sahnelerde kutu ayrımı
   Standart NMS=sil, Soft-NMS=puanı düşür
   Kaynak: "Deep Learning Based Face Detection in Crowded Scenes"

2. FaceAligner — 5 landmark ile yüz hizalama
   Affine transform → 112x112 normalize yüz
   SKYWATCH'ın InsightFace recognition modülüne hazır yüz üretir
   Kaynak: "MTCNN" + "Face Detection and Recognition Systems"
"""

from __future__ import annotations

import numpy as np
import cv2


# ════════════════════════════════════════════════════════════════════════
# Soft-NMS
# ════════════════════════════════════════════════════════════════════════

def soft_nms(
    dets: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
    iou_threshold: float = 0.3,
) -> np.ndarray:
    """Soft Non-Maximum Suppression.

    Standart NMS yerine, yüksek IoU'ya sahip kutuların puanını
    Gaussian veya lineer decay fonksiyonuyla düşürür.
    Kalabalık sahnelerde gerçek yüz kutularının silinmesini engeller.

    Kaynak: "Deep Learning Based Face Detection in Crowded Scenes"
             (Soft-NMS — Improving Object Detection with One Line of Code, ICCV 2017)

    Args:
        dets (np.ndarray): (N, 5) — [x1, y1, x2, y2, score]
        sigma (float): Gaussian sigma (method='gaussian' için).
        score_threshold (float): Bu puanın altındaki kutuları kaldır.
        method (str): 'gaussian' veya 'linear'.
        iou_threshold (float): Linear decay için IoU eşiği.

    Returns:
        np.ndarray: Filtre uygulanmış detections indeksleri (tutulacak).
    """
    N = dets.shape[0]
    if N == 0:
        return np.array([], dtype=int)

    x1 = dets[:, 0].copy()
    y1 = dets[:, 1].copy()
    x2 = dets[:, 2].copy()
    y2 = dets[:, 3].copy()
    scores = dets[:, 4].copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    order = scores.argsort()[::-1]  # Yüksek puandan düşüğe

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # i ile diğer tüm kutular arasında IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Puan güncelleme
        if method == "gaussian":
            weight = np.exp(-(iou ** 2) / sigma)  # Gaussian decay
        else:  # linear
            weight = np.where(iou > iou_threshold, 1 - iou, np.ones_like(iou))

        scores[order[1:]] *= weight

        # Eşiğin altına düşenleri kaldır
        order = order[1:][scores[order[1:]] > score_threshold]

    return np.array(keep, dtype=int)


def apply_soft_nms_to_results(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """SKYWATCH pipeline ile entegre Soft-NMS.

    Args:
        boxes (np.ndarray): (N, 4) — [x1, y1, x2, y2]
        scores (np.ndarray): (N,) — güven puanları
        sigma (float): Gaussian sigma.
        score_threshold (float): Minimum puan eşiği.

    Returns:
        tuple: (filtered_boxes, filtered_scores)
    """
    if len(boxes) == 0:
        return boxes, scores

    dets = np.concatenate([boxes, scores[:, None]], axis=1)
    keep = soft_nms(dets, sigma=sigma, score_threshold=score_threshold)
    return boxes[keep], scores[keep]


# ════════════════════════════════════════════════════════════════════════
# Face Aligner
# ════════════════════════════════════════════════════════════════════════

# InsightFace / ArcFace için standart hedef landmark konumları (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # sol göz merkezi
    [73.5318, 51.5014],  # sağ göz merkezi
    [56.0252, 71.7366],  # burun ucu
    [41.5493, 92.3655],  # sol ağız köşesi
    [70.7299, 92.2041],  # sağ ağız köşesi
], dtype=np.float32)


class FaceAligner:
    """5-Nokta Landmark Tabanlı Yüz Hizalama.

    Model tarafından tahmin edilen 5 yüz noktasını (göz, burun, ağız)
    kullanarak yüzü ArcFace/InsightFace standart konumuna hizalar.

    Çıktı: 112x112 boyutlu, gözler yatayda hizalanmış yüz görüntüsü.
    SKYWATCH recognition pipeline'ına doğrudan beslenebilir.

    Kaynak: "MTCNN" + "Face Detection and Recognition Systems"
    """

    def __init__(
        self,
        output_size: int = 112,
        dst_landmarks: np.ndarray | None = None,
    ):
        """FaceAligner'ı başlat.

        Args:
            output_size (int): Çıktı görüntü boyutu (kare).
            dst_landmarks (np.ndarray | None): Hedef landmark konumları.
                None ise ArcFace standardı kullanılır.
        """
        self.output_size = output_size
        self.dst = dst_landmarks if dst_landmarks is not None else ARCFACE_DST.copy()

        # dst'yi output_size'a göre ölçekle (eğer 112 değilse)
        if output_size != 112:
            scale = output_size / 112.0
            self.dst = self.dst * scale

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        return_transform: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Yüzü hizala.

        Args:
            image (np.ndarray): (H, W, 3) BGR görüntü.
            landmarks (np.ndarray): (5, 2) yüz noktaları — (x, y) piksel koordinatları.
                Sıralama: [sol_göz, sağ_göz, burun, sol_ağız, sağ_ağız]
            return_transform (bool): True ise transform matrisini de döndür.

        Returns:
            np.ndarray: (output_size, output_size, 3) hizalanmış yüz.
            tuple: Ek olarak transform matrisi (return_transform=True ise).
        """
        src = landmarks.astype(np.float32)
        dst = self.dst.astype(np.float32)

        # Affine transform matrisini hesapla (en küçük kareler yöntemi)
        M = self._estimate_affine(src, dst)

        # Warp uygula
        aligned = cv2.warpAffine(
            image, M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        if return_transform:
            return aligned, M
        return aligned

    @staticmethod
    def _estimate_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Partial affine transform (similarity transform) matrisini hesapla.

        Tam affine yerine similarity transform kullanılır — sadece
        scale, rotation ve translation. Yüz şekli korunur.

        Args:
            src (np.ndarray): (N, 2) kaynak noktalar.
            dst (np.ndarray): (N, 2) hedef noktalar.

        Returns:
            np.ndarray: (2, 3) transform matrisi.
        """
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        src_scale = np.sqrt((src_centered ** 2).sum() / len(src))
        dst_scale = np.sqrt((dst_centered ** 2).sum() / len(dst))

        if src_scale < 1e-10:
            return np.eye(2, 3, dtype=np.float32)

        src_norm = src_centered / src_scale
        dst_norm = dst_centered / dst_scale

        H = src_norm.T @ dst_norm
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        scale = dst_scale / src_scale
        t = dst_mean - scale * R @ src_mean

        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:2, 2] = t
        return M

    def get_face_angle(self, landmarks: np.ndarray) -> float:
        """Yüz eğim açısını hesapla (derece).

        İki gözün konumuna göre horizontal açıyı döndürür.
        ±15° üzeri = aşırı eğik yüz.

        Args:
            landmarks (np.ndarray): (5, 2) yüz noktaları.

        Returns:
            float: Eğim açısı (derece).
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle)

    def is_valid_face(
        self,
        landmarks: np.ndarray,
        max_angle: float = 30.0,
        min_eye_distance: float = 20.0,
    ) -> bool:
        """Yüzün hizalanabilir kalitede olup olmadığını kontrol et.

        Args:
            landmarks (np.ndarray): (5, 2) yüz noktaları.
            max_angle (float): Maksimum eğim açısı (derece).
            min_eye_distance (float): Minimum göz arası mesafe (piksel).

        Returns:
            bool: Geçerli yüz = True.
        """
        angle = abs(self.get_face_angle(landmarks))
        eye_dist = np.linalg.norm(landmarks[1] - landmarks[0])
        return angle <= max_angle and eye_dist >= min_eye_distance

    def align_batch(
        self,
        image: np.ndarray,
        all_landmarks: np.ndarray,
    ) -> list[np.ndarray]:
        """Birden fazla yüzü tek çağrıda hizala.

        Args:
            image (np.ndarray): (H, W, 3) kaynak görüntü.
            all_landmarks (np.ndarray): (N, 5, 2) tüm yüzlerin landmarkları.

        Returns:
            list[np.ndarray]: N adet hizalanmış yüz görüntüsü.
        """
        return [self.align(image, lm) for lm in all_landmarks]
