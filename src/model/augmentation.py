"""
SKYWATCH-Det Augmentation Pipeline

15 makaleden derlenen gelişmiş augmentation stratejisi:

1. CLAHE — Aydınlatma normalizasyonu (Handbook + Survey)
2. Motion Blur + Gaussian Blur — Düşük kalite simülasyonu (Low-Quality images paper)
3. Random Perspective — Farklı kamera açıları (Multi-view + Handbook)
4. Random Erasing — Kısmi oklüzyon simülasyonu (WIDER FACE + Local CNN)
5. JPEG Compression — Video codec bozulması (Low-Res Surveillance)
6. Downscale+Upsample — Düşük çözünürlük simülasyonu (Low-Res Surveillance)
7. GridMask — Kaba oklüzyon (WIDER FACE)
8. Color Jitter — Işık değişimi (Handbook)
9. Gaussian Noise — Sensör gürültüsü (Low-Res Surveillance)
10. Horizontal Flip — Profil simetri (Multi-view)
"""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: CLAHE
# ════════════════════════════════════════════════════════════════════════

class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization.

    Aydınlatma varyasyonlarını normalize eder.
    Gece çekimlerinde gizlenen yüzleri görünür kılar.

    Kaynak: "Handbook of Face Recognition" + "A Survey on Face Detection in the Wild"
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8, p: float = 0.4):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        # LAB color space'de sadece L kanalına uygula
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: Motion Blur
# ════════════════════════════════════════════════════════════════════════

class MotionBlur:
    """Hareket bulanıklığı simülasyonu.

    Güvenlik kameralarındaki hareket kaynaklı bulanıklığı taklit eder.

    Kaynak: "Face Detection in Low-quality Images via Deep Feature Enhancement"
    """

    def __init__(self, max_kernel: int = 15, p: float = 0.3):
        self.max_kernel = max_kernel
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        k = random.choice(range(3, self.max_kernel + 1, 2))  # Tek kernel boyutu
        # Yatay veya dikey hareket
        if random.random() < 0.5:
            kernel = np.zeros((k, k))
            kernel[k // 2, :] = 1.0 / k  # Yatay
        else:
            kernel = np.zeros((k, k))
            kernel[:, k // 2] = 1.0 / k  # Dikey
        return cv2.filter2D(img, -1, kernel)


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: Gaussian Blur
# ════════════════════════════════════════════════════════════════════════

class GaussianBlur:
    """Gaussian bulanıklık simülasyonu (odak dışı kamera)."""

    def __init__(self, max_kernel: int = 9, p: float = 0.25):
        self.max_kernel = max_kernel
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        k = random.choice(range(3, self.max_kernel + 1, 2))
        return cv2.GaussianBlur(img, (k, k), 0)


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: JPEG Compression Artifacts
# ════════════════════════════════════════════════════════════════════════

class JPEGCompression:
    """JPEG sıkıştırma bozulması simülasyonu.

    Düşük bit hızlı güvenlik kamerası yayınlarını taklit eder.

    Kaynak: "Face Detection in Low-Resolution Surveillance Images"
    """

    def __init__(self, quality_range: tuple[int, int] = (30, 70), p: float = 0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: Downscale + Upsample (Low-Res Simulation)
# ════════════════════════════════════════════════════════════════════════

class LowResSimulation:
    """Düşük çözünürlük + yeniden büyütme simülasyonu.

    Uzaktaki yüzlerin pikselleşmesini taklit eder.

    Kaynak: "Face Detection in Low-Resolution Surveillance Images"
    """

    def __init__(self, scale_range: tuple[float, float] = (0.4, 0.7), p: float = 0.25):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        h, w = img.shape[:2]
        scale = random.uniform(*self.scale_range)
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: Gaussian Noise
# ════════════════════════════════════════════════════════════════════════

class GaussianNoise:
    """Sensör gürültüsü simülasyonu."""

    def __init__(self, std_range: tuple[int, int] = (5, 25), p: float = 0.2):
        self.std_range = std_range
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        std = random.randint(*self.std_range)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════
# Yardımcı: Random Erasing (Oklüzyon Simülasyonu)
# ════════════════════════════════════════════════════════════════════════

class RandomErasing:
    """Rastgele bölge silme — kısmi oklüzyon simülasyonu.

    Maskelerin, gözlüklerin ve ellerin yüzü kapatmasını taklit eder.

    Kaynak: "WIDER FACE Benchmark" + "Robust Face Detection Using Local CNN and SVM"
    """

    def __init__(
        self,
        p: float = 0.4,
        scale: tuple[float, float] = (0.02, 0.2),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: int = 0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value  # 0 = siyah (maske sim.), 128 = gri

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        h, w = img.shape[:2]
        area = h * w
        img = img.copy()

        # 1-3 arasında rastgele silme
        n_erases = random.randint(1, 3)
        for _ in range(n_erases):
            erase_area = area * random.uniform(*self.scale)
            aspect = random.uniform(*self.ratio)
            eh = int(round((erase_area * aspect) ** 0.5))
            ew = int(round((erase_area / aspect) ** 0.5))
            if eh > h or ew > w:
                continue
            y = random.randint(0, h - eh)
            x = random.randint(0, w - ew)
            img[y:y + eh, x:x + ew] = self.value

        return img


# ════════════════════════════════════════════════════════════════════════
# Ana Pipeline
# ════════════════════════════════════════════════════════════════════════

class SkyWatchAugmentation:
    """SKYWATCH-Det için tam augmentation pipeline'ı.

    15 makaleden derlenen augmentation stratejisi.
    Ultralytics'in kendi augmentation'larına EK olarak çalışır.
    Ultralytics'in Mosaic, RandomPerspective, ColorJitter, Flip'i zaten var.
    Bu pipeline surveillance-specific eklemeler yapar.

    Kullanım:
        aug = SkyWatchAugmentation(mode='train')
        result = aug(labels_dict)  # Ultralytics labels formatı
    """

    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.enabled = (mode == "train")

        if self.enabled:
            self.transforms = [
                CLAHE(clip_limit=2.0, tile_size=8, p=0.4),
                MotionBlur(max_kernel=11, p=0.3),
                GaussianBlur(max_kernel=7, p=0.25),
                JPEGCompression(quality_range=(35, 75), p=0.3),
                LowResSimulation(scale_range=(0.45, 0.75), p=0.2),
                GaussianNoise(std_range=(5, 20), p=0.2),
                RandomErasing(p=0.35, scale=(0.02, 0.15)),
            ]

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Ultralytics labels dict üzerinde augmentation uygula.

        Args:
            labels (dict): Ultralytics augmentation pipeline'dan gelen dict.
                           'img' anahtarı numpy array içerir.

        Returns:
            dict: Augmentation uygulanmış labels.
        """
        if not self.enabled:
            return labels

        img = labels.get("img")
        if img is None:
            return labels

        for transform in self.transforms:
            img = transform(img)

        labels["img"] = img
        return labels

    def __repr__(self) -> str:
        transforms_str = "\n    ".join(
            f"{t.__class__.__name__}(p={t.p})" for t in self.transforms
        ) if self.enabled else "disabled"
        return f"SkyWatchAugmentation(mode={self.mode}):\n    {transforms_str}"
