"""
SKYWATCH Custom Trainer
Tez Katkisi #3: Okluzyon Augmentasyonu + SkyWatchDetectionLoss entegrasyonu

Bu dosya:
1. SkyWatchDetectionLoss'u egitim sirasinda etkinlestirir
2. Okluzyona duyarli augmentasyon ekler (maske, gozluk, blur sim.)
3. Egitilebilir trainer sinifi sunar

Kullanim:
    python src/model/skywatch_trainer.py
"""

import sys
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

# SKYWATCH'i parent path'e ekle
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "ultralytics"))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.augment import Compose, LetterBox

# Kendi loss'umuzu import et
sys.path.insert(0, str(Path(__file__).parent))
from skywatch_loss import SkyWatchDetectionLoss


# ──────────────────────────────────────────────────────────────────────
# Katkı #3: Oklüzyon Augmentasyonu
# ──────────────────────────────────────────────────────────────────────
class FaceOcclusionAug:
    """Guvenlik kamerasi senaryolari icin yuz okluzyonu simulasyonu.

    Egitim sirasinda tesadifi olarak:
    - Yuzun bir bolumunu siyah dikdortgenle kapat (maske/gozluk/el)
    - Gaussian blur uygula (hareket bulaniklik sim.)
    - Goruntu kalitesini dusur (dusuk cozunurluk sim.)

    Tez Katkisi #3 — Okluzyona Dayanikli Egitim
    """

    def __init__(
        self,
        occlusion_prob: float = 0.35,   # Her imaj icin okluzyon olasiligi
        blur_prob: float = 0.25,         # Blur uygulama olasiligi
        downscale_prob: float = 0.20,    # Dusuk cozunurluk sim. olasiligi
        min_occlude: float = 0.15,       # Bbox alaninin min okluzyon orani
        max_occlude: float = 0.50,       # Bbox alaninin max okluzyon orani
    ):
        self.occlusion_prob = occlusion_prob
        self.blur_prob = blur_prob
        self.downscale_prob = downscale_prob
        self.min_occlude = min_occlude
        self.max_occlude = max_occlude

    def apply_occlusion(self, img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Yuz bboxlari uzerine rastgele okluzyon uygula.

        Args:
            img: (H, W, 3) BGR goruntu
            bboxes: (N, 4) YOLO format normalized [cx, cy, w, h]

        Returns:
            img: Okluzyon uygulanmis goruntu
        """
        h, w = img.shape[:2]
        for bbox in bboxes:
            if random.random() > self.occlusion_prob:
                continue
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            bw_px = max(x2 - x1, 1)
            bh_px = max(y2 - y1, 1)

            # Okluzyon oranini sec
            occ_ratio = random.uniform(self.min_occlude, self.max_occlude)

            # Rastgele konum: ust (goz/alin), orta (burun/agiz), alt (agiz/cene)
            region = random.choice(["top", "mid", "bottom"])
            if region == "top":
                oy1 = y1
                oy2 = y1 + int(bh_px * occ_ratio)
            elif region == "mid":
                oy1 = y1 + int(bh_px * 0.25)
                oy2 = oy1 + int(bh_px * occ_ratio)
            else:
                oy1 = y2 - int(bh_px * occ_ratio)
                oy2 = y2

            oy1, oy2 = max(0, oy1), min(h, oy2)
            ox1, ox2 = max(0, x1), min(w, x2)

            if oy2 > oy1 and ox2 > ox1:
                img[oy1:oy2, ox1:ox2] = 0  # Siyah bant (maske)

        return img

    def apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Hareket bulaniklik simulasyonu."""
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def apply_downscale(self, img: np.ndarray) -> np.ndarray:
        """Dusuk cozunurluk kamera simulasyonu."""
        h, w = img.shape[:2]
        scale = random.uniform(0.4, 0.7)
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def __call__(self, labels: dict) -> dict:
        """Ultralytics augmentation pipeline ile uyumlu __call__."""
        img = labels.get("img")
        if img is None:
            return labels

        bboxes = labels.get("bboxes", np.zeros((0, 4)))

        if random.random() < self.occlusion_prob and len(bboxes) > 0:
            img = self.apply_occlusion(img, bboxes)

        if random.random() < self.blur_prob:
            img = self.apply_blur(img)

        if random.random() < self.downscale_prob:
            img = self.apply_downscale(img)

        labels["img"] = img
        return labels


# ──────────────────────────────────────────────────────────────────────
# Custom Trainer: SKYWATCH katkılarini entegre eder
# ──────────────────────────────────────────────────────────────────────
class SkyWatchTrainer(DetectionTrainer):
    """SKYWATCH-Det Egitim Sinifi.

    Standart DetectionTrainer'i su sekilde genisletir:
    1. BboxLoss -> SkyWatchBboxLoss (Katki #2)
    2. FaceOcclusionAug eklenir (Katki #3)
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Modeli yukle."""
        model = super().get_model(cfg, weights, verbose)
        return model

    def init_criterion(self):
        """Standard loss yerine SkyWatchDetectionLoss kullan."""
        return SkyWatchDetectionLoss(self.model)

    def build_dataset(self, img_path, mode="train", batch=None):
        """train modunda FaceOcclusionAug ekle."""
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            # Mevcut transform listesine okluzyon augmentasyonunu ekle
            try:
                occ_aug = FaceOcclusionAug(
                    occlusion_prob=0.35,
                    blur_prob=0.25,
                    downscale_prob=0.20,
                )
                # transforms listesinin sonuna ekle (normalize'dan once)
                if hasattr(dataset, "transforms") and dataset.transforms is not None:
                    if hasattr(dataset.transforms, "transforms"):
                        dataset.transforms.transforms.insert(-1, occ_aug)
                        print("[SKYWATCH] FaceOcclusionAug eklendi")
            except Exception as e:
                print(f"[SKYWATCH] Aug ekleme hatasi (devam ediliyor): {e}")
        return dataset


# ──────────────────────────────────────────────────────────────────────
# Ana Egitim Fonksiyonlari
# ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parents[2]
DATA_YAML = BASE_DIR / "src/model/data.yaml"
RUNS_DIR  = BASE_DIR / "src/model/runs"
SW_YAML   = BASE_DIR / "ultralytics/ultralytics/cfg/models/v8/skywatch-det.yaml"

COMMON_ARGS = dict(
    data=str(DATA_YAML.resolve()),
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    project=str(RUNS_DIR),
    save=True,
    save_period=10,
    plots=True,
    val=True,
    verbose=True,
    # Augmentasyon (SKYWATCH trainer ek olarak okluzyon da ekler)
    mosaic=1.0,
    mixup=0.1,
    degrees=15.0,
    fliplr=0.5,
    scale=0.6,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # Loss agirliklar
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Optimizer
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    weight_decay=0.0005,
)


def train_baseline():
    """Baseline: Standart YOLOv8s-P2, degisiklik yok."""
    print("\n" + "="*55)
    print("  BASELINE: yolov8s-p2 (standart)")
    print("="*55)
    # yaml'dan olustur, pretrained backbone kullan (yolov8s.pt'den)
    model = YOLO("yolov8s-p2.yaml")
    model.train(
        **COMMON_ARGS,
        name="baseline_yolov8s_p2",
        exist_ok=True,
        pretrained="yolov8s.pt",   # pretrained backbone indir/kullan
    )
    metrics = model.val(data=str(DATA_YAML.resolve()))
    return metrics


def train_skywatch():
    """SKYWATCH-Det: CBAM + SkyWatchBboxLoss + OcclusionAug."""
    print("\n" + "="*55)
    print("  SKYWATCH-Det: CBAM + Adaptive Loss + OcclusionAug")
    print("="*55)

    trainer = SkyWatchTrainer(
        overrides={
            **COMMON_ARGS,
            "model": str(SW_YAML),
            "name": "skywatch_det_full",
            "exist_ok": True,
            "pretrained": True,
        }
    )
    trainer.train()

    # Val metrikleri al
    model = YOLO(str(RUNS_DIR / "skywatch_det_full/weights/best.pt"))
    metrics = model.val(data=str(DATA_YAML.resolve()))
    return metrics


def compare(b, s):
    """Tablo olarak karsilastir."""
    print("\n" + "="*55)
    print("  KARSILASTIRMA TABLOSU")
    print("="*55)
    rows = [
        ("mAP@0.5",      b.box.map50, s.box.map50),
        ("mAP@0.5:0.95", b.box.map,   s.box.map),
        ("Precision",    b.box.mp,    s.box.mp),
        ("Recall",       b.box.mr,    s.box.mr),
    ]
    print(f"  {'Metrik':<20} {'Baseline':>10} {'SKYWATCH':>10} {'Delta':>8}")
    print("  " + "-"*50)
    for name, bv, sv in rows:
        d = sv - bv
        print(f"  {name:<20} {bv:>10.4f} {sv:>10.4f} {d:>+8.4f}")
    delta = s.box.map50 - b.box.map50
    print("="*55)
    print(f"  {'Katki: ' + ('+' if delta>0 else '') + f'{delta:.4f} mAP@0.5'}")
    print("="*55)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "skywatch", "both"], default="skywatch")
    args = p.parse_args()

    if args.mode == "baseline":
        train_baseline()
    elif args.mode == "skywatch":
        train_skywatch()
    elif args.mode == "both":
        b = train_baseline()
        s = train_skywatch()
        compare(b, s)
