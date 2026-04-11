"""
SKYWATCH-Det Eğitim Scripti
Model: skywatch-det.yaml (CBAM-Enhanced YOLOv8-P2)
Karşılaştırma: Önce baseline, sonra SKYWATCH-Det

Kullanım:
    python src/model/train_skywatch.py --mode baseline
    python src/model/train_skywatch.py --mode skywatch
    python src/model/train_skywatch.py --mode both   (önce baseline, sonra skywatch)
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR  = Path("C:/Users/musta/OneDrive/Desktop/SKYWATCH")
DATA_YAML = BASE_DIR / "src/model/data.yaml"
RUNS_DIR  = BASE_DIR / "src/model/runs"

# Ultralytics reposu üzerinden custom model yüklemek için
ULTRA_CFG = BASE_DIR / "ultralytics/ultralytics/cfg/models/v8/skywatch-det.yaml"

TRAIN_ARGS = dict(
    data=str(DATA_YAML.resolve()),
    epochs=100,
    imgsz=640,
    batch=16,      # RTX 3050 için → OOM olursa 8'e düşür
    workers=4,
    device=0,
    project=str(RUNS_DIR),
    save=True,
    save_period=10,
    plots=True,
    val=True,
    verbose=True,

    # ── Face-specific augmentasyon ──────────────────────────────────
    mosaic=1.0,           # Küçük yüz için kritik
    mixup=0.1,
    degrees=15.0,         # Güvenlik kamerası açıları
    fliplr=0.5,
    scale=0.6,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.3,          # Kısmi oklüzyon simülasyonu (maske/gözlük)
    blur=0.1,             # Hareket bulanıklığı

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    weight_decay=0.0005,

    # ── Küçük nesne odaklı loss ─────────────────────────────────────
    box=7.5,              # Box loss ağırlığı
    cls=0.5,              # Cls loss (tek sınıf = düşük)
    dfl=1.5,
)


def print_gpu_info():
    print("\n" + "="*55)
    print("  SKYWATCH-Det Eğitim")
    print("="*55)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM     : {vram:.1f} GB")
        if vram < 5:
            print("  ⚠️  VRAM düşük → batch=8 önerilir")
    print("="*55 + "\n")


def train_baseline():
    """Standart YOLOv8s-P2 — karşılaştırma için baseline."""
    print("🔵 BASELINE: yolov8s-p2 eğitimi başlıyor...")
    model = YOLO("yolov8s-p2.pt")  # pretrained weights
    model.train(
        **TRAIN_ARGS,
        name="baseline_yolov8s_p2",
        exist_ok=True,
    )
    metrics = model.val()
    print_results("BASELINE", metrics)
    return metrics


def train_skywatch():
    """SKYWATCH-Det: CBAM-Enhanced P2 modeli."""
    print("🔴 SKYWATCH-Det eğitimi başlıyor...")

    # Custom YAML'dan başla (pretrained backbone ağırlıkları ile)
    model = YOLO(str(ULTRA_CFG))
    model.train(
        **TRAIN_ARGS,
        name="skywatch_det_cbam",
        exist_ok=True,
        pretrained=True,   # ImageNet pretrained backbone kullan
    )
    metrics = model.val()
    print_results("SKYWATCH-Det", metrics)
    return metrics


def print_results(name: str, metrics):
    print(f"\n{'='*55}")
    print(f"  📊 {name} Sonuçları")
    print(f"{'='*55}")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print(f"{'='*55}\n")


def compare(baseline_m, skywatch_m):
    delta_map50 = skywatch_m.box.map50 - baseline_m.box.map50
    delta_map   = skywatch_m.box.map   - baseline_m.box.map
    print(f"\n{'='*55}")
    print(f"  📈 KARŞILAŞTIRMA")
    print(f"{'='*55}")
    print(f"  {'Metrik':<20} {'Baseline':>12} {'SKYWATCH':>12} {'Delta':>10}")
    print(f"  {'-'*54}")
    print(f"  {'mAP@0.5':<20} {baseline_m.box.map50:>12.4f} {skywatch_m.box.map50:>12.4f} {delta_map50:>+10.4f}")
    print(f"  {'mAP@0.5:0.95':<20} {baseline_m.box.map:>12.4f} {skywatch_m.box.map:>12.4f} {delta_map:>+10.4f}")
    print(f"  {'Precision':<20} {baseline_m.box.mp:>12.4f} {skywatch_m.box.mp:>12.4f} {skywatch_m.box.mp-baseline_m.box.mp:>+10.4f}")
    print(f"  {'Recall':<20} {baseline_m.box.mr:>12.4f} {skywatch_m.box.mr:>12.4f} {skywatch_m.box.mr-baseline_m.box.mr:>+10.4f}")
    print(f"{'='*55}")
    sign = "✅" if delta_map50 > 0 else "❌"
    print(f"  {sign} CBAM katkısı mAP@0.5'i {'artırdı' if delta_map50 > 0 else 'azalttı'}: {delta_map50:+.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "skywatch", "both"],
                        default="skywatch", help="Hangi modeli eğiteceğin")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA bulunamadı!")
        sys.exit(1)

    print_gpu_info()

    if args.mode == "baseline":
        train_baseline()

    elif args.mode == "skywatch":
        train_skywatch()

    elif args.mode == "both":
        b_metrics = train_baseline()
        s_metrics = train_skywatch()
        compare(b_metrics, s_metrics)
