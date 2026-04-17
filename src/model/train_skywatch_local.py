"""
SKYWATCH-Det — Lokal RTX 3070 Ti Eğitim Scripti
================================================
Dataset : C:/Users/musta/OneDrive/Desktop/SKYWATCH/archive/
Format  : YOLO normalize (cx cy w h) — labels/ klasörü
Model   : skywatch-det.yaml (C2f_CAM + FRM + P2 head)
GPU     : RTX 3070 Ti (8 GB VRAM)

Dataset İstatistikleri (analiz sonucundan):
  - 13,386 train / 3,347 val görüntü
  - Yüz büyüklüğü: %44.9 tiny, %40.2 small → %85.1 küçük yüz!
  - Ort. 3.5 yüz/görüntü, max 153 → kalabalık sahneler

Kullanım:
    python src/model/train_skywatch_local.py
    python src/model/train_skywatch_local.py --mode baseline
    python src/model/train_skywatch_local.py --resume runs/skywatch_det_local/weights/last.pt
"""

import sys
import argparse
from pathlib import Path

# ── PATH KURULUMU ──────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]   # SKYWATCH/
MODEL_DIR = Path(__file__).parent  # SKYWATCH/src/model/

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR))

# ══════════════════════════════════════════════════════════════════════
# YOLLAR
# ══════════════════════════════════════════════════════════════════════

DATASET_YAML = ROOT / "archive" / "data.yaml"
MODEL_YAML   = ROOT / "src" / "ultralytics_patch" / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"
RUNS_DIR     = MODEL_DIR / "runs"

# ══════════════════════════════════════════════════════════════════════
# HİPERPARAMETRELER
# ══════════════════════════════════════════════════════════════════════
# RTX 3070 Ti: 8GB VRAM
# batch=8 → nbs=64 → grad accum=8 — 64 batch efekti
# imgsz=640 — standart
#
# Dataset bulguları:
#   - %85.1 küçük yüz → SkyWatchBboxLoss + küçük yüz augm. kritik
#   - Max 153 yüz/görüntü → mosaic=1.0 ŞART
#   - 13K train → warmup kısa tutulabilir, patience=40 yeterli
# ══════════════════════════════════════════════════════════════════════

TRAIN_ARGS = dict(
    # ── Temel ────────────────────────────────────────────────────
    data       = str(DATASET_YAML),
    model      = str(MODEL_YAML),
    project    = str(RUNS_DIR),
    name       = "skywatch_det_local",
    exist_ok   = True,

    # ── Epoch & Batch ────────────────────────────────────────────
    epochs     = 150,          # 13K görüntü için yeterli
    imgsz      = 640,
    batch      = 8,            # RTX 3070 Ti 8GB için güvenli sınır
    # nbs=64: ultralytics nominal batch size — grad accumulation ile
    # etkin batch boyutu = batch * accum = 8 * 8 = 64
    nbs        = 64,
    workers    = 4,
    device     = 0,            # Tek GPU

    # ── Optimizer ────────────────────────────────────────────────
    optimizer       = "AdamW",
    lr0             = 0.001,   # Başlangıç LR
    lrf             = 0.01,    # Final LR = lr0 * lrf = 0.00001
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 5,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,
    patience        = 40,      # 13K dataset küçük → çok bekleme

    # ── Loss Ağırlıkları ─────────────────────────────────────────
    # %85 küçük yüz → box losu ağır tut
    box  = 8.5,
    cls  = 0.5,
    dfl  = 2.0,

    # ── Augmentasyon ─────────────────────────────────────────────
    # Kalabalık sahneler (max=153 yüz) → mosaic ŞART
    mosaic       = 1.0,
    mixup        = 0.05,        # Hafif mixup
    degrees      = 10.0,
    fliplr       = 0.5,
    flipud       = 0.0,
    scale        = 0.5,
    shear        = 0.0,
    perspective  = 0.0005,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    # erasing: küçük yüzlerin bozulmasını engellemek için hafif tut
    erasing      = 0.1,
    # Küçük yüz dataset → multi_scale faydalı ama OOM riski
    # VRAM izleyip açabilirsin
    multi_scale  = False,
    # Son 10 epoch'ta mosaici kapat → val metriği daha gerçekçi
    close_mosaic = 10,

    # ── Kaydetme ────────────────────────────────────────────────
    save        = True,
    save_period = 10,          # Her 10 epoch'ta .pt kaydet
    val         = True,
    plots       = True,
    verbose     = True,
)


# ══════════════════════════════════════════════════════════════════════
# BASELINE: Karşılaştırma için standart YOLOv8s-P2
# ══════════════════════════════════════════════════════════════════════

BASELINE_ARGS = dict(
    **{k: v for k, v in TRAIN_ARGS.items()
       if k not in ("model", "name")},
    model  = "yolov8s.pt",     # Pretrained backbone
    name   = "baseline_yolov8s",
)


# ══════════════════════════════════════════════════════════════════════
# PATCH KONTROLÜ
# ══════════════════════════════════════════════════════════════════════

def ensure_patch():
    """Lokal ultralytics patch'ini kontrol et ve uygula."""
    print("\n[PATCH] Ultralytics custom modüller kontrol ediliyor...")
    try:
        from ultralytics.nn.modules import C2f_CAM, FRM
        print("  ✅ C2f_CAM, FRM aktif")
        return True
    except ImportError:
        pass

    print("  ⚠️  Custom modüller bulunamadı, lokal patch deneniyor...")

    # Lokal patch klasörü
    patch_src = ROOT / "src" / "ultralytics_patch" / "nn" / "modules" / "skywatch_modules.py"
    if not patch_src.exists():
        print(f"  ❌ skywatch_modules.py bulunamadı: {patch_src}")
        return False

    try:
        import ultralytics
        import shutil
        ult_dir = Path(ultralytics.__file__).parent
        dst = ult_dir / "nn" / "modules" / "skywatch_modules.py"
        shutil.copy2(patch_src, dst)

        init_path = ult_dir / "nn" / "modules" / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        if "skywatch_modules" not in content:
            with open(init_path, "a", encoding="utf-8") as f:
                f.write("\nfrom .skywatch_modules import C2f_CAM, FRM\n")

        # tasks.py patch
        tasks_path = ult_dir / "nn" / "tasks.py"
        tasks_content = tasks_path.read_text(encoding="utf-8")
        import_line = "from ultralytics.nn.modules.skywatch_modules import C2f_CAM, FRM"
        if import_line not in tasks_content:
            import re
            tasks_content, n = re.subn(r'\bC2f\b(?=\s*,)', 'C2f, C2f_CAM, FRM', tasks_content, count=1)
            if n == 0:
                tasks_content = import_line + "\n" + tasks_content
            tasks_path.write_text(tasks_content, encoding="utf-8")

        # YAML kopyala
        src_yaml = ROOT / "src" / "ultralytics_patch" / "cfg" / "models" / "skywatch"
        dst_yaml = ult_dir / "cfg" / "models" / "skywatch"
        if src_yaml.exists():
            shutil.copytree(src_yaml, dst_yaml, dirs_exist_ok=True)

        # Modülleri sıfırla
        import importlib
        for key in list(sys.modules.keys()):
            if "ultralytics" in key:
                del sys.modules[key]

        from ultralytics.nn.modules import C2f_CAM, FRM
        print("  ✅ Patch uygulandı, C2f_CAM ve FRM aktif")
        return True

    except Exception as e:
        print(f"  ❌ Patch hatası: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# DATASET DOĞRULAMA
# ══════════════════════════════════════════════════════════════════════

def validate_dataset():
    """Dataset ve data.yaml kontrolü."""
    print("\n[DATA] Dataset doğrulanıyor...")

    if not DATASET_YAML.exists():
        print(f"  ❌ data.yaml bulunamadı: {DATASET_YAML}")
        print("     Önce archive/analyze_dataset.py çalıştır!")
        return False

    train_dir = ROOT / "archive" / "images" / "train"
    val_dir   = ROOT / "archive" / "images" / "val"

    train_n = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
    val_n   = len(list(val_dir.glob("*.jpg")))   + len(list(val_dir.glob("*.png")))

    print(f"  ✅ data.yaml: {DATASET_YAML}")
    print(f"  📁 Train: {train_n:,} görüntü")
    print(f"  📁 Val:   {val_n:,} görüntü")

    lbl_train = ROOT / "archive" / "labels" / "train"
    lbl_n = len(list(lbl_train.glob("*.txt"))) if lbl_train.exists() else 0
    print(f"  📋 Labels/train: {lbl_n:,} dosya")
    return True


def run_channel_preflight(input_channels: int = 3, imgsz: int = 640):
    """Eğitimden önce model kanal uyumunu doğrula."""
    try:
        from src.tools.channel_validator import validate_model_channels
    except Exception as e:
        print(f"\n[CHECK] Validator import hatası: {e}")
        print("[CHECK] Kanal kontrolü atlanıyor, eğitime devam ediliyor...")
        return

    print("\n[CHECK] Kanal preflight kontrolü...")
    try:
        result = validate_model_channels(
            model_yaml=MODEL_YAML,
            input_channels=input_channels,
            imgsz=imgsz,
            verbose=True,
        )
    except Exception as e:
        print(f"\n[CHECK] Doğrulama sırasında hata: {e}")
        print("[CHECK] Kanal kontrolü atlanıyor, eğitime devam ediliyor...")
        return

    if not result["ok"]:
        err = result.get("error", "Bilinmeyen hata")
        print(f"\n[CHECK] Doğrulama başarısız: {err}")
        print("[CHECK] Eğitime yine de devam ediliyor — ilk forward'da fail-fast yakalar.")


# ══════════════════════════════════════════════════════════════════════
# ANA EĞİTİM FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════

def train_skywatch(resume: str = ""):
    """SKYWATCH-Det: C2f_CAM + FRM + SkyWatchDetectionLoss + FaceOcclusionAug."""
    from skywatch_trainer import SkyWatchTrainer

    print("\n" + "="*60)
    print("  SKYWATCH-Det — Lokal Eğitim")
    print(f"  Model  : {MODEL_YAML.name}")
    print(f"  Epoch  : {TRAIN_ARGS['epochs']} | LR0: {TRAIN_ARGS['lr0']}")
    print(f"  Batch  : {TRAIN_ARGS['batch']} (nbs={TRAIN_ARGS.get('nbs', 64)} → effektif 64)")
    print(f"  Dataset: {DATASET_YAML}")
    print("="*60)

    # Eğitimden önce kanal-mimari uyumunu zorunlu doğrula
    run_channel_preflight(input_channels=3, imgsz=TRAIN_ARGS["imgsz"])

    overrides = dict(**TRAIN_ARGS)
    if resume:
        overrides["model"]  = resume
        overrides["resume"] = True
        print(f"  ↺ Resume: {resume}")

    trainer = SkyWatchTrainer(overrides=overrides)
    trainer.train()

    # Sonuçları raporla
    best = RUNS_DIR / TRAIN_ARGS["name"] / "weights" / "best.pt"
    if best.exists():
        print(f"\n  ✅ En iyi model: {best}")
        validate_model(str(best))
    return str(best)


def train_baseline():
    """Baseline YOLOv8s — karşılaştırma."""
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("  BASELINE — YOLOv8s")
    print("="*60)

    model = YOLO("yolov8s.pt")
    model.train(**BASELINE_ARGS)

    best = RUNS_DIR / "baseline_yolov8s" / "weights" / "best.pt"
    print(f"\n  ✅ Baseline en iyi: {best}")
    validate_model(str(best))
    return str(best)


def validate_model(weights: str):
    """Val seti üzerinde metrik ölç."""
    from ultralytics import YOLO

    print(f"\n[VAL] Validasyon: {Path(weights).parent.parent.name}")
    model   = YOLO(weights)
    metrics = model.val(data=str(DATASET_YAML), imgsz=640, verbose=False)

    box = metrics.box
    print(f"  mAP@50    : {box.map50:.4f}")
    print(f"  mAP@50:95 : {box.map:.4f}")
    print(f"  Precision : {box.mp:.4f}")
    print(f"  Recall    : {box.mr:.4f}")
    return metrics


def compare_results():
    """Mevcut run'ları karşılaştır."""
    sw_best   = RUNS_DIR / "skywatch_det_local"   / "weights" / "best.pt"
    base_best = RUNS_DIR / "baseline_yolov8s"     / "weights" / "best.pt"

    if not sw_best.exists() or not base_best.exists():
        print("Karşılaştırma için her iki model de eğitilmeli!")
        return

    from ultralytics import YOLO

    def get_metrics(path):
        m = YOLO(str(path)).val(data=str(DATASET_YAML), imgsz=640, verbose=False)
        return m.box

    print("\n" + "="*60)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("="*60)

    b = get_metrics(base_best)
    s = get_metrics(sw_best)

    rows = [
        ("mAP@0.5",      b.map50, s.map50),
        ("mAP@0.5:0.95", b.map,   s.map),
        ("Precision",    b.mp,    s.mp),
        ("Recall",       b.mr,    s.mr),
    ]
    print(f"  {'Metrik':<20} {'Baseline':>10} {'SKYWATCH':>10} {'Delta':>8}")
    print("  " + "-"*50)
    for name, bv, sv in rows:
        d = sv - bv
        print(f"  {name:<20} {bv:>10.4f} {sv:>10.4f} {d:>+8.4f}")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SKYWATCH-Det Lokal Eğitici")
    parser.add_argument(
        "--mode",
        choices=["skywatch", "baseline", "compare"],
        default="skywatch",
        help="Eğitim modu (varsayılan: skywatch)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Devam et: last.pt yolu (sadece --mode skywatch)"
    )
    args = parser.parse_args()

    # Patch + dataset kontrolü
    patch_ok = ensure_patch()
    if not patch_ok and args.mode == "skywatch":
        print("\n❌ Custom modüller yüklenemedi. Eğitim iptal.")
        return

    if not validate_dataset():
        return

    # Eğitim modunu çalıştır
    if args.mode == "skywatch":
        train_skywatch(resume=args.resume)
    elif args.mode == "baseline":
        train_baseline()
    elif args.mode == "compare":
        compare_results()


if __name__ == "__main__":
    main()
