"""
SKYWATCH-Det — Kaggle 2xT4 Eğitim Scripti
==========================================
2 Fazlı Eğitim Stratejisi:
  Faz 1 → Normal LR, başlangıç eğitimi
  Faz 2 → best.pt'den devam, düşük LR + yüksek epoch (fine-tune)

Kullanım (Kaggle Notebook):
  !python train_skywatch_kaggle.py --phase 1
  !python train_skywatch_kaggle.py --phase 2
  veya tek seferde:
  !python train_skywatch_kaggle.py --phase all
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

# ══════════════════════════════════════════════════════════
# KAGGLE YOLLARI — Veri setine göre düzelt
# ══════════════════════════════════════════════════════════

WORKING       = Path("/kaggle/working")
REPO_DIR      = WORKING / "SKYWATCH"
DATA_RAW      = Path("/kaggle/input/wider-face")   # ← Kaggle dataset adı
DATA_YOLO     = WORKING / "skywatch_data"           # Dönüştürülmüş YOLO verisi
RUNS_DIR      = WORKING / "runs"
CHECKPOINT_DIR = WORKING / "checkpoints"            # Zip checkpoint dizini
GITHUB_REPO   = "https://github.com/mustafabsnl/SKYWATCH.git"  # ← GitHub adresiniz

# Model YAML (repo içindeki yol)
MODEL_YAML    = REPO_DIR / "src/ultralytics_patch/cfg/models/skywatch/skywatch-det.yaml"
DATA_YAML     = DATA_YOLO / "data.yaml"

# Her kaç epoch'ta ZIP alınsın
CHECKPOINT_EVERY_N = 10

# ══════════════════════════════════════════════════════════
# FAZ 1 HİPERPARAMETRELERİ
# ══════════════════════════════════════════════════════════
PHASE1 = dict(
    epochs          = 100,
    imgsz           = 640,
    batch           = -1,          # Auto-batch (2xT4 → ~32)
    lr0             = 0.001,       # Başlangıç LR
    lrf             = 0.01,        # Final LR katsayısı (lr0 * lrf)
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 5,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,
    optimizer       = "AdamW",
    patience        = 50,
    save_period     = 10,
    # Augmentation
    mosaic          = 1.0,
    mixup           = 0.05,
    degrees         = 10.0,
    fliplr          = 0.5,
    perspective     = 0.0005,
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    erasing         = 0.4,
    multi_scale     = True,
    close_mosaic    = 10,
    # Loss ağırlıkları
    box             = 7.5,
    cls             = 0.5,
    dfl             = 1.5,
    pose            = 12.0,
    kobj            = 1.0,
    # Kaydetme
    name            = "skywatch_det_phase1",
    exist_ok        = True,
    plots           = True,
    save            = True,
    val             = True,
    verbose         = True,
)

# ══════════════════════════════════════════════════════════
# FAZ 2 HİPERPARAMETRELERİ (Fine-tune)
# ══════════════════════════════════════════════════════════
PHASE2 = dict(
    epochs          = 300,
    imgsz           = 640,
    batch           = 8,           # Faz 2'de sabit batch
    lr0             = 0.0002,      # Düşük LR (fine-tune)
    lrf             = 0.005,       # Çok küçük final LR
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 1.0,         # Minimal warmup
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.01,
    optimizer       = "AdamW",
    patience        = 80,          # Daha uzun sabır
    save_period     = 25,
    # Fine-tune: Daha az agresif augmentation
    mosaic          = 0.5,         # Azaltıldı
    mixup           = 0.0,         # Kapatıldı
    degrees         = 5.0,
    fliplr          = 0.5,
    perspective     = 0.0,
    hsv_h           = 0.01,
    hsv_s           = 0.4,
    hsv_v           = 0.3,
    erasing         = 0.2,         # Azaltıldı
    multi_scale     = False,       # Sabit ölçek (fine-tune)
    close_mosaic    = 30,
    # Loss ağırlıkları (aynı)
    box             = 7.5,
    cls             = 0.5,
    dfl             = 1.5,
    pose            = 12.0,
    kobj            = 1.0,
    # Kaydetme
    name            = "skywatch_det_phase2",
    exist_ok        = True,
    plots           = True,
    save            = True,
    val             = True,
    verbose         = True,
)


# ══════════════════════════════════════════════════════════
# ORTAM KURULUMU
# ══════════════════════════════════════════════════════════

def setup():
    """Repo clone + kaggle_setup.py kontrolu + path kurulumu."""
    print("\n" + "="*55)
    print("  ORTAM KURULUMU")
    print("="*55)

    # Repo clone
    if not REPO_DIR.exists():
        print(f"  GitHub clone: {GITHUB_REPO}")
        subprocess.run(["git", "clone", GITHUB_REPO, str(REPO_DIR)], check=True)
    else:
        print(f"  Repo mevcut: {REPO_DIR}")

    # Path'e ekle
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / "src" / "model"))

    # kaggle_setup.py calistir (C2f_CAM, FRM patch + YAML)
    setup_script = REPO_DIR / "kaggle_setup.py"
    if setup_script.exists():
        result = subprocess.run([sys.executable, str(setup_script)], capture_output=False)
        if result.returncode != 0:
            print("  [UYARI] kaggle_setup.py hata verdi, devam ediliyor...")
    else:
        print("  [UYARI] kaggle_setup.py bulunamadi!")

    # GPU kontrol
    import torch
    n = torch.cuda.device_count()
    print(f"  GPU: {n}x {torch.cuda.get_device_name(0) if n>0 else 'YOK'}")
    print(f"  PyTorch: {torch.__version__}")

    # Dogrula
    try:
        from ultralytics.nn.modules import C2f_CAM, FRM
        print("  Custom moduller aktif (C2f_CAM, FRM)")
    except ImportError as e:
        print(f"  [HATA] Custom modul yuklenemedi: {e}")

    return n  # GPU sayisi


# ══════════════════════════════════════════════════════════
# VERİ SETİ DÖNÜŞÜMÜ
# ══════════════════════════════════════════════════════════

def prepare_data():
    """WIDER FACE → YOLO Pose dönüşümü."""
    if DATA_YAML.exists():
        print(f"\n  Dataset hazır: {DATA_YOLO}")
        return

    print("\n" + "="*55)
    print("  VERİ SETİ DÖNÜŞÜMÜ")
    print("="*55)

    # WIDER FACE dizin yapısını bul (farklı Kaggle dataset'leri farklı yapıda olabilir)
    # Kullanıcı iç içe yapıyı düzeltti, düz yapıyı kullan
    wider_root = _find_wider_root()

    converter = REPO_DIR / "src/model/wider_face_converter.py"
    cmd = [
        sys.executable, str(converter),
        "--wider_root", str(wider_root),
        "--output_dir", str(DATA_YOLO),
        "--min_face_size", "5",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Dönüşüm başarısız!")

    print(f"\n  ✓ Dataset hazır: {DATA_YOLO}")


def _find_wider_root():
    """WIDER FACE root dizinini otomatik bul."""
    candidates = [
        DATA_RAW,
        DATA_RAW / "wider-face",
        Path("/kaggle/input/wider-face-dataset"),
        Path("/kaggle/input/widerface"),
    ]
    for c in candidates:
        ann = c / "wider_face_split" / "wider_face_train_bbx_gt.txt"
        if ann.exists():
            print(f"  WIDER FACE bulundu: {c}")
            return c
    raise FileNotFoundError(
        f"WIDER FACE bulunamadı! Denenen: {candidates}\n"
        "Lütfen Kaggle'a WIDER FACE dataseti ekleyin ve DATA_RAW yolunu güncelleyin."
    )


# ══════════════════════════════════════════════════════════
# FAZ 1 EĞİTİM
# ══════════════════════════════════════════════════════════

def train_phase1(n_gpu: int) -> str:
    """Faz 1: Pretrained backbone'dan baslayarak normal LR egitimi."""
    from ultralytics import YOLO
    from checkpoint_zipper import make_checkpoint_callback, make_final_zip

    print("\n" + "="*55)
    print("  FAZ 1 — Ilk Egitim")
    print(f"  Epoch: {PHASE1['epochs']} | LR0: {PHASE1['lr0']} | Batch: {PHASE1['batch']}")
    print(f"  ZIP her {CHECKPOINT_EVERY_N} epoch'ta: {CHECKPOINT_DIR / 'phase1'}")
    print("="*55)

    model = YOLO(str(MODEL_YAML))

    # Her 10 epoch'ta ZIP callback'i ekle
    ckpt_dir = CHECKPOINT_DIR / "phase1"
    model.add_callback(
        "on_fit_epoch_end",
        make_checkpoint_callback(ckpt_dir, every_n=CHECKPOINT_EVERY_N)
    )

    # Device ayari
    device = "0,1" if n_gpu >= 2 else "0"

    model.train(
        data    = str(DATA_YAML),
        project = str(RUNS_DIR),
        device  = device,
        **PHASE1,
    )

    # Faz 1 bitis ZIP'i
    run_dir = RUNS_DIR / PHASE1["name"]
    make_final_zip(run_dir, CHECKPOINT_DIR / "phase1", label="phase1_final")

    best = run_dir / "weights" / "best.pt"
    print(f"  En iyi model: {best}")
    return str(best)


# ══════════════════════════════════════════════════════════
# FAZ 2 EĞİTİM (Fine-tune)
# ══════════════════════════════════════════════════════════

def train_phase2(phase1_weights: str, n_gpu: int) -> str:
    """Faz 2: best.pt'den dusuk LR ile uzun fine-tune."""
    from ultralytics import YOLO
    from checkpoint_zipper import make_checkpoint_callback, make_final_zip

    print("\n" + "="*55)
    print("  FAZ 2 — Fine-tune (Dusuk LR)")
    print(f"  Baslangic:  {phase1_weights}")
    print(f"  Epoch: {PHASE2['epochs']} | LR0: {PHASE2['lr0']} | Batch: {PHASE2['batch']}")
    print(f"  Patience: {PHASE2['patience']}")
    print(f"  ZIP her {CHECKPOINT_EVERY_N} epoch'ta: {CHECKPOINT_DIR / 'phase2'}")
    print("="*55)

    # Faz 1'in best.pt'sini yukle
    model = YOLO(phase1_weights)

    # Her 10 epoch'ta ZIP callback'i ekle
    ckpt_dir = CHECKPOINT_DIR / "phase2"
    model.add_callback(
        "on_fit_epoch_end",
        make_checkpoint_callback(ckpt_dir, every_n=CHECKPOINT_EVERY_N)
    )

    device = "0,1" if n_gpu >= 2 else "0"

    model.train(
        data    = str(DATA_YAML),
        project = str(RUNS_DIR),
        device  = device,
        **PHASE2,
    )

    # Faz 2 bitis ZIP'i
    run_dir = RUNS_DIR / PHASE2["name"]
    make_final_zip(run_dir, CHECKPOINT_DIR / "phase2", label="phase2_final")

    best = run_dir / "weights" / "best.pt"
    print(f"  Final model: {best}")
    return str(best)


# ══════════════════════════════════════════════════════════
# VALİDASYON
# ══════════════════════════════════════════════════════════

def validate(weights: str, label: str = ""):
    """Model performansını ölç."""
    from ultralytics import YOLO

    print(f"\n  Validasyon: {label} [{Path(weights).parent.parent.name}]")
    model = YOLO(weights)
    metrics = model.val(data=str(DATA_YAML), imgsz=640, verbose=False)

    try:
        print(f"  mAP@50:     {metrics.pose.map50:.4f}")
        print(f"  mAP@50:95:  {metrics.pose.map:.4f}")
        print(f"  Precision:  {metrics.pose.mp:.4f}")
        print(f"  Recall:     {metrics.pose.mr:.4f}")
    except:
        try:
            print(f"  mAP@50:     {metrics.box.map50:.4f}")
            print(f"  mAP@50:95:  {metrics.box.map:.4f}")
        except:
            print("  Metrikler alınamadı")

    return metrics


# ══════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SKYWATCH-Det Kaggle Eğitici")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Hangi faz çalıştırılsın: 1, 2 veya all (varsayılan: all)"
    )
    parser.add_argument(
        "--phase1_weights",
        type=str,
        default="",
        help="Faz 2 için kullanılacak Faz 1 ağırlık dosyası (--phase 2 ise zorunlu)"
    )
    parser.add_argument(
        "--skip_data_prep",
        action="store_true",
        help="Veri dönüşümünü atla (zaten yapıldıysa)"
    )
    args = parser.parse_args()

    # Kurulum
    n_gpu = setup()

    # Veri hazırlama
    if not args.skip_data_prep:
        prepare_data()

    # Eğitim
    if args.phase == "1":
        p1_weights = train_phase1(n_gpu)
        validate(p1_weights, "Faz 1")

    elif args.phase == "2":
        if not args.phase1_weights:
            # Otomatik bul
            auto = RUNS_DIR / PHASE1["name"] / "weights" / "best.pt"
            if not auto.exists():
                raise ValueError("--phase1_weights belirtilmedi ve otomatik bulunamadı!")
            args.phase1_weights = str(auto)
            print(f"  Faz 1 ağırlıkları: {args.phase1_weights}")
        p2_weights = train_phase2(args.phase1_weights, n_gpu)
        validate(p2_weights, "Faz 2")

    elif args.phase == "all":
        # Sırayla: 1 → 2
        p1_weights = train_phase1(n_gpu)
        validate(p1_weights, "Faz 1 Sonucu")
        p2_weights = train_phase2(p1_weights, n_gpu)
        validate(p2_weights, "Faz 2 Sonucu (Final)")

        print("\n" + "="*55)
        print("  EĞİTİM TAMAMLANDI")
        print(f"  Final model: {p2_weights}")
        print("="*55)


if __name__ == "__main__":
    main()
