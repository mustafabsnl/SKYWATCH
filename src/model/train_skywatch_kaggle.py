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
# KAGGLE YOLLARI
# Dataset: lylmsc/wider-face-for-yolo-training
#   - 32,203 görüntü (train+val zaten ayrılmış)
#   - ~393,703 etiketli yüz | yüz boyutu: 10px ~ 500px
#   - Format: YOLO detection (class cx cy w h) — dönüşüm GEREKSİZ
#   - Erişim: kagglehub.dataset_download() ile otomatik indirilir
# ══════════════════════════════════════════════════════════

WORKING        = Path("/kaggle/working")
REPO_DIR       = WORKING / "SKYWATCH"
# DATA_RAW: setup() → download_dataset() tarafından kagglehub ile belirlenir
DATA_RAW       = None
DATASET_SLUG   = "lylmsc/wider-face-for-yolo-training"  # kagglehub slug
RUNS_DIR       = WORKING / "runs"
CHECKPOINT_DIR = WORKING / "checkpoints"
GITHUB_REPO    = "https://github.com/mustafabsnl/SKYWATCH.git"

# SKYWATCH-Det: 4-scale detection head (P2/P3/P4/P5) + C2f_CAM + FRM
# Pose head KULLANILMIYOR — dataset'te landmark yok
MODEL_YAML     = "skywatch-det.yaml"
DATA_YAML      = None   # prepare_data() tarafından belirlenir

# Her kaç epoch'ta ZIP alınsın
CHECKPOINT_EVERY_N = 10

# ══════════════════════════════════════════════════════════
# FAZ 1 HİPERPARAMETRELERİ
# ══════════════════════════════════════════════════════════
PHASE1 = dict(
    epochs          = 100,
    imgsz           = 640,
    batch           = 16,          # 2xT4: 8 per GPU (multi_scale OOM nedeniyle azaltildi)
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
    # Augmentation — WIDER FACE: sokak kamerası simülasyonu
    # mosaic: kalabalık sahneler için kritik
    # blur varyasyonu: haar-like 'Hard' difficulty için
    mosaic          = 1.0,
    mixup           = 0.05,
    degrees         = 10.0,
    fliplr          = 0.5,
    perspective     = 0.0005,
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    erasing         = 0.4,
    # Dataset 10px~500px yüz içeriyor → multi_scale önemli
    # OOM riski: True yapmak istersen batch=8'e düşür
    multi_scale     = False,
    close_mosaic    = 10,
    # Loss — Detection modu (pose/kobj yok)
    box             = 7.5,
    cls             = 0.5,
    dfl             = 1.5,
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
    mosaic          = 0.5,
    mixup           = 0.0,
    degrees         = 5.0,
    fliplr          = 0.5,
    perspective     = 0.0,
    hsv_h           = 0.01,
    hsv_s           = 0.4,
    hsv_v           = 0.3,
    erasing         = 0.2,
    multi_scale     = False,
    close_mosaic    = 30,
    # Loss — Detection modu (pose/kobj yok)
    box             = 7.5,
    cls             = 0.5,
    dfl             = 1.5,
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
# VERİ SETİ HAZIRLIĞI
# Dataset zaten YOLO formatında — dönüşüm gerekmez
# kagglehub ile otomatik indirilir
# ══════════════════════════════════════════════════════════

def download_dataset() -> Path:
    """kagglehub ile dataset'i indir, gerçek path'i döndür.

    Not: Kaggle ortamında internet açık olmalıdır.
    Dataset zaten cache'deyse yeniden indirilmez.
    """
    global DATA_RAW
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub", "-q"], check=True)
        import kagglehub

    print(f"  Dataset indiriliyor: {DATASET_SLUG}")
    print("  (Cache'de varsa anında hazır olur)")
    raw_path = kagglehub.dataset_download(DATASET_SLUG)
    DATA_RAW = Path(raw_path)
    print(f"  ✓ Dataset path: {DATA_RAW}")
    return DATA_RAW


def prepare_data():
    """kagglehub ile dataset'i indir, DATA_YAML global'ini ayarla.

    lylmsc/wider-face-for-yolo-training zaten YOLO formatında:
      images/train/, images/val/, labels/train/, labels/val/
    kagglehub cache'den yükler (zaten indirildiyse anında hazır).
    """
    global DATA_YAML

    # 1. Dataset'i indir (veya cache'den yükle)
    download_dataset()

    print("\n" + "="*55)
    print("  VERİ SETİ KONTROLÜ")
    print("="*55)
    print(f"  Kaynak: {DATA_RAW}")

    # 2. data.yaml bul veya oluştur
    DATA_YAML = _find_or_create_data_yaml()

    # İstatistik
    for split in ("train", "val"):
        img_dir = DATA_YAML.parent / "images" / split
        if img_dir.exists():
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            print(f"  {split:5s}: {len(imgs):,} görüntü")

    print(f"  ✓ data.yaml: {DATA_YAML}")


def _find_or_create_data_yaml() -> Path:
    """dataset.yaml / data.yaml dosyasını bul; yoksa oluştur."""
    # Dataset içinde olası YAML konumları
    candidates = [
        DATA_RAW / "dataset.yaml",
        DATA_RAW / "data.yaml",
        DATA_RAW / "wider_face_yolo" / "dataset.yaml",
        DATA_RAW / "wider_face_yolo" / "data.yaml",
    ]
    for c in candidates:
        if c.exists():
            print(f"  YAML bulundu: {c}")
            return c

    # Bulunamazsa images/train'i baz alarak oluştur
    img_root = _find_images_train_root()
    yaml_path = img_root / "data.yaml"
    yaml_content = (
        f"# SKYWATCH-Det — WIDER FACE YOLO Dataset\n"
        f"path: {str(img_root).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['face']\n\n"
        f"# lylmsc/wider-face-for-yolo-training\n"
        f"# 32,203 images | ~393,703 faces | face size: 10px ~ 500px\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml oluşturuldu: {yaml_path}")
    return yaml_path


def _find_images_train_root() -> Path:
    """images/train klasörünü içeren root dizini döndür."""
    candidates = [
        DATA_RAW,
        DATA_RAW / "wider_face_yolo",
    ]
    for root in candidates:
        train_dir = root / "images" / "train"
        if train_dir.exists() and any(train_dir.iterdir()):
            return root
    raise FileNotFoundError(
        f"\n[HATA] images/train dizini bulunamadı!\n"
        f"  Kaggle notebook'una şu dataset eklenmiş mi?\n"
        f"  → lylmsc/wider-face-for-yolo-training\n"
        f"  Aranan kök: {[str(c) for c in candidates]}"
    )


# ══════════════════════════════════════════════════════════
# FAZ 1 EĞİTİM
# ══════════════════════════════════════════════════════════

def start_zip_monitor(run_dir: Path, checkpoint_dir: Path,
                       every_n: int = 10, interval: int = 90):
    """
    DDP ile uyumlu arka plan ZIP monitoru.
    model.train() blocking cagrisini beklerken results.csv'yi izler,
    her N epoch'ta checkpoint ZIP'i olusturur.
    """
    import threading, csv, zipfile, time

    def _zip_run(epoch: int):
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ts = __import__('datetime').datetime.now().strftime('%H%M')
            zip_path = checkpoint_dir / f"skywatch_epoch{epoch:04d}_{ts}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for f in sorted(run_dir.rglob('*')):
                    if f.is_file():
                        zf.write(f, f.relative_to(run_dir.parent))
            mb = zip_path.stat().st_size / 1e6
            print(f"\n  [ZIP ✓] Epoch {epoch} → {zip_path.name} ({mb:.1f} MB)")
        except Exception as e:
            print(f"\n  [ZIP HATA] Epoch {epoch}: {e}")

    def monitor():
        last_zipped = 0
        print(f"  [ZIP MON] Basladi — her {every_n} epoch'ta zip, {interval}s aralık")
        while True:
            try:
                csv_path = run_dir / "results.csv"
                if csv_path.exists():
                    with open(csv_path, newline='') as f:
                        epoch_count = max(0, sum(1 for _ in csv.reader(f)) - 1)  # header sak
                    if epoch_count > last_zipped and epoch_count % every_n == 0:
                        # Weight dosyasinin yazilmasini bekle (max 30s)
                        w = run_dir / "weights" / f"epoch{epoch_count}.pt"
                        waited = 0
                        while not w.exists() and waited < 30:
                            time.sleep(2); waited += 2
                        _zip_run(epoch_count)
                        last_zipped = epoch_count
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True, name="ZipMonitor")
    t.start()
    return t


def train_phase1(n_gpu: int) -> str:
    """Faz 1: Pretrained backbone'dan baslayarak normal LR egitimi."""
    from ultralytics import YOLO
    from checkpoint_zipper import make_final_zip

    print("\n" + "="*55)
    print("  FAZ 1 — Ilk Egitim")
    print(f"  Epoch: {PHASE1['epochs']} | LR0: {PHASE1['lr0']} | Batch: {PHASE1['batch']}")
    print(f"  ZIP her {CHECKPOINT_EVERY_N} epoch'ta: {CHECKPOINT_DIR / 'phase1'}")
    print("="*55)

    model = YOLO(str(MODEL_YAML))

    # Device ayari
    device = "0,1" if n_gpu >= 2 else "0"

    # ── DDP-uyumlu ZIP monitoru ─────────────────────────────────
    # add_callback DDP subprocess'te calismaz → arka plan thread kullan
    run_dir = RUNS_DIR / PHASE1["name"]
    ckpt_dir = CHECKPOINT_DIR / "phase1"
    start_zip_monitor(run_dir, ckpt_dir, every_n=CHECKPOINT_EVERY_N, interval=90)

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

    # Veri hazırlama — DATA_YAML global'ini belirle
    if not args.skip_data_prep:
        prepare_data()
    else:
        # --skip_data_prep: manuel olarak DATA_YAML bul
        global DATA_YAML
        DATA_YAML = _find_or_create_data_yaml()
        print(f"  [skip] Dataset: {DATA_YAML}")

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
