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
    print("  (Cache'de varsa aninda hazir olur)")
    raw_path = kagglehub.dataset_download(DATASET_SLUG)
    DATA_RAW = Path(raw_path)
    print(f"  Dataset path: {DATA_RAW}")
    return DATA_RAW


def prepare_data():
    """kagglehub ile dataset'i indir, DATA_YAML global'ini ayarla.

    lylmsc/wider-face-for-yolo-training YAPISI (gercek):
      images/  [12,880 dosya - duz, train/val YOKU]
      labels/  [12,880 dosya - duz, train/val YOKU]

    Bu fonksiyon:
      1. kagglehub ile indirir
      2. Flat yapiyi tespit eder
      3. %%90 train / %%10 val olarak boler (symlink - hizli)
      4. data.yaml olusturur -> DATA_YAML global'ini ayarlar
    """
    global DATA_YAML

    # 1. Dataset'i indir (veya cache'den yukle)
    download_dataset()

    print("\n" + "="*55)
    print("  VERI SETI KONTROLU")
    print("="*55)
    print(f"  Kaynak: {DATA_RAW}")

    # 2. Yapiyi tespit et ve hazirla
    images_dir = DATA_RAW / "images"
    subdirs = [d for d in images_dir.iterdir() if d.is_dir()] if images_dir.exists() else []
    is_flat = images_dir.exists() and len(subdirs) == 0

    if is_flat:
        print("  Yapi: FLAT (train/val yok) -> otomatik bolunuyor...")
        DATA_YAML = _split_flat_dataset()
    else:
        print("  Yapi: SPLIT (train/val mevcut) -> direkt kullaniliyor")
        DATA_YAML = _create_data_yaml_for_split()

    # Istatistik
    for split in ("train", "val"):
        img_dir = DATA_YAML.parent / "images" / split
        if img_dir.exists():
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            print(f"  {split:5s}: {len(imgs):,} goruntu")

    print(f"  data.yaml: {DATA_YAML}")


def _split_flat_dataset() -> Path:
    """Flat dataset'i %%90 train / %%10 val olarak bol, symlink ile bagla.

    Cikti: /kaggle/working/skywatch_data/
      images/train/  <- symlink
      images/val/    <- symlink
      labels/train/  <- symlink
      labels/val/    <- symlink
      data.yaml
    """
    import random

    SPLIT_DIR = WORKING / "skywatch_data"
    data_yaml = SPLIT_DIR / "data.yaml"

    # Zaten bolunmusse atla
    train_dir = SPLIT_DIR / "images" / "train"
    if data_yaml.exists() and train_dir.exists() and any(train_dir.iterdir()):
        print(f"  Split data zaten hazir: {SPLIT_DIR}")
        return data_yaml

    images_src = DATA_RAW / "images"
    labels_src = DATA_RAW / "labels"

    # Tum gorsuntuleri listele
    all_images = sorted(images_src.glob("*.jpg")) + sorted(images_src.glob("*.png"))
    if not all_images:
        raise FileNotFoundError(f"images/ klasorunde goruntu bulunamadi: {images_src}")

    # %%90 / %%10 split (seed=42)
    import random
    random.seed(42)
    shuffled = all_images[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.9)
    splits = {"train": shuffled[:n_train], "val": shuffled[n_train:]}
    print(f"  Toplam: {len(all_images):,} goruntu")
    print(f"  Train:  {len(splits['train']):,} | Val: {len(splits['val']):,}")

    # Cikti dizinleri olustur
    for split in ("train", "val"):
        (SPLIT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Symlink ile bagla (kopyalama yok -> hizli)
    linked = 0
    for split, img_list in splits.items():
        for img_path in img_list:
            stem = img_path.stem
            dst_img = SPLIT_DIR / "images" / split / img_path.name
            if not dst_img.exists():
                os.symlink(str(img_path), str(dst_img))
            lbl_src = labels_src / f"{stem}.txt"
            dst_lbl = SPLIT_DIR / "labels" / split / f"{stem}.txt"
            if lbl_src.exists() and not dst_lbl.exists():
                os.symlink(str(lbl_src), str(dst_lbl))
                linked += 1
    print(f"  Symlink: {linked:,} label baglandi")

    # data.yaml olustur
    yaml_content = (
        f"# SKYWATCH-Det WIDER FACE YOLO\n"
        f"path: {str(SPLIT_DIR).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['face']\n\n"
        f"# lylmsc/wider-face-for-yolo-training\n"
        f"# {len(splits['train']):,} train | {len(splits['val']):,} val | face: 10px~500px\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml: {data_yaml}")
    return data_yaml


def _create_data_yaml_for_split() -> Path:
    """Train/val split'li dataset icin data.yaml bul veya olustur."""
    candidates = [
        DATA_RAW / "dataset.yaml",
        DATA_RAW / "data.yaml",
        WORKING / "skywatch_data" / "data.yaml",
    ]
    for c in candidates:
        if c.exists():
            print(f"  YAML bulundu: {c}")
            return c
    # images/train'i bul ve yaml olustur
    for root in [DATA_RAW, WORKING / "skywatch_data"]:
        if (root / "images" / "train").exists():
            yaml_path = root / "data.yaml"
            yaml_content = (
                f"path: {str(root).replace(chr(92), '/')}\n"
                f"train: images/train\nval: images/val\n\n"
                f"nc: 1\nnames: ['face']\n"
            )
            yaml_path.write_text(yaml_content, encoding="utf-8")
            return yaml_path
    raise FileNotFoundError("data.yaml ve images/train bulunamadi!")


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
