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

# ════════════════════════════════════════════════════════
# KAGGLE YOLLARI
# Dataset: fareselmenshawii/face-detection-dataset
#
#   YAPI (kagglehub ile indirildikten sonra):
#     <path>/
#       images/
#         train/   (13,386 görüntü)
#         val/     ( 3,347 görüntü)
#       labels/              ← YOLO normalize format (cx cy w h) ✔
#         train/   (13,386 .txt)
#         val/     ( 3,347 .txt)
#       labels2/             ← WIDER ham format — KULLANILMIYOR
#
#   Format: YOLO detection (class cx cy w h) — dönüşüm GEREKSİZ
#   nc: 1 | class_id: 0 | sinif: face
#
#   Yüz büyüklük dağılımı (analiz sonucundan):
#     tiny  (<5%) : %44.9  → P2 head KRİTİK
#     small (5-15%): %40.2  → SkyWatchBboxLoss small_w=2.0 DOGRULANMIS
#     TOPLAM %85.1 küçük yüz → mosaic=1.0 ŞART
# ════════════════════════════════════════════════════════

WORKING        = Path("/kaggle/working")
REPO_DIR       = WORKING / "SKYWATCH"
# DATA_RAW: setup() → download_dataset() tarafından kagglehub ile belirlenir
DATA_RAW       = None
# Kaggle dataset slug: fareselmenshawii/face-detection-dataset
# Yapı: images/train+val, labels/train+val (YOLO format)
DATASET_SLUG   = "fareselmenshawii/face-detection-dataset"
RUNS_DIR       = WORKING / "runs"
CHECKPOINT_DIR = WORKING / "checkpoints"
GITHUB_REPO    = "https://github.com/mustafabsnl/SKYWATCH.git"

# SKYWATCH-Det: 4-scale detection head (P2/P3/P4/P5) + C2f_CAM + FRM
# Pose head KULLANILMIYOR — dataset'te landmark yok
MODEL_YAML     = "skywatch-det.yaml"
DATA_YAML      = None   # prepare_data() tarafından belirlenir

# Her kaç epoch'ta snapshot alinsin (klasor yapisi: snapshot_epoch_N/)
CHECKPOINT_EVERY_N = 5

# ══════════════════════════════════════════════════════════
# FAZ 1 HİPERPARAMETRELERİ
# ══════════════════════════════════════════════════════════
PHASE1 = dict(
    epochs          = 100,
    imgsz           = 640,
    batch           = 8,           # 2xT4: 4 per GPU — OOM olmasin (nbs=64 ile grad accum)
    lr0             = 0.001,       # Başlangıç LR
    lrf             = 0.01,        # Final LR katsayısı (lr0 * lrf)
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 5,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,
    optimizer       = "AdamW",
    patience        = 50,
    save_period     = 5,           # ultralytics epoch5.pt, epoch10.pt... kaydetsin
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

    Train/Val zaten ayrılmış — split gerekmez.
    """
    global DATA_YAML

    # 1. Dataset'i indir (veya cache'den yukle)
    download_dataset()

    print("\n" + "="*55)
    print("  VERI SETI KONTROLU")
    print("="*55)
    print(f"  Kaynak: {DATA_RAW}")

    # 2. data.yaml bul veya olustur
    DATA_YAML = _find_or_create_data_yaml()
    print(f"  data.yaml: {DATA_YAML}")

    # Istatistik
    for split in ("train", "val", "validation"):  # her ikisini dene
        img_dir = DATA_YAML.parent / "images" / split
        if img_dir.exists():
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
            label = "val" if split != "train" else "train"
            print(f"  {label:5s}: {len(imgs):,} goruntu")


def _find_or_create_data_yaml() -> Path:
    """Dataset'e ait data.yaml'i her zaman tazenden olustur.

    KRITIK: Eski session'dan kalan /kaggle/working/skywatch_data/data.yaml
    farkli bir dataset'e ait olabilir. Bunu kullanma.
    Her zaman DATA_RAW'i baz alarak taze olustur.

    fareselmenshawii/face-detection-dataset yapisi:
      <path>/
        images/train/  [13,386 goruntu]
        images/val/    [ 3,347 goruntu]
        labels/train/  [YOLO normalize: cls cx cy w h]
        labels/val/    [YOLO normalize: cls cx cy w h]
        labels2/       [WIDER ham format - KULLANILMIYOR]
    """
    img_root = _find_images_train_root()

    # val klasoru: 'validation' mi 'val' mi?
    if (img_root / "images" / "validation").exists():
        val_dir = "images/validation"
    elif (img_root / "images" / "val").exists():
        val_dir = "images/val"
    else:
        val_dir = "images/val"  # fallback

    # Her zaman taze data.yaml olustur (eski session'dan kalan varsa uzerine yaz)
    yaml_path = WORKING / "skywatch_data" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = (
        f"# SKYWATCH-Det — iamtushara/face-detection-dataset\n"
        f"# Otomatik olusturuldu: {DATA_RAW}\n"
        f"path: {str(img_root).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: {val_dir}\n\n"
        f"nc: 1\n"
        f"names: ['face']\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml olusturuldu: {yaml_path}")
    print(f"    path : {img_root}")
    print(f"    train: images/train")
    print(f"    val  : {val_dir}")
    return yaml_path



def _fix_data_yaml_paths(yaml_path: Path) -> Path:
    """data.yaml icindeki relative path'leri mutlak yola cevir.

    Ornek: train: ../dataset/images/train  ->  path: /abs/path, train: images/train
    """
    import yaml as _yaml

    try:
        with open(yaml_path, encoding="utf-8") as f:
            d = _yaml.safe_load(f)
    except Exception:
        return yaml_path  # okunamadiysa oldugu gibi kullan

    # Zaten 'path' anahtari varsa ultralytics bunu halleder
    if "path" in d:
        return yaml_path

    # path yoksa: train/val degerlerinden root'u cikart
    train_val = d.get("train", "images/train")
    # Relative ise abs'e cevir
    if "images/train" in str(train_val):
        img_root = yaml_path.parent
        fixed_yaml = WORKING / "skywatch_data" / "data.yaml"
        fixed_yaml.parent.mkdir(parents=True, exist_ok=True)
        fixed_content = (
            f"path: {str(img_root).replace(chr(92), '/')}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"nc: {d.get('nc', 1)}\n"
            f"names: {d.get('names', ['face'])}\n"
        )
        fixed_yaml.write_text(fixed_content, encoding="utf-8")
        print(f"  Path duzeltildi: {fixed_yaml}")
        return fixed_yaml

    return yaml_path


def _find_images_train_root() -> Path:
    """images/train klasorunu iceren root dizini dondur (recursive).

    fareselmenshawii/face-detection-dataset yapisi:
      <DATA_RAW>/images/train/      ← dogrudan root'ta
      <DATA_RAW>/images/val/
      <DATA_RAW>/labels/train/      ← YOLO normalize format
      <DATA_RAW>/labels/val/

    Fallback olarak alt dizinleri de tarar.
    """
    known = [
        DATA_RAW,                              # dogrudan root (fareselmenshawii)
        DATA_RAW / "dataset",                  # alternatif alt klasor
        DATA_RAW / "merged",                   # eski iamtushara yapisi
        WORKING / "skywatch_data",
    ]
    for root in known:
        if root and (root / "images" / "train").exists():
            print(f"  images/train bulundu: {root}")
            return root

    for dirpath, dirnames, _ in os.walk(str(DATA_RAW)):
        depth = str(dirpath).replace(str(DATA_RAW), "").count(os.sep)
        if depth > 3:
            dirnames.clear()
            continue
        if "images" in dirnames and (Path(dirpath) / "images" / "train").exists():
            return Path(dirpath)

    # Hata: yapiyi goster
    print(f"\n  Dizin yapisi ({DATA_RAW}):")
    for dirpath, dirnames, files in os.walk(str(DATA_RAW)):
        depth = str(dirpath).replace(str(DATA_RAW), "").count(os.sep)
        if depth > 3:
            dirnames.clear()
            continue
        print(f"  {'  '*depth}{Path(dirpath).name}/ [{len(files)} dosya]")
    raise FileNotFoundError(
        f"images/train dizini bulunamadi! kagglehub path: {DATA_RAW}"
    )


# ══════════════════════════════════════════════════════════
# FAZ 1 EĞİTİM
# ══════════════════════════════════════════════════════════

def start_zip_monitor(run_dir: Path, checkpoint_dir: Path,
                       every_n: int = 5, interval: int = 90):
    """
    DDP ile uyumlu arka plan snapshot monitoru.
    model.train() blocking cagrisini beklerken results.csv'yi izler,
    her N epoch'ta tam snapshot klasoru olusturur:

      checkpoint_dir/
        snapshot_epoch_5/
          weights/
            best.pt
            last.pt
          args.yaml
          results.csv
          loss_curves.png / metric_curves.png / lr_curve.png
          snapshot_summary.json
          training_config.json
    """
    import threading, csv, shutil, time, json
    import datetime as _dt

    def _save_snapshot(epoch: int):
        """Her N epoch'ta cagrilanarak tam snapshot klasoru olusturur."""
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            snap_dir = checkpoint_dir / f"snapshot_epoch_{epoch}"
            snap_dir.mkdir(exist_ok=True)

            # ── 1. weights/ kopyala ──────────────────────────────
            w_src = run_dir / "weights"
            w_dst = snap_dir / "weights"
            w_dst.mkdir(exist_ok=True)
            for wname in ["best.pt", "last.pt", f"epoch{epoch}.pt"]:
                src = w_src / wname
                if src.exists():
                    shutil.copy2(src, w_dst / wname)

            # ── 2. Metadata dosyalari ────────────────────────────
            for fname in ["args.yaml", "results.csv"]:
                src = run_dir / fname
                if src.exists():
                    shutil.copy2(src, snap_dir / fname)

            # ── 3. PNG grafikler ─────────────────────────────────
            for png in run_dir.glob("*.png"):
                shutil.copy2(png, snap_dir / png.name)

            # ── 4. snapshot_summary.json ─────────────────────────
            metrics = {}
            csv_path = run_dir / "results.csv"
            if csv_path.exists():
                with open(csv_path, newline='') as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    metrics = {k.strip(): v.strip() for k, v in last.items()}

            summary = {
                "epoch": epoch,
                "timestamp": _dt.datetime.now().isoformat(),
                "saved_weights": [p.name for p in w_dst.iterdir() if p.suffix == ".pt"],
                "metrics": metrics,
            }
            with open(snap_dir / "snapshot_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # ── 5. training_config.json ──────────────────────────
            config = {
                "model": str(MODEL_YAML),
                "dataset": str(DATASET_SLUG),
                "checkpoint_every_n": every_n,
                "phase1": {k: str(v) if isinstance(v, Path) else v
                           for k, v in PHASE1.items()},
            }
            with open(snap_dir / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Boyut raporu
            total_mb = sum(p.stat().st_size for p in snap_dir.rglob("*") if p.is_file()) / 1e6
            n_weights = len(list(w_dst.glob("*.pt")))
            print(f"\n  [SNAP ✓] snapshot_epoch_{epoch}/ "
                  f"({n_weights} weight, {total_mb:.1f} MB)")

        except Exception as e:
            print(f"\n  [SNAP HATA] Epoch {epoch}: {e}")
            import traceback; traceback.print_exc()

    def monitor():
        last_snapped = 0
        print(f"  [ZIP MON] Basladi — her {every_n} epoch'ta snapshot, {interval}s aralik")
        while True:
            try:
                csv_path = run_dir / "results.csv"
                if csv_path.exists():
                    with open(csv_path, newline='') as f:
                        epoch_count = max(0, sum(1 for _ in csv.reader(f)) - 1)
                    if epoch_count > last_snapped and epoch_count % every_n == 0:
                        # Weight dosyasinin yazilmasini bekle (max 60s)
                        w = run_dir / "weights" / f"epoch{epoch_count}.pt"
                        waited = 0
                        while not w.exists() and waited < 60:
                            time.sleep(3); waited += 3
                        _save_snapshot(epoch_count)
                        last_snapped = epoch_count
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True, name="SnapshotMonitor")
    t.start()
    return t


def _build_trainer(overrides: dict) -> "SkyWatchTrainer":
    """SkyWatchTrainer için tek-nokta fabrika fonksiyonu.

    Her iki fazda aynı trainer tipini ve davranışını garanti eder.
    setup() sonrasında çağrılır (lazy import ile path bağımlılığını çözer).
    """
    from skywatch_trainer import SkyWatchTrainer
    return SkyWatchTrainer(overrides=overrides)


# ══════════════════════════════════════════════════════════
# FAZ 1 EĞİTİM
# ══════════════════════════════════════════════════════════

def train_phase1(n_gpu: int) -> str:
    """Faz 1: Pretrained backbone'dan baslayarak normal LR egitimi."""
    from checkpoint_zipper import make_final_zip

    print("\n" + "="*55)
    print("  FAZ 1 — Ilk Egitim (SkyWatchTrainer)")
    print(f"  Epoch: {PHASE1['epochs']} | LR0: {PHASE1['lr0']} | Batch: {PHASE1['batch']}")
    print(f"  Snapshot her {CHECKPOINT_EVERY_N} epoch'ta: {CHECKPOINT_DIR / 'phase1'}")
    print("="*55)

    device   = "0,1" if n_gpu >= 2 else "0"
    run_dir  = RUNS_DIR / PHASE1["name"]
    ckpt_dir = CHECKPOINT_DIR / "phase1"

    # DDP-uyumlu snapshot monitoru (add_callback DDP subprocess'te çalışmaz)
    start_zip_monitor(run_dir, ckpt_dir, every_n=CHECKPOINT_EVERY_N, interval=90)

    trainer = _build_trainer({
        "model":   str(MODEL_YAML),
        "data":    str(DATA_YAML),
        "project": str(RUNS_DIR),
        "device":  device,
        **PHASE1,
    })
    trainer.train()

    make_final_zip(run_dir, ckpt_dir, label="phase1_final")

    best = run_dir / "weights" / "best.pt"
    print(f"  En iyi model: {best}")
    return str(best)


# ══════════════════════════════════════════════════════════
# FAZ 2 EĞİTİM (Fine-tune)
# ══════════════════════════════════════════════════════════

def train_phase2(phase1_weights: str, n_gpu: int) -> str:
    """Faz 2: best.pt'den dusuk LR ile uzun fine-tune."""
    from checkpoint_zipper import make_final_zip

    print("\n" + "="*55)
    print("  FAZ 2 — Fine-tune (SkyWatchTrainer, Dusuk LR)")
    print(f"  Baslangic:  {phase1_weights}")
    print(f"  Epoch: {PHASE2['epochs']} | LR0: {PHASE2['lr0']} | Batch: {PHASE2['batch']}")
    print(f"  Patience: {PHASE2['patience']}")
    print(f"  Snapshot her {CHECKPOINT_EVERY_N} epoch'ta: {CHECKPOINT_DIR / 'phase2'}")
    print("="*55)

    device   = "0,1" if n_gpu >= 2 else "0"
    run_dir  = RUNS_DIR / PHASE2["name"]
    ckpt_dir = CHECKPOINT_DIR / "phase2"

    # Faz 1 ile aynı DDP-uyumlu mekanizma (add_callback DDP'de çalışmaz)
    start_zip_monitor(run_dir, ckpt_dir, every_n=CHECKPOINT_EVERY_N, interval=90)

    trainer = _build_trainer({
        "model":   phase1_weights,   # Faz 1 best.pt → fine-tune başlangıcı
        "data":    str(DATA_YAML),
        "project": str(RUNS_DIR),
        "device":  device,
        **PHASE2,
    })
    trainer.train()

    make_final_zip(run_dir, ckpt_dir, label="phase2_final")

    best = run_dir / "weights" / "best.pt"
    print(f"  Final model: {best}")
    return str(best)


# ══════════════════════════════════════════════════════════
# VALİDASYON
# ══════════════════════════════════════════════════════════

def validate(weights: str, label: str = "") -> object:
    """Detection modelinin performansını ölç ve raporla."""
    from ultralytics import YOLO

    print(f"\n  Validasyon: {label} [{Path(weights).parent.parent.name}]")
    model   = YOLO(weights)
    metrics = model.val(data=str(DATA_YAML), imgsz=640, verbose=False)

    box = metrics.box
    print(f"  mAP@50:     {box.map50:.4f}")
    print(f"  mAP@50:95:  {box.map:.4f}")
    print(f"  Precision:  {box.mp:.4f}")
    print(f"  Recall:     {box.mr:.4f}")

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
