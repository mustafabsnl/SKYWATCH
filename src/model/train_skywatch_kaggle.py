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
import csv
import json
from datetime import datetime
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
    epochs          = 70,
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
    mosaic          = 0.75,
    # Crowded scenes: Copy-Paste augmentation
    copy_paste      = 0.1,
    copy_paste_mode = "flip",
    mixup           = 0.05,
    degrees         = 10.0,
    fliplr          = 0.5,
    perspective     = 0.0005,
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    erasing         = 0.1,
    # Dataset 10px~500px yüz içeriyor → multi_scale önemli
    # OOM riski: True yapmak istersen batch=8'e düşür
    multi_scale     = False,
    close_mosaic    = 10,
    # Loss — Detection modu (pose/kobj yok)
    # SkyWatchBboxLoss zaten küçük yüzlere 2x ağırlık veriyor,
    # bu yüzden gain'leri biraz düşürüyoruz (standart: box=7.5, dfl=1.5)
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
    # Fine-tune: Copy-Paste'i azalt (daha temiz fine-tune)
    copy_paste      = 0.05,
    copy_paste_mode = "flip",
    mixup           = 0.0,
    degrees         = 5.0,
    fliplr          = 0.5,
    perspective     = 0.0,
    hsv_h           = 0.01,
    hsv_s           = 0.4,
    hsv_v           = 0.3,
    erasing         = 0.0,
    multi_scale     = False,
    close_mosaic    = 50,
    # Loss — Detection modu (pose/kobj yok)
    # Fine-tune: Loss gain'leri Phase 1 ile aynı
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

    # Path'e ekle (Mevcut islem icin)
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / "src" / "model"))

    # DDP (Distributed Data Parallel) Alt islemleri (subprocess) icin PYTHONPATH guncellemesi (Kritik Duzeltme)
    # Ultralytics DDP baslatirken yeni bir terminal acar, sys.path'i gormez.
    os.environ["PYTHONPATH"] = f"{str(REPO_DIR)}:{str(REPO_DIR / 'src' / 'model')}:" + os.environ.get("PYTHONPATH", "")


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
# SNAPSHOT SİSTEMİ (Callback-based)
# PC_EGITIM_KODLARI/train.py ile aynı mantık, period = 5 epoch
# ══════════════════════════════════════════════════════════

def _generate_training_plots(results_csv: Path, out_dir: Path):
    """results.csv'den loss, metrik ve LR grafiklerini üretir."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results_csv.exists():
        return

    data: dict[str, list[float]] = {}
    with open(str(results_csv), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                key = (key or "").strip()
                try:
                    data.setdefault(key, []).append(float(val))
                except (ValueError, TypeError):
                    pass

    if not data:
        return

    n = len(next(iter(data.values())))
    epochs = data.get("epoch", list(range(1, n + 1)))

    # ── Loss eğrileri ──
    loss_groups = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tk, vk, title) in zip(axes, loss_groups):
        if tk in data:
            ax.plot(epochs[: len(data[tk])], data[tk], label="train")
        if vk in data:
            ax.plot(epochs[: len(data[vk])], data[vk], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / "loss_curves.png"), dpi=150)
    plt.close()

    # ── Metrik eğrileri ──
    metric_pairs = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    available = [(k, t) for k, t in metric_pairs if k in data]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
        if len(available) == 1:
            axes = [axes]
        for ax, (key, title) in zip(axes, available):
            ax.plot(epochs[: len(data[key])], data[key], color="tab:blue", marker=".")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / "metric_curves.png"), dpi=150)
        plt.close()

    # ── Learning Rate ──
    lr_key = "lr/pg0"
    if lr_key in data:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs[: len(data[lr_key])], data[lr_key])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / "lr_curve.png"), dpi=150)
        plt.close()


def _generate_val_curves(metrics, snap_dir: Path, data: dict):
    """Validator metrics'ten BoxF1, BoxP, BoxPR, BoxR eğrilerini üretir."""
    try:
        from ultralytics.utils.metrics import plot_pr_curve, plot_mc_curve
    except ImportError:
        return

    names = data.get("names", {})
    box = getattr(metrics, "box", None)
    if box is None:
        return

    px = getattr(box, "px", None)
    if px is None or not hasattr(px, "__len__") or len(px) == 0:
        return

    prec_values = getattr(box, "prec_values", None)
    f1_curve = getattr(box, "f1_curve", None)
    p_curve = getattr(box, "p_curve", None)
    r_curve = getattr(box, "r_curve", None)
    all_ap = getattr(box, "all_ap", None)
    ap50 = all_ap[:, 0] if all_ap is not None and hasattr(all_ap, "ndim") and all_ap.ndim >= 2 else None

    if prec_values is not None:
        try:
            plot_pr_curve(px, prec_values, ap50, save_dir=snap_dir / "BoxPR_curve.png", names=names, on_plot=None)
        except Exception:
            pass

    for arr, fname, ylabel in [
        (f1_curve, "BoxF1_curve.png", "F1"),
        (p_curve, "BoxP_curve.png", "Precision"),
        (r_curve, "BoxR_curve.png", "Recall"),
    ]:
        if arr is not None:
            try:
                plot_mc_curve(px, arr, save_dir=snap_dir / fname, names=names, ylabel=ylabel, on_plot=None)
            except Exception:
                pass


def _make_snapshot_callbacks(period: int, save_dir: Path):
    """Her *period* epoch'ta tam bir snapshot oluşturan callback çifti döner.

    Returns:
        (on_train_start_cb, on_fit_epoch_end_cb) — iki callback fonksiyonu.
        on_train_start_cb: validator'a plots hook'u ekler.
        on_fit_epoch_end_cb: asıl snapshot işlemini yapar.
    """

    def _on_train_start(trainer):
        """Validator'a on_val_start hook'u ekle: snapshot epoch'larında plots=True zorla."""
        v = getattr(trainer, "validator", None)
        if v is None:
            return

        def _force_plots_for_snapshot(validator):
            epoch = trainer.epoch + 1
            if epoch % period == 0:
                validator.args.plots = True

        v.add_callback("on_val_start", _force_plots_for_snapshot)

    def _on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1  # 0-indexed → 1-indexed
        if epoch % period != 0:
            return

        snap_dir = save_dir / f"snapshot_epoch_{epoch}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Weights kopyala ──
        snap_weights = snap_dir / "weights"
        snap_weights.mkdir(exist_ok=True)
        src_weights = save_dir / "weights"
        for name in ("best.pt", "last.pt"):
            src = src_weights / name
            if src.exists():
                shutil.copy2(str(src), str(snap_weights / name))

        # ── 2. Mevcut tüm dosyaları kopyala (png, jpg, csv, yaml, json) ──
        for pattern in ("*.png", "*.jpg", "*.csv", "*.yaml", "*.json"):
            for src_file in save_dir.glob(pattern):
                if src_file.is_file():
                    shutil.copy2(str(src_file), str(snap_dir / src_file.name))

        # ── 3. results.png'yi Ultralytics ile oluştur ──
        try:
            from ultralytics.utils.plotting import plot_results as _ul_plot_results

            _ul_plot_results(file=snap_dir / "results.csv", dir=snap_dir, on_plot=None)
        except Exception:
            pass

        # ── 4. Confusion Matrix oluştur ──
        try:
            v = trainer.validator
            if v and hasattr(v, "confusion_matrix") and v.confusion_matrix is not None:
                v.confusion_matrix.plot(save_dir=snap_dir, normalize=False, on_plot=None)
                v.confusion_matrix.plot(save_dir=snap_dir, normalize=True, on_plot=None)
        except Exception as exc:
            print(f"  ⚠️  Confusion matrix hatası (epoch {epoch}): {exc}")

        # ── 5. F1 / P / R / PR eğrileri oluştur ──
        try:
            v = trainer.validator
            if v and hasattr(v, "metrics"):
                _generate_val_curves(v.metrics, snap_dir, trainer.data)
        except Exception as exc:
            print(f"  ⚠️  Curve grafik hatası (epoch {epoch}): {exc}")

        # ── 6. Kendi eğitim grafikleri (loss, metric, LR) ──
        try:
            _generate_training_plots(snap_dir / "results.csv", snap_dir)
        except Exception as exc:
            print(f"  ⚠️  Snapshot grafik hatası (epoch {epoch}): {exc}")

        # ── 7. Metrik özeti kaydet ──
        metrics = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in trainer.metrics.items()}

        summary = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(str(snap_dir / "snapshot_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📸 Snapshot kaydedildi: {snap_dir}")

    return _on_train_start, _on_fit_epoch_end


def _flush_gpu():
    """Tüm GPU belleğini agresif şekilde serbest bırak (faz geçişlerinde OOM önlemi)."""
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  [MEM] GPU flush tamamlandi (kalan alloc: {alloc:.2f} GB)")
    except Exception:
        pass


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

    trainer = _build_trainer({
        "model":   str(MODEL_YAML),
        "data":    str(DATA_YAML),
        "project": str(RUNS_DIR),
        "device":  device,
        **PHASE1,
    })
    snap_train_start, snap_epoch_end = _make_snapshot_callbacks(CHECKPOINT_EVERY_N, run_dir)
    trainer.add_callback("on_train_start", snap_train_start)
    trainer.add_callback("on_fit_epoch_end", snap_epoch_end)
    trainer.train()

    make_final_zip(run_dir, ckpt_dir, label="phase1_final")

    # Trainer'i GPU'dan temizle (Phase 2 OOM onlemi)
    del trainer
    _flush_gpu()

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

    trainer = _build_trainer({
        "model":   phase1_weights,   # Faz 1 best.pt → fine-tune başlangıcı
        "data":    str(DATA_YAML),
        "project": str(RUNS_DIR),
        "device":  device,
        **PHASE2,
    })
    snap_train_start, snap_epoch_end = _make_snapshot_callbacks(CHECKPOINT_EVERY_N, run_dir)
    trainer.add_callback("on_train_start", snap_train_start)
    trainer.add_callback("on_fit_epoch_end", snap_epoch_end)
    trainer.train()

    make_final_zip(run_dir, ckpt_dir, label="phase2_final")

    del trainer
    _flush_gpu()

    best = run_dir / "weights" / "best.pt"
    print(f"  Final model: {best}")
    return str(best)


# ══════════════════════════════════════════════════════════
# VALİDASYON
# ══════════════════════════════════════════════════════════

def validate(weights: str, label: str = "") -> dict:
    """Detection modelinin performansını ölç, raporla ve GPU'yu temizle."""
    from ultralytics import YOLO

    print(f"\n  Validasyon: {label} [{Path(weights).parent.parent.name}]")
    model   = YOLO(weights)
    metrics = model.val(data=str(DATA_YAML), imgsz=640, verbose=False)

    box = metrics.box
    result = {"map50": box.map50, "map": box.map, "mp": box.mp, "mr": box.mr}
    print(f"  mAP@50:     {box.map50:.4f}")
    print(f"  mAP@50:95:  {box.map:.4f}")
    print(f"  Precision:  {box.mp:.4f}")
    print(f"  Recall:     {box.mr:.4f}")

    del model, metrics
    _flush_gpu()

    return result


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
        # Sırayla: 1 → 2 (train_phase1 ve validate içinde otomatik cleanup var)
        p1_weights = train_phase1(n_gpu)
        validate(p1_weights, "Faz 1 Sonucu")

        print("\n  [MEM] Phase 2 başlatılıyor...")
        p2_weights = train_phase2(p1_weights, n_gpu)
        validate(p2_weights, "Faz 2 Sonucu (Final)")

        print("\n" + "="*55)
        print("  EĞİTİM TAMAMLANDI")
        print(f"  Final model: {p2_weights}")
        print("="*55)


if __name__ == "__main__":
    main()
