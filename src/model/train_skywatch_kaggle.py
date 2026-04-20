"""
SKYWATCH-Det — Kaggle 2xT4 Egitim Scripti
==========================================
Dataset : WIDER FACE (Kaggle Input: YUZLER/skywatch_wider)
Format  : YOLO normalize (cx cy w h) — 1 sinif: face
Istatistik: 12,880 train / 3,226 val goruntu, ~196k yuz

Egitim:
  Faz 1 → 50 epoch tam egitim (backbone + head, tam augmentasyon)
  Faz 2 → DEVRE DISI (epochs=0). Fine-tune gerekirse --phase 2 ile aktif et.

Fix #2: PHASE2 bilerek kapatildi; usteki yorum bunu yansitmiyordu.

Kullanim (Kaggle Notebook):
  !python train_skywatch_kaggle.py --phase 1          # Sadece Faz 1 (tavsiye edilen)
  !python train_skywatch_kaggle.py --phase 2          # Sadece Faz 2 (Faz 1 sonrasi)
  !python train_skywatch_kaggle.py --phase all        # 1 sonra 2 (Faz 2 skip eder)
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
# Dataset: WIDER FACE → YOLO (Kaggle'a "YUZLER" olarak yuklendi)
#
#   YAPI (Kaggle Input'tan dogrudan okunur — indirme GEREKMEZ):
#     /kaggle/input/yuzler/skywatch_wider/
#       images/
#         train/   (12,880 goruntu)
#         val/     ( 3,226 goruntu)
#       labels/
#         train/   (12,880 .txt — YOLO: 0 cx cy w h)
#         val/     ( 3,226 .txt)
#       data.yaml
#
#   nc: 1 | class_id: 0 | sinif: face
#   Toplam: 196,106 yuz (ort. 12.2 yuz/resim, max 1962)
# ════════════════════════════════════════════════════════

WORKING        = Path("/kaggle/working")
REPO_DIR       = WORKING / "SKYWATCH"
KAGGLE_INPUT   = Path("/kaggle/input")
RUNS_DIR       = WORKING / "runs"
CHECKPOINT_DIR = WORKING / "checkpoints"
GITHUB_REPO    = "https://github.com/mustafabsnl/SKYWATCH.git"

# SKYWATCH-Det: 4-scale detection head (P2/P3/P4/P5) + C2f_CAM + FRM
# Fix #3: kısa isim yerine tam absolute path — göreli isim çözümleme hatası önlenir.
MODEL_YAML     = str(REPO_DIR / "src" / "ultralytics_patch" / "cfg" / "models" / "skywatch" / "skywatch-det.yaml")
DATA_YAML      = None   # prepare_data() tarafindan belirlenir

# Her kaç epoch'ta snapshot alinsin (klasor yapisi: snapshot_epoch_N/)
CHECKPOINT_EVERY_N = 5  # 50 epoch için 5'te bir (epoch 5, 10, 15, ..., 50)

# ══════════════════════════════════════════════════════════
# FAZ 1 HİPERPARAMETRELERİ — 50 EPOCH TAM EĞİTİM
# Tek faz, 50 epoch, Faz 2 devre dışı
# ══════════════════════════════════════════════════════════
PHASE1 = dict(
    epochs          = 50,          # Tek faz tam eğitim
    imgsz           = 640,
    batch           = 6,           # 2xT4: YOLOv11m için 3 per GPU (m modeli s'den ~2x büyük)
    nbs             = 64,          # Nominal batch: efektif batch = 8 * (64/8) = 64
    lr0             = 0.001,       # 35 epoch için standart başlangıç LR
    lrf             = 0.01,        # Final LR = lr0 * lrf = 0.00001 (derin decay)
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 3,           # 50 epoch için uygun warmup (%6)
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,
    optimizer       = "AdamW",
    patience        = 25,          # 50 epoch için yeterli patience
    save_period     = 5,           # Her 5 epoch'ta weight kaydet
    # Augmentation — WIDER FACE: sokak kamerası simülasyonu
    # mosaic=1.0: kalabalık sahneler (%85 küçük yüz) için kritik, tam güç
    mosaic          = 1.0,
    # Crowded scenes: Copy-Paste augmentation
    copy_paste      = 0.15,
    copy_paste_mode = "flip",
    mixup           = 0.1,
    degrees         = 10.0,
    fliplr          = 0.5,
    perspective     = 0.0005,
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    erasing         = 0.1,
    # Dataset 10px~500px yüz içeriyor → multi_scale faydalı ama OOM riski
    multi_scale     = False,
    close_mosaic    = 15,          # Son 15 epoch mosaic kapat → val metriği gerçekçileşir
    # Loss ağırlıkları — %85 küçük yüz için box loss yüksek tutulur
    # SkyWatchBboxLoss zaten küçük yüzlere 1.5x ek ağırlık veriyor
    box             = 8.5,
    cls             = 0.5,
    dfl             = 2.0,
    # Kaydetme
    name            = "skywatch_det_phase1",
    exist_ok        = True,
    plots           = True,
    save            = True,
    val             = True,
    verbose         = True,
)

# ══════════════════════════════════════════════════════════
# FAZ 2 - FINE-TUNE (best.pt uzerinden, 30 epoch)
# ══════════════════════════════════════════════════════════
# Neden box_loss/dfl_loss Faz 1de yuksek kaldi?
#   1. SkyWatchBboxLoss kucuk yuzlere 3x agirlik -> sayi yapay sisiyor
#   2. DFL + P2(stride=4) 20px yuz icin inherent zor
#   3. Mosaic kapaninca (ep36) son 15 epoch yeniden converge etti
#
# Faz 2 hedefleri:
#   box_loss: ~1.63 -> ~1.35 | dfl_loss: ~1.57 -> ~1.30 | mAP50: ~0.70 -> ~0.74+
#
# Strateji:
#   - Warm restart: dusuk lr0=0.0003, cos_lr=True
#   - box weight: 8.5 ->  9.5  (recall koruyarak lokalizasyon vurgusu)
#   - dfl weight: 2.0 ->  1.5  (P2 DFL cezasi yumusatildi)
#   - mosaic=0.0               (fine-tune icin temiz bbox siniri)
#   - FaceOcclusionAug korunuyor (SkyWatchTrainer uzerinden)
# ══════════════════════════════════════════════════════════
PHASE2 = dict(
    epochs          = 30,          # Faz 1 best.pt uzerinden 30 epoch ince ayar
    imgsz           = 640,
    batch           = 6,           # Faz 1 ile ayni batch (6); OOM olursa 4e dus
    nbs             = 64,          # efektif batch sabit: 6 * (64/6) ~ 64
    lr0             = 0.0003,      # Warm restart LR (Faz 1 lr0'in 1/3u)
    lrf             = 0.01,        # Final LR = 0.0003 * 0.01 = 0.000003
    cos_lr          = True,        # Cosine LR schedule - fine-tune icin kritik
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 2,           # Kisa warmup (2/30 ~ %7)
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.01,
    optimizer       = "AdamW",
    patience        = 15,
    save_period     = 5,
    # Augmentation - Mosaic KAPALI (fine-tune box regresyon stabilitesi)
    # FaceOcclusionAug SkyWatchTrainer uzerinden aktif kalmaya devam eder.
    mosaic          = 0.0,         # Mosaic kapat: bbox kenari icin temiz imaj
    copy_paste      = 0.0,         # Kapat: FaceOcclusionAug zaten var, ust uste gurultu olur
    copy_paste_mode = "flip",
    mixup           = 0.0,         # Mixup kapat: DFL ogrenimi icin gurultu yaratmaz
    degrees         = 8.0,
    fliplr          = 0.5,
    perspective     = 0.0003,
    hsv_h           = 0.01,
    hsv_s           = 0.5,
    hsv_v           = 0.3,
    erasing         = 0.0,         # Erasing kapat: DFL icin temiz bbox kenari
    multi_scale     = False,
    close_mosaic    = 30,          # mosaic=0.0 zaten kapali - dummy param
    # box 8.5 -> 9.5: Lokalizasyon vurgusu (recall zararlanmamasi icin 11.0 kullanilmadi)
    # dfl 2.0 ->  1.5: P2/DFL cezasini yumusat -> dfl_loss dusurur
    box             = 9.5,
    cls             = 0.5,
    dfl             = 1.5,
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

    # Fix #4: Repo varsa sadece "mevcut" demek yetmiyor—eski kod çalışabilir.
    # git pull ile her zaman en güncel kodu çek; akşi durumu engelle.
    if not REPO_DIR.exists():
        print(f"  GitHub clone: {GITHUB_REPO}")
        subprocess.run(["git", "clone", GITHUB_REPO, str(REPO_DIR)], check=True)
    else:
        print(f"  Repo mevcut, güncelleniyor (git pull)...")
        result_pull = subprocess.run(
            ["git", "-C", str(REPO_DIR), "pull", "origin", "main"],
            capture_output=True, text=True
        )
        if result_pull.returncode == 0:
            print(f"  ✓ git pull OK: {result_pull.stdout.strip().splitlines()[-1] if result_pull.stdout.strip() else 'Degisiklik yok'}")
        else:
            # pull başarısızsa uyar — ama eğitimi durdurma (offline moda düşebilir)
            import warnings
            warnings.warn(
                f"[setup] git pull başarısız! Mevcut repo kodu kullanılıyor.\n"
                f"  Hata: {result_pull.stderr.strip()}",
                RuntimeWarning, stacklevel=2
            )

    # Path'e ekle (Mevcut islem icin)
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / "src" / "model"))

    # DDP (Distributed Data Parallel) Alt islemleri (subprocess) icin PYTHONPATH guncellemesi (Kritik Duzeltme)
    # Ultralytics DDP baslatirken yeni bir terminal acar, sys.path'i gormez.
    os.environ["PYTHONPATH"] = f"{str(REPO_DIR)}:{str(REPO_DIR / 'src' / 'model')}:" + os.environ.get("PYTHONPATH", "")


    # ── kaggle_setup.py — BAŞARISIZSA EĞİTİMİ DURDUR ──────────────
    # Fix #3: Patch hatası artık uyarı değil, RuntimeError.
    # Bozuk patch ile ilerlemek beklenen mimariden sapma demektir.
    setup_script = REPO_DIR / "kaggle_setup.py"
    if not setup_script.exists():
        raise FileNotFoundError(
            f"[HATA] kaggle_setup.py bulunamadı: {setup_script}\n"
            f"Bu dosya olmadan C2f_CAM/FRM ultralytics'e enjekte edilemez."
        )

    result = subprocess.run([sys.executable, str(setup_script)], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"[HATA] kaggle_setup.py başarısız! (returncode={result.returncode})\n"
            f"C2f_CAM ve FRM düzgün patchlenemedi. Eğitim durduruldu.\n"
            f"Loglarda hangi adımın başarısız olduğunu kontrol edin."
        )

    # ── GPU bilgisi ──────────────────────────────────────────────────
    import torch
    n = torch.cuda.device_count()
    print(f"  GPU: {n}x {torch.cuda.get_device_name(0) if n>0 else 'YOK'}")
    print(f"  PyTorch: {torch.__version__}")

    # ── Custom modül import doğrulaması — BAŞARISIZSA DURDUR ────────
    # patch başarılı görünse bile import çalışmazsa devam etme.
    try:
        from ultralytics.nn.modules import C2f_CAM, FRM
        print("  ✓ Custom moduller aktif: C2f_CAM, FRM")
    except ImportError as e:
        raise RuntimeError(
            f"[HATA] Custom modul import başarısız: {e}\n"
            f"kaggle_setup.py çalıştı ama ultralytics'e inject edilemedi.\n"
            f"Loglarda patch adımlarını kontrol edin."
        )

    return n  # GPU sayisi


# ══════════════════════════════════════════════════════════
# VERİ SETİ HAZIRLIĞI
# Dataset Kaggle'a Input olarak eklendi — indirme GEREKMEZ
# Kaggle Input: /kaggle/input/yuzler/skywatch_wider/
# ══════════════════════════════════════════════════════════

def _find_dataset_root() -> Path:
    """Kaggle Input'tan WIDER FACE YOLO dataset'ini bul.

    Recursive: /kaggle/input/ altinda images/train iceren klasoru dondurur.
    Kaggle yapisi: /kaggle/input/datasets/<user>/<dataset>/<folder>/images/train
    """
    for dirpath, dirnames, _ in os.walk(str(KAGGLE_INPUT)):
        depth = dirpath.replace(str(KAGGLE_INPUT), "").count(os.sep)
        if depth > 8:
            dirnames.clear()
            continue
        p = Path(dirpath)
        if p.name == "train" and p.parent.name == "images":
            root = p.parent.parent
            print(f"  Dataset bulundu: {root}")
            return root

    print("\n  [HATA] Dataset bulunamadi! Dizin yapisi:")
    for dirpath, dirnames, files in os.walk(str(KAGGLE_INPUT)):
        depth = dirpath.replace(str(KAGGLE_INPUT), "").count(os.sep)
        if depth > 8:
            dirnames.clear()
            continue
        indent = "  " * (depth + 1)
        print(f"  {indent}{Path(dirpath).name}/ [{len(files)} dosya, {len(dirnames)} klasor]")

    raise FileNotFoundError(
        f"Dataset bulunamadi! /kaggle/input/ altinda images/train yok.\n"
        f"Kaggle notebook'a YUZLER dataset'ini Input olarak eklediginizden emin olun."
    )


def prepare_data():
    """Kaggle Input'tan dataset'i bul, DATA_YAML olustur."""
    global DATA_YAML

    dataset_root = _find_dataset_root()

    print("\n" + "="*55)
    print("  VERI SETI KONTROLU")
    print("="*55)
    print(f"  Kaynak: {dataset_root}")

    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        if img_dir.exists():
            n = sum(1 for _ in img_dir.iterdir())
            print(f"  {split:5s}: {n:,} goruntu")

    yaml_path = WORKING / "skywatch_data" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = (
        f"# SKYWATCH-Det — WIDER FACE YOLO\n"
        f"path: {str(dataset_root).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['face']\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml: {yaml_path}")
    print(f"    path : {dataset_root}")
    print(f"    train: images/train")
    print(f"    val  : images/val")

    DATA_YAML = yaml_path


# ══════════════════════════════════════════════════════════
# SNAPSHOT SİSTEMİ (Thread-based Monitor)
# DDP modunda callback'ler subprocess'e aktarılmıyor!
# Bu yüzden dosya sistemini izleyen thread kullanıyoruz.
# ══════════════════════════════════════════════════════════

def start_snapshot_monitor(run_dir: Path, checkpoint_dir: Path,
                           every_n: int = 5, poll_interval: int = 30):
    """
    DDP-uyumlu arka plan snapshot monitörü.
    
    results.csv'yi izler, her N epoch'ta tam snapshot oluşturur.
    Callback'ler DDP subprocess'ine aktarılmadığı için bu yöntem kullanılır.
    
    Args:
        run_dir: Eğitim çıktı dizini (runs/skywatch_det_phase1)
        checkpoint_dir: Snapshot'ların kaydedileceği dizin
        every_n: Kaç epoch'ta bir snapshot alınacak
        poll_interval: Kaç saniyede bir kontrol edilecek
    """
    import threading
    import time as _time
    import datetime as _dt

    def _save_snapshot(epoch: int):
        """Tam snapshot klasörü oluşturur ve ZIP'ler (Kaggle output için)."""
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            snap_dir = checkpoint_dir / f"snapshot_epoch_{epoch}"
            snap_dir.mkdir(exist_ok=True)

            # 1. Weights kopyala
            w_src = run_dir / "weights"
            w_dst = snap_dir / "weights"
            w_dst.mkdir(exist_ok=True)
            for wname in ["best.pt", "last.pt", f"epoch{epoch}.pt"]:
                src = w_src / wname
                if src.exists():
                    shutil.copy2(src, w_dst / wname)

            # 2. Tüm dosyaları kopyala (png, jpg, csv, yaml, json)
            for pattern in ["*.png", "*.jpg", "*.jpeg", "*.csv", "*.yaml", "*.json"]:
                for f in run_dir.glob(pattern):
                    if f.is_file():
                        shutil.copy2(f, snap_dir / f.name)

            # 3. Training grafikleri oluştur
            try:
                _generate_training_plots(snap_dir / "results.csv", snap_dir)
            except Exception:
                pass

            # 4. Metrik özeti
            metrics = {}
            csv_path = run_dir / "results.csv"
            if csv_path.exists():
                with open(csv_path, newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    last = rows[-1]
                    metrics = {k.strip(): v.strip() for k, v in last.items() if k and v}

            summary = {
                "epoch": epoch,
                "timestamp": _dt.datetime.now().isoformat(),
                "saved_weights": [p.name for p in w_dst.glob("*.pt")],
                "metrics": metrics,
            }
            with open(snap_dir / "snapshot_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # 5. KRITIK: ZIP oluştur (Kaggle output için)
            import zipfile
            zip_name = f"snapshot_epoch_{epoch}.zip"
            zip_path = WORKING / zip_name  # /kaggle/working/ altına (output'ta görünür!)
            
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for f in snap_dir.rglob("*"):
                    if f.is_file():
                        arcname = f.relative_to(snap_dir.parent)
                        zf.write(f, arcname)
            
            zip_mb = zip_path.stat().st_size / 1e6
            snap_mb = sum(p.stat().st_size for p in snap_dir.rglob("*") if p.is_file()) / 1e6
            n_weights = len(list(w_dst.glob("*.pt")))
            print(f"\n  📸 [SNAP] snapshot_epoch_{epoch}.zip → {zip_mb:.1f} MB (klasör: {snap_mb:.1f} MB, {n_weights} weight)")

        except Exception as e:
            print(f"\n  ❌ [SNAP ERROR] Epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()

    def _monitor():
        """Arka planda results.csv'yi izle, snapshot al."""
        last_snapped = 0
        print(f"  [MONITOR] Başladı — her {every_n} epoch'ta snapshot, {poll_interval}s aralık")
        
        while True:
            try:
                csv_path = run_dir / "results.csv"
                if csv_path.exists():
                    with open(csv_path, newline='', encoding='utf-8') as f:
                        # Header satırını çıkar
                        epoch_count = max(0, sum(1 for _ in csv.reader(f)) - 1)
                    
                    # Snapshot zamanı geldi mi?
                    if epoch_count > last_snapped and epoch_count % every_n == 0:
                        # Weight dosyasının yazılmasını bekle (max 90s)
                        w_last = run_dir / "weights" / "last.pt"
                        waited = 0
                        while not w_last.exists() and waited < 90:
                            _time.sleep(5)
                            waited += 5
                        
                        _save_snapshot(epoch_count)
                        last_snapped = epoch_count
                        
            except Exception as _mon_err:
                # Fix #6: Sessiz pass kaldirildi — snapshot hatası logda görünsün.
                # Main thread'i etkilemesin ama sessizce yutulmasın.
                print(f"\n  ⚠️  [MONITOR] CSV okuma/kontrol hatası: {_mon_err}")
            
            _time.sleep(poll_interval)

    t = threading.Thread(target=_monitor, daemon=True, name="SnapshotMonitor")
    t.start()
    return t

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


def _run_channel_preflight(model_yaml: str, input_channels: int = 3, imgsz: int = 640) -> None:
    """Eğitim öncesi kanal doğrulama: önce kontrol, sonra train."""
    try:
        from src.tools.channel_validator import validate_model_channels
    except Exception as e:
        print(f"\n  [PREFLIGHT] Validator import hatası: {e}")
        print("  [PREFLIGHT] Kanal kontrolü atlanıyor, eğitime devam ediliyor...")
        return

    print("\n" + "=" * 55)
    print("  KANAL PREFLIGHT KONTROLÜ")
    print("=" * 55)
    try:
        result = validate_model_channels(
            model_yaml=model_yaml,
            input_channels=input_channels,
            imgsz=imgsz,
            verbose=True,
        )
    except Exception as e:
        print(f"\n  [PREFLIGHT] Doğrulama sırasında hata: {e}")
        print("  [PREFLIGHT] Kanal kontrolü atlanıyor, eğitime devam ediliyor...")
        return

    if not result["ok"]:
        err = result.get("error", "Bilinmeyen hata")
        # Fix #4: Preflight başarısızsa eğitimi başlatma.
        # Maliyetli Kaggle job'unu ilk epoch'ta patlatmak zaman kaybıdır.
        raise RuntimeError(
            f"\n{'='*55}\n"
            f"  [PREFLIGHT] KANAL DOĞRULAMASI BAŞARISIZ — EĞİTİM DURDURULDU\n"
            f"{'='*55}\n"
            f"  Hata   : {err}\n"
            f"  Neden  : Yanlış kanal boyutu ile eğitim anlamsız.\n"
            f"  Çözüm  : channel_validator.py çalıştır, YAML veya\n"
            f"           tasks.py patch'ini kontrol et.\n"
            f"{'='*55}"
        )


# ══════════════════════════════════════════════════════════
# FAZ 1 EĞİTİM
# ══════════════════════════════════════════════════════════

def train_phase1(n_gpu: int) -> str:
    """Faz 1: Pretrained backbone'dan baslayarak normal LR egitimi."""
    from checkpoint_zipper import make_final_zip

    print("\n" + "="*55)
    print(f"  FAZ 1 — {PHASE1['epochs']} Epoch Tam Egitim (SkyWatchTrainer)")
    print(f"  Epoch: {PHASE1['epochs']} | LR0: {PHASE1['lr0']} | LRf: {PHASE1['lrf']}")
    print(f"  Batch: {PHASE1['batch']} (nbs={PHASE1.get('nbs',64)} → efektif 64)")
    print(f"  Snapshot her {CHECKPOINT_EVERY_N} epoch'ta: {CHECKPOINT_DIR / 'phase1'}")
    print("="*55)

    # KRITIK SIRA:
    # 1) kanal sayısı belirlenir ve model buna göre doğrulanır
    # 2) ancak doğrulama geçerse eğitim başlar
    _run_channel_preflight(MODEL_YAML, input_channels=3, imgsz=PHASE1["imgsz"])

    device   = "0,1" if n_gpu >= 2 else "0"
    run_dir  = RUNS_DIR / PHASE1["name"]
    ckpt_dir = CHECKPOINT_DIR / "phase1"

    # Thread-based snapshot monitörü başlat (DDP uyumlu)
    # 50 epoch için 60 saniyede bir kontrol (daha uzun eğitim süresi)
    start_snapshot_monitor(run_dir, ckpt_dir, every_n=CHECKPOINT_EVERY_N, poll_interval=60)

    trainer = _build_trainer({
        "model":   str(MODEL_YAML),
        "data":    str(DATA_YAML),
        "project": str(RUNS_DIR),
        "device":  device,
        **PHASE1,
    })
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
    """Faz 2: Faz 1 best.pt'den devam, düşük LR ile fine-tune (varsayılan: devre dışı)."""
    from checkpoint_zipper import make_final_zip

    print("\n" + "="*55)
    print(f"  FAZ 2 — {PHASE2['epochs']} Epoch Fine-Tune (SkyWatchTrainer)")
    print(f"  Başlangıç ağırlıkları: {phase1_weights}")
    print(f"  Epoch: {PHASE2['epochs']} | LR0: {PHASE2['lr0']} | LRf: {PHASE2['lrf']}")
    print(f"  Mosaic: {PHASE2['mosaic']} (azaltılmış fine-tune modu)")
    print("="*55)

    if PHASE2["epochs"] == 0:
        print("  [ATLANDI] PHASE2 epochs=0 — Phase 1 ağırlıkları döndürülüyor.")
        return phase1_weights

    device  = "0,1" if n_gpu >= 2 else "0"
    run_dir = RUNS_DIR / PHASE2["name"]
    ckpt_dir = CHECKPOINT_DIR / "phase2"

    # Fine-tune snapshot monitörü
    start_snapshot_monitor(run_dir, ckpt_dir, every_n=CHECKPOINT_EVERY_N, poll_interval=60)

    trainer = _build_trainer({
        "model":   phase1_weights,   # Faz 1 en iyi ağırlıklardan başla
        "data":    str(DATA_YAML),
        "project": str(RUNS_DIR),
        "device":  device,
        **PHASE2,
    })
    trainer.train()

    make_final_zip(run_dir, ckpt_dir, label="phase2_final")

    del trainer
    _flush_gpu()

    best = run_dir / "weights" / "best.pt"
    print(f"  Faz 2 en iyi model: {best}")
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
    args = parser.parse_args()

    # Kurulum
    n_gpu = setup()

    # Veri hazirla — Kaggle Input'tan oku, DATA_YAML olustur
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
