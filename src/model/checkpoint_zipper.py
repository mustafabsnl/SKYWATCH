"""
SKYWATCH-Det Epoch Checkpoint Zipper
======================================
Her N epoch'ta Ultralytics run klasorunu komple zip'ler.
Zip icerigi egitim bittigindeki son hali ile ayni olur:
  - args.yaml
  - results.csv / results.png
  - BoxF1_curve.png, BoxP_curve.png, BoxPR_curve.png, BoxR_curve.png
  - confusion_matrix.png / confusion_matrix_normalized.png
  - labels.jpg
  - loss_curves.png / lr_curve.png / metric_curves.png
  - train_batch0.jpg, train_batch1.jpg, train_batch2.jpg
  - val_batch0_pred.jpg, val_batch0_labels.jpg, ...
  - training_config.json / snapshot_summary.json
  - weights/best.pt, weights/last.pt

Kaggle'da her 10 epoch'ta zip, Output'a kaydedilir.
Session sona erse bile checkpointлар kaybolmaz.
"""

import zipfile
import json
import shutil
from pathlib import Path
from datetime import datetime


# ── Her kac epoch'ta zip alinsin ─────────────────────────────
CHECKPOINT_EVERY_N_EPOCHS = 10
# ─────────────────────────────────────────────────────────────


def _collect_run_files(run_dir: Path) -> list[Path]:
    """Run klasorunun icindeki tum dosyalari listele."""
    files = []
    # Tum dosyalari recursive topla
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            files.append(f)
    return files


def _make_snapshot_summary(trainer, epoch: int) -> dict:
    """Anlık eğitim özetini JSON olarak döndür."""
    metrics = {}
    try:
        if hasattr(trainer, 'metrics') and trainer.metrics:
            for k, v in trainer.metrics.items():
                try:
                    metrics[str(k)] = float(v)
                except (TypeError, ValueError):
                    metrics[str(k)] = str(v)
    except Exception:
        pass

    return {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "model": "SKYWATCH-Det",
        "checkpoint_type": f"epoch_{epoch}",
        "metrics": metrics,
        "save_dir": str(trainer.save_dir) if hasattr(trainer, 'save_dir') else "",
    }


def _make_training_config(trainer) -> dict:
    """Egitim konfigurasyonunu JSON olarak döndür."""
    config = {}
    try:
        if hasattr(trainer, 'args'):
            args = trainer.args
            # En onemli parametreleri al
            keys = [
                'epochs', 'imgsz', 'batch', 'lr0', 'lrf', 'momentum',
                'weight_decay', 'warmup_epochs', 'optimizer', 'device',
                'patience', 'mosaic', 'mixup', 'degrees', 'fliplr',
                'box', 'cls', 'dfl', 'pose', 'kobj',
                'erasing', 'multi_scale', 'close_mosaic',
            ]
            for k in keys:
                v = getattr(args, k, None)
                if v is not None:
                    try:
                        config[k] = float(v) if isinstance(v, (int, float)) else str(v)
                    except Exception:
                        config[k] = str(v)
    except Exception:
        pass
    return config


def create_epoch_zip(trainer, epoch: int, checkpoint_dir: Path) -> Path | None:
    """
    Mevcut run klasorunu zip'le ve checkpoint_dir'e kaydet.

    Args:
        trainer: Ultralytics trainer nesnesi.
        epoch: Mevcut epoch numarasi.
        checkpoint_dir: Zip dosyalarinin kaydedilecegi dizin.

    Returns:
        Path: Olusturulan zip dosyasinin yolu.
    """
    try:
        run_dir = Path(trainer.save_dir)
        if not run_dir.exists():
            print(f"  [ZIP] Run dizini bulunamadi: {run_dir}")
            return None

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        zip_name = f"skywatch_det_epoch{epoch:04d}.zip"
        zip_path = checkpoint_dir / zip_name

        # Anlık snapshot ve config JSON'larini run_dir'e yaz
        snapshot_path = run_dir / "snapshot_summary.json"
        config_path = run_dir / "training_config.json"

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(_make_snapshot_summary(trainer, epoch), f, indent=2)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_make_training_config(trainer), f, indent=2)

        # Zip olustur
        files = _collect_run_files(run_dir)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in files:
                # Zip icinde relative yol kullan
                arcname = f.relative_to(run_dir.parent)
                zf.write(f, arcname)

        size_mb = zip_path.stat().st_size / 1e6
        print(f"\n  [ZIP] Epoch {epoch} checkpoint: {zip_name} ({size_mb:.1f} MB)")
        print(f"        Konum: {zip_path}")
        return zip_path

    except Exception as e:
        print(f"  [ZIP HATA] Epoch {epoch}: {e}")
        return None


def make_checkpoint_callback(checkpoint_dir: Path, every_n: int = CHECKPOINT_EVERY_N_EPOCHS):
    """
    Ultralytics callback factory.
    Her N epoch'ta ZIP checkpoint alir.

    Kullanim:
        cb = make_checkpoint_callback(Path("/kaggle/working/checkpoints"))
        model.add_callback("on_fit_epoch_end", cb)

    Args:
        checkpoint_dir: Zip dosyalarinin kaydedilecegi dizin.
        every_n: Kac epoch'ta bir zip alinsin.

    Returns:
        Callable: on_fit_epoch_end callback fonksiyonu.
    """
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1  # 0-indexed -> 1-indexed
        if epoch % every_n == 0:
            create_epoch_zip(trainer, epoch, checkpoint_dir)

    return on_fit_epoch_end


def make_final_zip(run_dir: Path, output_dir: Path, label: str = "final") -> Path | None:
    """
    Egitim tamamlandiginda son zip'i olustur.

    Args:
        run_dir: Ultralytics run klasoru.
        output_dir: Zip'in kaydedilecegi yer.
        label: Zip dosyasi etiketi ("final", "phase1", "phase2" vb.)

    Returns:
        Path: Zip dosyasi yolu.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        zip_name = f"skywatch_det_{label}_{ts}.zip"
        zip_path = output_dir / zip_name

        files = _collect_run_files(Path(run_dir))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in files:
                arcname = f.relative_to(Path(run_dir).parent)
                zf.write(f, arcname)

        size_mb = zip_path.stat().st_size / 1e6
        print(f"\n  [ZIP] Final checkpoint: {zip_name} ({size_mb:.1f} MB)")
        return zip_path

    except Exception as e:
        print(f"  [ZIP HATA] Final: {e}")
        return None
