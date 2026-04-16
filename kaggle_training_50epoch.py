# ══════════════════════════════════════════════════════════════════════════════
# SKYWATCH-Det 50 EPOCH TAM EĞİTİM KODU (Kaggle Notebook)
# ══════════════════════════════════════════════════════════════════════════════
# Bu kod Kaggle notebook'ta çalıştırılır.
#
# Eğitim Planı:
#   Faz 1: 50 epoch — Tek fazlı tam backbone eğitimi
#   Faz 2: Devre dışı
#
# Snapshot Stratejisi:
#   Her 5 epoch'ta ZIP → epoch 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
#   Final ZIP: Eğitim sonunda
#
# Hiperparametreler:
#   LR0=0.001 | LRf=0.01 | Warmup=3 | Patience=25
#   Mosaic=1.0 → son 15 epochta kapalı | Box=8.5 | DFL=2.0
#
# Tahmini süre (Kaggle 2xT4):
#   ~70-90 dakika (50 epoch × ~85 sn/epoch)
# ══════════════════════════════════════════════════════════════════════════════

import os
from pathlib import Path

WORKING_DIR = "/kaggle/working"
REPO_URL = "https://github.com/mustafabsnl/SKYWATCH.git"
REPO_DIR = os.path.join(WORKING_DIR, "SKYWATCH")

# ──────────────────────────────────────────────────────────────────────────────
# 1. ADIM: Depoyu Klonla veya Güncelle
# ──────────────────────────────────────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    print("="*70)
    print("  DEPO KLONLANIYOR...")
    print("="*70)
    !git clone {REPO_URL} {REPO_DIR}
else:
    print("="*70)
    print("  DEPO GÜNCELLENIYOR (git pull)...")
    print("="*70)
    %cd {REPO_DIR}
    !git reset --hard
    !git pull origin main
    print("  En son commit:")
    !git log -1 --oneline

print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. ADIM: PYTHONPATH Ayarları
# ──────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src/model"))

os.environ['PYTHONPATH'] = (
    f"{REPO_DIR}:{os.path.join(REPO_DIR, 'src/model')}:"
    + os.environ.get('PYTHONPATH', '')
)

print("="*70)
print("  PYTHONPATH AYARLANDI")
print(f"  REPO_DIR: {REPO_DIR}")
print("="*70)
print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. ADIM: 50 EPOCH EĞİTİM (sadece Faz 1)
# ──────────────────────────────────────────────────────────────────────────────
%cd {REPO_DIR}

print("="*70)
print("  50 EPOCH TAM EĞİTİM BAŞLATILIYOR (Faz 1)...")
print("  Epochs    : 50")
print("  LR0       : 0.001  →  LRf: 0.01  |  Optimizer: AdamW")
print("  Warmup    : 3 epoch  |  Patience: 25")
print("  Mosaic    : 1.0 (tam güç, son 15 epoch kapalı)")
print("  Box loss  : 8.5  |  DFL: 2.0  (küçük yüz odaklı)")
print("  Snapshot  : Epoch 5, 10, 15, 20, 25, 30, 35, 40, 45, 50")
print("  Tahmini süre: ~70-90 dakika")
print("="*70)
print("\n")

!python src/model/train_skywatch_kaggle.py --phase 1

print("\n")
print("="*70)
print("  50 EPOCH EĞİTİM TAMAMLANDI!")
print("")
print("  Output sekmesinde şu dosyaları göreceksin:")
print("    ✅  snapshot_epoch_5.zip")
print("    ✅  snapshot_epoch_10.zip")
print("    ✅  snapshot_epoch_15.zip")
print("    ✅  snapshot_epoch_20.zip")
print("    ✅  snapshot_epoch_25.zip")
print("    ✅  snapshot_epoch_30.zip")
print("    ✅  snapshot_epoch_35.zip")
print("    ✅  snapshot_epoch_40.zip")
print("    ✅  snapshot_epoch_45.zip")
print("    ✅  snapshot_epoch_50.zip")
print("    ✅  skywatch_det_phase1_final_<timestamp>.zip")
print("")
print("  En iyi model: /kaggle/working/runs/skywatch_det_phase1/weights/best.pt")
print("="*70)
