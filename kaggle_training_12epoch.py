# ══════════════════════════════════════════════════════════════════════════════
# SKYWATCH-Det 12 EPOCH HIZLI TEST KODU (Kaggle Notebook)
# ══════════════════════════════════════════════════════════════════════════════
# Bu kod Kaggle notebook'ta çalıştırılır.
# Özellikler:
#   - 12 epoch Phase 1 eğitimi
#   - Her 3 epoch'ta snapshot ZIP (epoch 3, 6, 9, 12)
#   - Phase 2 atlanır (hızlı test için)
#   - Tahmini süre: 15-20 dakika (2xT4 GPU)
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
    !git reset --hard  # Lokal değişiklikleri temizle
    !git pull origin main
    print(f"  En son commit: ")
    !git log -1 --oneline

print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. ADIM: PYTHONPATH ve Dizin Ayarları (Modüllerin Görünmesi İçin Şart)
# ──────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src/model"))

# DDP (Çift GPU) işlemleri için sistem ortam değişkenini güncelle
os.environ['PYTHONPATH'] = f"{REPO_DIR}:{os.path.join(REPO_DIR, 'src/model')}:" + os.environ.get('PYTHONPATH', '')

print("="*70)
print("  PYTHONPATH AYARLANDI")
print(f"  REPO_DIR: {REPO_DIR}")
print("="*70)
print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. ADIM: Eğitimi Başlat (12 EPOCH - Sadece Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
# NOT: Phase 2 devre dışı (12 epoch test için gereksiz)
# Sadece --phase 1 kullanıyoruz

%cd {REPO_DIR}

print("="*70)
print("  12 EPOCH EĞİTİM BAŞLATIILIYOR...")
print("  - Phase 1: 12 epoch")
print("  - Snapshot: Epoch 3, 6, 9, 12")
print("  - Phase 2: Atlandı (hızlı test)")
print("  - Tahmini süre: 15-20 dakika")
print("="*70)
print("\n")

!python src/model/train_skywatch_kaggle.py --phase 1

print("\n")
print("="*70)
print("  EĞİTİM TAMAMLANDI!")
print("  Output sekmesinde şu ZIP'leri göreceksin:")
print("    - snapshot_epoch_3.zip")
print("    - snapshot_epoch_6.zip")
print("    - snapshot_epoch_9.zip")
print("    - snapshot_epoch_12.zip")
print("    - skywatch_det_phase1_final_<timestamp>.zip")
print("="*70)
