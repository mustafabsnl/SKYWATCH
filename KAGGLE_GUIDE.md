# SKYWATCH-Det — Kaggle Eğitim Kılavuzu (2 Fazlı Strateji)

## 📁 Dosya Yapısı

```
SKYWATCH/
├── src/model/
│   ├── train_skywatch_kaggle.py    ← Ana eğitim scripti (2 faz)
│   ├── wider_face_converter.py     ← Veri dönüştürücü
│   └── verify_model.py             ← Model doğrulama
└── ultralytics_skywatch/
    └── ultralytics/cfg/models/skywatch/
        └── skywatch-det.yaml       ← Model mimarisi
```

---

## 🎯 Eğitim Stratejisi

```
Faz 1 ─── Normal LR (0.001) ──────── 100 epoch
             ↓ best.pt
Faz 2 ─── Düşük LR (0.0002) ─────── 300 epoch + patience=80
             ↓ best.pt
           FINAL MODEL ✅
```

| Parametre | Faz 1 | Faz 2 |
|---|---|---|
| **Epochs** | 100 | 300 |
| **LR0** | 0.001 | 0.0002 |
| **LRf** | 0.01 | 0.005 |
| **Batch** | -1 (auto) | 8 |
| **Warmup** | 5 epoch | 1 epoch |
| **Patience** | 50 | 80 |
| **Mosaic** | 1.0 | 0.5 |
| **Multi-scale** | Açık | Kapalı |

---

## 🖥️ ADIM 1 — Yerel Veri Dönüşümü (Windows'ta)

WIDER FACE veri setini YOLO formatına dönüştür:

```powershell
cd C:\Users\musta\OneDrive\Desktop\SKYWATCH

python src/model/wider_face_converter.py `
    --wider_root "C:\Users\musta\OneDrive\Desktop\SKYWATCH\VeriSeti" `
    --output_dir "C:\Users\musta\OneDrive\Desktop\SKYWATCH\dataset_yolo" `
    --splits train val `
    --min_face_size 5
```

**Beklenen çıktı:**
```
  Train: 12,880 görüntü, ~159,420 yüz
  Val:   3,226 görüntü, ~39,708 yüz
  data.yaml oluşturuldu
```

---

## 🌐 ADIM 2 — GitHub'a Push

```powershell
cd C:\Users\musta\OneDrive\Desktop\SKYWATCH
git add .
git commit -m "feat: SKYWATCH-Det v1 — 2-phase training ready"
git push origin main
```

---

## 📓 ADIM 3 — Kaggle Notebook Kurulumu

### Notebook Ayarları
| Ayar | Değer |
|---|---|
| **Accelerator** | GPU T4 × 2 |
| **Internet** | On |
| **Session** | 12 Hours |

### WIDER FACE Dataseti Ekle
1. Kaggle → Add Data → Search: **"wider face"**
2. Dataset'i notebook'a ekle → `/kaggle/input/wider-face/`

---

## 📓 Notebook Hücreleri

### Hücre 1 — Repo + Kurulum
```python
import subprocess, sys

# SKYWATCH repo clone
!git clone https://github.com/mustafabsnl/SKYWATCH.git /kaggle/working/SKYWATCH

# ultralytics_skywatch'u kur (custom modüller dahil)
!pip install -e /kaggle/working/SKYWATCH/ultralytics_skywatch -q

# Path ekle
sys.path.insert(0, '/kaggle/working/SKYWATCH')
sys.path.insert(0, '/kaggle/working/SKYWATCH/ultralytics_skywatch')

print("✓ Kurulum tamamlandı")
```

### Hücre 2 — Modeli Doğrula
```python
!python /kaggle/working/SKYWATCH/src/model/verify_model.py
# Beklenen: 4/4 test geçti 🚀
```

### Hücre 3 — Veri Dönüşümü
```python
!python /kaggle/working/SKYWATCH/src/model/wider_face_converter.py \
    --wider_root /kaggle/input/wider-face \
    --output_dir /kaggle/working/skywatch_data \
    --splits train val \
    --min_face_size 5
```

### Hücre 4 — FAZ 1 Eğitimi
```python
!python /kaggle/working/SKYWATCH/src/model/train_skywatch_kaggle.py \
    --phase 1 \
    --skip_data_prep
```

### Hücre 5 — FAZ 2 Fine-tune
```python
# Faz 1'in best.pt yolunu kontrol et
import os
phase1_best = "/kaggle/working/runs/skywatch_det_phase1/weights/best.pt"
print(f"Faz 1 ağırlık: {phase1_best}")
print(f"Mevcut mu: {os.path.exists(phase1_best)}")
```

```python
!python /kaggle/working/SKYWATCH/src/model/train_skywatch_kaggle.py \
    --phase 2 \
    --phase1_weights /kaggle/working/runs/skywatch_det_phase1/weights/best.pt \
    --skip_data_prep
```

### Hücre 6 — Tek Seferde Tüm Eğitim
```python
# Alternatif: 2 fazı arka arkaya çalıştır
!python /kaggle/working/SKYWATCH/src/model/train_skywatch_kaggle.py \
    --phase all \
    --skip_data_prep
```

### Hücre 7 — Sonuçları Kaydet
```python
import shutil
from pathlib import Path

# Faz 2 final modeli
src = "/kaggle/working/runs/skywatch_det_phase2/weights/best.pt"
dst = "/kaggle/working/skywatch-det-final.pt"

if Path(src).exists():
    shutil.copy(src, dst)
    print(f"✓ Final model kaydedildi: {dst}")
else:
    # Faz 1 yeterli olduysa
    src = "/kaggle/working/runs/skywatch_det_phase1/weights/best.pt"
    shutil.copy(src, "/kaggle/working/skywatch-det-phase1.pt")
    print("Faz 2 bulunamadı, Faz 1 kaydedildi")
```

---

## ⚙️ Hiperparametre Açıklamaları

### Faz 1 — Neden bu değerler?
| Param | Değer | Neden |
|---|---|---|
| `lr0=0.001` | Standart | AdamW için optimal başlangıç |
| `batch=-1` | Auto | 2xT4 RAM'e göre otomatik |
| `mosaic=1.0` | Tam | Küçük yüz öğrenimi için kritik |
| `multi_scale=True` | Açık | 320-1280px varyasyon |
| `patience=50` | Orta | Erken duraksama için |

### Faz 2 — Neden bu değerler?
| Param | Değer | Neden |
|---|---|---|
| `lr0=0.0002` | 5× düşük | Fine-tune overfitting engeller |
| `lrf=0.005` | Çok küçük | LR neredeyse sabit kalır |
| `epochs=300` | Yüksek | patience ile erken duracak |
| `patience=80` | Yüksek | Daha uzun iyileşme şansı |
| `warmup=1.0` | Minimal | Model zaten iyi başlıyor |
| `mosaic=0.5` | Azaltıldı | Fine-tune'da daha temiz örnekler |
| `multi_scale=False` | Kapalı | Sabit ölçek ile kararlı fine-tune |

---

## 📊 Beklenen Eğitim Süresi (2×T4)

| Faz | Epoch | Süre |
|---|---|---|
| Faz 1 | 100 | ~5-7 saat |
| Faz 2 | 300 (patience=80) | ~6-10 saat |
| **Toplam** | **~400 (erken durabilir)** | **~12-15 saat** |

> 💡 Kaggle 12 saatlik limitinde sığdırmak için: `--phase 1` çalıştır, kaydet, yeni oturumda `--phase 2` ile devam et.

---

## 🔧 Sorun Giderme

### Kaggle'da WIDER FACE dizini bulunamadı
```python
import os
print(os.listdir('/kaggle/input/'))
# Çıktıdaki gerçek ismi DATA_RAW değişkenine yaz:
# train_skywatch_kaggle.py → DATA_RAW = Path("/kaggle/input/GERÇEK-İSİM")
```

### OOM (Out of Memory)
```python
# train_skywatch_kaggle.py içinde PHASE2 batch değerini düşür:
# batch = 8  →  batch = 4
```

### Faz 2 best.pt bulunamadı
```python
import glob
# En son best.pt'yi bul
files = glob.glob('/kaggle/working/runs/**/best.pt', recursive=True)
print(files)
```
