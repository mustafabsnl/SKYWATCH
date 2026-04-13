# SKYWATCH-Det — Kaggle Eğitim Kılavuzu

## 📦 Kullanılan Dataset

**`lylmsc/wider-face-for-yolo-training`**

| Özellik | Değer |
|---|---|
| Görüntü Sayısı | 32,203 (train + val ayrılmış) |
| Etiketli Yüz | ~393,703 |
| Yüz Boyut Aralığı | 10px ~ 500px |
| Format | YOLO Detection (`class cx cy w h`) |
| Zorluk Seviyeleri | Easy / Medium / Hard (gömülü) |
| Dönüşüm Gerekiyor mu? | ❌ Hayır — direkt kullanılır |

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
| **Batch** | 16 | 8 |
| **Warmup** | 5 epoch | 1 epoch |
| **Patience** | 50 | 80 |
| **Mosaic** | 1.0 | 0.5 |
| **Multi-scale** | ❌ (OOM riski) | ❌ |

> 💡 `multi_scale=True` yapmak istersen batch'i 8'e düşür.

---

## 📐 Model Mimarisi

**SKYWATCH-Det** (`skywatch-det.yaml`)
- 4-scale Detection Head: **P2 / P3 / P4 / P5**
- P2 head: ≤20px yüzler (Hard difficulty için kritik)
- **C2f_CAM**: Kalabalık sahnelerde contextual attention
- **FRM**: Bulanık kamera görüntülerinde özellik iyileştirme
- Tek sınıf: `face`

---

## 📓 Kaggle Notebook Hücreleri

### ⚙️ Hücre 0 — Notebook Ayarları

| Ayar | Değer |
|---|---|
| **Accelerator** | GPU T4 × 2 |
| **Internet** | ✅ Açık (kagglehub için zorunlu) |
| **Session** | 12 Hours |

> ❌ Artık `+ Add Data` ile dataset eklemeye gerek yok.
> `kagglehub.dataset_download()` scriptin içinde otomatik indirir.

---

### Hücre 1 — Repo Clone + Kurulum

```python
import os, sys

# 1. SKYWATCH repo'yu klonla
!git clone https://github.com/mustafabsnl/SKYWATCH.git /kaggle/working/SKYWATCH

# 2. Custom modülleri kur (C2f_CAM, FRM, skywatch-det.yaml)
!python /kaggle/working/SKYWATCH/kaggle_setup.py

# 3. Path'e ekle
sys.path.insert(0, '/kaggle/working/SKYWATCH')
sys.path.insert(0, '/kaggle/working/SKYWATCH/src/model')

print("✓ Kurulum tamamlandı")
```

---

### Hücre 2 — Ortam Doğrulama

```python
import torch

# GPU kontrolü
n = torch.cuda.device_count()
print(f"GPU: {n}x {torch.cuda.get_device_name(0)}")

# Custom modül kontrolü
from ultralytics.nn.modules import C2f_CAM, FRM
print("✓ C2f_CAM & FRM yüklendi")

# Dataset kontrolü
import os
p = "/kaggle/input/wider-face-for-yolo-training"
print(f"Dataset: {os.path.exists(p)}")
print("Train:", len(os.listdir(f"{p}/images/train")), "görüntü")
print("Val  :", len(os.listdir(f"{p}/images/val")), "görüntü")
```

Beklenen çıktı:
```
GPU: 2x Tesla T4
✓ C2f_CAM & FRM yüklendi
Dataset: True
Train: ~28000 görüntü
Val  : ~4200 görüntü
```

---

### Hücre 3 — FAZ 1 Eğitimi (100 epoch)

```python
!python /kaggle/working/SKYWATCH/src/model/train_skywatch_kaggle.py \
    --phase 1
```

> Script otomatik olarak `dataset.yaml` / `data.yaml` bulur.
> Eğer bulamazsa `/kaggle/input/wider-face-for-yolo-training/` altında oluşturur.

---

### Hücre 4 — FAZ 2 Fine-tune (300 epoch)

```python
# Faz 1 best.pt kontrolü
import os
p1 = "/kaggle/working/runs/skywatch_det_phase1/weights/best.pt"
print(f"Faz 1 ağırlık mevcut: {os.path.exists(p1)}")
```

```python
!python /kaggle/working/SKYWATCH/src/model/train_skywatch_kaggle.py \
    --phase 2 \
    --skip_data_prep
```

---

### Hücre 5 — TEK SEFERDE Tüm Eğitim

```python
# Faz 1 + Faz 2 arka arkaya
!python /kaggle/working/SKYWATCH/src/model/train_skywatch_kaggle.py \
    --phase all
```

---

### Hücre 6 — Final Modeli Kaydet

```python
import shutil
from pathlib import Path

candidates = [
    ("/kaggle/working/runs/skywatch_det_phase2/weights/best.pt", "skywatch-det-final.pt"),
    ("/kaggle/working/runs/skywatch_det_phase1/weights/best.pt", "skywatch-det-phase1.pt"),
]

for src, dst_name in candidates:
    if Path(src).exists():
        dst = f"/kaggle/working/{dst_name}"
        shutil.copy(src, dst)
        size_mb = Path(dst).stat().st_size / 1e6
        print(f"✓ {dst_name} kaydedildi ({size_mb:.1f} MB)")
        break
```

---

## 📊 Beklenen Eğitim Süresi (2×T4)

| Faz | Epoch | Tahmini Süre |
|---|---|---|
| Faz 1 | 100 | ~6–8 saat |
| Faz 2 | 300 (patience=80) | ~10–15 saat |
| **Toplam** | **~400** | **~16–22 saat** |

> 💡 **12 saat limitini aşmamak için:**
> Oturum 1: `--phase 1` → Checkpoint'i indir
> Oturum 2: `--phase 2 --skip_data_prep` → Fine-tune

---

## 🔧 Sorun Giderme

### ❌ Dataset bulunamadı
```python
import os
# Kaggle'daki gerçek klasör adını kontrol et
print(os.listdir('/kaggle/input/'))
# Çıktıda 'wider-face-for-yolo-training' görünmüyorsa,
# notebook'a dataset eklenmemiş demektir.
```

### ❌ OOM (Out of Memory) Hatası
```python
# train_skywatch_kaggle.py → PHASE1 içinde:
# batch = 16  →  batch = 8
# multi_scale = False  (zaten False)
```

### ❌ C2f_CAM / FRM bulunamadı
```python
# kaggle_setup.py'yi tekrar çalıştır:
!python /kaggle/working/SKYWATCH/kaggle_setup.py
```

### ❌ Faz 2 best.pt bulunamadı
```python
import glob
files = glob.glob('/kaggle/working/runs/**/best.pt', recursive=True)
print(files)
# En son oluşturulan dosyayı --phase1_weights argümanına ver
```

---

## 📁 Proje Yapısı (Eğitim İlgili)

```
SKYWATCH/
├── kaggle_setup.py                     ← Custom modül kurulumu (C2f_CAM, FRM)
└── src/
    ├── model/
    │   ├── train_skywatch_kaggle.py    ← 2 fazlı eğitim scripti
    │   └── wider_face_converter.py     ← Artık kullanılmıyor (dataset hazır)
    └── ultralytics_patch/
        ├── nn/modules/
        │   └── skywatch_modules.py     ← C2f_CAM + FRM implementasyonu
        └── cfg/models/skywatch/
            └── skywatch-det.yaml       ← 4-scale Detection mimarisi
```
