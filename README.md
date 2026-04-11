# 🔍 SKYWATCH — Akıllı Güvenlik Platformu

**Gerçek Zamanlı Yüz Tanıma Tabanlı Suçlu Tespiti ve Kişi Takip Sistemi**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-grey)](LICENSE)

---

## 🎯 Ne Yapar?

SKYWATCH, güvenlik kameralarından gelen görüntüyü gerçek zamanlı olarak işleyerek kameradaki kişilerin yüzlerini **suçlu veritabanıyla** karşılaştırır. Eşleşme tespit edildiğinde anlık uyarı üretir.

| Özellik | Açıklama |
|---------|----------|
| **Suçlu Tespiti** | Kameradaki yüzler veritabanındaki suçlu kayıtlarıyla karşılaştırılır |
| **Anlık Uyarı** | Eşleşme bulunduğunda alarm ve tespit kaydı üretilir |
| **Kişi Takibi** | DeepSORT ile her kişiye sabit ID atanır, frameler arası takip sağlanır |
| **Hareket Analizi** | Hız, yön, bekleme süresiyle şüpheli davranış skorlaması yapılır |
| **Aktif Arama** | *(Geliştirme aşamasında)* Fotoğraf yükleyerek kameralarda kişi arama |
| **Multi-Camera** | *(Geliştirme aşamasında)* Kameralar arası geçiş takibi |

---

## 🔄 Sistem Akışı

```
Kamera Görüntüsü
      │
      ▼
  Yüz Algılama         ← GPU ile gerçek zamanlı (50ms/frame)
      │
      ▼
  Kişi Takibi          ← DeepSORT (Track ID atama)
      │
      ▼
  Suçlu DB Sorgusu     ← Yüz vektörü → SQLite embedding karşılaştırması
      │
   ┌──┴──┐
   │     │
Eşleşme  Eşleşme Yok
   │     │
🔴 ALARM  🟢 Temiz
+ Kayıt
```

---

## 👤 Kişi Takip Sistemi (Tracking)

Takip sistemi, SKYWATCH'ın en kritik bileşenlerinden biridir. Yalnızca yüz tanımak yeterli değildir — sistemin kişiyi **sürekli izlemesi**, **kaybetmemesi** ve **davranışını analiz etmesi** gerekmektedir.

### Track ID Mantığı

Kameraya giren her kişiye **benzersiz bir Track ID** atanır. Bu ID şu özelliklere sahiptir:

- Kişi kısa süre görüş alanından çıkıp geri dönse bile **aynı ID korunur**
- Yüz embedding'i ile ilişkilendirildiğinden suçlu eşleşmesi **her frame'de tekrar sorgulanmaz** — bir kez eşleşti mi o track için kayıt açılır
- Kişi tamamen kaybolduğunda track sonlandırılır ve log kaydı oluşturulur

### Hareket Analizi

Her aktif track için `MovementAnalyzer` modülü şu verileri hesaplar:

| Metrik | Açıklama |
|--------|----------|
| **Anlık Hız** | Son 2 frame arası piksel/frame mesafesi |
| **Ortalama Hız** | Son 30 frame üzerinden hesaplanan ortalama |
| **Yön Vektörü** | Hareket yönü (dx, dy) |
| **Bekleme Süresi** | Kişi hareket etmeden aynı bölgede kaldığı süre (sn) |
| **Ani Yön Değişimi** | Son 10 frame içinde açısal sapma 90°'yi aşarsa tetiklenir |
| **Rota** | Son 30 noktanın (x, y) koordinat geçmişi |

### Davranış Skoru

Her kişiye 0.0 – 1.0 arasında bir **davranış skoru** hesaplanır:

```
Koşma (hız > eşik)        → +0.35
Hızlı hareket              → +0.20
Uzun bekleme               → +0.15
Ani yön değişimi           → +0.10
─────────────────────────────────
Skor ≥ 0.5  → SUSPICIOUS (Şüpheli)
Skor < 0.5  → NORMAL
```

### Planlanan Geliştirmeler

- **Kameralar Arası Takip:** Kişi bir kameradan çıkıp komşu kameraya girdiğinde yüz embedding'i ile aynı kişi olarak tanınacak ve Track ID sürekliliği sağlanacak
- **Rota Görselleştirme:** Kişinin kamera içindeki hareket rotası ekranda çizilecek
- **Zaman Bazlı Analiz:** Belirli bir bölgede anormal sıklıkta görünen kişiler otomatik işaretlenecek

---

## 🗄️ Suçlu Veritabanı

Sistemde yalnızca **suçlu kayıtları** tutulur. Her kayıt şunları içerir:

| Alan | Açıklama |
|------|----------|
| Ad / Soyad | Kişi kimlik bilgisi |
| Suç Türü | İşlenen suçun kategorisi |
| Tehlike Seviyesi | LOW / MEDIUM / HIGH / CRITICAL |
| Durum | WANTED (Aranan) / CRIMINAL (Sabıkalı) |
| Yüz Vektörü | 512 boyutlu ArcFace embedding |
| Fotoğraf | Referans yüz fotoğrafı |

Veritabanına yeni suçlu eklemek için:

```python
from database.db import Database
db = Database(config, logger)
db.add_criminal(
    name="Ad Soyad",
    embedding=face_embedding,   # FaceAnalyzer ile çıkarılır
    crime_type="Hırsızlık",
    danger_level="HIGH",
    status="WANTED"
)
```

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------|
| Yüz Algılama & Embedding | ArcFace mimarisi (geliştirme aşamasında) |
| GPU Hızlandırma | ONNX Runtime + CUDA 12.6 |
| Kişi Takibi | DeepSORT Realtime |
| Hareket Analizi | Özel `MovementAnalyzer` modülü |
| Veritabanı | SQLite (embedding blob desteği) |
| Dil | Python 3.11+ |

---

## 🚀 Kurulum

### Gereksinimler
- **GPU:** NVIDIA (CUDA 12.6 uyumlu)
- **Python:** 3.11+

```bash
# 1. Sanal ortam
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Bağımlılıklar
pip install -r requirements.txt

# 3. Çalıştır
python src/main.py
```

> **Not:** `config/config.yaml` dosyasındaki `source` alanını kamera numaranız (0, 1) veya video dosya yoluyla güncelleyin.

---

## ⚙️ Konfigürasyon

```yaml
cameras:
  - id: "CAM_01"
    source: 0             # 0 = webcam, "video.mp4" = dosya
    name: "Ana Giriş"

face:
  similarity_threshold: 0.45   # Suçlu eşleşme eşiği

movement:
  speed_threshold_fast: 50
  dwell_time_threshold: 120
```

---

## 📁 Proje Yapısı

```
SKYWATCH/
├── config/
│   └── config.yaml            # Tüm sistem ayarları
├── src/
│   ├── main.py                # Giriş noktası (GPU setup + threaded pipeline)
│   ├── core/
│   │   ├── face_analyzer.py   # Yüz algılama & embedding çıkarımı
│   │   ├── tracker.py         # DeepSORT wrapper (kişi takibi)
│   │   ├── movement.py        # Hareket & davranış analizi
│   │   └── models.py          # Veri modelleri (dataclass)
│   ├── database/
│   │   └── db.py              # Suçlu kaydı, embedding, tespit logu
│   ├── engine/
│   │   ├── pipeline.py        # Ana orkestratör (Facade pattern)
│   │   ├── renderer.py        # Overlay çizimi (SRP)
│   │   ├── camera_manager.py  # Kamera akışı yönetimi
│   │   └── decision.py        # Karar motoru (CLEAN/SUSPICIOUS/WANTED)
│   └── utils/
│       ├── config.py          # AppConfig
│       ├── logger.py          # EventLogger
│       └── gpu_setup.py       # cuDNN DLL path fix (Windows)
├── database/                  # SQLite DB + suçlu fotoğrafları
└── logs/                      # Event logları + tespit ekran görüntüleri
```

---

## 📌 Geliştirme Durumu

- [x] AŞAMA 1: Ortam & Temel Altyapı
- [x] AŞAMA 2: Yüz Algılama + GPU
- [x] AŞAMA 3: Suçlu Veritabanı + Embedding Eşleştirme
- [x] AŞAMA 4: Kişi Takibi + Threaded Pipeline + Overlay
- [ ] AŞAMA 5: Multi-Camera + Aktif Arama
- [ ] AŞAMA 6: GUI Arayüz
- [ ] AŞAMA 7: Model Geliştirme & Entegrasyon Test

---

## 📄 Lisans

MIT © 2025 mustafabsnl
