# 🔍 SKYWATCH — Akıllı Güvenlik Platformu

**Gerçek Zamanlı Yüz Tanıma Tabanlı Kişi Tespiti ve Takip Sistemi**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-grey)](LICENSE)

---

## 🎯 Proje Hakkında

SKYWATCH, güvenlik kameralarından gelen görüntüyü gerçek zamanlı olarak işleyerek kişileri tespit eden ve frameler arası kesintisiz takibini sağlayan bir bilgisayarlı görü projesidir.

Algılama tarafında klasik YOLOv8 mimarisini olduğu gibi kullanmak yerine, projenin ihtiyaçları doğrultusunda ek katmanlar (CBAM Attention, P2 Head, Feature Refinement) entegre edilerek model özelleştirilmiştir. ONNX Runtime ve CUDA ile GPU hızlandırması sayesinde frame başına ortalama **~50ms** işlem süresine ulaşılmıştır.

---

## 🔄 Sistem Akışı

```
Kamera Görüntüsü
      │
      ▼
  Yüz Algılama              ← SKYWATCH-Det (Custom YOLOv8 + CBAM)
      │
      ▼
  Embedding Çıkarımı         ← InsightFace ArcFace (buffalo_l, 512-d)
      │
      ▼
  Kişi Takibi                ← DeepSORT (benzersiz Track ID)
      │
      ▼
  Veritabanı Karşılaştırması ← SQLite üzerinde cosine similarity
      │
   ┌──┴──┐
   │     │
Eşleşme  Eşleşme Yok
   │     │
🔴 ALARM  🟢 Temiz
+ Kayıt
```

---

## 🧠 Model Mimarisi — SKYWATCH-Det

Standart bir YOLO modeli kullanmak yerine, güvenlik kamerası senaryolarına özel bir mimari tasarlandı:

| Bileşen | Açıklama |
|---------|----------|
| **Base** | YOLOv11m-equivalent backbone (depth=0.67, width=0.75) |
| **P2 Detection Head** | 4 ölçekli algılama (P2/P3/P4/P5) — 10×10px kadar küçük yüzleri yakalamak için |
| **C2f_CAM** | P2 seviyesinde CBAM tabanlı bağlamsal dikkat — kalabalık sahnelerde yan yana yüzleri ayırmak için |
| **FRM** | Backbone sonrası Feature Refinement — bulanık kamera görüntülerinde kenar/doku detayını kurtarmak için |
| **Custom Loss** | Adaptif büyüklük ağırlıklı loss — küçük yüzlere daha yüksek loss ağırlığı atayan özel fonksiyon |

Eğitim WIDER FACE veri seti üzerinde (32.203 görüntü, ~393K yüz) gerçekleştirilmiştir.

---

## 👤 Kişi Takip Sistemi

Her kişiye **DeepSORT** ile benzersiz bir Track ID atanır. Kişi görüş alanından kısa süre çıkıp geri dönse bile bu ID korunur. Takip verisinin üzerine, özel `MovementAnalyzer` modülü ile davranış analizi yapılır:

| Metrik | Açıklama |
|--------|----------|
| **Anlık / Ortalama Hız** | Piksel/frame mesafesi (son 30 frame ortalaması) |
| **Yön Vektörü** | Hareket yönü (dx, dy) |
| **Bekleme Süresi** | Aynı bölgede hareketsiz kalma süresi |
| **Ani Yön Değişimi** | 90°'yi aşan açısal sapma tespiti |
| **Davranış Skoru** | 0.0–1.0 arası şüpheli davranış skorlaması |

---

## 🗄️ Veritabanı ve Tanıma

Embedding vektörleri (512-d ArcFace) SQLite üzerinde saklanır. Her frame'de tespit edilen yüzler, veritabanındaki kayıtlarla cosine similarity ile karşılaştırılır. Sistem iki modda çalışır:

- **GENERAL** — Tüm veritabanı taranır, eşleşen kişi anlık olarak işaretlenir
- **PERSON_SEARCH** — Belirli bir kişi hedef alınarak aktif arama yapılır

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------| 
| Yüz Algılama | SKYWATCH-Det (Custom YOLOv8 + CBAM + P2 Head) |
| Yüz Tanıma / Embedding | InsightFace ArcFace (buffalo_l) |
| GPU Hızlandırma | ONNX Runtime + CUDA 12.6 |
| Kişi Takibi | DeepSORT Realtime |
| Hareket Analizi | Özel MovementAnalyzer modülü |
| Veritabanı | SQLite (embedding blob desteği) |
| GUI | PyQt5 |
| Dil | Python 3.11+ |

---

## ⚙️ Mimari Tasarım

Sistem, **Facade pattern** ile orkestre edilen, threaded pipeline üzerine kurulu modüler bir yapıdadır. Sorumluluklar katmanlara ayrılmıştır:

```
SKYWATCH/
├── config/
│   └── config.yaml              # Tüm sistem ayarları
├── src/
│   ├── main.py                  # Giriş noktası (GPU setup + threaded pipeline)
│   ├── core/
│   │   ├── face_analyzer.py     # YOLO algılama + InsightFace embedding
│   │   ├── tracker.py           # DeepSORT wrapper (kişi takibi)
│   │   ├── movement.py          # Hareket & davranış analizi
│   │   ├── gmc.py               # Global Motion Compensation
│   │   └── models.py            # Veri modelleri (dataclass)
│   ├── database/
│   │   └── db.py                # Kişi kaydı, embedding, tespit logu
│   ├── engine/
│   │   ├── pipeline.py          # Ana orkestratör (Facade pattern)
│   │   ├── renderer.py          # Overlay çizimi
│   │   ├── camera_manager.py    # Kamera akışı yönetimi
│   │   └── decision.py          # Karar motoru (CLEAN/SUSPICIOUS/WANTED)
│   ├── model/
│   │   ├── train_skywatch.py    # Model eğitim scripti
│   │   ├── skywatch_loss.py     # Adaptif küçük yüz ağırlıklı custom loss
│   │   └── skywatch_trainer.py  # Trainer konfigürasyonu
│   ├── gui/
│   │   └── main_window.py       # PyQt5 masaüstü arayüz
│   └── utils/
│       ├── config.py            # AppConfig
│       └── logger.py            # EventLogger
├── database/                    # SQLite DB + kişi fotoğrafları
└── logs/                        # Event logları + tespit ekran görüntüleri
```

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
venv\Scripts\python.exe src\run_gui.py
```

> **Not:** `config/config.yaml` dosyasındaki `source` alanını kamera numaranız (0, 1) veya video dosya yoluyla güncelleyin.

---

## 📌 Geliştirme Durumu

- [x] Ortam & Temel Altyapı
- [x] Yüz Algılama + GPU Hızlandırma
- [x] Veritabanı + Embedding Eşleştirme
- [x] Kişi Takibi + Threaded Pipeline + Overlay
- [x] Custom Model (CBAM + P2 Head + FRM)
- [x] Custom Loss (Adaptif Küçük Yüz Ağırlıklı)
- [x] GUI Arayüz (PyQt5)
- [ ] Kameralar Arası Geçiş Takibi
- [ ] Model Optimizasyonu & Entegrasyon Test

---

## 📄 Lisans

MIT © 2026 mustafabsnl
