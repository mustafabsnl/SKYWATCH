# 🔍 SKYWATCH — Akıllı Güvenlik Platformu

**Yüz Tanıma Tabanlı Sabıka Sorgulama & Kişi Arama Sistemi**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)](https://developer.nvidia.com/cuda-toolkit)
[![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange)](https://github.com/deepinsight/insightface)
[![License](https://img.shields.io/badge/License-MIT-grey)](LICENSE)

---

## 🎯 Ne Yapar?

| Özellik | Açıklama |
|---------|----------|
| **Pasif Gözetleme** | Kameradan gelen yüzleri gerçek zamanlı sabıka veritabanında sorgular |
| **Tracking** | DeepSORT ile kişileri frameler arası stabil ID ile takip eder |
| **Hareket Analizi** | Hız, yön, bekleme süresiyle şüpheli davranış tespiti |
| **Aktif Arama** | *(Yakında)* Fotoğraf ile kişi arama |
| **Multi-Camera** | *(Yakında)* Kameralar arası geçiş takibi |

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------|
| Yüz Algılama | RetinaFace (InsightFace `buffalo_l`) |
| Yüz Embedding | ArcFace 512-dim |
| GPU Hızlandırma | ONNX Runtime + CUDA 12.6 |
| Tracking | DeepSORT Realtime |
| Veritabanı | SQLite |
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
# source venv/bin/activate   # Linux/Mac

# 2. Bağımlılıklar
pip install -r requirements.txt

# 3. Çalıştır
python src/main.py
```

> **Not:** `config/config.yaml` dosyasındaki `source` alanını kamera numaranız (0, 1) veya video dosya yoluyla güncelleyin.

---

## ⚙️ Konfigürasyon

`config/config.yaml` tek konfigürasyon noktasıdır:

```yaml
cameras:
  - id: "CAM_01"
    source: 0             # 0 = webcam, "video.mp4" = dosya
    name: "Ana Giriş"

face:
  recognition_model: "buffalo_l"
  similarity_threshold: 0.45

movement:
  speed_threshold_fast: 50
  dwell_time_threshold: 120
```

---

## 📁 Proje Yapısı

```
SKYWATCH/
├── config/
│   └── config.yaml          # Tüm sistem ayarları
├── models/
│   └── models/buffalo_l/    # InsightFace modeli (otomatik indirilir)
├── src/
│   ├── main.py              # Giriş noktası (GPU setup + threaded pipeline)
│   ├── core/
│   │   ├── face_analyzer.py # InsightFace wrapper (GPU destekli)
│   │   ├── tracker.py       # DeepSORT wrapper
│   │   ├── movement.py      # Hareket & davranış analizi
│   │   └── models.py        # Veri modelleri (dataclass)
│   ├── database/
│   │   └── db.py            # SQLite CRUD işlemleri
│   ├── engine/
│   │   ├── pipeline.py      # Ana orkestratör (Facade pattern)
│   │   ├── renderer.py      # Overlay çizimi (SRP)
│   │   ├── camera_manager.py# Kamera akışı yönetimi
│   │   └── decision.py      # Karar motoru (Strategy pattern)
│   └── utils/
│       ├── config.py        # AppConfig
│       ├── logger.py        # EventLogger
│       └── gpu_setup.py     # cuDNN DLL path fix (Windows)
├── database/                # SQLite DB + yüz fotoğrafları
└── logs/                    # Event logları + tespit ekran görüntüleri
```

---

## 🏗️ Mimari

```
main.py
  ├── [Thread-Display]  → imshow() — hiç bloke olmaz
  └── [Thread-Pipeline] → CameraManager → Pipeline → OverlayRenderer
                               │
                    ┌──────────┴──────────┐
                    │                     │
              FaceAnalyzer          DeepSORT Tracker
              (InsightFace GPU)     (movement analizi)
                    │
               Database lookup
               (embedding cache)
                    │
               DecisionEngine
               (CLEAN/SUSPICIOUS/WANTED)
```

---

## 📌 Geliştirme Durumu

- [x] AŞAMA 1: Ortam & Temel Altyapı
- [x] AŞAMA 2: Yüz Algılama + GPU (InsightFace + CUDA 12.6)
- [x] AŞAMA 3: Veritabanı + Embedding Eşleştirme
- [x] AŞAMA 4: Tracking + Threaded Pipeline + Overlay
- [ ] AŞAMA 5: Multi-Camera + Aktif Arama
- [ ] AŞAMA 6: GUI Arayüz (PyQt5)
- [ ] AŞAMA 7: Entegrasyon & Test

---

## 📄 Lisans

MIT © 2025 mustafabsnl
