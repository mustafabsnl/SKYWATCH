"""
SKYWATCH — Suçlu Ekleme Aracı
Komut satırından veya interaktif olarak çalıştır.

Kullanim:
    python src/tools/add_criminal.py

    # veya argüman ile:
    python src/tools/add_criminal.py --photo foto.jpg --name "Ali Veli" --crime "Hirsizlik" --level HIGH
"""

import sys
import argparse
import shutil
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# GPU setup (insightface'ten önce)
import utils.gpu_setup  # noqa: F401

from utils.config import AppConfig
from utils.logger import EventLogger, EventType
from database.db import Database
from core.face_analyzer import FaceAnalyzer


def add_criminal_interactive():
    """Adım adım interaktif suçlu ekleme."""
    print("\n" + "="*55)
    print("  SKYWATCH — Suçlu Kayıt Sistemi")
    print("="*55)

    # Fotoğraf yolu
    while True:
        photo_path = input("\n  Fotoğraf yolu (tam yol veya dosya adı): ").strip().strip('"')
        p = Path(photo_path)
        if p.exists():
            break
        print(f"  [HATA] Dosya bulunamadı: {photo_path}")

    # Kişi bilgileri
    name = input("  Ad Soyad: ").strip()
    if not name:
        name = "Bilinmeyen"

    print("\n  Suç Türü Örnekleri: Hirsizlik, Dolandiricilik, Uyusturucu, Siddet, Cinayet")
    crime_type = input("  Suç Türü: ").strip()
    if not crime_type:
        crime_type = "Belirtilmedi"

    print("\n  Tehlike Seviyesi: LOW / MEDIUM / HIGH / CRITICAL")
    danger = input("  Tehlike Seviyesi [HIGH]: ").strip().upper()
    if danger not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        danger = "HIGH"

    print("\n  Durum: WANTED (Aranan) / CRIMINAL (Sabıkalı)")
    status = input("  Durum [WANTED]: ").strip().upper()
    if status not in ["WANTED", "CRIMINAL"]:
        status = "WANTED"

    return Path(photo_path), name, crime_type, danger, status


def add_criminal(photo_path: Path, name: str, crime_type: str,
                 danger_level: str, status: str):
    """Suçluyu sisteme ekler."""

    print(f"\n  [1/4] Sistem başlatılıyor...")
    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)
    analyzer = FaceAnalyzer(config)

    print(f"  [2/4] Fotoğraftan yüz embedding'i çıkarılıyor...")
    import cv2
    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"  [HATA] Fotoğraf okunamadı: {photo_path}")
        return False

    embedding = analyzer.extract_embedding(img)
    if embedding is None:
        print("  [HATA] Fotoğrafta yüz tespit edilemedi!")
        print("  → Daha net, yüzü tam görünen bir fotoğraf kullanın.")
        return False

    print(f"  [3/4] Veritabanına kaydediliyor...")

    # Fotoğrafı database/photos/ klasörüne kopyala
    photos_dir = config.get_photos_dir()
    dest_photo = photos_dir / photo_path.name
    shutil.copy2(str(photo_path), str(dest_photo))

    criminal_id = db.add_criminal(
        name=name,
        embedding=embedding,
        crime_type=crime_type,
        danger_level=danger_level,
        status=status,
        photo_path=str(dest_photo)
    )

    if criminal_id < 0:
        print("  [HATA] Veritabanına eklenemedi!")
        return False

    print(f"\n  [4/4] ✅ BAŞARILI!")
    print("="*55)
    print(f"  ID           : {criminal_id}")
    print(f"  Ad           : {name}")
    print(f"  Suç Türü     : {crime_type}")
    print(f"  Tehlike      : {danger_level}")
    print(f"  Durum        : {status}")
    print(f"  Embedding    : {embedding.shape} boyutunda vektör")
    print("="*55)
    print(f"\n  Sistem şimdi bu kişiyi kamerada TANIYACAK.")
    print(f"  Durum '{status}' → {'KIRMIZI kutu (Aranan)' if status == 'WANTED' else 'SARI kutu (Sabıkalı)'}")
    print()

    logger.log(EventType.SYSTEM_START, f"Suçlu eklendi: {name} (ID:{criminal_id})")
    return True


def list_criminals():
    """Mevcut suçluları listele."""
    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)

    import sqlite3
    with sqlite3.connect(str(config.get_db_path())) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, name, crime_type, danger_level, status, created_at FROM criminals ORDER BY id")
        rows = cur.fetchall()

    if not rows:
        print("\n  Veritabanı boş — henüz suçlu eklenmemiş.")
        return

    print("\n" + "="*70)
    print(f"  {'ID':<5} {'Ad':<20} {'Suç':<18} {'Tehlike':<10} {'Durum':<10}")
    print("  " + "-"*65)
    for r in rows:
        print(f"  {r['id']:<5} {r['name']:<20} {r['crime_type']:<18} {r['danger_level']:<10} {r['status']:<10}")
    print("="*70)
    print(f"  Toplam: {len(rows)} kişi\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKYWATCH Suçlu Ekleme")
    parser.add_argument("--photo", type=str, help="Fotoğraf yolu")
    parser.add_argument("--name", type=str, help="Ad Soyad")
    parser.add_argument("--crime", type=str, help="Suç türü")
    parser.add_argument("--level", type=str, default="HIGH",
                        choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    parser.add_argument("--status", type=str, default="WANTED",
                        choices=["WANTED", "CRIMINAL"])
    parser.add_argument("--list", action="store_true", help="Mevcut suçluları listele")
    args = parser.parse_args()

    if args.list:
        list_criminals()

    elif args.photo and args.name and args.crime:
        # Argüman modunda çalıştır
        add_criminal(
            photo_path=Path(args.photo),
            name=args.name,
            crime_type=args.crime,
            danger_level=args.level,
            status=args.status
        )
    else:
        # İnteraktif mod
        photo_path, name, crime_type, danger, status = add_criminal_interactive()
        add_criminal(photo_path, name, crime_type, danger, status)
