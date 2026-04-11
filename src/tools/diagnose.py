"""
SKYWATCH — Tanı Aracı
DB'deki embedding'lerle kameradan alınan yüzü karşılaştırır.
Gerçek cosine similarity skorunu gösterir.

Kullanim:
    python src/tools/diagnose.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import utils.gpu_setup  # noqa
from utils.config import AppConfig
from utils.logger import EventLogger, EventType
from database.db import Database
from core.face_analyzer import FaceAnalyzer


def main():
    print("\n" + "="*55)
    print("  SKYWATCH — Eşleşme Tanısı")
    print("="*55)

    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)
    analyzer = FaceAnalyzer(config)

    # 1) DB'yi kontrol et
    embeddings = db.get_all_embeddings()
    print(f"\n  DB'deki suçlu embedding sayısı: {len(embeddings)}")

    if not embeddings:
        print("  [HATA] DB BOŞ! Önce suçlu ekle:")
        print("  python src/tools/add_criminal.py")
        return

    # 2) Kameradan tek kare al
    print("\n  Kamera açılıyor... 's' tuşuna bas = fotoğraf çek, 'q' = çık")
    cap = cv2.VideoCapture(0)

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Tani - 's' ile fotoğraf çek", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            captured = frame.copy()
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured is None:
        print("  İptal edildi.")
        return

    # 3) Fotoğraftaki yüzleri bul
    print("\n  Yüz analiz ediliyor...")
    faces = analyzer.detect_faces(captured)
    print(f"  Tespit edilen yüz sayısı: {len(faces)}")

    if not faces:
        print("  [HATA] Yüz tespit edilemedi! Daha net ve yakın bir pozisyon dene.")
        return

    # 4) Her yüzü DB'deki herkesle karşılaştır
    print("\n" + "-"*55)
    print("  GERÇEK COSINE SIMILARITY SKORLARI:")
    print("-"*55)

    for i, face in enumerate(faces):
        print(f"\n  [Kameradaki Yüz {i+1}] det_score={face.det_score:.3f}")
        for cid, db_emb in embeddings:
            score = analyzer.compare(face.embedding, db_emb)

            # DB'den isim al
            import sqlite3
            with sqlite3.connect(str(config.get_db_path())) as conn:
                cur = conn.cursor()
                cur.execute("SELECT name, status FROM criminals WHERE id=?", (cid,))
                row = cur.fetchone()
                name = row[0] if row else f"ID:{cid}"
                status = row[1] if row else "?"

            threshold = analyzer.threshold
            match = "✅ EŞLEŞTİ" if score >= threshold else "❌ eşleşmedi"
            print(f"    vs {name:<20} ({status}) → Skor: {score:.4f} {match}")

    print("\n" + "-"*55)
    print(f"  Mevcut threshold: {analyzer.threshold}")
    print(f"  Eşleşmesi için en az bu skor gerekli")
    print("-"*55)
    print()


if __name__ == "__main__":
    main()
