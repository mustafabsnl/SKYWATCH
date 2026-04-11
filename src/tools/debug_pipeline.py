"""
SKYWATCH — Debug Pipeline
Tam olarak hangi aşamada sorun olduğunu gösterir.
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
from core.tracker import Tracker


def main():
    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)
    analyzer = FaceAnalyzer(config)
    tracker = Tracker(config.tracking)

    # DB embeddings
    db_embeddings = db.get_all_embeddings()
    print(f"\nDB'de {len(db_embeddings)} embedding var")

    cap = cv2.VideoCapture(0)
    frame_count = 0

    print("\nDEBUG MODU - Her frame'de detaylı bilgi:")
    print("="*70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Her 5 frame'de bir algılama
        if frame_count % 5 == 0:
            faces = analyzer.detect_faces(frame)
        else:
            faces = []

        # Tracker güncelle
        tracks = tracker.update("CAM_01", faces, frame)

        for track in tracks:
            emb = track.face_embedding
            has_emb = emb is not None
            has_match = track.criminal_match is not None

            # Eğer embedding varsa, DB ile karşılaştır
            best_score = 0.0
            best_name = ""
            if has_emb and db_embeddings:
                for cid, db_emb in db_embeddings:
                    score = analyzer.compare(emb, db_emb)
                    if score > best_score:
                        best_score = score
                        import sqlite3
                        with sqlite3.connect(str(config.get_db_path())) as conn:
                            cur = conn.cursor()
                            cur.execute("SELECT name FROM criminals WHERE id=?", (cid,))
                            row = cur.fetchone()
                            best_name = row[0] if row else f"ID:{cid}"

            color = (0, 255, 0)
            label = f"ID:{track.track_id}"

            if has_match:
                color = (0, 0, 255)
                label += f" MATCH(cid:{track.criminal_match.criminal_id})"
            elif has_emb:
                label += f" emb:OK score:{best_score:.2f}"
                if best_score > 0.3:
                    color = (0, 165, 255)  # Turuncu - eşleşmeli ama match atanmamış
                    label += " !!BUG!!"
            else:
                label += " emb:NONE"
                color = (128, 128, 128)

            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if frame_count % 15 == 0:
                print(f"  Track:{track.track_id} | is_new:{track.is_new} | "
                      f"emb:{'YES' if has_emb else 'NO'} | "
                      f"match:{'YES' if has_match else 'NO'} | "
                      f"best_score:{best_score:.3f} ({best_name})")

        # İstatistik
        info = f"Frame:{frame_count} | Tracks:{len(tracks)}"
        cv2.putText(frame, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        cv2.imshow("DEBUG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
