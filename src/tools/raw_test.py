"""
Ham model testi — Sıfır filtre, sıfır HUD.
best.pt'nin kendi çıktısı, hiçbir şeye dokunulmadan.
"""
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

MODEL  = PROJECT_ROOT / "best.pt"
VIDEO  = PROJECT_ROOT / "Video1.webm"

model = YOLO(str(MODEL))
cap   = cv2.VideoCapture(str(VIDEO))

cv2.namedWindow("RAW", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RAW", 540, 960)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # ── Sıfır filtre: conf=0.01, iou=0.99 ──
    # conf=0.01 → Modelin en zayıf tespitini bile göster
    # iou=0.99  → NMS neredeyse kapalı, hiçbir kutu silinmiyor
    results = model.predict(frame, conf=0.01, iou=0.99, verbose=False)

    # Ultralytics'in kendi plot() fonksiyonu — bizden hiç dokunuş yok
    annotated = results[0].plot()

    cv2.imshow("RAW", annotated)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
