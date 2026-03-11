"""
SKYWATCH — Ana Uygulama Giriş Noktası
GPU kurulumu → Config → Pipeline → Threaded döngü
"""

import sys
import os
import cv2
import time
import threading
from pathlib import Path

# ── 1. GPU DLL PATH (insightface/onnxruntime'dan önce) ──────────────
_venv = Path(sys.executable).parent.parent
for _sub in ("cudnn", "cublas"):
    _d = _venv / "Lib" / "site-packages" / "nvidia" / _sub / "bin"
    if _d.exists():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + ";" + os.environ.get("PATH", "")

# ── 2. Proje Kökü ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import AppConfig
from utils.logger import EventLogger, EventType
from engine.pipeline import Pipeline
from engine.camera_manager import CameraManager
from engine.renderer import OverlayRenderer


def _init_dirs(config: AppConfig):
    """Gerekli dizinleri oluşturur."""
    for d in [config.get_db_path().parent,
              config.get_photos_dir(),
              config.get_log_dir(),
              config.get_screenshot_dir()]:
        d.mkdir(parents=True, exist_ok=True)


def run():
    """SKYWATCH ana döngüsü."""
    print("=" * 55)
    print("  SKYWATCH — Akıllı Güvenlik Platformu")
    print("=" * 55)

    config = AppConfig()
    _init_dirs(config)
    logger = EventLogger(config)
    logger.log(EventType.SYSTEM_START, "SKYWATCH başlatıldı")

    pipeline = Pipeline(config, logger)
    renderer = OverlayRenderer()
    cam_manager = CameraManager(config, logger)
    cam_manager.start_all()

    cam_id = config.cameras[0]["id"]
    print(f"  Kamera: {cam_id} | Çıkış: 'q'")
    time.sleep(2)

    # Paylaşımlı durum (thread-safe)
    _lock = threading.Lock()
    _display_frame = [None]
    _running = [True]

    def pipeline_loop():
        while _running[0]:
            frame = cam_manager.get_frame(cam_id)
            if frame is None:
                time.sleep(0.01)
                continue
            results = pipeline.process_frame(cam_id, frame)
            overlay = renderer.draw(frame, results, pipeline.stats)
            # Küçült (max 1280px)
            h, w = overlay.shape[:2]
            if w > 1280:
                overlay = cv2.resize(overlay, (1280, int(h * 1280 / w)))
            with _lock:
                _display_frame[0] = overlay

    worker = threading.Thread(target=pipeline_loop, daemon=True)
    worker.start()

    cv2.namedWindow("SKYWATCH", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SKYWATCH", 1280, 720)

    while True:
        with _lock:
            frame = _display_frame[0]

        if frame is not None:
            cv2.imshow("SKYWATCH", frame)
        else:
            import numpy as np
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Baslatiliyor...", (450, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("SKYWATCH", blank)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    _running[0] = False
    worker.join(timeout=2)
    cam_manager.stop_all()
    cv2.destroyAllWindows()
    logger.log(EventType.SYSTEM_START, "SKYWATCH kapatıldı")
    print("Kapatıldı.")


if __name__ == "__main__":
    run()
