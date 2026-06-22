"""
Smoke test: Renderer kutu + etiket çıktısı (canlı kamera yok).

Çalıştır:
  python tools/test_renderer_overlay.py

Çıktı:
  logs/debug/renderer_overlay_test.png
"""
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from core.models import DecisionResult
from engine.renderer import OverlayRenderer


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("renderer_test")

    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    names = {7: "Kerim"}
    dec = DecisionResult(
        track_id=12,
        bbox=[100, 100, 220, 220],
        status="WANTED",
        danger_level="HIGH",
        color=(0, 0, 255),
        criminal_id=7,
        confidence=0.91,
        behavior_label="normal",
        global_id=None,
        time_since_update=0,
    )

    r = OverlayRenderer(trace_logger=log)
    out = r.draw(
        frame,
        [dec],
        {"active_tracks": 1},
        names,
        camera_id="TEST_CAM",
        trace_logger=log,
    )

    out_dir = PROJECT_ROOT / "logs" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "renderer_overlay_test.png"
    ok = cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path} ok={ok}")
    if not ok:
        raise SystemExit(1)
    print("[OK] Beklenen: kirmizi kutu ve 'Kerim %91' etiketi.")


if __name__ == "__main__":
    main()
