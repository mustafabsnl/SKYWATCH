"""
YOLO tahminlerine Soft-NMS uygula (crowded sahneler icin).

Ultralytics'te Soft-NMS dogrudan bir arguman olarak yok. Bu script,
standart YOLO predict sonucundan bbox+score alir, Soft-NMS uygular ve
filtrelenmis kutulari geri verir.

Kullanim (ornek):
    python src/model/yolo_softnms_predict.py --model runs/.../best.pt --source path/to/images --conf 0.001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ultralytics import YOLO

from soft_nms_aligner import apply_soft_nms_to_results


def _to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="best.pt / last.pt yolu")
    p.add_argument("--source", type=str, required=True, help="Goruntu klasoru veya dosya")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--score_thr", type=float, default=0.01)
    args = p.parse_args()

    model = YOLO(args.model)
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)

    # Soft-NMS uygula (her image icin)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = _to_numpy(r.boxes.xyxy)
        conf = _to_numpy(r.boxes.conf)
        keep_xyxy, keep_conf = apply_soft_nms_to_results(
            xyxy, conf, sigma=args.sigma, score_threshold=args.score_thr
        )

        # Bilgi yazdir (isteyen buradan devam edip kaydetme ekleyebilir)
        src = getattr(r, "path", "")
        print(f"{Path(str(src)).name}: {len(xyxy)} -> {len(keep_xyxy)} (Soft-NMS)")


if __name__ == "__main__":
    main()

