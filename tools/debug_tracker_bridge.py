import argparse
from pathlib import Path
import cv2

from src.utils.config import AppConfig
from src.utils.logger import EventLogger
from src.core.face_analyzer import FaceAnalyzer
from src.core.tracker import Tracker


def main():
    parser = argparse.ArgumentParser(description="Debug FaceAnalyzer -> Tracker bridge")
    parser.add_argument("--source", required=True, help="Video source path or camera index")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    args = parser.parse_args()

    cfg = AppConfig()
    logger = EventLogger(cfg)
    analyzer = FaceAnalyzer(cfg)
    tracker = Tracker(cfg.tracking)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    frame_idx = 0
    try:
        while frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            faces = analyzer.detect_faces(frame)
            embedding_ok_count = sum(1 for f in faces if f.embedding is not None)
            tracks = tracker.update("DEBUG_CAM", faces, frame, (0.0, 0.0))
            dbg = tracker.last_debug.get("DEBUG_CAM", {})
            decisions_count = len(tracks)

            print(
                f"frame_idx={frame_idx} "
                f"faces_count={len(faces)} "
                f"embedding_ok_count={embedding_ok_count} "
                f"tracker_prepared_detections={dbg.get('tracker_prepared_detections', 0)} "
                f"tracker_output_tracks={dbg.get('tracker_output_tracks', 0)} "
                f"decisions_count={decisions_count} "
                f"rejected_no_embedding={dbg.get('tracker_rejected_no_embedding', 0)} "
                f"rejected_bad_bbox={dbg.get('tracker_rejected_bad_bbox', 0)}"
            )
            frame_idx += 1
    finally:
        cap.release()


if __name__ == "__main__":
    main()
