"""
SKYWATCH — Ana Uygulama Giriş Noktası
GPU kurulumu → Config → Pipeline → Threaded döngü
"""

import sys
import os
import cv2
import time
import threading
import traceback
import numpy as np
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
from utils.run_logger import RunLogger
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

    run_logger = None
    try:
        config = AppConfig()
        _init_dirs(config)
        logger = EventLogger(config)
        logger.log(EventType.SYSTEM_START, "SKYWATCH başlatıldı")
        run_logger = RunLogger(config, logger)
        logger.set_run_logger(run_logger)

        pipeline = Pipeline(config, logger)
        renderer = OverlayRenderer()
        cam_manager = CameraManager(config, logger)
        cam_manager.start_all()
        active_cameras = [c.get("id") for c in config.get_active_cameras() if c.get("id")]
        if not active_cameras:
            raise RuntimeError("Aktif kamera bulunamadı. config/config.yaml kontrol edin.")
        layout = config.get_camera_layout()
        rows = max(1, int(layout.get("rows", 2)))
        cols = max(1, int(layout.get("cols", 2)))
        print(f"  Aktif kameralar: {', '.join(active_cameras)} | Çıkış: 'q'")
        run_logger.log_system(
            "main_mode_started",
            selected_mode="main",
            active_cameras=active_cameras,
            grid_rows=rows,
            grid_cols=cols,
        )
        time.sleep(2)

        _lock = threading.Lock()
        _display_frame = [None]
        _running = [True]
        _last_frames: dict[str, np.ndarray] = {}
        _perf_state = {
            "last_perf_ts": 0.0,
            "last_cam_ts": 0.0,
            "loop_ms": 0.0,
            "display_fps": 0.0,
            "last_loop_t": time.time(),
            "no_signal_counts": {cid: 0 for cid in active_cameras},
        }

        def _no_signal_frame(cam_id: str, w: int = 640, h: int = 360):
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(blank, f"{cam_id} - NO SIGNAL", (40, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            return blank

        def _compose_grid(frames: list[np.ndarray], cam_ids: list[str]):
            if len(frames) == 1:
                return frames[0]
            tile_h, tile_w = 360, 640
            resized = []
            for i, frame in enumerate(frames):
                tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                cv2.putText(tile, f"Kamera: {cam_ids[i]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                resized.append(tile)
            total_tiles = rows * cols
            while len(resized) < total_tiles:
                resized.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            grid_rows = []
            for r in range(rows):
                row_tiles = resized[r * cols:(r + 1) * cols]
                if row_tiles:
                    grid_rows.append(cv2.hconcat(row_tiles))
            return cv2.vconcat(grid_rows) if grid_rows else resized[0]

        def pipeline_loop():
            while _running[0]:
                loop_start = time.time()
                pipeline.begin_cycle()
                frames = []
                cam_ids = []
                try:
                    for cam_id in active_cameras:
                        frame = cam_manager.get_frame(cam_id)
                        stream = cam_manager.streams.get(cam_id)
                        if frame is None:
                            frame = _last_frames.get(cam_id)
                        has_signal = frame is not None
                        if frame is None:
                            frame = _no_signal_frame(cam_id)
                            _perf_state["no_signal_counts"][cam_id] += 1
                            frames.append(frame)
                            cam_ids.append(cam_id)
                            continue
                        _last_frames[cam_id] = frame
                        t_proc = time.perf_counter()
                        results = pipeline.process_frame(cam_id, frame)
                        proc_ms = (time.perf_counter() - t_proc) * 1000.0
                        overlay = renderer.draw(frame, results, pipeline.stats)
                        frames.append(overlay)
                        cam_ids.append(cam_id)

                        now = time.time()
                        if now - _perf_state["last_cam_ts"] >= run_logger.camera_interval:
                            run_logger.log_camera_status(
                                cam_id,
                                has_signal=has_signal,
                                stream_fps=round(stream.fps, 2) if stream else 0.0,
                                frame_shape=list(frame.shape) if frame is not None else [],
                                process_ms=round(proc_ms, 3),
                                decision_count=len(results),
                                no_signal_count=_perf_state["no_signal_counts"].get(cam_id, 0),
                            )
                        profile = pipeline.last_profile.get(cam_id, {})
                        if profile and now - _perf_state["last_perf_ts"] >= run_logger.perf_interval:
                            run_logger.log_pipeline_profile(cam_id, **profile)
                finally:
                    pipeline.end_cycle()

                if not frames:
                    time.sleep(0.01)
                    continue
                composed = _compose_grid(frames, cam_ids)
                h, w = composed.shape[:2]
                if w > 1280:
                    composed = cv2.resize(composed, (1280, int(h * 1280 / w)))
                with _lock:
                    _display_frame[0] = composed

                now = time.time()
                _perf_state["loop_ms"] = (now - loop_start) * 1000.0
                dt = max(1e-6, now - _perf_state["last_loop_t"])
                _perf_state["display_fps"] = 1.0 / dt
                _perf_state["last_loop_t"] = now
                if now - _perf_state["last_perf_ts"] >= run_logger.perf_interval:
                    run_logger.log_performance(
                        display_fps=round(_perf_state["display_fps"], 3),
                        loop_ms=round(_perf_state["loop_ms"], 3),
                        active_camera_count=len(active_cameras),
                        active_cameras=active_cameras,
                        grid_rows=rows,
                        grid_cols=cols,
                        total_faces_scanned=pipeline.stats.get("total_faces_scanned", 0),
                        total_matches=pipeline.stats.get("total_matches", 0),
                        active_tracks=pipeline.stats.get("active_tracks", 0),
                        reid_hits=pipeline.stats.get("reid_hits", 0),
                        velocity_rejected=pipeline.stats.get("velocity_rejected", 0),
                        selected_cameras=active_cameras,
                    )
                    _perf_state["last_perf_ts"] = now
                if now - _perf_state["last_cam_ts"] >= run_logger.camera_interval:
                    _perf_state["last_cam_ts"] = now

        worker = threading.Thread(target=pipeline_loop, daemon=True)
        worker.start()
        run_logger.log_system("main_pipeline_worker_started", worker_name=worker.name)

        cv2.namedWindow("SKYWATCH", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SKYWATCH", 1280, 720)

        while True:
            with _lock:
                frame = _display_frame[0]

            if frame is not None:
                cv2.imshow("SKYWATCH", frame)
            else:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "Baslatiliyor...", (450, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("SKYWATCH", blank)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        _running[0] = False
        worker.join(timeout=2)
        run_logger.log_system("main_pipeline_worker_stopped", worker_name=worker.name)
        cam_manager.stop_all()
        cv2.destroyAllWindows()
        logger.log(EventType.SYSTEM_START, "SKYWATCH kapatıldı")
        print("Kapatıldı.")
    except Exception as e:
        if run_logger is not None:
            run_logger.log_error("Unhandled exception in main run", e, traceback=traceback.format_exc())
        raise
    finally:
        if run_logger is not None:
            run_logger.close()


if __name__ == "__main__":
    run()
