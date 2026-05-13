"""
SKYWATCH — Run Logger
Her uygulama çalıştırmasında izole run klasörü ve performans kayıtları üretir.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml


class RunLogger:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.enabled = bool(config.logging.get("enable_run_logging", True))
        self.async_logging = bool(config.logging.get("async_logging", True))
        self.perf_interval = float(config.logging.get("perf_log_interval_sec", 1.0))
        self.camera_interval = float(config.logging.get("camera_log_interval_sec", 2.0))
        self.keep_last_runs = int(config.logging.get("keep_last_runs", 30))
        self.start_ts = time.time()
        self.start_iso = datetime.now().isoformat(timespec="seconds")
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        base_rel = config.logging.get("run_logs_dir", "logs/runs/")
        self.base_dir = config.project_root / base_rel
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.paths = {
            "system": self.run_dir / "system.log",
            "performance": self.run_dir / "performance.jsonl",
            "camera_status": self.run_dir / "camera_status.jsonl",
            "pipeline_profile": self.run_dir / "pipeline_profile.jsonl",
            "events": self.run_dir / "events.log",
            "errors": self.run_dir / "errors.log",
            "summary": self.run_dir / "summary.json",
            "config_snapshot": self.run_dir / "config_snapshot.yaml",
        }
        for p in (
            self.paths["system"],
            self.paths["performance"],
            self.paths["camera_status"],
            self.paths["pipeline_profile"],
            self.paths["events"],
            self.paths["errors"],
        ):
            p.touch(exist_ok=True)

        self._q: queue.Queue = queue.Queue(maxsize=5000)
        self._stop_evt = threading.Event()
        self._writer = None
        self._warning_count = 0
        self._error_count = 0
        self._dropped_log_count = 0
        self._perf_samples: list[float] = []
        self._loop_ms_samples: list[float] = []
        self._max_proc_ewma_ms = 0.0
        self._final_perf_level = 0
        self._total_alerts = 0
        self._camera_no_signal_counts: dict[str, int] = {}
        self._last_metrics: dict = {}

        self._cleanup_old_runs()
        self.write_config_snapshot()
        self._start_writer()
        self._log_startup_context()

    @property
    def run_dir_path(self) -> Path:
        return self.run_dir
        
    def get_run_dir(self) -> Path:
        return self.run_dir
        
    def resolve_run_path(self, filename: str) -> Path:
        """
        Aktif run klasörü içinde filename için güvenli dosya yolu döndürür.
        filename sadece dosya adı veya relative path olabilir.
        Absolute path verilirse bile güvenli şekilde run klasörüne normalize edilir.
        """
        name = Path(filename).name
        return self.run_dir / name

    def _start_writer(self):
        if not self.enabled:
            return
        if self.async_logging:
            self._writer = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer.start()

    def _enqueue(self, kind: str, record: dict):
        if not self.enabled:
            return
        if self.async_logging:
            try:
                self._q.put_nowait((kind, record))
            except queue.Full:
                # Kuyruk doluysa sistemi bloklamadan kritik olmayan logları düş.
                self._dropped_log_count += 1
        else:
            self._write_one(kind, record)

    def _writer_loop(self):
        while not self._stop_evt.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            kind, record = item
            self._write_one(kind, record)
            self._q.task_done()

        # Kuyrukta kalanları flush et
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            kind, record = item
            self._write_one(kind, record)
            self._q.task_done()

    def _base_record(self, category: str, level: str, message: str, fields: dict | None = None):
        return {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "category": category,
            "level": level,
            "message": message,
            "fields": fields or {},
        }

    def _write_one(self, kind: str, record: dict):
        if kind == "system":
            with self.paths["system"].open("a", encoding="utf-8") as f:
                f.write(f"{record['timestamp_iso']} | {record['level']} | {record['message']} | {json.dumps(record['fields'], ensure_ascii=False)}\n")
            return

        if kind == "event":
            with self.paths["events"].open("a", encoding="utf-8") as f:
                f.write(f"{record['timestamp_iso']} | {record['level']} | {record['message']} | {json.dumps(record['fields'], ensure_ascii=False)}\n")
            return

        if kind == "error":
            with self.paths["errors"].open("a", encoding="utf-8") as f:
                f.write(f"{record['timestamp_iso']} | ERROR | {record['message']} | {json.dumps(record['fields'], ensure_ascii=False)}\n")
            return

        if kind in ("performance", "camera_status", "pipeline_profile"):
            p = self.paths[kind]
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_system(self, message: str, **fields):
        """Updates _last_metrics keys used by summary.json (active_mode, target_person_id, selected_cameras)."""
        # Merge canonical mode from either field (GUI uses active_mode in runtime_context)
        mode = fields.get("active_mode")
        if mode is None:
            mode = fields.get("selected_mode")
        if mode is not None:
            self._last_metrics["active_mode"] = mode
            if mode != "PERSON_SEARCH":
                self._last_metrics["target_person_id"] = None
                self._last_metrics["target_person_ids"] = []

        if "selected_cameras" in fields:
            self._last_metrics["selected_cameras"] = list(fields["selected_cameras"] or [])

        ps_context = (
            mode == "PERSON_SEARCH"
            or fields.get("selected_mode") == "PERSON_SEARCH"
            or fields.get("active_mode") == "PERSON_SEARCH"
        )

        def _as_int_ids(xs) -> list[int]:
            out: list[int] = []
            if not isinstance(xs, (list, tuple)):
                return out
            for x in xs:
                try:
                    out.append(int(x))
                except (TypeError, ValueError):
                    continue
            return out

        if isinstance(fields.get("target_person_ids"), (list, tuple)) and ps_context:
            clean = _as_int_ids(fields["target_person_ids"])
            if clean:
                self._last_metrics["target_person_ids"] = clean
                self._last_metrics["target_person_id"] = clean[0]

        tid = fields.get("target_person_id")
        if tid is not None:
            self._last_metrics["target_person_id"] = tid

        opts = fields.get("options")
        if isinstance(opts, dict) and ps_context:
            ot = opts.get("target_person_ids")
            if isinstance(ot, (list, tuple)):
                clean = _as_int_ids(ot)
                if clean:
                    self._last_metrics["target_person_ids"] = clean
                    self._last_metrics["target_person_id"] = clean[0]
            if opts.get("target_person_id") is not None and not self._last_metrics.get("target_person_ids"):
                try:
                    lone = int(opts["target_person_id"])
                    self._last_metrics["target_person_id"] = lone
                    self._last_metrics["target_person_ids"] = [lone]
                except (TypeError, ValueError):
                    pass

        self._enqueue("system", self._base_record("system", "INFO", message, fields))

    def log_event(self, event_type: str, message: str, **fields):
        level = fields.pop("level", "INFO")
        if str(level).upper().startswith("WARN"):
            self._warning_count += 1
        self._enqueue("event", self._base_record(event_type, str(level).upper(), message, fields))

    def log_error(self, message: str, exc: Exception | None = None, **fields):
        self._error_count += 1
        if exc is not None:
            fields["exception"] = repr(exc)
            fields["traceback"] = traceback.format_exc()
        self._enqueue("error", self._base_record("error", "ERROR", message, fields))

    def log_performance(self, **metrics):
        fps = metrics.get("display_fps")
        target_fps = metrics.get("worker_target_fps")
        loop_ms = metrics.get("display_loop_ms", metrics.get("loop_ms"))
        if isinstance(fps, (int, float)):
            fps_val = float(fps)
            if isinstance(target_fps, (int, float)) and target_fps > 0:
                fps_val = min(fps_val, float(target_fps))
            self._perf_samples.append(fps_val)
        if isinstance(loop_ms, (int, float)):
            self._loop_ms_samples.append(float(loop_ms))
        if isinstance(metrics.get("proc_ewma_ms"), (int, float)):
            self._max_proc_ewma_ms = max(self._max_proc_ewma_ms, float(metrics["proc_ewma_ms"]))
        if isinstance(metrics.get("perf_level"), int):
            self._final_perf_level = metrics["perf_level"]
        if isinstance(metrics.get("total_alerts"), int):
            self._total_alerts = max(self._total_alerts, metrics["total_alerts"])
        self._last_metrics.update(metrics)
        self._enqueue("performance", self._base_record("performance", "INFO", "performance_sample", metrics))

    def log_camera_status(self, camera_id: str, **metrics):
        if metrics.get("has_signal") is False:
            self._camera_no_signal_counts[camera_id] = self._camera_no_signal_counts.get(camera_id, 0) + 1
        payload = {"camera_id": camera_id, **metrics}
        self._enqueue("camera_status", self._base_record("camera_status", "INFO", "camera_status_sample", payload))

    def log_pipeline_profile(self, camera_id: str, **metrics):
        metrics = dict(metrics or {})
        metrics.pop("camera_id", None)
        payload = {"camera_id": camera_id, **metrics}
        self._enqueue("pipeline_profile", self._base_record("pipeline_profile", "INFO", "pipeline_profile_sample", payload))

    def write_config_snapshot(self):
        try:
            src = Path(getattr(self.config, "_config_path", ""))
            if src.exists():
                shutil.copy2(src, self.paths["config_snapshot"])
                return
            with self.paths["config_snapshot"].open("w", encoding="utf-8") as f:
                yaml.safe_dump(getattr(self.config, "_data", {}), f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            self.log_error("Failed to write config snapshot", e)

    def _cleanup_old_runs(self):
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            runs = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda p: p.name,
            )
            if self.keep_last_runs <= 0:
                return
            extra = len(runs) - self.keep_last_runs
            if extra <= 0:
                return
            for old in runs[:extra]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Run cleanup failed: {e}")

    def _safe_gpu_info(self):
        info = {"torch_cuda": "unknown", "gpu_name": "unknown", "onnx_providers": "unknown"}
        try:
            import torch

            info["torch_version"] = torch.__version__
            info["torch_cuda"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        try:
            import onnxruntime as ort

            info["onnx_providers"] = ort.get_available_providers()
        except Exception:
            pass
        return info

    def _log_startup_context(self):
        try:
            import cv2
            import numpy as np
            try:
                from video_sources import ACTIVE_VIDEO_SOURCE_IDS
                active_video_source_ids = list(ACTIVE_VIDEO_SOURCE_IDS)
            except Exception:
                active_video_source_ids = []

            enabled_cameras = [c.get("id") for c in self.config.get_enabled_cameras()]
            default_active_cameras = [c.get("id") for c in self.config.get_active_cameras()]
            configured_max_active_cameras = int(self.config.get_max_active_cameras())
            base = {
                "run_id": self.run_id,
                "start_time": self.start_iso,
                "python_version": sys.version,
                "platform": platform.platform(),
                "os": os.name,
                "working_directory": str(Path.cwd()),
                "project_root": str(self.config.project_root),
                "config_path": str(getattr(self.config, "_config_path", "unknown")),
                "max_active_cameras": configured_max_active_cameras,
                "configured_max_active_cameras": configured_max_active_cameras,
                "runtime_camera_mode": "2-camera-default",
                "hard_runtime_cap": 2,
                "infrastructure_supported_cameras": 4,
                "layout": self.config.get_camera_layout(),
                "enabled_cameras": enabled_cameras,
                "active_cameras": default_active_cameras,
                "default_active_cameras": default_active_cameras,
                "selected_cameras": default_active_cameras,
                "active_video_source_ids": active_video_source_ids,
                "recognition_model": self.config.face.get("recognition_model", "unknown"),
                "face_model_path": str(self.config.project_root / "best.pt"),
                "similarity_threshold": self.config.face.get("similarity_threshold", "unknown"),
                "tracking": self.config.tracking,
                "performance": getattr(self.config, "performance", {}),
                "cv2_version": cv2.__version__,
                "numpy_version": np.__version__,
            }
            base.update(self._safe_gpu_info())
            self.log_system("SKYWATCH run started", **base)
            torch_cuda = base.get("torch_cuda")
            onnx_providers = base.get("onnx_providers")
            if torch_cuda is False:
                self.log_event(
                    "GPU_WARNING",
                    "PyTorch CUDA is not available. YOLO .pt inference may run on CPU and cause severe slowdown.",
                    level="WARNING",
                )
                if isinstance(onnx_providers, list) and "CUDAExecutionProvider" in onnx_providers:
                    self.log_event(
                        "GPU_WARNING",
                        "ONNX Runtime CUDA provider exists, but PyTorch CUDA is disabled. Consider exporting YOLO model to ONNX/TensorRT or installing CUDA-enabled torch.",
                        level="WARNING",
                    )
        except Exception as e:
            self.log_error("Failed to collect startup context", e)

    def _summary_payload(self):
        end_ts = time.time()
        duration = max(0.0, end_ts - self.start_ts)
        avg_fps = sum(self._perf_samples) / len(self._perf_samples) if self._perf_samples else 0.0
        avg_loop = sum(self._loop_ms_samples) / len(self._loop_ms_samples) if self._loop_ms_samples else 0.0
        max_loop = max(self._loop_ms_samples) if self._loop_ms_samples else 0.0
        min_fps = min(self._perf_samples) if self._perf_samples else 0.0
        
        debug_cfg = self.config.get("debug", {}) if hasattr(self.config, "get") else {}
        ps_enabled = bool(debug_cfg.get("person_search_trace_enabled", False))
        ps_path = str(self.resolve_run_path(debug_cfg.get("person_search_jsonl_file", "person_search_trace.jsonl"))) if ps_enabled else None
        
        return {
            "run_id": self.run_id,
            "start_time": self.start_iso,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(duration, 3),
            "run_dir": str(self.run_dir),
            "selected_cameras": self._last_metrics.get("selected_cameras", []),
            "active_mode": self._last_metrics.get("active_mode", "UNKNOWN"),
            "target_person_id": self._last_metrics.get("target_person_id", None),
            "target_person_ids": self._last_metrics.get("target_person_ids", []),
            "person_search_trace_enabled": ps_enabled,
            "person_search_trace_path": ps_path,
            "max_active_cameras": self.config.get_max_active_cameras(),
            "avg_fps": round(avg_fps, 3),
            "min_fps": round(min_fps, 3),
            "max_loop_ms": round(max_loop, 3),
            "avg_loop_ms": round(avg_loop, 3),
            "max_proc_ewma_ms": round(self._max_proc_ewma_ms, 3),
            "final_perf_level": self._final_perf_level,
            "total_faces_scanned": int(self._last_metrics.get("total_faces_scanned", 0)),
            "total_matches": int(self._last_metrics.get("total_matches", 0)),
            "total_alerts": int(self._total_alerts),
            "camera_no_signal_counts": self._camera_no_signal_counts,
            "error_count": self._error_count,
            "warning_count": self._warning_count,
            "dropped_log_count": int(self._dropped_log_count),
            "inference_restart_count": int(self._last_metrics.get("inference_restart_count", 0)),
            "inference_error_count": int(self._last_metrics.get("inference_error_count", 0)),
            "inference_last_success_age_sec": float(self._last_metrics.get("inference_last_success_age_sec", 0.0)),
            "inference_loop_count": int(self._last_metrics.get("inference_loop_count", 0)),
            "config_snapshot_path": str(self.paths["config_snapshot"]),
        }

    def close(self):
        if not self.enabled:
            return
        self.log_system("SKYWATCH run stopping")
        if self.async_logging and self._writer is not None:
            self._stop_evt.set()
            try:
                self._q.put_nowait(None)
            except Exception:
                pass
            self._writer.join(timeout=2.0)

        try:
            with self.paths["summary"].open("w", encoding="utf-8") as f:
                json.dump(self._summary_payload(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to write summary: {e}")
