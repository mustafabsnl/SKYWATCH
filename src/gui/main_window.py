"""SKYWATCH — Ana Pencere (Sidebar Mimarisi)"""

import sys
import time
import threading
import copy
import logging
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QApplication, QMessageBox
)
from PyQt6.QtGui import QFont

from gui.styles.theme import GLOBAL_STYLE, BG_APP
from gui.widgets.sidebar import Sidebar
from gui.pages.dashboard    import DashboardPage
from gui.pages.mode_select  import ModePage
from gui.pages.add_criminal import AddCriminalPage
from gui.pages.criminal_list import CriminalListPage
from utils.config import AppConfig
from utils.logger import EventLogger
from utils.run_logger import RunLogger
from engine.pipeline import Pipeline
from engine.renderer import OverlayRenderer
from engine.camera_manager import CameraManager
from core.local_bbox_tracker import LocalBBoxTracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_sk_log = logging.getLogger("SKYWATCH")


def _options_person_search_ids(options: dict | None) -> list[int]:
    options = dict(options or {})
    raw = options.get("target_person_ids")
    out: list[int] = []
    if isinstance(raw, (list, tuple)):
        for x in raw:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
    if not out and options.get("target_person_id") is not None:
        try:
            out.append(int(options["target_person_id"]))
        except (TypeError, ValueError):
            pass
    return out


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SKYWATCH")
        self.setMinimumSize(1280, 760)
        self.resize(1440, 900)
        self.setStyleSheet(GLOBAL_STYLE)
        self.statusBar().hide()

        # ── Runtime durumu (lifecycle / logger) ── BEFORE widget creation ──
        self._cfg = None
        self._logger = None
        self._run_logger: RunLogger | None = None
        self._pipeline = None
        self._renderer = None
        self._local_bbox_tracker: LocalBBoxTracker | None = None
        self._max_active_cameras = 4
        self._grid_rows = 2
        self._grid_cols = 2

        # Eagerly create cfg + EventLogger + RunLogger so child widgets can log
        # (ModePage logs at construction time). If this fails we keep going so the UI
        # at least appears, but diagnostics will print to stdout.
        self._ensure_runtime_logging()
        if self._logger:
            self._logger.info("[MAIN_RUNTIME_LOGGING_READY]")

        # ── Merkezi widget ──────────────────────────────────────────────────
        central = QWidget()
        central.setStyleSheet(f"background: {BG_APP};")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sol Sidebar ─────────────────────────────────────────────────────
        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self._goto)
        root.addWidget(self.sidebar)

        # ── Sayfa Yığını ────────────────────────────────────────────────────
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {BG_APP};")
        root.addWidget(self.stack, 1)

        self.pg_dash = DashboardPage()
        self.pg_mode = ModePage()
        mp_msg = (
            f"[MAIN_MODEPAGE_CREATED] object_id={id(self.pg_mode)} "
            f"type={type(self.pg_mode).__module__}.{type(self.pg_mode).__name__}"
        )
        print(mp_msg, flush=True)
        _sk_log.info(mp_msg)
        if self._logger:
            self._logger.info(mp_msg)
        if self._run_logger:
            self._run_logger.log_event(
                "MAIN_MODEPAGE_CREATED",
                mp_msg,
                object_id=id(self.pg_mode),
                type=f"{type(self.pg_mode).__module__}.{type(self.pg_mode).__name__}",
            )

        # ModePage'e runtime logger'ları geçir — _start() doğrudan run_logger/event_logger'a yazsın
        try:
            self.pg_mode.attach_runtime_logger(self._logger, self._run_logger)
        except Exception as e:
            print(f"[MAIN_MODEPAGE_ATTACH_LOGGER_FAILED] error={e!s}", flush=True)

        self.pg_add  = AddCriminalPage()
        self.pg_list = CriminalListPage()

        self.stack.addWidget(self.pg_dash)   # 0
        self.stack.addWidget(self.pg_mode)   # 1
        self.stack.addWidget(self.pg_add)    # 2
        self.stack.addWidget(self.pg_list)   # 3

        # ── Bağlantılar ─────────────────────────────────────────────────────
        self.pg_mode.system_start.connect(self._on_start)
        self.pg_mode.system_start.connect(self._debug_mode_start_signal)
        sig_msg = (
            f"[MAIN_SIGNAL_CONNECTED] ModePage.system_start -> MainWindow._on_start "
            f"(modepage_object_id={id(self.pg_mode)})"
        )
        print(sig_msg, flush=True)
        _sk_log.info(sig_msg)
        if self._logger:
            self._logger.info(sig_msg)
        if self._run_logger:
            self._run_logger.log_event(
                "MAIN_SIGNAL_CONNECTED",
                sig_msg,
                modepage_object_id=id(self.pg_mode),
            )

        self.pg_mode.system_stop.connect(self._on_stop)
        self.pg_add.person_added.connect(self._on_person_added)
        # Video oynatma durumu
        self._cam_manager: CameraManager | None = None
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._display_tick)
        self._selected_cameras = []
        self._active_mode = None
        self._active_mode_options = {}
        self._system_started_from_mode_page = False
        self._display_last_tick = 0.0
        self._display_fps_smooth = 0.0
        self._alerted_tracks: set[tuple[str, int, str]] = set()
        self._last_cam_frames: dict[str, np.ndarray] = {}
        self._last_raw_frames: dict[str, np.ndarray] = {}
        self._last_display_frames: dict[str, np.ndarray] = {}
        self._last_overlay_frames: dict[str, np.ndarray] = {}
        self._last_decisions: dict[str, list] = {}
        self._last_criminal_names: dict[str, dict] = {}
        self._last_inference_ts: dict[str, float] = {}
        self._last_process_ms: dict[str, float] = {}
        self._last_pipeline_profiles: dict[str, dict] = {}
        self._last_local_tracker_metrics: dict[str, dict] = {}
        self._last_seen_frame_seq: dict[str, int] = {}
        self._new_display_frame_counts: dict[str, int] = {}
        self._stale_display_frame_counts: dict[str, int] = {}
        self._raw_frame_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._inference_thread = None
        self._inference_running = False
        self._inference_round_robin_idx = 0
        self._inference_last_heartbeat_ts = 0.0
        self._inference_last_success_ts = 0.0
        self._inference_loop_count = 0
        self._inference_error_count = 0
        self._inference_consecutive_errors = 0
        self._inference_restart_count = 0
        self._inference_restarting = False
        self._inference_watchdog_timer = QTimer(self)
        self._inference_watchdog_timer.timeout.connect(self._check_inference_watchdog)
        self._latest_frame = None
        self._latest_stats = (0.0, 0, 0, 0)
        self._pending_alerts: list[tuple[int, str, str]] = []
        self._proc_ewma_ms = 35.0
        self._perf_level = 0  # 0: kalite, 1: dengeli, 2: hizli
        self._display_frame_count = 0

        # Display ve inference ayarlari
        self._display_target_fps = 15
        self._display_timer_interval_ms = int(1000 / self._display_target_fps)
        self._torch_cuda_available = False

        # Pipeline / Renderer / LocalBBoxTracker are lazy via _ensure_pipeline();
        # cfg + EventLogger + RunLogger were already created at the top of __init__.
        self._last_perf_log_ts = 0.0
        self._last_camera_log_ts = 0.0
        self._warn_cooldowns: dict[tuple[str, str], float] = {}
        self._no_signal_since: dict[str, float] = {}
        self._stream_online_since: dict[str, float] = {}
        self._active_start_watch_gen = 0

        # DB sayısını ilk göster
        self._update_db_count()

    # ── Navigasyon ────────────────────────────────────────────────────────────
    def _goto(self, index: int):
        self.stack.setCurrentIndex(index)
        if index == 3:
            self.pg_list.refresh()
            self._update_db_count()

    # ── Pipeline sinyalleri ──────────────────────────────────────────────────
    def _debug_mode_start_signal(self, mode, cameras, options):
        """Debug: confirms ModePage.system_start reaches MainWindow (see [MAIN_SIGNAL_RECEIVED_DEBUG])."""
        opt = dict(options or {})
        msg = f"[MAIN_SIGNAL_RECEIVED_DEBUG] mode={mode} cameras={list(cameras or [])} options={opt}"
        print(msg, flush=True)
        _sk_log.info(msg)
        if self._logger:
            self._logger.info(msg)

    def _schedule_active_start_watch(self, mode: str, cameras: list):
        """Task 10: if inference never runs, fail loudly."""
        self._active_start_watch_gen = int(getattr(self, "_active_start_watch_gen", 0)) + 1
        gen = self._active_start_watch_gen
        cams = list(cameras or [])

        def _check():
            if gen != getattr(self, "_active_start_watch_gen", 0):
                return
            if int(getattr(self, "_inference_loop_count", 0) or 0) > 0:
                return
            line = (
                f"[ACTIVE_START_FAILED] reason=inference_loop_count_still_zero mode={mode} "
                f"selected_cameras={cams} inference_loop_count={getattr(self, '_inference_loop_count', 0)}"
            )
            print(line, flush=True)
            _sk_log.error(line)
            if self._logger:
                self._logger.error(line)
            QMessageBox.warning(
                self,
                "SKYWATCH",
                "Sistem aktif başlatılamadı. Logları kontrol edin.",
            )

        QTimer.singleShot(2000, _check)

    def _on_start(self, mode: str, cameras: list, options: dict = None):
        print("[MAIN_ON_START_ENTER]", mode, cameras, options, flush=True)

        mode = str(mode or "GENERAL").upper()
        if mode not in ("GENERAL", "PERSON_SEARCH"):
            mode = "GENERAL"
        options = dict(options or {})
        cam_list = list(cameras or [])

        _sk_log.info(f"[MAIN_ON_START_ENTER] mode={mode} cameras={cam_list} options={options}")

        if not self._ensure_pipeline():
            QMessageBox.critical(self, "Hata", "Sistem bileşenleri başlatılamadı. Lütfen logları kontrol edin.")
            return

        if self._logger:
            self._logger.info(f"[MAIN_ON_START_ENTER] mode={mode} cameras={cam_list} options={options}")

        self._active_mode = mode
        self._active_mode_options = dict(options)

        if self._logger:
            self._logger.info(f"[GUI_ON_START] mode={mode} options={options} cameras={cam_list}")

            if mode == "PERSON_SEARCH":
                tids = _options_person_search_ids(options)
                self._logger.info(
                    f"[PERSON_SEARCH_SELECT] target_person_ids={tids} "
                    f"name(s)={options.get('target_person_names')!r}"
                )
                self._logger.info(f"[PERSON_SEARCH_START] target_person_ids={tids}")
                self._logger.person_search_trace("UI_MODE_SELECT", mode=mode)
                self._logger.person_search_trace(
                    "UI_PERSON_SELECTED",
                    selected_person_ids=tids,
                    selected_person_names=options.get("target_person_names"),
                    selected_person_id=options.get("target_person_id"),
                    selected_person_name=options.get("target_person_name"),
                )
                self._logger.person_search_trace(
                    "UI_START_REQUEST",
                    mode=mode,
                    selected_person_ids=tids,
                    selected_cameras=cam_list,
                )
            elif mode == "GENERAL":
                self._logger.info(f"[GENERAL_MODE_START] cameras={cam_list}")

        if self._run_logger:
            ps_ids = _options_person_search_ids(options) if mode == "PERSON_SEARCH" else []
            self._run_logger.log_system(
                "runtime_context",
                active_mode=mode,
                target_person_id=(
                    (ps_ids[0] if ps_ids else None) if mode == "PERSON_SEARCH" else None
                ),
                target_person_ids=(ps_ids if mode == "PERSON_SEARCH" else None),
                target_person_names=(
                    options.get("target_person_names") if mode == "PERSON_SEARCH" else None
                ),
                selected_cameras=cam_list,
                options=options,
            )

        selected = list(cameras or self.pg_mode.get_camera_ids(only_checked=True))
        self._alerted_tracks.clear()
        self._pending_alerts.clear()

        if not self._start_video_mode(selected, mode, options):
            self._system_started_from_mode_page = False
            self._active_mode = None
            self._active_mode_options = {}
            self.sidebar.set_running(False)
            self.pg_dash.set_mode("", False)
            try:
                self.pg_mode.release_start_after_main_failure()
            except Exception:
                pass
            if self._run_logger:
                self._run_logger.log_system(
                    "gui_video_mode_start_failed",
                    selected_mode="UNKNOWN",
                    active_mode="UNKNOWN",
                    options={},
                )
            return

        self.sidebar.set_running(True)
        self.pg_dash.set_mode(mode, True)
        self.pg_dash.clear_alerts()

        self._goto(0)
        self.sidebar._select(0)

    def _on_stop(self):
        self._stop_video_mode()
        self.sidebar.set_running(False)
        self.pg_dash.set_mode("", False)

    def _start_video_mode(self, cameras: list, mode: str = "GENERAL", options: dict = None):
        mode = str(mode or "GENERAL").upper()
        if mode not in ("GENERAL", "PERSON_SEARCH"):
            mode = "GENERAL"
        options = dict(options or {})

        cam_in = list(cameras or [])
        msg = f"[GUI_VIDEO_MODE_START] mode={mode} cameras={cam_in} options={options}"
        if self._logger:
            self._logger.info(msg)
        else:
            print(msg, flush=True)

        if mode == "PERSON_SEARCH":
            ps_targets = _options_person_search_ids(options)
            if len(ps_targets) < 1:
                if self._logger:
                    self._logger.error(
                        '[PERSON_SEARCH_START_BLOCKED] reason="missing_target_person_ids_in_mainwindow"'
                    )
                QMessageBox.warning(
                    self,
                    "Uyarı",
                    "Kişi Ara modunda en az bir kişi seçmelisiniz.",
                )
                return False

        preview_before = (
            not self._system_started_from_mode_page and self._video_timer.isActive()
        )

        if not self._ensure_pipeline():
            return False

        self._system_started_from_mode_page = True
        pv_msg = f"[PREVIEW_TO_ACTIVE_TRANSITION] from_preview={str(preview_before).lower()} mode={mode}"
        if self._logger:
            self._logger.info(pv_msg)
        else:
            print(pv_msg, flush=True)

        normalized = self._normalize_selected_cameras(cameras or [])
        if self._logger:
            self._logger.info(f"[CAMERA_SELECTION_NORMALIZED] selected_cameras={normalized}")

        if not normalized:
            self._system_started_from_mode_page = False
            QMessageBox.warning(self, "Uyarı", "En az bir kamera seçin.")
            return False

        self._selected_cameras = normalized
        self._stop_inference_worker()
        self._sync_camera_sources()

        active_streams = [
            cid for cid in self._selected_cameras if self._cam_manager and cid in self._cam_manager.streams
        ]
        if self._logger:
            self._logger.info(
                f"[CAMERA_SYNC_DONE] selected_cameras={self._selected_cameras} active_streams={active_streams}"
            )

        if self._cam_manager is None or not any(
            cid in self._cam_manager.streams for cid in self._selected_cameras
        ):
            self._system_started_from_mode_page = False
            QMessageBox.warning(self, "Hata", "Seçilen kameralar için kaynak bulunamadı.")
            return False

        if self._run_logger is not None:
            self._run_logger.log_system(
                "gui_video_mode_started",
                selected_mode=mode,
                active_mode=mode,
                selected_cameras=self._selected_cameras,
                max_active_cameras=self._max_active_cameras,
                layout={"rows": self._grid_rows, "cols": self._grid_cols},
                options=options,
            )

        if self._pipeline:
            if self._logger:
                tlist = _options_person_search_ids(options)
                self._logger.info(
                    f"[GUI_PIPELINE_SET_MODE_CALL] mode={mode} target_person_ids={tlist} "
                    f"legacy_target_person_id={options.get('target_person_id')}"
                )

            if not self._pipeline.set_mode(mode, options):
                self._system_started_from_mode_page = False
                QMessageBox.warning(self, "Hata", "Seçilen mod başlatılamadı. Lütfen logları kontrol edin.")
                return False

            if self._logger:
                self._logger.info(
                    f"[PIPELINE_MODE_SET_DONE] mode={mode} current_mode={getattr(self._pipeline, 'current_mode', None)}"
                )

            if mode == "PERSON_SEARCH" and self._logger:
                tlist = _options_person_search_ids(options)
                self._logger.person_search_trace(
                    "UI_PIPELINE_SET_MODE_CALL",
                    mode=mode,
                    target_person_ids=tlist,
                    target_person_id=options.get("target_person_id"),
                )

        if self._logger:
            self._logger.info(f"[INFERENCE_WORKER_START_REQUEST] selected_cameras={self._selected_cameras}")

        self._start_inference_worker()

        if self._logger:
            self._logger.info("[INFERENCE_WORKER_STARTED]")

        self._start_display_timer()

        if self._logger:
            self._logger.info(f"[DISPLAY_TIMER_STARTED] interval_ms={getattr(self, '_display_timer_interval_ms', 0)}")

        self._inference_watchdog_timer.start(2000)

        if self._logger:
            self._logger.info(f"[ACTIVE_SYSTEM_STARTED] mode={mode} cameras={self._selected_cameras}")

        self._schedule_active_start_watch(mode, list(self._selected_cameras))

        return True

    def _start_display_timer(self):
        self._display_last_tick = time.time()
        self._display_fps_smooth = 0.0
        perf = self._cfg.performance if self._cfg else {}
        display_fps = int(perf.get("display_fps_gpu", perf.get("display_fps", 15))) if self._torch_cuda_available else int(perf.get("display_fps_cpu", perf.get("display_fps", 12)))
        self._display_target_fps = max(5, min(30, display_fps))
        ui_interval = int(1000 / self._display_target_fps)
        self._display_timer_interval_ms = ui_interval
        if self._run_logger is not None:
            self._run_logger.log_event(
                "DISPLAY_TIMER_CONFIG",
                "Display timer configured",
                display_target_fps=self._display_target_fps,
                display_timer_interval_ms=ui_interval,
                torch_cuda_available=self._torch_cuda_available,
            )
        if self._video_timer.isActive():
            self._video_timer.stop()
        self._video_timer.start(ui_interval)

    def _start_camera_preview_only(self, camera_ids: list):
        """
        Dashboard'da sadece ham kamera görüntüsü göstermek için kullanılır.
        Inference worker başlatmaz. Pipeline mode değiştirmez.
        """
        if not self._ensure_pipeline():
            return
        self._selected_cameras = self._normalize_selected_cameras(camera_ids or [])
        if self._logger:
            self._logger.info(f"[DASH_CAMERA_PREVIEW_START] selected_cameras={self._selected_cameras} reason=\"system_not_started_from_mode_page\"")
            
        self._sync_camera_sources()
        self._start_display_timer()
        self.sidebar.set_running(False)
        self.pg_dash.set_mode("PREVIEW", False)

    def _restart_active_system_with_cameras(self, camera_ids: list):
        self._stop_inference_worker()
        self._ensure_pipeline()
        self._selected_cameras = self._normalize_selected_cameras(camera_ids or [])
        self._sync_camera_sources()
        self._inference_round_robin_idx = 0
        
        mode = self._active_mode or "GENERAL"
        opts = dict(self._active_mode_options or {})
        
        if self._logger:
            self._logger.info(f"[DASH_CAMERA_SELECTION_CHANGED] preserve_mode={mode} options={opts}")
            self._logger.info(
                f"[DASH_PIPELINE_MODE_REAPPLY] mode={mode} target_person_ids="
                f"{_options_person_search_ids(opts)} legacy_id={opts.get('target_person_id')}"
            )
            
        self.pg_dash.set_mode(mode, True)
        if self._pipeline:
            if not self._pipeline.set_mode(mode, opts):
                if self._logger:
                    self._logger.error(f"[DASH_PIPELINE_MODE_REAPPLY_FAILED] mode={mode}")
                return
            
        self._start_inference_worker()
        self._start_display_timer()
        self.sidebar.set_running(True)

    def _stop_video_mode(self):
        if self._inference_watchdog_timer.isActive():
            self._inference_watchdog_timer.stop()
        self._stop_inference_worker()
        if self._video_timer.isActive():
            self._video_timer.stop()
        if self._cam_manager is not None:
            self._cam_manager.stop_all()
        self._last_cam_frames.clear()
        self._last_raw_frames.clear()
        self._last_display_frames.clear()
        self._last_overlay_frames.clear()
        self._last_decisions.clear()
        self._last_criminal_names.clear()
        self._last_inference_ts.clear()
        self._last_process_ms.clear()
        self._last_pipeline_profiles.clear()
        self._last_local_tracker_metrics.clear()
        if self._run_logger is not None:
            self._run_logger.log_system("gui_video_mode_stopped", selected_cameras=self._selected_cameras)
        self.pg_dash.update_stats(0.0, 0, 0, 0)
        self.pg_dash.clear_frame()
        self._system_started_from_mode_page = False
        self._active_mode = None
        self._active_mode_options = {}

    def _camera_source_map(self) -> dict:
        allowed_ids = set()
        cfg_sources = {}
        if self._cfg is not None:
            for cam in self._cfg.get_enabled_cameras():
                cid = cam.get("id")
                if cid:
                    allowed_ids.add(cid)
                    cfg_sources[cid] = cam.get("source")
        try:
            from video_sources import VIDEO_SOURCES
            src = dict(VIDEO_SOURCES)
            if allowed_ids:
                src = {cid: s for cid, s in src.items() if cid in allowed_ids}
                # video_sources içinde olmayan aktif kameralar için config source fallback
                for cid in allowed_ids:
                    if cid not in src and cid in cfg_sources:
                        src[cid] = cfg_sources[cid]
            return src
        except Exception:
            return cfg_sources

    def _ensure_camera_manager(self):
        if self._cam_manager is not None:
            return
        self._cam_manager = CameraManager(config=self._cfg, logger=self._logger, autoload_config=False)

    def _ensure_runtime_logging(self) -> bool:
        """Initialize cfg + EventLogger + RunLogger early so any widget can log."""
        if self._cfg is not None and self._logger is not None:
            return True
        try:
            self._cfg = AppConfig()
            self._max_active_cameras = self._cfg.get_max_active_cameras()
            layout = self._cfg.get_camera_layout()
            self._grid_rows = int(layout.get("rows", 2))
            self._grid_cols = int(layout.get("cols", 2))
            self._logger = EventLogger(self._cfg)
            if self._cfg.logging.get("enable_run_logging", True):
                self._run_logger = RunLogger(self._cfg, self._logger)
                self._logger.set_run_logger(self._run_logger)
            return True
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[RUNTIME_LOGGING_INIT_FAILED] error={e!s} traceback={tb}", flush=True)
            return False

    def _ensure_pipeline(self) -> bool:
        if self._pipeline is not None and self._renderer is not None:
            return True
        try:
            if not self._ensure_runtime_logging():
                return False
            self._pipeline = Pipeline(self._cfg, self._logger)
            self._local_bbox_tracker = LocalBBoxTracker(self._cfg.tracking, logger=self._logger)
            try:
                debug_cfg = self._cfg.get("debug", {}) or {}
                overlay_cfg = self._cfg.get("overlay", {}) or {}
                dedup_iou = float(debug_cfg.get("overlay_dedup_iou", 0.45))
                draw_predicted_tracks = bool(debug_cfg.get("draw_predicted_tracks", False))
                fallback_suppress_center_factor = float(overlay_cfg.get("fallback_suppress_center_factor", 0.90))
                fallback_suppress_min_center_px = float(overlay_cfg.get("fallback_suppress_min_center_px", 40))
                fallback_suppress_iou = float(overlay_cfg.get("fallback_suppress_iou", 0.20))
            except Exception:
                dedup_iou = 0.45
                draw_predicted_tracks = False
                fallback_suppress_center_factor = 0.90
                fallback_suppress_min_center_px = 40.0
                fallback_suppress_iou = 0.20
            self._renderer = OverlayRenderer(
                dedup_iou=dedup_iou,
                draw_predicted_tracks=draw_predicted_tracks,
                fallback_suppress_center_factor=fallback_suppress_center_factor,
                fallback_suppress_min_center_px=fallback_suppress_min_center_px,
                fallback_suppress_iou=fallback_suppress_iou,
            )
            try:
                import torch
                self._torch_cuda_available = bool(torch.cuda.is_available())
            except Exception:
                self._torch_cuda_available = False
                
            self._ensure_camera_manager()
            return True
            
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"[PIPELINE_INIT_FAILED] error={e!s} traceback={tb}"
            _sk_log.error(err_msg)
            if self._logger:
                self._logger.error(err_msg)
            else:
                print(err_msg, flush=True)

            if self._run_logger:
                self._run_logger.log_error("PIPELINE_INIT_FAILED", e)
                
            self._pipeline = None
            self._renderer = None
            self._local_bbox_tracker = None
            return False

    def _warn_with_cooldown(self, event_name: str, camera_id: str, message: str, cooldown_sec: float = 5.0, **fields):
        now = time.time()
        key = (event_name, camera_id)
        last_ts = self._warn_cooldowns.get(key, 0.0)
        if now - last_ts < cooldown_sec:
            return
        self._warn_cooldowns[key] = now
        if self._run_logger is not None:
            self._run_logger.log_event(event_name, message, level="WARNING", camera_id=camera_id, **fields)

    def _normalize_selected_cameras(self, camera_ids: list[str]) -> list[str]:
        if self._cfg is None:
            return list(camera_ids or [])[:self._max_active_cameras]
        allowed = [c.get("id") for c in self._cfg.get_enabled_cameras() if c.get("id")]
        requested = list(camera_ids or [])
        filtered = [cid for cid in requested if cid in set(allowed)]
        if not filtered:
            filtered = allowed[:]
        if len(filtered) > self._max_active_cameras:
            if self._logger:
                self._logger.warning(f"Maksimum {self._max_active_cameras} aktif kamera destekleniyor.")
            if self._run_logger is not None:
                self._run_logger.log_event(
                    "CONFIG_CAMERA_TRUNCATED",
                    "Configured enabled cameras exceed max_active_cameras; truncating to first 2.",
                    level="WARNING",
                    requested_cameras=filtered,
                    max_active_cameras=self._max_active_cameras,
                )
            filtered = filtered[:self._max_active_cameras]
        return filtered

    def _sync_camera_sources(self):
        """Seçili kameraları CameraManager'a senkronize eder."""
        self._ensure_camera_manager()
        source_map = self._camera_source_map()
        max_cap = int(self._max_active_cameras)
        normalized = self._normalize_selected_cameras(self._selected_cameras)
        selected_list = list(normalized)[:max_cap]
        selected = set(selected_list)
        self._selected_cameras = selected_list
        added_stream_ids: list[str] = []
        removed_stream_ids: list[str] = []

        # Artık seçili olmayan kameraları kaldır + pipeline state'ini temizle
        for cam_id in list(self._cam_manager.streams.keys()):
            if cam_id not in selected:
                self._cam_manager.remove_source(cam_id)
                removed_stream_ids.append(cam_id)
                self._last_cam_frames.pop(cam_id, None)
                # Pipeline'daki kamera-spesifik state'i temizle
                if self._pipeline is not None:
                    try:
                        self._pipeline.reset_camera_state(cam_id)
                    except Exception:
                        pass
                if self._local_bbox_tracker is not None:
                    try:
                        self._local_bbox_tracker.reset_camera(cam_id)
                    except Exception:
                        pass

        # Seçili kameraları source_map ile senkronize et
        for cam_id in selected_list:
            src = source_map.get(cam_id)
            if src is None or src == "":
                continue

            if cam_id in self._cam_manager.streams:
                stream = self._cam_manager.streams[cam_id]

                # Kaynak değiştiyse stream'i yeniden oluştur
                if stream.source != src:
                    self._cam_manager.remove_source(cam_id)
                    self._cam_manager.add_source(cam_id, src, f"Kamera {cam_id}")
                    added_stream_ids.append(cam_id)
                else:
                    # Stream var ama çalışmıyorsa başlat
                    if not stream.is_running:
                        self._cam_manager.start_camera(cam_id)
            else:
                self._cam_manager.add_source(cam_id, src, f"Kamera {cam_id}")
                added_stream_ids.append(cam_id)

        # Round-robin index'i sıfırla — kamera seti değiştiğinde bias önle
        self._inference_round_robin_idx = 0
        active_stream_ids = sorted(list(self._cam_manager.streams.keys()))[:max_cap]
        if self._run_logger is not None:
            self._run_logger.log_event(
                "CAMERA_SYNC",
                "Camera sources synchronized to selection",
                selected=selected_list,
                active_streams=active_stream_ids,
                max_active_cameras=max_cap,
                removed_stream_ids=sorted(removed_stream_ids),
                added_stream_ids=sorted(added_stream_ids),
            )

    def _start_inference_worker(self):
        if self._inference_thread is not None and self._inference_thread.is_alive():
            if self._run_logger is not None:
                self._run_logger.log_event(
                    "INFERENCE_WORKER_ALREADY_ALIVE",
                    "Inference worker start requested but thread already alive",
                    level="WARNING",
                    thread_name=self._inference_thread.name,
                )
            return
        if self._inference_running:
            return
        if self._run_logger is not None:
            self._run_logger.log_event(
                "INFERENCE_WORKER_START",
                "Starting inference worker",
                selected_cameras=list(self._selected_cameras),
                torch_cuda_available=self._torch_cuda_available,
                inference_fps_total=round(1.0 / self._get_inference_sleep(), 3),
            )
        self._inference_running = True
        self._inference_loop_count = 0
        self._inference_last_heartbeat_ts = time.time()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="InferenceWorker")
        self._inference_thread.start()
        if self._run_logger is not None:
            self._run_logger.log_event(
                "INFERENCE_WORKER_STARTED",
                "Inference worker started",
                thread_name=self._inference_thread.name,
                thread_alive=bool(self._inference_thread.is_alive()),
            )

    def _stop_inference_worker(self):
        if self._run_logger is not None:
            self._run_logger.log_event(
                "INFERENCE_WORKER_STOP_REQUEST",
                "Stopping inference worker",
                thread_alive=bool(self._inference_thread.is_alive()) if self._inference_thread is not None else False,
            )
        if not self._inference_running and self._inference_thread is None:
            return
        self._inference_running = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=1.5)
            if self._inference_thread.is_alive():
                if self._run_logger is not None:
                    self._run_logger.log_event(
                        "INFERENCE_WORKER_JOIN_TIMEOUT",
                        "Inference worker did not stop within join timeout",
                        level="WARNING",
                        thread_name=self._inference_thread.name,
                    )
            else:
                self._inference_thread = None

    def _restart_inference_worker(self, reason: str):
        if self._inference_restarting:
            return
        self._inference_restarting = True
        try:
            self._inference_restart_count += 1
            if self._run_logger is not None:
                self._run_logger.log_event(
                    "INFERENCE_WORKER_RESTART",
                    "Restarting inference worker",
                    level="WARNING",
                    reason=reason,
                    restart_count=self._inference_restart_count,
                )
            self._stop_inference_worker()
            time.sleep(0.05)
            self._start_inference_worker()
        finally:
            self._inference_restarting = False

    def _check_inference_watchdog(self):
        if not self._selected_cameras:
            return
        if self._inference_restarting:
            return
        if not self._inference_running:
            self._restart_inference_worker("not_running")
            return
        if self._inference_thread is None or not self._inference_thread.is_alive():
            if self._run_logger is not None:
                self._run_logger.log_event(
                    "INFERENCE_WORKER_DEAD",
                    "Inference worker thread is dead",
                    level="WARNING",
                )
            self._restart_inference_worker("dead_thread")
            return
        age = time.time() - self._inference_last_heartbeat_ts
        if age > 3.0:
            if self._run_logger is not None:
                self._run_logger.log_event(
                    "INFERENCE_HEARTBEAT_STALE",
                    "Inference heartbeat stale",
                    level="WARNING",
                    heartbeat_age_sec=round(age, 3),
                )
            self._restart_inference_worker("heartbeat_stale")

    def _get_inference_sleep(self) -> float:
        perf = self._cfg.performance if self._cfg else {}
        fps_key = "inference_fps_total_gpu" if self._torch_cuda_available else "inference_fps_total_cpu"
        fps_total = float(perf.get(fps_key, perf.get("inference_fps_total", 2)))
        fps_total = max(0.5, fps_total)
        return 1.0 / fps_total

    def _prepare_inference_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        perf = self._cfg.performance if self._cfg else {}
        cam_count = len(self._selected_cameras)
        hw = "gpu" if self._torch_cuda_available else "cpu"
        if cam_count <= 1:
            max_proc_w = int(perf.get(f"max_proc_width_single_{hw}", perf.get("max_proc_width_single", 640)))
        elif cam_count == 2:
            max_proc_w = int(perf.get(f"max_proc_width_2cam_{hw}", perf.get("max_proc_width_2cam", 512)))
        else:
            max_proc_w = int(perf.get(f"max_proc_width_4cam_{hw}", perf.get("max_proc_width_4cam", 416)))
        if w > max_proc_w:
            scale = max_proc_w / float(w)
            proc_h = int(h * scale)
            proc = cv2.resize(frame, (max_proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            return proc, {"scaled": True, "scale_x": w / float(max_proc_w), "scale_y": h / float(proc_h), "proc_w": max_proc_w}
        return frame, {"scaled": False, "scale_x": 1.0, "scale_y": 1.0, "proc_w": w}

    def _get_scheduler_params(self):
        perf = self._cfg.performance if self._cfg else {}
        cam_count = len(self._selected_cameras)
        tier = "1cam" if cam_count <= 1 else ("2cam" if cam_count == 2 else "4cam")
        hw = "gpu" if self._torch_cuda_available else "cpu"
        detect_every_n = int(perf.get(f"detect_every_n_{tier}_{hw}", 1))
        db_check_interval = int(perf.get(f"db_check_interval_{tier}_{hw}", 8))
        return max(1, detect_every_n), max(1, db_check_interval), hw

    def _scale_decisions_to_original(self, decisions: list, scale_info: dict):
        if not scale_info.get("scaled"):
            return decisions
        sx = float(scale_info["scale_x"])
        sy = float(scale_info["scale_y"])
        scaled = []
        for d in decisions:
            nd = copy.copy(d)
            if hasattr(nd, "bbox") and nd.bbox is not None:
                x1, y1, x2, y2 = nd.bbox
                nd.bbox = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
            scaled.append(nd)
        return scaled

    def _inference_loop(self):
        thread_name = threading.current_thread().name
        if self._run_logger is not None:
            self._run_logger.log_event(
                "INFERENCE_LOOP_ENTER",
                "Inference loop entered",
                thread_name=thread_name,
            )
        while self._inference_running:
            loop_t0 = time.time()
            try:
                self._inference_last_heartbeat_ts = loop_t0
                self._inference_loop_count += 1
                self._inference_loop_once(loop_t0)
            except Exception as e:
                self._inference_error_count += 1
                self._inference_consecutive_errors += 1
                if self._run_logger is not None:
                    self._run_logger.log_error(
                        "inference_loop_unhandled_exception",
                        e,
                        loop_count=self._inference_loop_count,
                        selected_cameras=list(self._selected_cameras),
                        thread_name=thread_name,
                    )
                time.sleep(0.1)
                if self._inference_consecutive_errors >= 5:
                    if self._run_logger is not None:
                        self._run_logger.log_event(
                            "INFERENCE_WORKER_TOO_MANY_ERRORS",
                            "Inference worker hit too many consecutive errors",
                            level="WARNING",
                            consecutive_errors=self._inference_consecutive_errors,
                        )
                    self._inference_running = False
                    break
            
            # Rate limiting
            elapsed = time.time() - loop_t0
            sleep_time = max(0.001, self._get_inference_sleep() - elapsed)
            time.sleep(sleep_time)

        if self._run_logger is not None:
            self._run_logger.log_event(
                "INFERENCE_LOOP_EXIT",
                "Inference loop exited",
                thread_name=thread_name,
                loop_count=self._inference_loop_count,
                error_count=self._inference_error_count,
            )

    def _inference_loop_once(self, loop_t0: float):
        selected = list(self._selected_cameras)
        if not selected:
            self._warn_with_cooldown("INFERENCE_SKIP_NO_SELECTED", "global", "Inference skipped: no selected cameras")
            time.sleep(0.02)
            return
        if self._cam_manager is None:
            self._warn_with_cooldown("INFERENCE_SKIP_NO_CAM_MANAGER", "global", "Inference skipped: camera manager is None")
            time.sleep(0.02)
            return
        if self._pipeline is None:
            self._warn_with_cooldown("INFERENCE_SKIP_NO_PIPELINE", "global", "Inference skipped: pipeline is None")
            time.sleep(0.02)
            return

        if self._inference_loop_count < 5:
            print(f"[INFERENCE_LOOP_ONCE] loop={self._inference_loop_count} selected={selected}")

        cam_id = selected[self._inference_round_robin_idx % len(selected)]
        self._inference_round_robin_idx += 1

        if self._inference_loop_count < 5:
            print(f"[INFERENCE_CAMERA_PICK] camera_id={cam_id}")

        stream = self._cam_manager.streams.get(cam_id)
        if stream is None:
            self._warn_with_cooldown("INFERENCE_SKIP_NO_STREAM", cam_id, "Inference skipped: no stream")
            time.sleep(0.01)
            return
        frame = stream.get_frame()
        if frame is None:
            self._warn_with_cooldown("INFERENCE_SKIP_NO_FRAME", cam_id, "Inference skipped: stream returned no frame")
            time.sleep(0.005)
            return

        if self._inference_loop_count < 5:
            print(f"[INFERENCE_FRAME_OK] camera_id={cam_id} shape={frame.shape}")

        proc_frame, scale_info = self._prepare_inference_frame(frame)
        process_t0 = time.time()
        decisions = []
        detect_every_n, db_check_interval, hw = self._get_scheduler_params()
        self._pipeline._detect_every_n = detect_every_n
        self._pipeline._db_check_interval = db_check_interval
        try:
            self._pipeline.begin_cycle()
            decisions = self._pipeline.process_frame(cam_id, proc_frame)
        except Exception as e:
            self._inference_error_count += 1
            self._inference_consecutive_errors += 1
            if self._run_logger is not None:
                self._run_logger.log_error("INFERENCE_PROCESS_FAILED", e, camera_id=cam_id)
            decisions = []
        finally:
            self._pipeline.end_cycle()
        process_ms = (time.time() - process_t0) * 1000.0

        try:
            decisions = self._scale_decisions_to_original(decisions, scale_info)
        except Exception as e:
            self._warn_with_cooldown("INFERENCE_SCALE_FAILED", cam_id, "Inference scale decision conversion failed")
            if self._run_logger is not None:
                self._run_logger.log_error("INFERENCE_SCALE_FAILED", e, camera_id=cam_id)
            decisions = []
        if self._local_bbox_tracker is not None:
            try:
                self._local_bbox_tracker.update_from_detections(cam_id, frame, decisions)
                self._last_local_tracker_metrics[cam_id] = self._local_bbox_tracker.get_camera_metrics(cam_id)
            except Exception as e:
                self._warn_with_cooldown("LOCAL_TRACKER_UPDATE_FAILED", cam_id, "Local tracker update failed")
                if self._run_logger is not None:
                    self._run_logger.log_error("LOCAL_TRACKER_UPDATE_FAILED", e, camera_id=cam_id)

        criminal_names = {}
        for d in decisions:
            if d.criminal_id is None:
                continue
            if d.criminal_id not in criminal_names:
                info = self._pipeline.db.get_criminal_info(d.criminal_id)
                criminal_names[d.criminal_id] = (info or {}).get("name", "")
            if d.status in ("WANTED", "CRIMINAL", "HEDEF BULUNDU", "TARGET_FOUND"):
                # TARGET_FOUND / HEDEF BULUNDU / WANTED / CRIMINAL → dashboard alert
                key = (cam_id, d.track_id, d.status)
                if key not in self._alerted_tracks:
                    self._alerted_tracks.add(key)
                    self._pending_alerts.append((d.track_id, d.status, criminal_names[d.criminal_id]))
        with self._inference_lock:
            self._last_decisions[cam_id] = decisions
            self._last_criminal_names[cam_id] = criminal_names
            self._last_inference_ts[cam_id] = time.time()
            self._last_process_ms[cam_id] = process_ms
            self._last_pipeline_profiles[cam_id] = self._pipeline.last_profile.get(cam_id, {})

        self._inference_last_success_ts = time.time()
        self._inference_consecutive_errors = 0
        inference_fps_total_target = round(1.0 / self._get_inference_sleep(), 3)
        profile = dict(self._last_pipeline_profiles.get(cam_id, {}) or {})
        local_metrics = dict(self._last_local_tracker_metrics.get(cam_id, {}) or {})
        inference_last_success_age_sec = max(0.0, time.time() - self._inference_last_success_ts)
        inference_last_heartbeat_age_sec = max(0.0, time.time() - self._inference_last_heartbeat_ts)
        if self._run_logger is not None:
            try:
                self._run_logger.log_performance(
                    inference_camera_id=cam_id,
                    inference_process_ms=round(process_ms, 3),
                    inference_fps_total_target=inference_fps_total_target,
                    torch_cuda_available=self._torch_cuda_available,
                    selected_hardware_profile=hw,
                    inference_round_robin_idx=self._inference_round_robin_idx,
                    scheduler_detect_every_n=detect_every_n,
                    scheduler_db_check_interval=db_check_interval,
                    selected_camera_count=len(selected),
                    estimated_camera_inference_period_sec=round(len(selected) / max(0.001, (1.0 / self._get_inference_sleep())), 3),
                    estimated_face_detection_period_sec=round((len(selected) / max(0.001, (1.0 / self._get_inference_sleep()))) * detect_every_n, 3),
                    proc_width=int(scale_info.get("proc_w", frame.shape[1])),
                    decisions_count=len(decisions),
                    faces_count=profile.get("faces_count", 0),
                    tracks_count=profile.get("tracks_count", 0),
                    pipeline_total_process_ms=profile.get("total_process_ms", 0.0),
                    face_detection_ms=profile.get("face_detection_ms", 0.0),
                    tracker_ms=profile.get("tracker_ms", 0.0),
                    db_search_ms=profile.get("db_search_ms", 0.0),
                    movement_ms=profile.get("movement_ms", 0.0),
                    decision_ms=profile.get("decision_ms", 0.0),
                    inference_loop_count=self._inference_loop_count,
                    inference_thread_alive=bool(self._inference_thread.is_alive()) if self._inference_thread is not None else False,
                    inference_restart_count=self._inference_restart_count,
                    inference_consecutive_errors=self._inference_consecutive_errors,
                    inference_last_success_age_sec=round(inference_last_success_age_sec, 3),
                    inference_last_heartbeat_age_sec=round(inference_last_heartbeat_age_sec, 3),
                    inference_error_count=self._inference_error_count,
                    active_mode=self._active_mode or (self._pipeline.current_mode if self._pipeline else "UNKNOWN"),
                    target_person_id=(
                        (
                            _options_person_search_ids(self._active_mode_options)[0]
                            if _options_person_search_ids(self._active_mode_options)
                            else self._active_mode_options.get("target_person_id")
                        )
                        if (self._active_mode == "PERSON_SEARCH" and self._active_mode_options)
                        else None
                    ),
                    target_person_ids=(
                        _options_person_search_ids(self._active_mode_options)
                        if (self._active_mode == "PERSON_SEARCH" and self._active_mode_options)
                        else None
                    ),
                    total_faces_scanned=int(self._pipeline.stats.get("total_faces_scanned", 0)),
                    total_matches=int(self._pipeline.stats.get("total_matches", 0)),
                )
                profile_payload = dict(profile)
                profile_payload.pop("camera_id", None)
                profile_payload.update({
                    "inference_process_ms": round(process_ms, 3),
                    "selected_hardware_profile": hw,
                    "scheduler_detect_every_n": detect_every_n,
                    "scheduler_db_check_interval": db_check_interval,
                    "selected_camera_count": len(selected),
                    "proc_width": int(scale_info.get("proc_w", frame.shape[1])),
                    "torch_cuda_available": self._torch_cuda_available,
                    "inference_fps_total_target": inference_fps_total_target,
                    "inference_loop_count": self._inference_loop_count,
                    "inference_restart_count": self._inference_restart_count,
                    "inference_consecutive_errors": self._inference_consecutive_errors,
                })
                profile_payload.update(local_metrics)
                self._run_logger.log_pipeline_profile(cam_id, **profile_payload)
            except Exception as e:
                self._warn_with_cooldown(
                    "INFERENCE_LOGGING_FAILED",
                    cam_id,
                    "Inference profile logging failed",
                    cooldown_sec=5.0,
                    exception=repr(e),
                )
        elapsed = time.time() - loop_t0
        time.sleep(max(0.0, self._get_inference_sleep() - elapsed))

    def _display_tick(self):
        now = time.time()
        t0 = now
        if not self._selected_cameras or self._cam_manager is None:
            self.pg_dash.update_stats(0.0, 0, 0, 0)
            return
        frames = []
        cam_ids = []
        active_total = 0
        active_streams = self._cam_manager.get_active_streams()
        selected_set = set(self._selected_cameras)
        stream_effective_fps_values: list[float] = []
        max_last_frame_age_ms = 0.0
        total_new_display_frames = 0
        total_stale_display_frames = 0
        ttl_sec = float(self._cfg.performance.get("decision_ttl_sec", 2.0)) if self._cfg else 2.0
        camera_log_interval = float(self._cfg.logging.get("camera_log_interval_sec", 2.0)) if self._cfg else 2.0
        should_log_camera_status = (
            self._run_logger is not None
            and (now - self._last_camera_log_ts) >= camera_log_interval
        )
        for cam_id in self._selected_cameras:
            stream = active_streams.get(cam_id)
            frame_info = stream.get_frame_info() if stream is not None and hasattr(stream, "get_frame_info") else {}
            frame = stream.get_frame() if stream is not None else None
            latest_raw = self._last_raw_frames.get(cam_id)
            if frame is None:
                frame = latest_raw
            seq = int(frame_info.get("frame_seq", -1))
            prev_seq = self._last_seen_frame_seq.get(cam_id, -1)
            is_new_frame = seq >= 0 and prev_seq != seq
            if is_new_frame:
                self._new_display_frame_counts[cam_id] = self._new_display_frame_counts.get(cam_id, 0) + 1
                self._last_seen_frame_seq[cam_id] = seq
                total_new_display_frames += 1
            else:
                self._stale_display_frame_counts[cam_id] = self._stale_display_frame_counts.get(cam_id, 0) + 1
                total_stale_display_frames += 1
            stream_effective_fps = float(frame_info.get("effective_fps", 0.0))
            stream_effective_fps_values.append(stream_effective_fps)
            last_frame_age_ms = float(frame_info.get("last_frame_age_ms", 1e9))
            max_last_frame_age_ms = max(max_last_frame_age_ms, last_frame_age_ms)
            with self._inference_lock:
                decisions = list(self._last_decisions.get(cam_id, []))
                criminal_names = dict(self._last_criminal_names.get(cam_id, {}))
                last_ts = self._last_inference_ts.get(cam_id, 0.0)
                last_process_ms = self._last_process_ms.get(cam_id, 0.0)
            age = time.time() - last_ts if last_ts else 1e9
            displayed_with_cached_decisions = bool(decisions and age <= ttl_sec)
            if frame is None:
                draw_frame = self._no_signal_frame(cam_id)
                self._no_signal_since.setdefault(cam_id, time.time())
            else:
                self._last_raw_frames[cam_id] = frame
                draw_frame = frame
                self._no_signal_since.pop(cam_id, None)
                if displayed_with_cached_decisions and self._renderer is not None and self._pipeline is not None:
                    original_decisions_count = len(decisions)
                    render_decisions = decisions
                    local_tracker_display_failed = False
                    if self._local_bbox_tracker is not None:
                        try:
                            predicted_decisions = self._local_bbox_tracker.predict_on_frame(cam_id, frame, decisions)
                            if predicted_decisions is not None:
                                render_decisions = predicted_decisions
                            self._last_local_tracker_metrics[cam_id] = self._local_bbox_tracker.get_camera_metrics(cam_id)
                        except Exception as e:
                            # FAIL-OPEN: local tracker patlarsa orijinal decisions ile çizime devam et
                            render_decisions = decisions
                            local_tracker_display_failed = True
                            self._last_local_tracker_metrics[cam_id] = {
                                "local_tracker_enabled": True,
                                "local_tracker_failed_count": 1,
                                "local_tracker_display_failed": True,
                                "local_tracker_input_decisions": original_decisions_count,
                                "local_tracker_output_decisions": len(render_decisions),
                                "renderer_input_decisions": len(render_decisions),
                            }
                            if self._run_logger is not None:
                                self._run_logger.log_error("LOCAL_TRACKER_DISPLAY_FAILED", e, camera_id=cam_id)
                    try:
                        draw_frame = frame.copy()
                        draw_frame = self._renderer.draw(draw_frame, render_decisions, self._pipeline.stats, criminal_names)
                        active_total += len(render_decisions)
                    except Exception as e:
                        if self._run_logger is not None:
                            self._run_logger.log_error("display_renderer_failed", e, camera_id=cam_id)
                    decisions = render_decisions
                    if self._run_logger is not None and local_tracker_display_failed:
                        self._run_logger.log_event(
                            "LOCAL_TRACKER_FAIL_OPEN_RENDER",
                            "Local tracker failed, rendered with original decisions",
                            level="WARNING",
                            camera_id=cam_id,
                            local_tracker_input_decisions=original_decisions_count,
                            renderer_input_decisions=len(render_decisions),
                        )
            frames.append(draw_frame)
            cam_ids.append(cam_id)
            if should_log_camera_status and self._run_logger is not None:
                local_metrics = dict(self._last_local_tracker_metrics.get(cam_id, {}) or {})
                self._run_logger.log_camera_status(
                    cam_id,
                    has_signal=frame is not None,
                    stream_running=bool(stream.is_running) if stream is not None else False,
                    stream_fps=round(getattr(stream, "fps", 0.0), 3) if stream else 0.0,
                    frame_shape=list(frame.shape) if frame is not None else None,
                    runtime_frame_shape=frame_info.get("runtime_frame_shape", list(frame.shape) if frame is not None else None),
                    source_downscaled=bool(frame is not None and stream is not None and getattr(stream, "max_frame_width", None) and frame.shape[1] <= int(getattr(stream, "max_frame_width"))),
                    stream_source=getattr(stream, "source", None) if stream is not None else None,
                    selected=cam_id in selected_set,
                    active_stream_count=len(self._cam_manager.streams) if self._cam_manager is not None else 0,
                    frame_seq=seq,
                    is_new_frame=is_new_frame,
                    stream_effective_fps=round(stream_effective_fps, 3),
                    stream_target_fps=round(float(frame_info.get("stream_target_fps", 0.0)), 3),
                    last_frame_age_ms=round(last_frame_age_ms, 3),
                    last_read_ms=round(float(frame_info.get("last_read_ms", 0.0)), 3),
                    last_resize_ms=round(float(frame_info.get("last_resize_ms", 0.0)), 3),
                    original_frame_shape=frame_info.get("original_frame_shape", None),
                    new_display_frame_count=self._new_display_frame_counts.get(cam_id, 0),
                    stale_display_frame_count=self._stale_display_frame_counts.get(cam_id, 0),
                    last_inference_age_sec=round(age, 3),
                    inference_thread_alive=bool(self._inference_thread.is_alive()) if self._inference_thread is not None else False,
                    inference_restart_count=self._inference_restart_count,
                    last_process_ms=round(float(last_process_ms), 3),
                    decision_count=len(decisions),
                    displayed_with_cached_decisions=displayed_with_cached_decisions if frame is not None else False,
                    local_tracker_enabled=local_metrics.get("local_tracker_enabled", False),
                    local_tracker_display_failed=local_metrics.get("local_tracker_display_failed", False),
                    local_tracker_input_decisions=local_metrics.get("local_tracker_input_decisions", len(decisions)),
                    local_tracker_output_decisions=local_metrics.get("local_tracker_output_decisions", len(decisions)),
                    local_tracker_updated_count=local_metrics.get("local_tracker_updated_count", 0),
                    local_tracker_failed_count=local_metrics.get("local_tracker_failed_count", 0),
                    renderer_input_decisions=local_metrics.get("renderer_input_decisions", len(decisions)),
                )
        if should_log_camera_status:
            self._last_camera_log_ts = now
        composed = self._compose_grid(frames, cam_ids)
        now = time.time()
        dt = max(1e-6, now - self._display_last_tick)
        self._display_last_tick = now
        inst_fps = 1.0 / dt
        self._display_fps_smooth = inst_fps if self._display_fps_smooth == 0.0 else (0.9 * self._display_fps_smooth + 0.1 * inst_fps)
        self._display_frame_count += 1
        total_faces = int(self._pipeline.stats.get("total_faces_scanned", 0)) if self._pipeline else 0
        total_alerts = len(self._alerted_tracks)
        self.pg_dash.update_frame(composed)
        self.pg_dash.update_stats(self._display_fps_smooth, active_total, total_faces, total_alerts)
        for tid, status, name in list(self._pending_alerts):
            self.pg_dash.add_alert(tid, status, name)
        self._pending_alerts.clear()
        display_loop_ms = (time.time() - t0) * 1000.0
        self._proc_ewma_ms = 0.9 * self._proc_ewma_ms + 0.1 * display_loop_ms
        if self._proc_ewma_ms > 80:
            self._perf_level = 2
        elif self._proc_ewma_ms > 50:
            self._perf_level = 1
        elif self._proc_ewma_ms < 30:
            self._perf_level = 0
        perf_interval = float(self._cfg.logging.get("perf_log_interval_sec", 1.0)) if self._cfg else 1.0
        if self._run_logger is not None and (now - self._last_perf_log_ts) >= perf_interval:
            self._run_logger.log_performance(
                display_loop_ms=round(display_loop_ms, 3),
                display_fps=round(self._display_fps_smooth, 3),
                display_target_fps=self._display_target_fps,
                display_timer_interval_ms=getattr(self, "_display_timer_interval_ms", int(1000 / max(1, self._display_target_fps))),
                selected_cameras=self._selected_cameras,
                active_stream_ids=sorted(list(active_streams.keys())),
                avg_stream_effective_fps=round(sum(stream_effective_fps_values) / max(1, len(stream_effective_fps_values)), 3),
                min_stream_effective_fps=round(min(stream_effective_fps_values), 3) if stream_effective_fps_values else 0.0,
                max_last_frame_age_ms=round(max_last_frame_age_ms, 3),
                total_new_display_frames=total_new_display_frames,
                total_stale_display_frames=total_stale_display_frames,
                grid_rows=self._grid_rows,
                grid_cols=self._grid_cols,
                display_canvas_width=int((self._cfg.performance or {}).get("display_canvas_width", 1280)) if self._cfg else 1280,
                display_canvas_height=int((self._cfg.performance or {}).get("display_canvas_height", 720)) if self._cfg else 720,
                display_frame_count=self._display_frame_count,
                cam_count=len(self._selected_cameras),
                total_faces_scanned=int(self._pipeline.stats.get("total_faces_scanned", total_faces)) if self._pipeline else total_faces,
                total_matches=int(self._pipeline.stats.get("total_matches", 0)) if self._pipeline else 0,
                total_alerts=total_alerts,
                perf_level=self._perf_level,
                proc_ewma_ms=round(self._proc_ewma_ms, 3),
            )
            self._last_perf_log_ts = now
            
        # ─── PERSON SEARCH 1-SECOND SUMMARY LOG ───
        if self._pipeline and self._pipeline.current_mode == "PERSON_SEARCH":
            if not hasattr(self, "_ps_last_summary_ts"):
                self._ps_last_summary_ts = 0.0
                
            if now - self._ps_last_summary_ts >= 1.0:
                self._ps_last_summary_ts = now
                if self._logger:
                    stats = self._pipeline.stats
                    self._logger.person_search_trace(
                        "SYSTEM_SUMMARY_1S",
                        mode="PERSON_SEARCH",
                        target_ids=list(getattr(self._pipeline, "target_person_ids", []) or []),
                        target_id=getattr(self._pipeline, "target_person_id", None),
                        has_embedding=bool(
                            getattr(self._pipeline, "_person_search_embeddings", None)
                        ),
                        cameras=str(self._selected_cameras),
                        display_fps=round(self._display_fps_smooth, 1),
                        active_tracks=active_total,
                        reid_hits=stats.get("reid_hits", 0),
                        total_faces=total_faces,
                        total_alerts=total_alerts,
                    )

    def _compose_grid(self, frames: list, cam_ids: list[str]):
        if len(frames) == 1:
            one = frames[0]
            perf = self._cfg.performance if self._cfg else {}
            canvas_w = int(perf.get("display_single_max_width", perf.get("display_canvas_width", 1280)))
            canvas_h = int(perf.get("display_canvas_height", 720))
            one = self._fit_frame_to_tile(one, canvas_w, canvas_h, bg_color=(0, 0, 0))
            self._draw_cam_label(one, cam_ids[0] if cam_ids else "CAM")
            return one

        n = len(frames)
        rows = max(1, self._grid_rows)
        cols = max(1, self._grid_cols)
        perf = self._cfg.performance if self._cfg else {}
        canvas_w = int(perf.get("display_canvas_width", 1280))
        canvas_h = int(perf.get("display_canvas_height", 720))

        tile_w = max(180, canvas_w // cols)
        tile_h = max(120, canvas_h // rows)
        resized = []
        for i, f in enumerate(frames):
            tile = self._fit_frame_to_tile(f, tile_w, tile_h, bg_color=(0, 0, 0))
            self._draw_cam_label(tile, cam_ids[i] if i < len(cam_ids) else f"CAM_{i+1}")
            resized.append(tile)

        total = rows * cols
        if resized:
            blank = np.zeros_like(resized[0])
            while len(resized) < total:
                resized.append(blank)

        row_imgs = []
        for r in range(rows):
            row = resized[r * cols:(r + 1) * cols]
            row_imgs.append(cv2.hconcat(row))
        grid = cv2.vconcat(row_imgs)
        return grid  # zaten doğru boyutta, tekrar resize gereksiz

    def _fit_frame_to_tile(self, frame, target_w, target_h, bg_color=(0, 0, 0)):
        if frame is None:
            return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        if h == target_h and w == target_w:
            return frame

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _draw_cam_label(self, frame, cam_id: str):
        label = f"Kamera: {cam_id}"
        cv2.rectangle(frame, (12, 12), (260, 54), (0, 0, 0), -1)
        cv2.rectangle(frame, (12, 12), (260, 54), (0, 140, 255), 2)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def _no_signal_frame(self, cam_id: str, w: int = 640, h: int = 360):
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(blank, f"{cam_id} - NO SIGNAL", (35, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return blank

    def _on_person_added(self, name: str):
        self.pg_list.refresh()
        self.pg_mode.refresh_persons()
        self._update_db_count()

    def _update_db_count(self):
        try:
            import sqlite3
            db = PROJECT_ROOT / "database" / "skywatch.db"
            if db.exists():
                conn  = sqlite3.connect(str(db), check_same_thread=False)
                count = conn.execute("SELECT COUNT(*) FROM criminals").fetchone()[0]
                conn.close()
                self.sidebar.set_db_count(count)
        except Exception:
            pass

    # ── Pipeline Entegrasyon API ─────────────────────────────────────────────
    def push_frame(self, bgr_frame):
        self.pg_dash.update_frame(bgr_frame)

    def push_stats(self, fps, active, total, alerts):
        self.pg_dash.update_stats(fps, active, total, alerts)

    def push_alert(self, tid, status, name=""):
        self.pg_dash.add_alert(tid, status, name)

    def closeEvent(self, event):
        try:
            self._stop_inference_worker()
            if self._cam_manager is not None:
                self._cam_manager.stop_all()
            if self._run_logger is not None:
                self._run_logger.log_system("gui_window_closed")
                self._run_logger.close()
        finally:
            super().closeEvent(event)


def launch_gui():
    app = QApplication(sys.argv)
    app.setApplicationName("SKYWATCH")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
