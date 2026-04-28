"""SKYWATCH — Ana Pencere (Sidebar Mimarisi)"""

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QApplication
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
from engine.pipeline import Pipeline
from engine.renderer import OverlayRenderer
from engine.camera_manager import CameraManager, GRID_FPS, DETAIL_FPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SKYWATCH")
        self.setMinimumSize(1280, 760)
        self.resize(1440, 900)
        self.setStyleSheet(GLOBAL_STYLE)
        self.statusBar().hide()

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
        self.pg_add  = AddCriminalPage()
        self.pg_list = CriminalListPage()

        self.stack.addWidget(self.pg_dash)   # 0
        self.stack.addWidget(self.pg_mode)   # 1
        self.stack.addWidget(self.pg_add)    # 2
        self.stack.addWidget(self.pg_list)   # 3

        # ── Bağlantılar ─────────────────────────────────────────────────────
        self.pg_mode.system_start.connect(self._on_start)
        self.pg_mode.system_stop.connect(self._on_stop)
        self.pg_add.person_added.connect(self._on_person_added)
        self.pg_dash.camera_selection_changed.connect(self._on_dashboard_camera_selection_changed)

        # Video oynatma durumu
        self._cam_manager: CameraManager | None = None
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._ui_refresh)
        self._selected_cameras = []
        self._last_tick = 0.0
        self._fps_smooth = 0.0
        self._alerted_tracks: set[tuple[str, int, str]] = set()
        self._camera_round_robin_idx = 0
        self._last_cam_frames: dict[str, np.ndarray] = {}
        self._state_lock = threading.Lock()
        self._worker_thread = None
        self._worker_running = False
        self._latest_frame = None
        self._latest_stats = (0.0, 0, 0, 0)
        self._pending_alerts: list[tuple[int, str, str]] = []
        self._proc_ewma_ms = 35.0
        self._perf_level = 0  # 0: kalite, 1: dengeli, 2: hizli

        # Worker FPS sınırlama — grid'de 15, tek kamerada 25
        self._worker_target_fps = GRID_FPS
        self._worker_frame_interval = 1.0 / self._worker_target_fps

        # Tespit/track pipeline (GUI izleme için)
        self._cfg = None
        self._logger = None
        self._pipeline = None
        self._renderer = None

        # DB sayısını ilk göster
        self._update_db_count()
        self._sync_dashboard_camera_options()

    # ── Navigasyon ────────────────────────────────────────────────────────────
    def _goto(self, index: int):
        self.stack.setCurrentIndex(index)
        if index == 0:
            self._sync_dashboard_camera_options()
        if index == 3:
            self.pg_list.refresh()
            self._update_db_count()

    # ── Pipeline sinyalleri ──────────────────────────────────────────────────
    def _on_start(self, mode: str, cameras: list):
        self.sidebar.set_running(True)
        self.pg_dash.set_mode(mode, True)
        self.pg_dash.clear_alerts()
        selected = list(cameras or self.pg_mode.get_camera_ids(only_checked=True))
        self._start_video_mode(selected)
        self._sync_dashboard_camera_options(active_only=True)
        self._goto(0)
        self.sidebar._select(0)

    def _on_stop(self):
        self._stop_video_mode()
        self.sidebar.set_running(False)
        self.pg_dash.set_mode("", False)
        self._sync_dashboard_camera_options()

    def _sync_dashboard_camera_options(self, active_only: bool = False):
        if active_only and self._selected_cameras:
            camera_ids = list(self._selected_cameras)
        else:
            camera_ids = self.pg_mode.get_camera_ids()
        self.pg_dash.set_camera_options(camera_ids, self._selected_cameras)

    def _start_video_mode(self, cameras: list):
        self._selected_cameras = list(cameras or [])
        self._ensure_pipeline()
        self._sync_camera_sources()
        self._last_tick = time.time()
        self._fps_smooth = 0.0
        self._start_worker()
        ui_interval = 40 if len(self._selected_cameras) <= 2 else 66
        if not self._video_timer.isActive():
            self._video_timer.start(ui_interval)

    def _stop_video_mode(self):
        self._stop_worker()
        if self._video_timer.isActive():
            self._video_timer.stop()
        if self._cam_manager is not None:
            self._cam_manager.stop_all()
        self._last_cam_frames.clear()
        self.pg_dash.update_stats(0.0, 0, 0, 0)
        self.pg_dash.clear_frame()

    def _camera_source_map(self) -> dict:
        try:
            from video_sources import VIDEO_SOURCES
            return dict(VIDEO_SOURCES)
        except Exception:
            return {}

    def _ensure_camera_manager(self):
        if self._cam_manager is not None:
            return
        self._cam_manager = CameraManager(logger=self._logger)

    def _ensure_pipeline(self):
        if self._pipeline is not None and self._renderer is not None:
            return
        try:
            self._cfg = AppConfig()
            self._logger = EventLogger(self._cfg)
            self._pipeline = Pipeline(self._cfg, self._logger)
            self._renderer = OverlayRenderer()
        except Exception:
            self._pipeline = None
            self._renderer = None
        self._ensure_camera_manager()

    def _sync_camera_sources(self):
        """Seçili kameraları CameraManager'a senkronize eder."""
        self._ensure_camera_manager()
        source_map = self._camera_source_map()
        selected = set(self._selected_cameras)

        # Artık seçili olmayan kameraları kaldır
        for cam_id in list(self._cam_manager.streams.keys()):
            if cam_id not in selected:
                self._cam_manager.remove_source(cam_id)
                self._last_cam_frames.pop(cam_id, None)

        # Yeni kameraları ekle (add_source içinde start() çağrılır)
        for cam_id in self._selected_cameras:
            src = source_map.get(cam_id)
            if src and cam_id not in self._cam_manager.streams:
                self._cam_manager.add_source(cam_id, src, f"Kamera {cam_id}")

    def _on_dashboard_camera_selection_changed(self, camera_ids: list):
        self._selected_cameras = list(camera_ids or [])
        if self._selected_cameras:
            self._ensure_pipeline()
            self._sync_camera_sources()
            self._start_worker()
        else:
            if self._cam_manager is not None:
                self._cam_manager.stop_all()

        # Izleme ekraninda manuel kamera seciminde de oynatma baslasin
        if self._selected_cameras and not self._video_timer.isActive():
            self._last_tick = time.time()
            self._fps_smooth = 0.0
            self._video_timer.start(66)
            self.sidebar.set_running(True)
            self.pg_dash.set_mode("GENERAL", True)

        if not self._selected_cameras and self._video_timer.isActive():
            self._video_timer.stop()
            self._stop_worker()
            self.pg_dash.update_stats(0.0, 0, 0, 0)
            self.pg_dash.clear_frame()
            self.sidebar.set_running(False)
            self.pg_dash.set_mode("", False)

    def _start_worker(self):
        if self._worker_running:
            return
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _stop_worker(self):
        if not self._worker_running:
            return
        self._worker_running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def _worker_loop(self):
        while self._worker_running:
            t0 = time.time()
            self._tick_video()
            # FPS sınırlama: hedef frame_interval kadar süre geçmemişse bekle
            elapsed = time.time() - t0
            sleep_t = self._worker_frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # İşlem çok uzun sürdü, en azından CPU'ya nefes aldır
                time.sleep(0.001)

    def _ui_refresh(self):
        with self._state_lock:
            frame = self._latest_frame
            fps, active, total, alerts = self._latest_stats
            pending = list(self._pending_alerts)
            self._pending_alerts.clear()

        if frame is not None:
            self.pg_dash.update_frame(frame)
        self.pg_dash.update_stats(fps, active, total, alerts)
        for tid, status, name in pending:
            self.pg_dash.add_alert(tid, status, name)

    def _tick_video(self):
        t_loop0 = time.time()
        if not self._selected_cameras or self._cam_manager is None:
            with self._state_lock:
                self._latest_stats = (0.0, 0, 0, 0)
            return

        cam_count = len(self._selected_cameras)

        # ── Hedef FPS ─────────────────────────────────────────────────
        if cam_count <= 1:
            self._worker_target_fps = DETAIL_FPS
        elif cam_count <= 4:
            self._worker_target_fps = GRID_FPS
        else:
            self._worker_target_fps = 15
        self._worker_frame_interval = 1.0 / self._worker_target_fps

        # CameraManager FPS'ini güncelle
        cam_fps = DETAIL_FPS if cam_count <= 1 else GRID_FPS
        self._cam_manager.set_all_target_fps(cam_fps)

        # ── İnference bütçesi ──────────────────────────────────────────
        if cam_count >= 10:
            infer_budget = 1
        elif cam_count >= 6:
            infer_budget = 2
        elif cam_count >= 3:
            infer_budget = 3
        else:
            infer_budget = cam_count

        if self._perf_level == 1:
            infer_budget = max(1, infer_budget - 1)
        elif self._perf_level >= 2:
            infer_budget = 1

        infer_cam_ids = set()
        if cam_count > 0 and infer_budget > 0:
            start = self._camera_round_robin_idx % cam_count
            for i in range(infer_budget):
                infer_cam_ids.add(self._selected_cameras[(start + i) % cam_count])
            self._camera_round_robin_idx = (start + infer_budget) % cam_count

        # ── Pipeline ayarları ─────────────────────────────────────────
        if self._pipeline is not None:
            if cam_count >= 10:
                self._pipeline._detect_every_n = 12 if self._perf_level >= 1 else 10
            elif cam_count >= 6:
                self._pipeline._detect_every_n = 10 if self._perf_level >= 2 else 8
            else:
                self._pipeline._detect_every_n = 5 if self._perf_level >= 2 else 4

            fa = self._pipeline.face_analyzer
            if cam_count <= 2 and self._perf_level == 0:
                fa._tile_enabled = True
                fa._tile_splits = min(max(int(getattr(fa, "_tile_splits", 3)), 2), 3)
            else:
                fa._tile_enabled = False

        # ── CameraManager'dan frame al (decode thread'lerde zaten yapıldı) ─
        frames = []
        frame_cam_ids = []
        total_active = 0
        for cam_id in self._selected_cameras:
            frame = self._cam_manager.get_frame(cam_id)
            if frame is not None:
                self._last_cam_frames[cam_id] = frame
            else:
                frame = self._last_cam_frames.get(cam_id)
            if frame is None:
                continue

            draw_frame = frame
            if (self._pipeline is not None and self._renderer is not None
                    and cam_id in infer_cam_ids):
                try:
                    ph, pw = frame.shape[:2]
                    # İşleme çözünürlüğü — kamera sayısına göre düşür
                    if cam_count >= 10:
                        max_proc_w = 320
                    elif cam_count >= 6:
                        max_proc_w = 416
                    else:
                        max_proc_w = 640
                    if self._perf_level >= 2:
                        max_proc_w = min(max_proc_w, 320)
                    elif self._perf_level == 1:
                        max_proc_w = min(max_proc_w, 384)
                    if pw > max_proc_w:
                        scale = max_proc_w / float(pw)
                        proc_frame = cv2.resize(frame, (max_proc_w, int(ph * scale)),
                                                interpolation=cv2.INTER_LINEAR)
                    else:
                        proc_frame = frame

                    decisions = self._pipeline.process_frame(cam_id, proc_frame)
                    total_active += len(decisions)

                    criminal_names = {}
                    for d in decisions:
                        if d.criminal_id is None:
                            continue
                        if d.criminal_id not in criminal_names:
                            info = self._pipeline.db.get_criminal_info(d.criminal_id)
                            criminal_names[d.criminal_id] = (info or {}).get("name", "")

                        if d.status in ("WANTED", "CRIMINAL"):
                            key = (cam_id, d.track_id, d.status)
                            if key not in self._alerted_tracks:
                                self._alerted_tracks.add(key)
                                self._pending_alerts.append(
                                    (d.track_id, d.status, criminal_names[d.criminal_id]))

                    draw_frame = self._renderer.draw(
                        proc_frame, decisions, self._pipeline.stats, criminal_names)
                except Exception:
                    draw_frame = frame

            frames.append(draw_frame)
            frame_cam_ids.append(cam_id)

        if not frames:
            with self._state_lock:
                self._latest_stats = (0.0, 0, 0, 0)
            return

        now = time.time()
        dt = max(1e-6, now - self._last_tick)
        inst_fps = 1.0 / dt
        self._last_tick = now
        self._fps_smooth = (inst_fps if self._fps_smooth == 0.0
                            else 0.9 * self._fps_smooth + 0.1 * inst_fps)

        composed = self._compose_grid(frames, frame_cam_ids)
        total_metric = 0
        if self._pipeline is not None:
            total_metric = int(self._pipeline.stats.get("total_faces_scanned", 0))
        with self._state_lock:
            self._latest_frame = composed
            self._latest_stats = (self._fps_smooth, total_active, total_metric, 0)

        # Adaptif perf seviyesi güncelle
        loop_ms = (time.time() - t_loop0) * 1000.0
        self._proc_ewma_ms = 0.9 * self._proc_ewma_ms + 0.1 * loop_ms
        if self._proc_ewma_ms > 80:
            self._perf_level = 2
        elif self._proc_ewma_ms > 50:
            self._perf_level = 1
        elif self._proc_ewma_ms < 30:
            self._perf_level = 0

    def _compose_grid(self, frames: list, cam_ids: list[str]):
        if len(frames) == 1:
            one = frames[0]
            h, w = one.shape[:2]
            max_w = 960
            if w > max_w:
                scale = max_w / w
                one = cv2.resize(one, (max_w, int(h * scale)),
                                 interpolation=cv2.INTER_LINEAR)
            one = one.copy()
            self._draw_cam_label(one, cam_ids[0] if cam_ids else "CAM")
            return one

        n = len(frames)
        # Secili kamera sayisina gore dinamik bolunme
        # 2 -> 1x2, 3 -> 1x3, 4 -> 2x2, 5-6 -> 2x3, 7-9 -> 3x3 ...
        if n <= 3:
            cols, rows = n, 1
        else:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))

        # Çok kamerada canvas çözünürlüğünü düşür → daha az resize işi
        if n >= 10:
            canvas_w, canvas_h = 640, 360
        elif n >= 6:
            canvas_w, canvas_h = 1024, 576
        else:
            canvas_w, canvas_h = 1280, 720

        tile_w = max(180, canvas_w // cols)
        tile_h = max(120, canvas_h // rows)
        resized = []
        for i, f in enumerate(frames):
            # INTER_LINEAR: INTER_AREA'dan ~3x hızlı, grid tile'da fark görünmez
            tile = cv2.resize(f, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
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

    def _draw_cam_label(self, frame, cam_id: str):
        label = f"Kamera: {cam_id}"
        cv2.rectangle(frame, (12, 12), (260, 54), (0, 0, 0), -1)
        cv2.rectangle(frame, (12, 12), (260, 54), (0, 140, 255), 2)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

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
