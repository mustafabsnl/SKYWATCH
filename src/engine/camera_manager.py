"""
SKYWATCH — CameraManager
Tüm kameraların yönetimini, frame okuma işlemlerini ve
canlı yayın sürekliliğini sağlar. Sistemin Tek Frame Okuma noktası.
"""

import cv2
import time
import threading
import numpy as np

from utils.config import AppConfig
from utils.logger import EventLogger, EventType

# ── FPS Sınırlama Sabitleri ──────────────────────────────────────────────────
# Grid ekranda çoklu kamera için 15 FPS yeterli; detay ekranında 25 FPS.
GRID_FPS   = 15
DETAIL_FPS = 25
_DEFAULT_FRAME_INTERVAL = 1.0 / GRID_FPS


class CameraStream:
    """Tek bir kamera akışını bağımsız bir thread'de yönetir."""

    class _NullLogger:
        """Logger yoksa sessiz çalışmayı sağlar."""
        def log(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def debug(self, *a, **kw): pass
        def warning(self, *a, **kw): pass

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        name: str,
        logger=None,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
        target_fps: int = GRID_FPS,
        video_file_realtime: bool = True,
        video_file_loop: bool = True,
        drop_old_frames: bool = True,
    ):
        self.camera_id = camera_id
        self.source = source
        self.name = name
        self.logger = logger or self._NullLogger()
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self._video_file_realtime = bool(video_file_realtime)
        self._video_file_loop = bool(video_file_loop)
        self._drop_old_frames = bool(drop_old_frames)
        self._is_video_file = isinstance(source, str) and not str(source).lower().startswith(("rtsp://", "http://", "https://", "rtmp://"))
        
        self.cap = None
        self.is_running = False
        self.thread = None
        
        # Son alınan kareyi tut (Thread-safe okuma için)
        self.current_frame = None
        self.lock = threading.Lock()
        
        # FPS hesaplama için
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Yeniden bağlanma süresi (sn)
        self.reconnect_delay = 5.0
        
        # FPS sınırlama — hedef FPS ve frame aralığı
        self._target_fps = max(1, int(target_fps))
        self._frame_interval = 1.0 / self._target_fps

        # Frame tazelik / decode metrikleri
        self.frame_seq = 0
        self.last_frame_ts = 0.0
        self.last_read_ms = 0.0
        self.last_decode_ms = 0.0
        self.last_resize_ms = 0.0
        self.effective_fps = 0.0
        self.new_frame_count = 0
        self.start_ts = time.time()
        self.last_frame_shape = None
        self.original_frame_shape = None
        self.runtime_frame_shape = None
        self._last_eof_log_ts = 0.0
        self._high_res_warned = False
        
    def start(self):
        """Kamera akışını başlatır."""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Kamera akışını durdurur."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_target_fps(self, fps: int):
        """Hedef FPS'i çalışma sırasında değiştirir (grid ↔ detay geçişi)."""
        self._target_fps = fps
        self._frame_interval = 1.0 / fps
            
    def get_frame(self) -> np.ndarray | None:
        """En son okunan kareyi thread-safe olarak döndürür (kopya yok → performans)."""
        with self.lock:
            return self.current_frame

    def get_frame_info(self) -> dict:
        with self.lock:
            last_ts = float(self.last_frame_ts or 0.0)
            age_ms = (time.time() - last_ts) * 1000.0 if last_ts > 0 else 1e9
            return {
                "frame_seq": int(self.frame_seq),
                "last_frame_ts": last_ts,
                "last_frame_age_ms": round(age_ms, 3),
                "effective_fps": round(float(self.effective_fps), 3),
                "last_read_ms": round(float(self.last_read_ms), 3),
                "last_decode_ms": round(float(self.last_decode_ms), 3),
                "last_resize_ms": round(float(self.last_resize_ms), 3),
                "original_frame_shape": list(self.original_frame_shape) if self.original_frame_shape else None,
                "runtime_frame_shape": list(self.runtime_frame_shape) if self.runtime_frame_shape else None,
                "stream_target_fps": float(self._target_fps),
                "target_interval_ms": round(float(self._frame_interval * 1000.0), 3),
                "video_file_realtime": bool(self._video_file_realtime),
                "source_is_video_file": bool(self._is_video_file),
                "source": self.source,
                "is_running": bool(self.is_running),
            }
            
    def _connect(self) -> bool:
        """Kameraya bağlanır."""
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self.logger.warning(
                    f"{self.camera_id} source could not be opened. Showing NO SIGNAL."
                )
                return False
                
            # Performans için buffer boyutunu küçült
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # FPS sınırlama: Native FPS'i al ama TARGET_FPS'ten yüksek yapma
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self._is_video_file and src_fps and 0 < src_fps <= 120:
                # Video dosyası: native FPS ile target FPS'in küçüğünü kullan
                effective_fps = min(src_fps, self._target_fps)
            else:
                # Webcam / RTSP: doğrudan target FPS'i uygula
                effective_fps = self._target_fps
            self._frame_interval = 1.0 / effective_fps
            if not self._is_video_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            
            self.logger.log(EventType.CAMERA_ONLINE, f"{self.name} bağlandı",
                            camera_id=self.camera_id)
            return True
            
        except Exception as e:
            self.logger.error(f"{self.name} bağlantı hatası: {e}")
            return False

    def _update_loop(self):
        """Sürekli olarak frame okuyan arka plan döngüsü."""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if not self._connect():
                    self.logger.log(EventType.CAMERA_OFFLINE,
                                    f"{self.name} ulaşılamıyor. Yeniden deneniyor...",
                                    camera_id=self.camera_id)
                    time.sleep(self.reconnect_delay)
                    continue
            
            t0 = time.time()
            t_read0 = time.perf_counter()
            ret, frame = self.cap.read()
            self.last_read_ms = (time.perf_counter() - t_read0) * 1000.0
            self.last_decode_ms = self.last_read_ms
            
            if not ret or frame is None:
                # Video dosyasıysa başa sar, kameraysa yeniden bağlan
                if self._is_video_file and self._video_file_loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    now = time.time()
                    if now - self._last_eof_log_ts > 5.0:
                        self._last_eof_log_ts = now
                        self.logger.info(f"{self.camera_id} video EOF reached, rewinding to frame 0")
                    continue
                self.logger.log(EventType.CAMERA_OFFLINE,
                                f"{self.name} görüntü kesildi.",
                                camera_id=self.camera_id)
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue
                
            original_shape = frame.shape[:2]
            if (not self._high_res_warned) and (original_shape[1] > 1920 or original_shape[0] > 1080):
                self._high_res_warned = True
                self.logger.warning(
                    f"{self.camera_id}: High resolution video source detected. Consider using 720p proxy for smoother playback."
                )
            t_resize0 = time.perf_counter()
            frame = self._resize_frame_for_runtime(frame)
            self.last_resize_ms = (time.perf_counter() - t_resize0) * 1000.0
            runtime_shape = frame.shape[:2] if frame is not None else None
            with self.lock:
                self.current_frame = frame
                self.frame_seq += 1
                self.last_frame_ts = time.time()
                self.new_frame_count += 1
                elapsed_run = max(0.001, self.last_frame_ts - self.start_ts)
                self.effective_fps = self.new_frame_count / elapsed_run
                self.original_frame_shape = original_shape
                self.runtime_frame_shape = runtime_shape
                self.last_frame_shape = runtime_shape
                
            # FPS hesapla
            self.frame_count += 1
            now = time.time()
            if now - self.start_time >= 1.0:
                self.fps = self.frame_count / (now - self.start_time)
                self.frame_count = 0
                self.start_time = now

            # FPS sınırlama: frame_interval kadar geçmesi gereken süreyi hesapla
            elapsed = time.time() - t0
            if self._is_video_file and not self._video_file_realtime:
                continue
            sleep_t = self._frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _resize_frame_for_runtime(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        max_w = self.max_frame_width
        max_h = self.max_frame_height
        if not max_w and not max_h:
            return frame
        lim_w = max_w if max_w else w
        lim_h = max_h if max_h else h
        scale = min(lim_w / float(w), lim_h / float(h), 1.0)
        if scale >= 1.0:
            return frame
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class CameraManager:
    """Tüm kameraları yöneten ana sınıf (TEK NOKTADAN OKUMA KURALI)."""

    def __init__(self, config: AppConfig = None, logger=None, autoload_config: bool = True):
        self.config = config
        self.logger = logger
        self.streams: dict[str, CameraStream] = {}
        self.max_active_cameras = 4
        self._active_camera_ids: list[str] = []
        self._max_frame_width = None
        self._max_frame_height = None
        self._camera_target_fps = GRID_FPS
        self._video_file_realtime = True
        self._video_file_loop = True
        self._drop_old_frames = True
        
        # Config varsa kameraları yükle
        if config is not None:
            self.max_active_cameras = config.get_max_active_cameras()
            perf = getattr(config, "performance", {}) or {}
            w = perf.get("display_source_max_width")
            h = perf.get("display_source_max_height")
            self._max_frame_width = int(w) if isinstance(w, (int, float)) and w > 0 else None
            self._max_frame_height = int(h) if isinstance(h, (int, float)) and h > 0 else None
            try:
                import torch
                torch_cuda = bool(torch.cuda.is_available())
            except Exception:
                torch_cuda = False
            fps_key = "camera_target_fps_gpu" if torch_cuda else "camera_target_fps_cpu"
            self._camera_target_fps = int(perf.get(fps_key, perf.get("camera_target_fps", GRID_FPS)))
            self._video_file_realtime = bool(perf.get("video_file_realtime", True))
            self._video_file_loop = bool(perf.get("video_file_loop", True))
            self._drop_old_frames = bool(perf.get("drop_old_frames", True))
            if autoload_config:
                self._init_cameras()

    def _running_stream_count(self) -> int:
        return sum(1 for s in self.streams.values() if s.is_running)

    def _emit_camera_limit_warning(self, requested_camera: str):
        if not self.logger:
            return
        active_streams = sorted([cid for cid, s in self.streams.items() if s.is_running])
        self.logger.warning(
            "CAMERA_LIMIT_REACHED "
            f"requested_camera={requested_camera} "
            f"max_active_cameras={self.max_active_cameras} "
            f"active_streams={active_streams}"
        )
        
    def _init_cameras(self):
        """Config dosyasındaki kameraları oluşturur ancak başlatmaz."""
        enabled_cameras = self.config.get_enabled_cameras()
        self._active_camera_ids = [c.get("id", "") for c in enabled_cameras[:self.max_active_cameras] if c.get("id")]
        ignored = enabled_cameras[self.max_active_cameras:]

        if self.logger:
            if self._active_camera_ids:
                self.logger.info(f"Active cameras: {', '.join(self._active_camera_ids)}")
            for cam_cfg in ignored:
                cid = cam_cfg.get("id", "UNKNOWN")
                self.logger.warning(f"{cid} ignored because max_active_cameras={self.max_active_cameras}")

        for cam_cfg in enabled_cameras[:self.max_active_cameras]:
            cam_id = cam_cfg.get("id")
            if not cam_id:
                continue
            source = cam_cfg.get("source")
            name = cam_cfg.get("name", cam_id)
            
            # Kaynak string ise (örn: RTSP) veya int ise (örn: 0) düzgün aktar
            # int dönüşümü yapmaya çalış (webcam için)
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass # Bırak string kalsın
                
            stream = CameraStream(
                cam_id,
                source,
                name,
                self.logger,
                max_frame_width=self._max_frame_width,
                max_frame_height=self._max_frame_height,
                target_fps=self._camera_target_fps,
                video_file_realtime=self._video_file_realtime,
                video_file_loop=self._video_file_loop,
                drop_old_frames=self._drop_old_frames,
            )
            self.streams[cam_id] = stream
            
    def start_all(self):
        """Tüm kameraları başlatır."""
        started = 0
        for cam_id, stream in self.streams.items():
            if started >= self.max_active_cameras:
                self._emit_camera_limit_warning(cam_id)
                continue
            stream.start()
            started += 1
            
    def stop_all(self):
        """Tüm kameraları durdurur."""
        for stream in self.streams.values():
            stream.stop()
            
    def start_camera(self, camera_id: str) -> bool:
        """Belirli bir kamerayı başlatır."""
        if camera_id in self.streams:
            stream = self.streams[camera_id]
            if not stream.is_running and self._running_stream_count() >= self.max_active_cameras:
                self._emit_camera_limit_warning(camera_id)
                return False
            self.streams[camera_id].start()
            return True
        return False
        
    def stop_camera(self, camera_id: str):
        """Belirli bir kamerayı durdurur."""
        if camera_id in self.streams:
            self.streams[camera_id].stop()
            
    def get_frame(self, camera_id: str) -> np.ndarray | None:
        """Bir kameradan son kareyi alır (Pipeline için)."""
        if camera_id in self.streams:
            return self.streams[camera_id].get_frame()
        return None
        
    def get_active_cameras(self) -> list[str]:
        """Çalışan kameraların ID listesi."""
        active = []
        for cam_id, stream in self.streams.items():
            if stream.is_running and stream.current_frame is not None:
                active.append(cam_id)
        return active

    def get_active_streams(self) -> dict[str, CameraStream]:
        """
        Yalnızca çalışan akışların shallow-copy dict'ini döndürür.
        MainWindow bu API üzerinden döngü kurarak kaldırılmış
        kameralardan frame almayı engeller.
        """
        return {cid: s for cid, s in self.streams.items() if s.is_running}
        
    def get_camera_info(self, camera_id: str) -> dict:
        """Kameranın o anki durumu okur (UI için)."""
        if camera_id not in self.streams:
            return {}
            
        stream = self.streams[camera_id]
        return {
            "id": stream.camera_id,
            "name": stream.name,
            "is_running": stream.is_running,
            "fps": round(stream.fps, 1),
            "online": stream.cap is not None and stream.cap.isOpened()
        }

    def set_all_target_fps(self, fps: int):
        """Tüm kamera akışlarının hedef FPS'ini değiştirir."""
        for stream in self.streams.values():
            stream.set_target_fps(fps)

    # ── Dinamik Kaynak Yönetimi ──────────────────────────────────────────
    def add_source(self, cam_id: str, source: str | int, name: str = ""):
        """Çalışma sırasında yeni kamera kaynağı ekler ve başlatır."""
        if cam_id in self.streams:
            return  # Zaten var
        if self._running_stream_count() >= self.max_active_cameras:
            self._emit_camera_limit_warning(cam_id)
            return
        stream = CameraStream(
            cam_id,
            source,
            name or cam_id,
            self.logger,
            max_frame_width=self._max_frame_width,
            max_frame_height=self._max_frame_height,
            target_fps=self._camera_target_fps,
            video_file_realtime=self._video_file_realtime,
            video_file_loop=self._video_file_loop,
            drop_old_frames=self._drop_old_frames,
        )
        self.streams[cam_id] = stream
        stream.start()

    def remove_source(self, cam_id: str):
        """Kamera kaynağını durdurur ve kaldırır."""
        if cam_id in self.streams:
            self.streams[cam_id].stop()
            del self.streams[cam_id]
