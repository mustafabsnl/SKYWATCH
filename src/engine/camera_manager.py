"""
SKYWATCH — CameraManager
Tüm kameraların yönetimini, frame okuma işlemlerini ve
canlı yayın sürekliliğini sağlar. Sistemin Tek Frame Okuma noktası.
"""

import cv2
import time
import threading
from collections import deque
import numpy as np

from utils.config import AppConfig
from utils.logger import EventLogger, EventType


class CameraStream:
    """Tek bir kamera akışını bağımsız bir thread'de yönetir."""

    def __init__(self, camera_id: str, source: str | int, name: str, logger: EventLogger):
        self.camera_id = camera_id
        self.source = source
        self.name = name
        self.logger = logger
        
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
        
        # Frame okuma hızı (saniye cinsinden gecikme)
        # _connect() çağrılınca kaynağa göre güncellenir
        self._frame_delay = 0.033  # varsayılan ~30fps
        
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
            
    def get_frame(self) -> np.ndarray | None:
        """En son okunan kareyi thread-safe olarak döndürür."""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def _connect(self) -> bool:
        """Kameraya bağlanır."""
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                return False
                
            # Performans için buffer boyutunu küçült
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Video dosyası mı? Native FPS'i al → frame_delay hesapla
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if isinstance(self.source, str) and src_fps and 0 < src_fps <= 120:
                self._frame_delay = 1.0 / src_fps
            else:
                self._frame_delay = 0.01  # Webcam: sadece CPU tasarrufu
            
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
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                # Video dosyasıysa başa sar, kameraysa yeniden bağlan
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.logger.log(EventType.CAMERA_OFFLINE,
                                f"{self.name} görüntü kesildi.",
                                camera_id=self.camera_id)
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue
                
            with self.lock:
                self.current_frame = frame
                
            # FPS hesapla
            self.frame_count += 1
            now = time.time()
            if now - self.start_time >= 1.0:
                self.fps = self.frame_count / (now - self.start_time)
                self.frame_count = 0
                self.start_time = now

            # Native FPS'e göre bekle (video: 30fps → 33ms, webcam: 10ms)
            sleep_t = self._frame_delay - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)


class CameraManager:
    """Tüm kameraları yöneten ana sınıf (TEK NOKTADAN OKUMA KURALI)."""

    def __init__(self, config: AppConfig, logger: EventLogger):
        self.config = config
        self.logger = logger
        self.streams: dict[str, CameraStream] = {}
        
        # Config'deki tüm kameraları yükle
        self._init_cameras()
        
    def _init_cameras(self):
        """Config dosyasındaki kameraları oluşturur ancak başlatmaz."""
        for cam_cfg in self.config.cameras:
            cam_id = cam_cfg.get("id")
            source = cam_cfg.get("source")
            name = cam_cfg.get("name", cam_id)
            
            # Kaynak string ise (örn: RTSP) veya int ise (örn: 0) düzgün aktar
            # int dönüşümü yapmaya çalış (webcam için)
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass # Bırak string kalsın
                
            stream = CameraStream(cam_id, source, name, self.logger)
            self.streams[cam_id] = stream
            
    def start_all(self):
        """Tüm kameraları başlatır."""
        for stream in self.streams.values():
            stream.start()
            
    def stop_all(self):
        """Tüm kameraları durdurur."""
        for stream in self.streams.values():
            stream.stop()
            
    def start_camera(self, camera_id: str) -> bool:
        """Belirli bir kamerayı başlatır."""
        if camera_id in self.streams:
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
