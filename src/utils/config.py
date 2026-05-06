"""
SKYWATCH — Config Yükleyici
Tek AppConfig sınıfı — tüm modüllere constructor'dan geçirilir.
"""

import os
import yaml
import warnings
from pathlib import Path


class AppConfig:
    """Tüm sistem konfigürasyonunu yükler ve yönetir."""
    _STABLE_RUNTIME_CAMERA_CAP = 2

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Proje kök dizinini bul (src/utils/ → src/ → SKYWATCH/)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self._config_path = Path(config_path)
        self._data = self._load()
        self._cameras_cfg = self._data.get("cameras", [])

        # Alt konfigürasyonları attribute olarak ata
        self.cameras = self._normalize_cameras(self._cameras_cfg)
        self.regions = self._data.get("regions", {})
        self.face = self._data.get("face", {})
        self.tracking = self._data.get("tracking", {})
        self.cross_camera = self._data.get("cross_camera", {})
        self.movement = self._data.get("movement", {})
        self.database = self._data.get("database", {})
        self.search = self._data.get("search", {})
        self.logging = self._data.get("logging", {})
        self.notifications = self._data.get("notifications", {})
        self.performance = self._data.get("performance", {})
        self._apply_runtime_camera_cap_policy()

        # Proje kök dizini (diğer modüller için)
        self.project_root = Path(__file__).parent.parent.parent

    def _apply_runtime_camera_cap_policy(self):
        """
        Stabil test/demo modu için runtime cap'i 2 kamerada tutar.
        Altyapı kamera tanımlarını silmez; sadece aktif kullanım limitlenir.
        """
        cap = self._STABLE_RUNTIME_CAMERA_CAP
        cameras_cfg = self._data.get("cameras", {}) or {}
        if isinstance(cameras_cfg, dict):
            cfg_cap = cameras_cfg.get("max_active_cameras")
            if isinstance(cfg_cap, int) and cfg_cap > cap:
                warnings.warn("current project stable mode uses 2 cameras; overriding cameras.max_active_cameras to 2")
                cameras_cfg["max_active_cameras"] = cap
        perf_cfg = self._data.get("performance", {}) or {}
        if isinstance(perf_cfg, dict):
            perf_cap = perf_cfg.get("max_active_cameras")
            if isinstance(perf_cap, int) and perf_cap > cap:
                warnings.warn("current project stable mode uses 2 cameras; overriding performance.max_active_cameras to 2")
                perf_cfg["max_active_cameras"] = cap
        search_cfg = self._data.get("search", {}) or {}
        if isinstance(search_cfg, dict):
            concurrent = search_cfg.get("max_concurrent_cameras")
            if isinstance(concurrent, int) and concurrent > cap:
                warnings.warn("current project stable mode uses 2 cameras; overriding search.max_concurrent_cameras to 2")
                search_cfg["max_concurrent_cameras"] = cap

        # Enabled kamera sayısı > cap ise ilk cap kaydı açık bırak, kalanını pasifleştir.
        enabled_count = 0
        for cam in self.cameras:
            if cam.get("enabled", True):
                enabled_count += 1
                if enabled_count > cap:
                    cam["enabled"] = False
        if enabled_count > cap:
            warnings.warn("Configured enabled cameras exceed max_active_cameras; truncating to first 2.")

    def _load(self) -> dict:
        """YAML config dosyasını yükle."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Config dosyası bulunamadı: {self._config_path}"
            )

        with open(self._config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Config dosyası boş: {self._config_path}")

        return data

    def get(self, key: str, default=None):
        """Üst seviye config değeri al."""
        return self._data.get(key, default)

    def _normalize_cameras(self, cameras_cfg) -> list[dict]:
        """Legacy liste ve yeni {items: []} formatlarını tek listeye indirger."""
        if isinstance(cameras_cfg, dict):
            items = cameras_cfg.get("items", []) or []
            return list(items)
        if isinstance(cameras_cfg, list):
            return list(cameras_cfg)
        return []

    def get_camera_by_id(self, camera_id: str) -> dict | None:
        """Kamera ID'sine göre kamera config'ini döndür."""
        for cam in self.cameras:
            if cam.get("id") == camera_id:
                return cam
        return None

    def get_enabled_cameras(self) -> list[dict]:
        """enabled=true olan kamera kayıtlarını döndürür (alan yoksa true varsayılır)."""
        enabled = []
        for cam in self.cameras:
            if cam.get("enabled", True):
                enabled.append(cam)
        return enabled

    def get_max_active_cameras(self) -> int:
        """Maksimum aktif kamera sayısını döndürür."""
        if isinstance(self._cameras_cfg, dict):
            val = self._cameras_cfg.get("max_active_cameras")
            if isinstance(val, int) and val > 0:
                return val
        perf_val = self.performance.get("max_active_cameras")
        if isinstance(perf_val, int) and perf_val > 0:
            return perf_val
        return 4

    def get_active_cameras(self) -> list[dict]:
        """Enabled kameraları max_active_cameras ile sınırlandırır."""
        max_active = self.get_max_active_cameras()
        return self.get_enabled_cameras()[:max_active]

    def get_camera_layout(self) -> dict:
        """
        Kamera grid düzenini döndürür.
        Varsayılan:
          1 kamera -> 1x1
          2 kamera -> 1x2
          3-4 kamera -> 2x2
        """
        if isinstance(self._cameras_cfg, dict):
            layout = self._cameras_cfg.get("layout", {}) or {}
            rows = layout.get("rows")
            cols = layout.get("cols")
            if isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0:
                return {"rows": rows, "cols": cols}

        n = len(self.get_active_cameras())
        if n <= 1:
            return {"rows": 1, "cols": 1}
        if n == 2:
            return {"rows": 1, "cols": 2}
        return {"rows": 2, "cols": 2}

    def get_cameras_in_region(self, region_id: str) -> list[str]:
        """Bölgedeki kamera ID'lerini döndür."""
        region = self.regions.get(region_id, {})
        return region.get("cameras", [])

    def get_db_path(self) -> Path:
        """Veritabanı dosya yolunu mutlak olarak döndür."""
        db_path = self.database.get("path", "database/skywatch.db")
        return self.project_root / db_path

    def get_photos_dir(self) -> Path:
        """Fotoğraf dizin yolunu mutlak olarak döndür."""
        photos = self.database.get("photos_dir", "database/photos/")
        return self.project_root / photos

    def get_log_dir(self) -> Path:
        """Log dizin yolunu mutlak olarak döndür."""
        log_dir = self.logging.get("log_dir", "logs/")
        return self.project_root / log_dir

    def get_screenshot_dir(self) -> Path:
        """Tespit ekran görüntüleri dizinini döndür."""
        ss_dir = self.logging.get("screenshot_dir", "logs/detections/")
        return self.project_root / ss_dir

    def __repr__(self) -> str:
        return (
            f"AppConfig("
            f"cameras={len(self.cameras)}, "
            f"regions={len(self.regions)}, "
            f"face_model={self.face.get('recognition_model', 'N/A')}"
            f")"
        )
