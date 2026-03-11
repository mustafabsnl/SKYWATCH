"""
SKYWATCH — Config Yükleyici
Tek AppConfig sınıfı — tüm modüllere constructor'dan geçirilir.
"""

import os
import yaml
from pathlib import Path


class AppConfig:
    """Tüm sistem konfigürasyonunu yükler ve yönetir."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Proje kök dizinini bul (src/utils/ → src/ → SKYWATCH/)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self._config_path = Path(config_path)
        self._data = self._load()

        # Alt konfigürasyonları attribute olarak ata
        self.cameras = self._data.get("cameras", [])
        self.regions = self._data.get("regions", {})
        self.face = self._data.get("face", {})
        self.tracking = self._data.get("tracking", {})
        self.cross_camera = self._data.get("cross_camera", {})
        self.movement = self._data.get("movement", {})
        self.database = self._data.get("database", {})
        self.search = self._data.get("search", {})
        self.logging = self._data.get("logging", {})
        self.notifications = self._data.get("notifications", {})

        # Proje kök dizini (diğer modüller için)
        self.project_root = Path(__file__).parent.parent.parent

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

    def get_camera_by_id(self, camera_id: str) -> dict | None:
        """Kamera ID'sine göre kamera config'ini döndür."""
        for cam in self.cameras:
            if cam.get("id") == camera_id:
                return cam
        return None

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
