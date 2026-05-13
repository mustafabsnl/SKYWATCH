"""
SKYWATCH — Event Logger & Bildirim Sistemi
Tüm olayları loglar + bildirim gönderir.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from enum import Enum


class EventType(Enum):
    """Olay türleri."""
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    FACE_SCANNED = "FACE_SCANNED"
    CRIMINAL_DETECTED = "CRIMINAL_DETECTED"
    WANTED_FOUND = "WANTED_FOUND"
    SEARCH_STARTED = "SEARCH_STARTED"
    SEARCH_COMPLETED = "SEARCH_COMPLETED"
    CAMERA_ONLINE = "CAMERA_ONLINE"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    TRACK_CREATED = "TRACK_CREATED"
    TRACK_LOST = "TRACK_LOST"
    CROSS_CAMERA_MATCH = "CROSS_CAMERA_MATCH"


class EventLogger:
    """Merkezi olay kayıt ve bildirim sistemi."""

    def __init__(self, config):
        self._config = config
        self._log_dir = Path(config.get_log_dir())
        self._screenshot_dir = Path(config.get_screenshot_dir())

        # Dizinleri oluştur
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Python logger ayarla
        self._logger = self._setup_logger()

        # Olay callback'leri (GUI için)
        self._callbacks: list[callable] = []
        self._run_logger = None

    def set_run_logger(self, run_logger):
        """RunLogger köprüsü ekler; EventLogger loglarını run klasörüne yansıtır."""
        self._run_logger = run_logger
        
        if self._run_logger and hasattr(self._run_logger, "run_dir"):
            run_dir = str(self._run_logger.run_dir)
            self.info(f"[RUN_LOG_DIR_READY] run_id={self._run_logger.run_id} run_dir=\"{run_dir}\"")
            
            cfg = self._config.get("debug", {})
            if bool(cfg.get("person_search_trace_enabled", False)):
                jsonl_file = cfg.get("person_search_jsonl_file", "person_search_trace.jsonl")
                txt_file = cfg.get("person_search_text_file", "person_search_trace.log")
                j_path = self._resolve_log_output_path(jsonl_file)
                t_path = self._resolve_log_output_path(txt_file) if txt_file else None
                self.info(f"[PERSON_SEARCH_TRACE_READY] jsonl=\"{j_path}\" text=\"{t_path}\"")

    def _mirror_event(self, event_type: str, message: str, level: str = "INFO", **fields):
        if self._run_logger is None:
            return
        try:
            self._run_logger.log_event(
                event_type,
                message,
                level=level,
                source="EventLogger",
                **fields,
            )
        except Exception:
            # EventLogger'ın ana akışını log köprüsü yüzünden bozma
            pass

    def _setup_logger(self) -> logging.Logger:
        """Python logging yapılandırması."""
        logger = logging.getLogger("SKYWATCH")
        logger.setLevel(logging.DEBUG)

        # Konsol handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S"
        )
        console.setFormatter(console_fmt)

        # Dosya handler
        log_file = self._config.logging.get("log_file", "logs/events.log")
        log_path = self._config.project_root / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            str(log_path), encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)

        # Handler'ları ekle (tekrar eklemeyi önle)
        if not logger.handlers:
            logger.addHandler(console)
            logger.addHandler(file_handler)

        return logger

    def log(self, event_type: EventType, message: str, **kwargs):
        """Olay kaydet.

        Args:
            event_type: Olay türü (EventType enum)
            message: Olay açıklaması
            **kwargs: Ek veriler (camera_id, track_id, vb.)
        """
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        full_msg = f"[{event_type.value}] {message}"
        if extra:
            full_msg += f" | {extra}"

        # Seviyeye göre logla
        if event_type in (EventType.WANTED_FOUND, EventType.CRIMINAL_DETECTED):
            self._logger.warning(full_msg)
            level = "WARNING"
        elif event_type in (EventType.CAMERA_OFFLINE,):
            self._logger.error(full_msg)
            level = "ERROR"
        else:
            self._logger.info(full_msg)
            level = "INFO"

        # Callback'leri çağır (GUI güncellemesi için)
        event_data = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.now(),
            **kwargs
        }
        for callback in self._callbacks:
            try:
                callback(event_data)
            except Exception as e:
                self._logger.error(f"Callback hatası: {e}")
                if self._run_logger is not None:
                    self._run_logger.log_error("event_callback_error", e, event_type=event_type.value)

        self._mirror_event(event_type.value, message, level=level, **kwargs)

    def on_event(self, callback: callable):
        """Olay dinleyici ekle (GUI için).

        Args:
            callback: fn(event_data: dict) — her olayda çağrılır
        """
        self._callbacks.append(callback)

    def info(self, message: str):
        """Genel bilgi logu."""
        self._logger.info(message)
        self._mirror_event("INFO", message, level="INFO")

    def warning(self, message: str):
        """Uyarı logu."""
        self._logger.warning(message)
        self._mirror_event("WARNING", message, level="WARNING")

    def error(self, message: str):
        """Hata logu."""
        self._logger.error(message)
        self._mirror_event("ERROR", message, level="ERROR")

    def debug(self, message: str):
        """Debug logu."""
        self._logger.debug(message)
        self._mirror_event("DEBUG", message, level="DEBUG")

    def _resolve_log_output_path(self, configured_path: str) -> Path:
        """
        RunLogger aktifse dosyayı run klasörünün içine yazar.
        Config içinde logs/person_search_trace.jsonl gibi global path verilmişse
        sadece dosya adını alır ve aktif run klasörüne taşır.
        """
        if self._run_logger is not None and hasattr(self._run_logger, "resolve_run_path"):
            return self._run_logger.resolve_run_path(configured_path)
            
        p = Path(configured_path)
        # Eğer config'de sadece dosya adı varsa, diagnostics klasörüne atalım (run_logger yoksa)
        if str(p.parent) == "." or not str(p.parent):
            return self._config.project_root / "logs" / "diagnostics" / p.name
            
        return self._config.project_root / configured_path

    def person_search_trace(self, event: str, **fields):
        """
        Person Search (Kişi Ara) moduna özel teşhis ve izleme logu yazar.
        Sistemi yavaşlatmamak için JSONL (ve isterseniz ayrı TEXT) formatında yazar.
        """
        cfg = self._config.get("debug", {})
        if not bool(cfg.get("person_search_trace_enabled", False)):
            return
            
        import json
        
        ts = datetime.now().isoformat()
        log_data = {"ts": ts, "event": event, **fields}
        
        # 1. Ana events.log'a sadece kısa metin yaz (opsiyonel ama isteniyor)
        fields_str = " ".join(f"{k}={v}" for k, v in fields.items())
        msg = f"[{event}] {fields_str}"
        self._logger.debug(msg) # events.log'a yazar (DEBUG seviyesinde)

        # 2. JSONL log
        if bool(cfg.get("person_search_jsonl_enabled", True)):
            jsonl_file = cfg.get("person_search_jsonl_file", "person_search_trace.jsonl")
            jsonl_path = self._resolve_log_output_path(jsonl_file)
            try:
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception as e:
                self._logger.error(f"Failed to write person_search jsonl trace: {e}")

        # 3. TEXT log
        txt_file = cfg.get("person_search_text_file", "person_search_trace.log")
        if txt_file:
            txt_path = self._resolve_log_output_path(txt_file)
            try:
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {msg}\n")
            except Exception as e:
                pass

