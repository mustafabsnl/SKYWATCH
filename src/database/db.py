"""
SKYWATCH — Database
SQLite veritabanı işlemleri (tek dosya).
"""

import sqlite3
import numpy as np
import io
import os
from datetime import datetime
from pathlib import Path

from utils.config import AppConfig
from utils.logger import EventLogger


def adapt_array(arr: np.ndarray) -> bytes:
    """Numpy dizisini SQLite BLOB formatına dönüştürür."""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return out.read()

def convert_array(text: bytes) -> np.ndarray:
    """SQLite BLOB formatını Numpy dizisine dönüştürür."""
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# SQLite için numpy dönüştürücülerini kaydet
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


class Database:
    """Tüm veritabanı işlemlerini yöneten sınıf."""

    def __init__(self, config: AppConfig, logger: EventLogger):
        self.config = config
        self.logger = logger
        self.db_path = config.get_db_path()
        self.photos_dir = config.get_photos_dir()
        
        # Dizinlerin kurulu olduğundan emin ol
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        
        # Tabloları oluştur
        self._init_db()
        
    def _get_conn(self) -> sqlite3.Connection:
        """Yeni bir veritabanı bağlantısı döndürür."""
        # detect_types, parse_decltypes numpy array dönüşümü için önemli
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Eksik tabloları oluşturur."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # CRIMINALS tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS criminals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    gender TEXT,
                    age INTEGER,
                    crime_type TEXT NOT NULL,
                    danger_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    photo_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # EMBEDDINGS tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    criminal_id INTEGER NOT NULL,
                    embedding array NOT NULL,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(criminal_id) REFERENCES criminals(id)
                )
            """)
            
            # DETECTIONS tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    criminal_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    screenshot_path TEXT,
                    confidence REAL,
                    FOREIGN KEY(criminal_id) REFERENCES criminals(id)
                )
            """)
            
            # SEARCH_REQUESTS tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_path TEXT NOT NULL,
                    target_cameras TEXT,
                    status TEXT NOT NULL DEFAULT 'SEARCHING',
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    found_at TEXT
                )
            """)
            
            conn.commit()

    # --- CRIMINALS (Sabıkalılar) ---

    def add_criminal(self, name: str, embedding: np.ndarray, 
                    crime_type: str, danger_level: str, 
                    status: str = "WANTED", photo_path: str = "") -> int:
        """Sabıkalı kişi ekler ve criminal_id döndürür."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                # Kişiyi criminals tablosuna ekle
                cursor.execute("""
                    INSERT INTO criminals 
                    (name, crime_type, danger_level, status, photo_path)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, crime_type, danger_level, status, photo_path))
                
                criminal_id = cursor.lastrowid
                
                # Embedding'i tabloya ekle
                cursor.execute("""
                    INSERT INTO embeddings (criminal_id, embedding)
                    VALUES (?, ?)
                """, (criminal_id, embedding))
                
                conn.commit()
                self.logger.info(f"DB: '{name}' başarıyla eklendi (ID: {criminal_id})")
                return criminal_id
        except Exception as e:
            self.logger.error(f"DB: Sabıkalı eklenirken hata - {e}")
            return -1

    def get_criminal_info(self, criminal_id: int) -> dict | None:
        """Belirtilen ID'ye sahip kişinin bilgilerini getirir."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM criminals WHERE id = ?", (criminal_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_criminal_status(self, criminal_id: int, status: str):
        """Kişinin durumunu (WANTED/CLEARED vb.) günceller."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE criminals SET status = ? WHERE id = ?", 
                (status, criminal_id)
            )
            conn.commit()

    # --- EMBEDDINGS (Yüz Verileri) ---

    def get_all_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """
        Tüm embedding'leri döndürür.
        UYARI: Karşılaştırma burada YAPILMAZ.
        Returns:
            list[tuple[criminal_id, embedding_array]]
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # Sadece ARANIYOR (WANTED) veya SABIKALI (CRIMINAL) detaylara sahip olanları çek.
                # Durumu TEMIZ (CLEARED) olanları sorgulamaya gerek yok.
                cursor.execute("""
                    SELECT e.criminal_id, e.embedding 
                    FROM embeddings e
                    JOIN criminals c ON e.criminal_id = c.id
                    WHERE c.status IN ('WANTED', 'CRIMINAL')
                """)
                
                rows = cursor.fetchall()
                # (criminal_id, numpy_dizisi) formatında liste
                return [(row[0], row[1]) for row in rows]
        except Exception as e:
            self.logger.error(f"DB: Embedding okunurken hata - {e}")
            return []

    # --- DETECTIONS (Tespit Kayıtları) ---

    def log_detection(self, criminal_id: int, camera_id: str, 
                     screenshot_path: str, confidence: float) -> int:
        """Sistem bir kişiyi tespit ettiğinde kayıt oluşturur."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO detections 
                    (criminal_id, camera_id, screenshot_path, confidence)
                    VALUES (?, ?, ?, ?)
                """, (criminal_id, camera_id, screenshot_path, confidence))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"DB: Tespit kaydedilemedi - {e}")
            return -1

    def get_criminal_history(self, criminal_id: int, limit: int = 10) -> list[dict]:
        """Bir kişinin en son nerelerde görüldüğünü getirir."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT camera_id, timestamp, screenshot_path, confidence
                FROM detections
                WHERE criminal_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (criminal_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]

    # --- SEARCH REQUESTS (Aktif Arama) ---

    def create_search_request(self, photo_path: str, target_cameras: list[str]) -> int:
        """Yeni bir aktif arama talebi oluşturur."""
        cameras_str = ",".join(target_cameras)
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_requests (photo_path, target_cameras)
                VALUES (?, ?)
            """, (photo_path, cameras_str))
            conn.commit()
            return cursor.lastrowid

    def update_search_status(self, request_id: int, status: str, found_camera: str = None):
        """Arama sonucunu günceller."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if found_camera:
                found_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("""
                    UPDATE search_requests 
                    SET status = ?, found_at = ?
                    WHERE id = ?
                """, (status, f"{found_camera} - {found_time}", request_id))
            else:
                cursor.execute("""
                    UPDATE search_requests 
                    SET status = ?
                    WHERE id = ?
                """, (status, request_id))
            conn.commit()
