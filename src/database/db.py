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


def _coerce_embedding(value, logger=None, person_id="unknown") -> np.ndarray | None:
    """
    SQLite'tan gelen embedding değerini güvenli şekilde np.ndarray'e çevirir.
    """
    if logger:
        logger.person_search_trace("EMBEDDING_COERCE_BEGIN", person_id=person_id, source_type=type(value).__name__)
        
    if value is None:
        if logger:
            logger.person_search_trace("EMBEDDING_COERCE_FAIL", person_id=person_id, reason="value_is_none")
        return None

    try:
        if isinstance(value, np.ndarray):
            arr = value
            source_type = "ndarray"
        elif isinstance(value, memoryview):
            arr = np.load(io.BytesIO(value.tobytes()), allow_pickle=False)
            source_type = "memoryview"
        elif isinstance(value, (bytes, bytearray)):
            arr = np.load(io.BytesIO(value), allow_pickle=False)
            source_type = "bytes"
        else:
            if logger:
                logger.warning(f"[EMBEDDING_SKIP] person_id={person_id} reason='parse_error' type='{type(value)}'")
            return None
    except Exception as e:
        if logger:
            logger.warning(f"[EMBEDDING_SKIP] person_id={person_id} reason='parse_error' error='{e}'")
        return None

    try:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    except Exception as e:
        if logger:
            logger.warning(f"[EMBEDDING_SKIP] person_id={person_id} reason='reshape_error' error='{e}'")
        return None

    if arr.shape != (512,):
        if logger:
            logger.warning(f"[EMBEDDING_SKIP] person_id={person_id} reason='invalid_shape' shape={arr.shape}")
        return None

    if not np.isfinite(arr).all():
        if logger:
            logger.warning(f"[EMBEDDING_SKIP] person_id={person_id} reason='nan_or_inf'")
        return None

    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        if logger:
            logger.warning(f"[EMBEDDING_SKIP] person_id={person_id} reason='zero_norm'")
        return None

    arr = arr / norm
    
    if logger:
        logger.person_search_trace(
            "EMBEDDING_COERCE_OK", 
            person_id=person_id, 
            shape=str(arr.shape), 
            dtype=str(arr.dtype), 
            norm_before=f"{norm:.3f}", 
            norm_after=f"{np.linalg.norm(arr):.3f}"
        )
        # logger.debug(f"[EMBEDDING_COERCE] person_id={person_id} source_type={source_type} shape={arr.shape}")
        
    return arr



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
                # GENERAL mode compares every valid live face against all saved embeddings.
                cursor.execute("""
                    SELECT e.criminal_id, e.embedding 
                    FROM embeddings e
                    JOIN criminals c ON e.criminal_id = c.id
                """)
                
                rows = cursor.fetchall()
                # (criminal_id, numpy_dizisi) formatında liste
                embeddings: list[tuple[int, np.ndarray]] = []
                for row in rows:
                    emb = _coerce_embedding(row[1], logger=self.logger, person_id=row[0])
                    if emb is not None:
                        embeddings.append((row[0], emb))
                return embeddings
        except Exception as e:
            if self.logger:
                self.logger.error(f"DB: Embedding okunurken hata - {e}")
            return []

    def get_all_person_embeddings_for_general(self) -> list[dict]:
        """
        GENERAL mod için: status filtresiz, embedding'i geçerli olan tüm satırlar.
        Aynı kişiden birden fazla embedding satırı olabilir; her biri ayrı dict olarak döner.
        Her candidate'e embedding_hash eklenir (duplicate tespiti için).
        """
        import hashlib

        out: list[dict] = []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT c.id AS person_id, c.name, c.status, e.embedding
                    FROM embeddings e
                    JOIN criminals c ON e.criminal_id = c.id
                    """
                )
                rows = cursor.fetchall()
            for row in rows:
                pid = int(row["person_id"])
                emb = _coerce_embedding(row["embedding"], logger=self.logger, person_id=pid)
                if emb is None:
                    continue
                emb_copy = np.asarray(emb, dtype=np.float32).reshape(-1).copy()
                emb_hash = hashlib.sha1(emb_copy.tobytes()).hexdigest()[:12]
                out.append(
                    {
                        "person_id": pid,
                        "name": row["name"] or "",
                        "status": (row["status"] or ""),
                        "embedding": emb_copy,
                        "embedding_hash": emb_hash,
                    }
                )
            return out
        except Exception as e:
            if self.logger:
                self.logger.error(f"DB: get_all_person_embeddings_for_general hatası - {e}")
            return []

    def get_embedding(self, criminal_id: int) -> np.ndarray | None:
        """Belirtilen kişinin embedding verisini döndürür."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT embedding FROM embeddings WHERE criminal_id = ?", (criminal_id,))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return _coerce_embedding(row[0], logger=self.logger, person_id=criminal_id)
                return None
        except Exception as e:
            self.logger.error(f"DB: get_embedding hatası - {e}")
            return None

    def get_person_embedding_for_search(self, person_id: int) -> dict | None:
        """
        Kişi Ara modunda seçilen kişinin detaylı bilgilerini ve güvenli embedding'ini getirir.
        """
        self.logger.person_search_trace("DB_PERSON_SEARCH_QUERY", person_id=person_id)
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.id, c.name, c.crime_type, c.danger_level, c.status, e.embedding
                    FROM criminals c
                    LEFT JOIN embeddings e ON e.criminal_id = c.id
                    WHERE c.id = ?
                """, (person_id,))
                row = cursor.fetchone()
                
            if not row:
                self.logger.person_search_trace("DB_PERSON_SEARCH_ROW", person_id=person_id, found=False)
                return None
                
            info = dict(row)
            raw_emb = info.pop("embedding", None)
            
            self.logger.person_search_trace(
                "DB_PERSON_SEARCH_ROW", 
                person_id=person_id, 
                found=True, 
                status=info.get("status"), 
                has_raw_embedding=(raw_emb is not None),
                raw_type=type(raw_emb).__name__
            )
            
            emb = _coerce_embedding(raw_emb, logger=self.logger, person_id=person_id)
            info["embedding"] = emb
            
            if emb is None:
                self.logger.warning(f"[PERSON_SEARCH_TARGET_MISSING] person_id={person_id} reason='no_embedding_or_parse_error'")
            else:
                self.logger.info(f"[PERSON_SEARCH_TARGET_LOAD] person_id={person_id} has_embedding=true shape={emb.shape}")
                
            self.logger.person_search_trace("DB_PERSON_SEARCH_RESULT", person_id=person_id, has_embedding=(emb is not None))
            return info
            
        except Exception as e:
            self.logger.error(f"DB: get_person_embedding_for_search hatası - {e}")
            return None

    def find_duplicate_person(self, new_embedding: np.ndarray, threshold: float, uncertain_low: float) -> dict | None:
        """
        Yeni bir yüzün veritabanındakilerle mükerrer olup olmadığını kontrol eder.
        
        Args:
            new_embedding: 512-d normalize numpy array
            threshold: Bu değerin üzerindeki benzerlik "duplicate" (kesin) sayılır.
            uncertain_low: Bu değer ile threshold arası "uncertain" (şüpheli) sayılır.
            
        Returns:
            dict | None: Mükerrer/şüpheli kayıt varsa bilgileri döndürür. Yoksa None döner.
        """
        emb1 = _coerce_embedding(new_embedding, logger=self.logger, person_id="new")
        if emb1 is None:
            self.logger.error("[DUP_CHECK_ERROR] new_embedding is invalid")
            return None

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.id, c.name, c.crime_type, c.danger_level, c.status, e.embedding
                    FROM criminals c
                    JOIN embeddings e ON e.criminal_id = c.id
                """)
                rows = cursor.fetchall()

            best_match = None
            best_score = 0.0
            
            valid_count = 0
            skipped_count = 0
            
            self.logger.info(f"[DUP_CHECK_START] db_rows={len(rows)} threshold={threshold:.2f} uncertain_low={uncertain_low:.2f}")

            for row in rows:
                rid = row["id"]
                name = row["name"]
                
                db_emb = _coerce_embedding(row["embedding"], logger=self.logger, person_id=rid)
                if db_emb is None:
                    skipped_count += 1
                    continue
                    
                valid_count += 1

                # Cosine similarity (already normalized)
                score = float(np.clip(np.dot(emb1, db_emb), -1.0, 1.0))
                
                # Her aday için log
                self.logger.info(f"[DUP_CHECK_SCORE] candidate_id={rid} name='{name}' similarity={score:.3f} percent={score*100:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_match = dict(row)
            
            if best_match is None:
                self.logger.info(f"[DUP_CHECK_SUMMARY] valid={valid_count} skipped={skipped_count} best_id=none best_similarity=0.0 level='no_match'")
                return None

            percent = max(0.0, min(1.0, best_score)) * 100
            
            if best_score >= threshold:
                level = "duplicate"
            elif best_score >= uncertain_low:
                level = "uncertain"
            else:
                level = "no_match"
                
            self.logger.info(f"[DUP_CHECK_SUMMARY] valid={valid_count} skipped={skipped_count} best_id={best_match['id']} best_similarity={best_score:.3f} level='{level}'")

            if level == "no_match":
                return None

            return {
                "person_id": best_match["id"],
                "name": best_match["name"],
                "crime_type": best_match["crime_type"],
                "danger_level": best_match["danger_level"],
                "status": best_match["status"],
                "similarity": best_score,
                "percent": percent,
                "level": level
            }

        except Exception as e:
            self.logger.error(f"DB: find_duplicate_person hatası - {e}")
            return None

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

    def list_persons_for_search(self) -> list[dict]:
        """Kişi Ara (Person Search) modu için veritabanındaki kişileri listeler.
        
        Returns:
            list[dict]: id, name, status, created_at, has_embedding
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        c.id, 
                        c.name, 
                        c.status, 
                        c.created_at,
                        CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END as has_embedding
                    FROM criminals c
                    LEFT JOIN embeddings e ON c.id = e.criminal_id
                    ORDER BY c.name ASC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"DB: list_persons_for_search hatası - {e}")
            return []
