"""
SKYWATCH — Veri Modelleri
Sistem genelinde kullanılan temel veri yapıları.
"""

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class MatchResult:
    """Sabıka veritabanı eşleşme sonucu."""
    criminal_id: int
    confidence: float


@dataclass
class FaceResult:
    """Tek bir karede algılanan yüz."""
    bbox: list[int]          # [x1, y1, x2, y2]
    embedding: np.ndarray    # 512-dim vektör
    det_score: float         # Algılama güven skoru (0-1)
    age: int = 0
    gender: str = ""


@dataclass
class MovementReport:
    """Bir Track'in hareket analiz sonucu."""
    speed: float = 0.0              # Anlık hız (px/frame)
    avg_speed: float = 0.0          # Ortalama hız
    direction: tuple = (0.0, 0.0)   # Yön vektörü (dx, dy)
    dwell_time: float = 0.0         # Bekleme süresi (sn)
    route: list[tuple] = field(default_factory=list)  # Geçilen (x,y) noktaları
    behavior_score: float = 0.0     # 0.0 - 1.0 (1.0 = çok şüpheli)
    behavior_label: str = "normal"  # normal / suspicious / running


@dataclass
class Track:
    """DeepSORT tarafından takip edilen kişi."""
    track_id: int
    bbox: list[int]
    is_new: bool = False
    
    # Zaman ve Durum
    age: int = 0
    is_confirmed: bool = False
    time_since_update: int = 0
    
    # Yüz ve Kimlik
    face_embedding: np.ndarray | None = None
    criminal_match: MatchResult | None = None
    global_id: str | None = None    # Cross-camera id
    
    # Hareket
    movement: MovementReport = field(default_factory=MovementReport)
    
    # Kamera
    camera_id: str = ""


@dataclass
class DecisionResult:
    """Karar motoru çıktısı."""
    track_id: int
    bbox: list[int]
    status: str             # CLEAN / SUSPICIOUS / WANTED
    danger_level: str       # LOW / MEDIUM / HIGH / CRITICAL
    color: tuple            # BGR renk kodu (overlay için)
    criminal_id: int | None = None
    confidence: float = 0.0
    behavior_label: str = "normal"
    global_id: str | None = None
