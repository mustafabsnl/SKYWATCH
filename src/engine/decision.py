"""
SKYWATCH — Karar Motoru
Track verilerini ve DB eşleşmelerini alarak son kararı
(Sabıkalı, Temiz, Aranan vb.) ve ekranda gösterilecek rengi belirler.
"""

from core.models import Track, DecisionResult, MatchResult, MovementReport


class DecisionEngine:
    """Tüm analizlerin birleşip sonuçlandığı karar mekanizması."""

    def __init__(self):
        # Duruma göre BGR renk kodları (OpenCV için)
        self.colors = {
            "CLEAN": (0, 255, 0),        # Yeşil
            "CRIMINAL": (0, 204, 255),   # Sarı/Turuncu
            "WANTED": (0, 0, 255),       # Kırmızı
            "SUSPICIOUS": (255, 0, 255), # Mor (hızlı/şüpheli hareket)
            "UNKNOWN": (200, 200, 200)   # Gri
        }

    def evaluate(self, track: Track, criminal_info: dict | None = None) -> DecisionResult:
        """
        Track üzerindeki eşleşme (match) ve hareket (movement)
        verilerini değerlendirerek nihai kararı verir.
        
        Args:
            track: Üzerinde çalışılan kişi track'i
            criminal_info: Eğer eşleşme varsa DB'den gelen detaylar
                           (danger_level, status vb.)
                           
        Returns:
            DecisionResult: UI overlay ve loglama için gerekli veriler
        """
        status = "CLEAN"
        danger_level = "LOW"
        color = self.colors["CLEAN"]
        
        # 1. Sabıka Veritabanı Eşleşmesi Öncelikli
        if track.criminal_match is not None and criminal_info is not None:
            # DB durumu (WANTED veya CRIMINAL)
            db_status = criminal_info.get("status", "CRIMINAL").upper()
            danger_level = criminal_info.get("danger_level", "LOW").upper()
            
            if db_status == "WANTED":
                status = "WANTED"
                color = self.colors["WANTED"]
            elif db_status == "CRIMINAL":
                status = "CRIMINAL"
                
                # Çok tehlikeli sabıkalılar da kırmızı işaretlenebilir
                if danger_level in ["HIGH", "CRITICAL"]:
                    color = self.colors["WANTED"]
                else:
                    color = self.colors["CRIMINAL"]
                    
        # 2. Hareket Analizi (Sabıkası yok ama hareketleri şüpheliyse)
        elif track.movement.behavior_score >= 0.7 or track.movement.behavior_label == "running":
            status = "SUSPICIOUS"
            color = self.colors["SUSPICIOUS"]
            
        # 3. Yeterince net değilse (Örn: Track yeni başladı)
        if track.age < 3: # henuz 3 frame olmadiysa gri
            color = self.colors["UNKNOWN"]

        return DecisionResult(
            track_id=track.track_id,
            bbox=track.bbox.copy(),
            status=status,
            danger_level=danger_level,
            color=color,
            criminal_id=track.criminal_match.criminal_id if track.criminal_match else None,
            confidence=track.criminal_match.confidence if track.criminal_match else 0.0,
            behavior_label=track.movement.behavior_label,
            global_id=track.global_id
        )
