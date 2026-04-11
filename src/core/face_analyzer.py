"""
SKYWATCH — FaceAnalyzer (Yüz Algılama ve Embedding)
InsightFace kütüphanesi wrapper sınıfı.
"""

# İLK import: cuDNN DLL'lerini PATH'e ekler
import utils.gpu_setup  # noqa: F401

import os
import sys
import numpy as np
from pathlib import Path

import insightface
from insightface.app import FaceAnalysis

from .models import FaceResult
from utils.config import AppConfig


class FaceAnalyzer:
    """InsightFace model yöneticisi ve embedding çıkarıcı."""

    def __init__(self, config: dict | AppConfig):
        # Eğer AppConfig geldiyse face ayarlarını al
        if isinstance(config, AppConfig):
            self.cfg = config.face
        else:
            self.cfg = config
            
        self.model_name = self.cfg.get('recognition_model', 'buffalo_l')
        self.threshold = self.cfg.get('similarity_threshold', 0.45)
        self.min_size = self.cfg.get('min_face_size', 30)

        # Modeli başlat
        self.app = FaceAnalysis(
            name=self.model_name,
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 640x640 algılama — kalite/hız dengesi
        self._det_size = (640, 640)
        self.app.prepare(ctx_id=0, det_size=self._det_size, det_thresh=0.5)
        
        # Frame küçültme: 0.75 = %44 daha hızlı, kalite yeterli
        self._scale = 0.75

    def detect_faces(self, frame: np.ndarray) -> list[FaceResult]:
        """Kameradan alınan bir kare (frame) içindeki tüm yüzleri algılar.
        
        Args:
            frame: BGR formatında OpenCV görüntüsü
        
        Returns:
            list[FaceResult]: Algılanan yüzlerin listesi (bbox, embedding vb.)
        """
        import cv2
        results = []
        
        # PERFORMANS: Frame'i küçült (algılama için)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))
        
        # InsightFace'e küçültülmüş frame'i gönder
        faces = self.app.get(small)
        
        # Ölçek faktörü (bbox'ları orijinal boyuta geri çevirmek için)
        inv_scale = 1.0 / self._scale
        
        for face in faces:
            # Sınırlayıcı kutu — orijinal frame boyutuna scale et
            bbox = face.bbox.astype(float)
            bbox = [int(bbox[0] * inv_scale), int(bbox[1] * inv_scale),
                    int(bbox[2] * inv_scale), int(bbox[3] * inv_scale)]
            
            # Yüz genişliği veya yüksekliği min_size'dan küçükse yoksay
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < self.min_size or height < self.min_size:
                continue
                
            res = FaceResult(
                bbox=bbox,
                embedding=face.normed_embedding,
                det_score=float(face.det_score),
                age=int(face.age) if hasattr(face, 'age') else 0,
                gender="M" if hasattr(face, 'gender') and face.gender == 1 else "F"
            )
            results.append(res)
            
        return results

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray | None:
        """Kırpılmış tek bir yüz fotoğrafından embedding çıkarır.
        Özellikle Aktif Arama (Mod 2) ve sabıkalı kaydı eklerken kullanılır.
        
        Args:
            face_image: Yüzü içeren fotoğraf (BGR)
            
        Returns:
            np.ndarray veya None (yüz bulunamazsa)
        """
        faces = self.app.get(face_image)
        if not faces:
            return None
            
        # Eğer birden fazla yüz varsa, merkeze en yakın olanı veya en büyüğünü seçebiliriz.
        # Basitlik için en yüksek güven skoruna sahip olanı (genellikle ilk eleman) dönüyoruz.
        return faces[0].normed_embedding

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """İki yüz embedding'i arasındaki benzeliği ölçer (Cosine Similarity).
        TÜM SİSTEMDE KARŞILAŞTIRMANIN YAPILDIĞI TEK NOKTA BURASIDIR.
        
        Args:
            emb1: 1. yüz vektörü (512,)
            emb2: 2. yüz vektörü (512,)
            
        Returns:
            float: 0.0 - 1.0 arası benzerlik skoru
        """
        # Normed embedding oldukları için direkt dot product yapabiliriz
        # Ancak güvenlik amaçlı yine de tam formülü uyguluyoruz
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        similarity = dot_product / (norm_a * norm_b)
        
        # (-1, 1) aralığını kontrol et
        return float(np.clip(similarity, -1.0, 1.0))
        
    def is_match(self, emb1: np.ndarray, emb2: np.ndarray, custom_threshold: float = None) -> bool:
        """İki yüzün aynı kişiye ait olup olmadığını kontrol eder."""
        thresh = custom_threshold if custom_threshold is not None else self.threshold
        return self.compare(emb1, emb2) >= thresh
