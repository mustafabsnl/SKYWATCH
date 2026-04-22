"""
SKYWATCH — Yüz Analiz Modülü (YOLO best.pt tabanlı)

InsightFace/buffalo_l KULLANILMIYOR.
Tüm aşamalar best.pt ile çalışır:
  - Yüz algılama → YOLO best.pt (ultralytics)
  - Embedding     → Kırpılmış yüz crop → normalize piksel vektörü (512-d)
"""

import numpy as np
import cv2
from pathlib import Path

from .models import FaceResult

# ── Sabitler ─────────────────────────────────────────────────────────────────
_EMBED_SIZE  = 64          # crop boyutu → 64×64×1 = 4096 → PCA ile 512'ye indir
_EMBED_DIM   = 512         # çıkış embedding boyutu
_MODEL_PATH  = Path(__file__).resolve().parents[3] / "best.pt"


def _make_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Kırpılmış yüz görüntüsünden 512-d L2-normalize embedding üret.

    Yöntem:
      1. Gri tonlamaya çevir
      2. 64×64'e yeniden boyutlandır
      3. Histogram eşitleme (CLAHE) → aydınlatma bağımsızlığı
      4. 4096-d düzleştir
      5. DCT tabanlı sıkıştırma → 512-d
      6. L2-normalize
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (_EMBED_SIZE, _EMBED_SIZE),
                         interpolation=cv2.INTER_AREA)

    # CLAHE: aydınlatma normalize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(resized).astype(np.float32) / 255.0

    # 2D DCT → frekans bileşenleri daha ayırt edici
    dct = cv2.dct(eq)

    # Düzleştir ve ilk 512 katsayıyı al
    flat = dct.flatten()[:_EMBED_DIM]

    # Pad (gerekirse)
    if len(flat) < _EMBED_DIM:
        flat = np.pad(flat, (0, _EMBED_DIM - len(flat)))

    # L2-normalize
    norm = np.linalg.norm(flat)
    if norm > 1e-6:
        flat = flat / norm

    return flat.astype(np.float32)


class FaceAnalyzer:
    """
    YOLO best.pt tabanlı yüz algılama + embedding üretici.
    InsightFace'e bağımlılık yoktur.
    """

    def __init__(self, config=None):
        from ultralytics import YOLO

        if config is not None:
            if hasattr(config, 'face'):
                cfg = config.face
            else:
                cfg = config if isinstance(config, dict) else {}
        else:
            cfg = {}

        self.threshold = cfg.get('similarity_threshold', 0.60)
        self.min_size  = cfg.get('min_face_size', 30)
        self._conf     = cfg.get('det_conf', 0.40)

        model_path = _MODEL_PATH
        if not model_path.exists():
            # Fallback: çalışma dizininde ara
            fallback = Path("best.pt")
            if fallback.exists():
                model_path = fallback

        self._yolo = YOLO(str(model_path))
        self._yolo.overrides['verbose'] = False

    # ── Algılama ─────────────────────────────────────────────────────────────
    def detect_faces(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Frame içindeki yüzleri/kişileri tespit et ve embedding üret.

        Args:
            frame: BGR formatında OpenCV görüntüsü
        Returns:
            list[FaceResult]
        """
        results = self._yolo(frame, conf=self._conf, verbose=False)
        face_results: list[FaceResult] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                if w < self.min_size or h < self.min_size:
                    continue

                # Yüzü kırp
                pad    = 8
                cx1    = max(0, x1 - pad)
                cy1    = max(0, y1 - pad)
                cx2    = min(frame.shape[1], x2 + pad)
                cy2    = min(frame.shape[0], y2 + pad)
                crop   = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    continue

                emb = _make_embedding(crop)

                face_results.append(FaceResult(
                    bbox=[x1, y1, x2, y2],
                    embedding=emb,
                    det_score=float(box.conf[0]),
                    age=0,
                    gender="?"
                ))

        return face_results

    # ── Tek fotoğraftan embedding ─────────────────────────────────────────────
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        Yüzü içeren bir fotoğraftan embedding çıkar.
        Kişi ekleme sayfasında kullanılır.

        Args:
            face_image: BGR formatında tam fotoğraf (yüz kırpık olmak zorunda değil)
        Returns:
            512-d numpy array veya None
        """
        faces = self.detect_faces(face_image)
        if not faces:
            return None
        # En yüksek güven skoruna sahip yüzü seç
        best = max(faces, key=lambda f: f.det_score)
        return best.embedding

    # ── Karşılaştırma ─────────────────────────────────────────────────────────
    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity (normed vektörler için dot product yeterli)."""
        dot   = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))

    def is_match(self, emb1: np.ndarray, emb2: np.ndarray,
                 custom_threshold: float = None) -> bool:
        thresh = custom_threshold if custom_threshold is not None else self.threshold
        return self.compare(emb1, emb2) >= thresh
