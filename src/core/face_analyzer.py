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


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


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
        self.min_size  = cfg.get('min_face_size', 18)
        self._conf     = cfg.get('det_conf', 0.22)
        self._conf_fallback = float(cfg.get('det_conf_fallback', 0.15))
        # Varsayilan kapali: coklu kamera senaryolarinda darbogaz olusmasin.
        self._tile_enabled = cfg.get('tile_inference_enabled', False)
        self._tile_splits = int(cfg.get('tile_vertical_splits', 3))
        self._tile_overlap = float(cfg.get('tile_overlap_ratio', 0.12))
        self._dedup_iou = float(cfg.get('tile_dedup_iou', 0.45))

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
        h, w = frame.shape[:2]
        detections: list[tuple[int, int, int, int, float]] = []

        def collect_from(img: np.ndarray, ox: int = 0, oy: int = 0, conf: float | None = None):
            run_conf = self._conf if conf is None else conf
            results = self._yolo(img, conf=run_conf, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 += ox; x2 += ox
                    y1 += oy; y2 += oy
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    detections.append((x1, y1, x2, y2, conf))

        # 1) Tam kare tespit
        collect_from(frame)

        # 2) Dikey tiling ile detay tarama (uzak/kucuk yuzler icin)
        if self._tile_enabled and self._tile_splits >= 2 and w >= 320:
            splits = max(2, self._tile_splits)
            tile_w = int(np.ceil(w / splits))
            overlap = int(tile_w * max(0.0, min(0.45, self._tile_overlap)))
            for i in range(splits):
                sx = max(0, i * tile_w - overlap)
                ex = min(w, (i + 1) * tile_w + overlap)
                if ex - sx < self.min_size:
                    continue
                tile = frame[:, sx:ex]
                collect_from(tile, ox=sx, oy=0)

        # 2.5) Hic kutu yoksa daha dusuk conf ile ikinci sans
        if not detections and self._conf_fallback < self._conf:
            collect_from(frame, conf=self._conf_fallback)

        # 3) NMS benzeri dedup (ayni yuze birden fazla kutu gelmesini azalt)
        detections.sort(key=lambda d: d[4], reverse=True)
        kept: list[tuple[int, int, int, int, float]] = []
        for det in detections:
            box = (det[0], det[1], det[2], det[3])
            if any(_iou_xyxy(box, (k[0], k[1], k[2], k[3])) >= self._dedup_iou for k in kept):
                continue
            kept.append(det)

        face_results: list[FaceResult] = []
        for x1, y1, x2, y2, conf in kept:
            bw = x2 - x1
            bh = y2 - y1
            if bw < self.min_size or bh < self.min_size:
                continue

            pad = 8
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            emb = _make_embedding(crop)
            face_results.append(FaceResult(
                bbox=[x1, y1, x2, y2],
                embedding=emb,
                det_score=conf,
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
