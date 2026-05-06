"""
SKYWATCH — Yüz Analiz Modülü (Hibrit Mimari)

Algılama  → YOLO best.pt (ultralytics)
Embedding → InsightFace / ArcFace / buffalo_l (gerçek yüz kimliği)

InsightFace yoksa DCT fallback kullanılır (düşük doğruluk uyarısı verilir).
"""

import numpy as np
import cv2
from pathlib import Path
import warnings
import os
import torch

# nvidia-cudnn-cu12 pip paketinden cuDNN DLL'lerini PATH'e ekle
try:
    import nvidia.cudnn
    _cudnn_bin = Path(nvidia.cudnn.__path__[0]) / "bin"
    if _cudnn_bin.exists() and str(_cudnn_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_cudnn_bin) + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

from .models import FaceResult

# ── Sabitler ─────────────────────────────────────────────────────────────────
_EMBED_SIZE  = 64          # DCT fallback crop boyutu
_EMBED_DIM   = 512         # çıkış embedding boyutu
_MODEL_PATH  = Path(__file__).resolve().parents[3] / "best.pt"
_INSIGHTFACE_MODEL = "buffalo_l"


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


def _dct_embedding_fallback(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback: DCT tabanlı embedding (InsightFace yokken).
    DİKKAT: Bu yöntem yüz kimliği tanımak için yetersizdir.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (_EMBED_SIZE, _EMBED_SIZE),
                         interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(resized).astype(np.float32) / 255.0
    dct = cv2.dct(eq)
    flat = dct.flatten()[:_EMBED_DIM]
    if len(flat) < _EMBED_DIM:
        flat = np.pad(flat, (0, _EMBED_DIM - len(flat)))
    norm = np.linalg.norm(flat)
    if norm > 1e-6:
        flat = flat / norm
    return flat.astype(np.float32)


class FaceAnalyzer:
    """
    YOLO best.pt tabanlı yüz algılama + InsightFace ArcFace embedding.

    Algılama: YOLO best.pt → bbox
    Embedding: InsightFace buffalo_l → 512-d ArcFace vektör
    """

    def __init__(self, config=None):
        from ultralytics import YOLO

        self._app_cfg = config
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
        self._gpu_cfg = getattr(config, "get", lambda *_: {})("gpu", {}) if config is not None else {}
        self.device = "cpu"
        self.torch_cuda_available = bool(torch.cuda.is_available())
        self.torch_version = getattr(torch, "__version__", "unknown")
        self.torch_version_cuda = getattr(torch.version, "cuda", None)
        self.gpu_name = torch.cuda.get_device_name(0) if self.torch_cuda_available else "cpu"
        self._resolve_device()

        model_path = _MODEL_PATH
        if not model_path.exists():
            # Fallback: çalışma dizininde ara
            fallback = Path("best.pt")
            if fallback.exists():
                model_path = fallback

        self._yolo = YOLO(str(model_path))
        self._yolo.overrides['verbose'] = False
        self.yolo_model_path = str(model_path)
        try:
            self._yolo.to(self.device)
        except Exception as e:
            if self.device.startswith("cuda"):
                if self._gpu_cfg.get("allow_cpu_fallback", False):
                    warnings.warn(f"[SKYWATCH] YOLO cuda transfer failed, fallback to CPU: {e}")
                    self.device = "cpu"
                    self._yolo.to(self.device)
                else:
                    raise RuntimeError(f"YOLO could not be moved to {self.device}: {e}") from e
            else:
                raise
        self.yolo_device = self.device

        # ── InsightFace ArcFace Embedding Modeli ─────────────────────────
        self._insightface_app = None
        self._use_insightface = True
        self.insightface_providers_requested = []
        self.insightface_ctx_id = 0
        self.insightface_device_mode = "CPU"

        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            requested = self._gpu_cfg.get("insightface_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
            if not isinstance(requested, list) or not requested:
                requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in requested if p in available_providers]
            if not providers:
                providers = ['CPUExecutionProvider']
            self.insightface_providers_requested = requested
            self.insightface_device_mode = "CUDA" if providers and providers[0] == "CUDAExecutionProvider" else "CPU"
            if self.device.startswith("cuda") and providers[0] != "CUDAExecutionProvider":
                warnings.warn("[SKYWATCH] InsightFace CUDA provider unavailable. Falling back to CPUExecutionProvider.")
            if providers[0] == "CUDAExecutionProvider":
                print("[SKYWATCH] InsightFace CUDA modu seçildi")
            else:
                print("[SKYWATCH] CUDA bulunamadı → InsightFace CPU modunda çalışacak")

            app = FaceAnalysis(
                name=_INSIGHTFACE_MODEL,
                providers=providers
            )
            self.insightface_ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
            app.prepare(ctx_id=self.insightface_ctx_id, det_size=(160, 160))
            self._insightface_app = app
            print(f"[SKYWATCH] InsightFace ArcFace yüklendi ✓ ({providers[0]})")
        except Exception as e:
            self._use_insightface = False
            warnings.warn(
                f"[SKYWATCH] InsightFace yüklenemedi: {e}\n"
                f"  → Embedding üretilemeyecek, yüz tanıma devre dışı!\n"
                f"  → Düzeltmek için: pip install insightface onnxruntime-gpu"
            )
        if self.torch_cuda_available and self.yolo_device.startswith("cpu"):
            warnings.warn("CUDA is available but YOLO appears to be running on CPU.")

    def _resolve_device(self):
        gpu_enabled = bool(self._gpu_cfg.get("enabled", True))
        require_cuda = bool(self._gpu_cfg.get("require_cuda", False))
        allow_cpu_fallback = bool(self._gpu_cfg.get("allow_cpu_fallback", True))
        requested_device = str(self._gpu_cfg.get("device", "cuda:0"))

        if not gpu_enabled:
            self.device = "cpu"
            return

        if self.torch_cuda_available:
            self.device = requested_device or "cuda:0"
            return

        if require_cuda and not allow_cpu_fallback:
            raise RuntimeError("CUDA is required by config.gpu.require_cuda=true but torch.cuda.is_available() is false.")
        if require_cuda and allow_cpu_fallback:
            warnings.warn("[SKYWATCH] CUDA required but unavailable. Falling back to CPU.")
        self.device = "cpu"

    # ── InsightFace Embedding ─────────────────────────────────────────────
    def _insightface_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        InsightFace ArcFace ile yüz crop'undan embedding çıkar.
        InsightFace kendi algılamasını yapar → en büyük yüzün embedding'ini döner.
        """
        if self._insightface_app is None:
            return None

        try:
            # InsightFace 112x112 hizalanmış yüz bekler, ama
            # app.get() kendi algılama + hizalama yapıyor
            faces = self._insightface_app.get(crop_bgr)
            if not faces:
                return None

            # En büyük yüzü seç
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            emb = best.embedding  # 512-d ArcFace vektör
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            return emb.astype(np.float32)
        except Exception:
            return None

    # ── Algılama ─────────────────────────────────────────────────────────────
    def detect_faces(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Frame içindeki yüzleri/kişileri tespit et ve embedding üret.

        Algılama: YOLO best.pt
        Embedding: InsightFace ArcFace (varsa) veya DCT fallback

        Args:
            frame: BGR formatında OpenCV görüntüsü
        Returns:
            list[FaceResult]
        """
        h, w = frame.shape[:2]
        detections: list[tuple[int, int, int, int, float]] = []

        def collect_from(img: np.ndarray, ox: int = 0, oy: int = 0, conf: float | None = None):
            run_conf = self._conf if conf is None else conf
            results = self._yolo.predict(
                img,
                conf=run_conf,
                device=self.device,
                verbose=False,
            )
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

            # Yüz çevresinde biraz padding bırak (InsightFace daha iyi çalışır)
            pad = max(8, int(min(bw, bh) * 0.15))
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # InsightFace varsa ArcFace embedding — yoksa bu yüzü atla
            if self._use_insightface:
                emb = self._insightface_embedding(crop)
                if emb is None:
                    # InsightFace crop'ta yüz bulamadı → güvenilir embedding yok, atla
                    continue
            else:
                # InsightFace yüklü değil → DB karşılaştırma güvenilir değil, atla
                continue

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
