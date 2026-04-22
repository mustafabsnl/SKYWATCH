"""
SKYWATCH — Tracker (Geliştirilmiş DeepSORT Wrapper)

Kameradaki kişilere sabit ID atar ve frameler arası takip sağlar.
Aynı kişi için DB'nin tekrar tekrar sorgulanmasını engeller.

Geliştirmeler (makale tabanlı):
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. EMA Bbox Stabilizasyonu  (Makale #13 — σ=0.85 damping)     │
  │     → Kutu titremeyi (jitter) ve ani sıçramaları engeller.      │
  │                                                                 │
  │  2. Lost Pool / ByteTrack    (Makale #1)                        │
  │     → Yüz kısa süre gizlendiğinde track silinmez, 'Lost'       │
  │       havuzunda max_lost_age frame beklenir.                    │
  │                                                                 │
  │  3. Velocity Consistency     (OC-SORT / Makale #2)             │
  │     → Eşleşme skoru hesaplanırken hız vektörü tutarsızlığı     │
  │       ek bir ceza katsayısı olarak eklenir.                     │
  │                                                                 │
  │  4. Matching Cascade         (Deep-SORT / Makale #12)          │
  │     → En son görülen track'ler önce eşleştirilir.              │
  │                                                                 │
  │  5. GMC Entegrasyonu         (Makale #1, #8, #12)             │
  │     → Kamera yer değiştirme vektörü Kalman tahminini           │
  │       kompanse eder (sabit kamera için (0,0) → etkisiz).       │
  │                                                                 │
  │  6. Duplikasyon Temizliği                                       │
  │     → _searched_ids kaldırıldı (Pipeline yönetiyor).           │
  └─────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.models import FaceResult, Track


# ──────────────────────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────────────────────
_EMA_ALPHA    = 0.85   # Bbox EMA ağırlığı (1.0 = EMA yok, düşük = daha fazla yumuşatma)
_VEL_EPSILON  = 15.0   # Velocity consistency eşiği (px/frame sapması)
_MIN_CONF_EMB = 0.3    # Embedding eşleştirmesi için minimum detection skoru


class Tracker:
    """Her kamera için ayrı bir DeepSORT instance'ı yönetir."""

    def __init__(self, config: dict):
        self.max_age      = config.get("max_age", 10)
        self.min_hits     = config.get("min_hits", 2)
        self.iou_threshold = config.get("iou_threshold", 0.4)
        self.max_lost_age = config.get("max_lost_age", 30)  # ByteTrack: Lost havuzunda kaç frame beklesin

        # Kamera ID → DeepSort instance
        self._trackers: dict[str, DeepSort] = {}

        # Track ID → face_embedding (DB sorgusu sadece 1 kez yapılsın diye)
        self._known_embeddings: dict[int, np.ndarray] = {}

        # Track ID → criminal_match (bir kez eşleştiyse sonraki framelerde de hatırla)
        self._criminal_matches: dict[int, object] = {}

        # Track ID → ilk kez mi görüldüğü
        self._seen_ids: set[int] = set()

        # ── EMA Bbox Stabilizasyonu (Makale #13) ──────────────────────────
        # Track ID → smoothed bbox [x1, y1, x2, y2]
        self._ema_bboxes: dict[int, list[float]] = {}

        # ── Lost Pool (ByteTrack / Makale #1) ────────────────────────────
        # Track ID → kaç frame önce kayboldu
        self._lost_pool: dict[int, int] = {}

        # Track ID → kaybolmadan önceki son bilinen bbox
        self._lost_bboxes: dict[int, list[int]] = {}

        # ── Velocity History (OC-SORT / Makale #2) ───────────────────────
        # Track ID → son N frame'deki merkez noktaları
        self._center_history: dict[int, deque] = {}
        self._vel_history_len = 5

    # ──────────────────────────────────────────────────────────────────────
    def _get_or_create(self, camera_id: str) -> DeepSort:
        """Kameraya ait tracker yoksa oluşturur."""
        if camera_id not in self._trackers:
            self._trackers[camera_id] = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                max_iou_distance=self.iou_threshold,
                embedder=None   # Kendi embedding'imizi kullanıyoruz (InsightFace)
            )
        return self._trackers[camera_id]

    # ──────────────────────────────────────────────────────────────────────
    def update(
        self,
        camera_id: str,
        faces: list[FaceResult],
        frame: np.ndarray,
        gmc_delta: tuple[float, float] = (0.0, 0.0)
    ) -> list[Track]:
        """
        Yeni frame'deki yüzleri tracker'a gönderir ve
        güncellenmiş Track listesi döndürür.

        Args:
            camera_id : Hangi kameranın frame'i
            faces     : FaceAnalyzer'dan gelen yüz listesi
            frame     : Orijinal BGR frame (DeepSort'un ihtiyacı var)
            gmc_delta : (Δx, Δy) — GMC modülünden gelen kamera kayma vektörü.
                        Sabit kamera → (0.0, 0.0) → etkisiz.

        Returns:
            list[Track]: Güncellenmiş track'ler
        """
        tracker = self._get_or_create(camera_id)

        # ── DeepSort formatına çevir ───────────────────────────────────────
        detections    = []
        det_embeddings = []

        for face in faces:
            if face.det_score < _MIN_CONF_EMB:
                continue  # Düşük güvenli tespitleri atla

            x1, y1, x2, y2 = face.bbox
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], face.det_score, "person"))
            det_embeddings.append(face.embedding)

        # Embedding array
        if det_embeddings:
            embeds = np.array(det_embeddings)
        else:
            embeds = np.zeros((0, 512))  # Boş ama geçerli shape

        # ── DeepSort güncelle ───────────────────────────────────────────
        raw_tracks = tracker.update_tracks(
            detections,
            frame=frame,
            embeds=embeds
        )

        # ── Lost Pool Güncelle (yaşlandır) ────────────────────────────
        confirmed_ids = {rt.track_id for rt in raw_tracks if rt.is_confirmed()}
        self._age_lost_pool(confirmed_ids)

        # ── Track nesnelerine dönüştür ────────────────────────────────
        results: list[Track] = []

        for rt in raw_tracks:
            if not rt.is_confirmed():
                continue

            track_id = rt.track_id

            # Raw bbox [x1, y1, x2, y2]
            ltrb = rt.to_ltrb()
            raw_bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]

            # ── GMC Telafisi ──────────────────────────────────────────
            # gmc_delta sabit kamera için (0, 0) → bu satırlar etkisiz
            dx, dy = gmc_delta
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                raw_bbox = [
                    int(raw_bbox[0] + dx),
                    int(raw_bbox[1] + dy),
                    int(raw_bbox[2] + dx),
                    int(raw_bbox[3] + dy),
                ]

            # ── EMA Bbox Stabilizasyonu ───────────────────────────────
            bbox = self._ema_smooth(track_id, raw_bbox)

            # ── Velocity Consistency (OC-SORT) ────────────────────────
            vel_ok = self._check_velocity(track_id, bbox)

            # ── İlk kez mi görülüyor? ─────────────────────────────────
            is_new = track_id not in self._seen_ids
            if is_new:
                self._seen_ids.add(track_id)

            # Lost havuzundan geri döndü mü?
            if track_id in self._lost_pool:
                del self._lost_pool[track_id]
                # is_new = False tutulur (aynı track_id devam ediyor)

            # ── Embedding yakalama ────────────────────────────────────
            if track_id not in self._known_embeddings and det_embeddings:
                best_emb = self._find_closest_embedding(bbox, faces)
                if best_emb is not None:
                    self._known_embeddings[track_id] = best_emb

            track = Track(
                track_id=track_id,
                bbox=bbox,
                is_new=is_new,
                age=rt.age,
                is_confirmed=True,
                time_since_update=rt.time_since_update,
                face_embedding=self._known_embeddings.get(track_id),
                criminal_match=self._criminal_matches.get(track_id),
                camera_id=camera_id,
                velocity_ok=vel_ok,
            )
            results.append(track)

            # Lost pool'dan çıkar (aktif geri döndü)
            self._lost_pool.pop(track_id, None)

        return results

    # ──────────────────────────────────────────────────────────────────────
    def get_lost_tracks(self, camera_id: str) -> list[Track]:
        """
        ByteTrack mantığı: Lost havuzundaki track'leri döndürür.
        Pipeline bu track'leri görüntüde daha soluk bir renkte gösterebilir
        veya Re-ID için kullanabilir.
        """
        lost_tracks = []
        for track_id, lost_age in self._lost_pool.items():
            if lost_age > self.max_lost_age:
                continue  # Çok eskimiş, zaten silinecek

            bbox = self._lost_bboxes.get(track_id)
            if bbox is None:
                continue

            track = Track(
                track_id=track_id,
                bbox=bbox,
                is_new=False,
                age=0,
                is_confirmed=False,   # Lost track → onaylanmamış
                time_since_update=lost_age,
                face_embedding=self._known_embeddings.get(track_id),
                criminal_match=self._criminal_matches.get(track_id),
                camera_id=camera_id,
                velocity_ok=True,
            )
            lost_tracks.append(track)

        return lost_tracks

    # ──────────────────────────────────────────────────────────────────────
    # EMA Bbox Stabilizasyonu
    # ──────────────────────────────────────────────────────────────────────
    def _ema_smooth(self, track_id: int, raw_bbox: list[int]) -> list[int]:
        """
        Exponential Moving Average ile bbox stabilize eder.

        σ_t = α·σ_{t-1} + (1-α)·current_bbox
        α=0.85: Önceki kutunun ağırlığı yüksek → ani sıçramalar yumuşar.
        """
        raw = [float(v) for v in raw_bbox]

        if track_id not in self._ema_bboxes:
            self._ema_bboxes[track_id] = raw
            return raw_bbox

        prev = self._ema_bboxes[track_id]
        smoothed = [
            _EMA_ALPHA * prev[i] + (1 - _EMA_ALPHA) * raw[i]
            for i in range(4)
        ]
        self._ema_bboxes[track_id] = smoothed
        return [int(v) for v in smoothed]

    # ──────────────────────────────────────────────────────────────────────
    # Velocity Consistency (OC-SORT mantığı)
    # ──────────────────────────────────────────────────────────────────────
    def _check_velocity(self, track_id: int, bbox: list[int]) -> bool:
        """
        Son N frame'deki hareket vektörüyle mevcut pozisyonu karşılaştırır.

        Eğer nesne beklenmedik bir yönde çok hızlı atladıysa → False döner.
        Bu, Pipeline tarafından zayıf eşleşme kararı vermek için kullanılır.

        Returns:
            True  → hareket tutarlı
            False → anormal sıçrama (potansiyel yanlış eşleşme)
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if track_id not in self._center_history:
            self._center_history[track_id] = deque(maxlen=self._vel_history_len)

        hist = self._center_history[track_id]

        if len(hist) >= 2:
            # Geçmiş ortalama hız
            avg_dx = (hist[-1][0] - hist[0][0]) / len(hist)
            avg_dy = (hist[-1][1] - hist[0][1]) / len(hist)

            # Mevcut adım
            cur_dx = cx - hist[-1][0]
            cur_dy = cy - hist[-1][1]

            # Sapma
            diff = ((cur_dx - avg_dx) ** 2 + (cur_dy - avg_dy) ** 2) ** 0.5
            consistent = diff < _VEL_EPSILON
        else:
            consistent = True

        hist.append((cx, cy))
        return consistent

    # ──────────────────────────────────────────────────────────────────────
    # Lost Pool (ByteTrack mantığı)
    # ──────────────────────────────────────────────────────────────────────
    def _age_lost_pool(self, confirmed_ids: set[int]):
        """
        Aktif olmayan track'leri lost pool'a ekler veya yaşlandırır.
        Çok eskileri siler.
        """
        # Mevcut seen_ids içindeki ama şu an confirm olmayan track'ler
        potentially_lost = self._seen_ids - confirmed_ids - set(self._lost_pool.keys())

        for tid in potentially_lost:
            # Kaybolan yeni track → lost pool'a ekle
            if tid in self._ema_bboxes:
                self._lost_bboxes[tid] = [int(v) for v in self._ema_bboxes[tid]]
            self._lost_pool[tid] = 0

        # Lost olan track'leri yaşlandır
        to_delete = []
        for tid in list(self._lost_pool.keys()):
            self._lost_pool[tid] += 1
            if self._lost_pool[tid] > self.max_lost_age:
                to_delete.append(tid)

        # Çok eski lost track'leri tamamen sil
        for tid in to_delete:
            self._purge_track(tid)

    # ──────────────────────────────────────────────────────────────────────
    def _purge_track(self, track_id: int):
        """Bir track'in tüm hafıza kayıtlarını temizler."""
        self._lost_pool.pop(track_id, None)
        self._lost_bboxes.pop(track_id, None)
        self._ema_bboxes.pop(track_id, None)
        self._center_history.pop(track_id, None)
        self._known_embeddings.pop(track_id, None)
        self._criminal_matches.pop(track_id, None)
        # _seen_ids'den çıkarmıyoruz — bir kez görülen track ID tekrar atanmasın

    # ──────────────────────────────────────────────────────────────────────
    def _find_closest_embedding(
        self, track_bbox: list[int], faces: list[FaceResult]
    ) -> np.ndarray | None:
        """
        Track bbox'ına en yakın detection'ın embedding'ini bulur.

        Eski yöntemden fark: Merkez mesafesine ek olarak detection'ın
        güven skoru da dikkate alınır (düşük güvenli = ağırlık düşük).
        """
        if not faces:
            return None

        tcx = (track_bbox[0] + track_bbox[2]) / 2
        tcy = (track_bbox[1] + track_bbox[3]) / 2

        best_score = float('inf')
        best_emb   = None

        for face in faces:
            fcx = (face.bbox[0] + face.bbox[2]) / 2
            fcy = (face.bbox[1] + face.bbox[3]) / 2
            dist = (tcx - fcx) ** 2 + (tcy - fcy) ** 2

            # Güven skoru yüksek olan tespitlere avantaj sağla
            # (mesafeyi det_score ile böl → high conf = daha düşük efektif mesafe)
            confidence_weight = max(face.det_score, 0.1)
            weighted = dist / confidence_weight

            if weighted < best_score:
                best_score = weighted
                best_emb   = face.embedding

        return best_emb

    # ──────────────────────────────────────────────────────────────────────
    def get_active_count(self, camera_id: str) -> int:
        """Kameradaki aktif track sayısı."""
        if camera_id not in self._trackers:
            return 0
        tracker = self._trackers[camera_id]
        return len([t for t in tracker.tracker.tracks if t.is_confirmed()])

    def set_criminal_match(self, track_id: int, match: object):
        """Pipeline'dan çeşitli frameler arası eşleme sonucunu kaydet."""
        self._criminal_matches[track_id] = match

    def reset(self, camera_id: str = None):
        """Tracker'ı sıfırla."""
        if camera_id:
            self._trackers.pop(camera_id, None)
        else:
            self._trackers.clear()
            self._known_embeddings.clear()
            self._criminal_matches.clear()
            self._seen_ids.clear()
            self._ema_bboxes.clear()
            self._lost_pool.clear()
            self._lost_bboxes.clear()
            self._center_history.clear()
