"""
SKYWATCH — Tracker (DeepSORT Wrapper)
Kameradaki kişilere sabit ID atar ve frameler arası takip sağlar.
Aynı kişi için DB'nin tekrar tekrar sorgulanmasını engeller.
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.models import FaceResult, Track


class Tracker:
    """Her kamera için ayrı bir DeepSORT instance'ı yönetir."""

    def __init__(self, config: dict):
        self.max_age = config.get("max_age", 30)
        self.min_hits = config.get("min_hits", 3)
        self.iou_threshold = config.get("iou_threshold", 0.3)

        # Kamera ID → DeepSort instance
        self._trackers: dict[str, DeepSort] = {}

        # Track ID → face_embedding (DB sorgusu sadece 1 kez yapılsın diye)
        self._known_embeddings: dict[int, np.ndarray] = {}

        # Track ID → ilk kez mi görüldüğü (is_new flag)
        self._seen_ids: set[int] = set()

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

    def update(self, camera_id: str, faces: list[FaceResult], frame: np.ndarray) -> list[Track]:
        """
        Yeni frame'deki yüzleri tracker'a gönderir ve
        güncellenmiş Track listesi döndürür.

        Args:
            camera_id: Hangi kameranın frame'i
            faces: FaceAnalyzer'dan gelen yüz listesi
            frame: Orijinal BGR frame (DeepSort'un ihtiyacı var)

        Returns:
            list[Track]: Güncellenmiş track'ler (is_new, bbox, embedding vb.)
        """
        tracker = self._get_or_create(camera_id)

        # DeepSort formatına çevir: [([x1, y1, w, h], confidence, class), ...]
        detections = []
        det_embeddings = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], face.det_score, "person"))
            det_embeddings.append(face.embedding)

        # Embedding'leri numpy dizisine çevir
        # ÖNEMLİ: deep_sort_realtime, embeds=None iken embedder=None olunca
        # detections boş bile olsa hata fırlatır. Bu yüzden HER ZAMAN array gönderiyoruz.
        if det_embeddings:
            embeds = np.array(det_embeddings)
        else:
            embeds = np.zeros((0, 512))  # Boş ama geçerli shape

        # DeepSort güncelle
        raw_tracks = tracker.update_tracks(
            detections,
            frame=frame,
            embeds=embeds
        )

        # Track nesnelerine dönüştür
        results: list[Track] = []

        # Detection → embedding eşleştirmesi için dict oluştur
        # (DeepSort track'e atanan detection'ın indexini verir)
        for rt in raw_tracks:
            if not rt.is_confirmed():
                continue

            track_id = rt.track_id

            # Bbox [x1, y1, x2, y2] formatına çevir
            ltrb = rt.to_ltrb()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]

            # İlk kez mi görülüyor?
            is_new = track_id not in self._seen_ids
            if is_new:
                self._seen_ids.add(track_id)

            # Embedding'i kaydet — orijinal detection'dan al
            if is_new and det_embeddings:
                # Yeni track için en yakın detection'ın embedding'ini bul
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
                camera_id=camera_id
            )
            results.append(track)

        return results

    def _find_closest_embedding(self, track_bbox: list[int], faces: list[FaceResult]) -> np.ndarray | None:
        """Track bbox'ına en yakın detection'ın embedding'ini bulur."""
        if not faces:
            return None

        # Track merkezi
        tcx = (track_bbox[0] + track_bbox[2]) / 2
        tcy = (track_bbox[1] + track_bbox[3]) / 2

        best_dist = float('inf')
        best_emb = None

        for face in faces:
            fcx = (face.bbox[0] + face.bbox[2]) / 2
            fcy = (face.bbox[1] + face.bbox[3]) / 2
            dist = (tcx - fcx) ** 2 + (tcy - fcy) ** 2

            if dist < best_dist:
                best_dist = dist
                best_emb = face.embedding

        return best_emb

    def get_active_count(self, camera_id: str) -> int:
        """Kameradaki aktif track sayısı."""
        if camera_id not in self._trackers:
            return 0
        tracker = self._trackers[camera_id]
        return len([t for t in tracker.tracker.tracks if t.is_confirmed()])

    def reset(self, camera_id: str = None):
        """Tracker'ı sıfırla. camera_id verilmezse tümünü sıfırla."""
        if camera_id:
            self._trackers.pop(camera_id, None)
        else:
            self._trackers.clear()
            self._known_embeddings.clear()
            self._seen_ids.clear()
