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
import time
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.models import FaceResult, Track


# ──────────────────────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────────────────────
_EMA_ALPHA    = 0.85   # Bbox EMA ağırlığı (1.0 = EMA yok, düşük = daha fazla yumuşatma)
_VEL_EPSILON  = 15.0   # Velocity consistency eşiği (px/frame sapması)
_MIN_CONF_EMB_DEFAULT = 0.18    # Embedding eşleştirmesi için minimum detection skoru


class Tracker:
    """Her kamera için ayrı bir DeepSORT instance'ı yönetir."""

    def __init__(self, config: dict):
        self.max_age      = config.get("max_age", 10)
        self.min_hits     = config.get("min_hits", 2)
        self.iou_threshold = config.get("iou_threshold", 0.4)
        self.max_lost_age = config.get("max_lost_age", 30)  # ByteTrack: Lost havuzunda kaç frame beklesin
        self.min_conf_emb = float(config.get("min_conf_emb", _MIN_CONF_EMB_DEFAULT))
        self.draw_tentative_tracks = bool(config.get("draw_tentative_tracks", True))
        self.render_stale_deepsort_tracks = bool(config.get("render_stale_deepsort_tracks", True))
        self.max_render_time_since_update = int(config.get("max_render_time_since_update", 0) or 0)
        self.allow_no_embedding_fallback = bool(config.get("allow_no_embedding_fallback", True))
        self.no_embedding_fallback_mode = str(config.get("no_embedding_fallback_mode", "raw"))
        self.bbox_prediction_enabled = bool(config.get("bbox_prediction_enabled", True))
        self.prediction_render_enabled = bool(config.get("prediction_render_enabled", False))
        self.prediction_update_existing_bbox = bool(config.get("prediction_update_existing_bbox", True))
        self.detection_snap_alpha = float(config.get("detection_snap_alpha", 1.0))
        self.max_prediction_gap_sec = float(config.get("max_prediction_gap_sec", 0.25))
        self.max_prediction_missed_frames = int(config.get("max_prediction_missed_frames", 2))
        self.keep_lost_tracks_for_sec = float(config.get("keep_lost_tracks_for_sec", 0.35))
        self.bbox_prediction_max_velocity_px_per_sec = float(config.get("bbox_prediction_max_velocity_px_per_sec", 500.0))
        self.bbox_prediction_max_shift_px = float(config.get("bbox_prediction_max_shift_px", 70.0))
        self.velocity_smoothing_alpha = float(config.get("velocity_smoothing_alpha", 0.6))
        self.reacquire_iou_threshold = float(config.get("reacquire_iou_threshold", 0.4))
        self.reacquire_center_distance_px = float(config.get("reacquire_center_distance_px", 45.0))
        self.reacquire_time_window_sec = float(config.get("reacquire_time_window_sec", 0.4))
        self.reacquire_min_score = float(config.get("reacquire_min_score", 0.4))
        # PREDICTED kutu confidence decay alt sınırı (bunun altındaki tahminler atılır)
        self.predicted_min_confidence = float(config.get("predicted_min_confidence", 0.35))

        # Kamera ID → DeepSort instance
        self._trackers: dict[str, DeepSort] = {}

        # (camera_id, track_id) → face_embedding
        self._known_embeddings: dict[tuple[str, int], np.ndarray] = {}

        # (camera_id, track_id) → criminal_match
        self._criminal_matches: dict[tuple[str, int], object] = {}

        # (camera_id, track_id) → ilk kez mi görüldüğü
        self._seen_ids: set[tuple[str, int]] = set()

        # ── EMA Bbox Stabilizasyonu (Makale #13) ──────────────────────────
        # (camera_id, track_id) → smoothed bbox [x1, y1, x2, y2]
        self._ema_bboxes: dict[tuple[str, int], list[float]] = {}

        # ── Lost Pool (ByteTrack / Makale #1) ────────────────────────────
        # (camera_id, track_id) → kaç frame önce kayboldu
        self._lost_pool: dict[tuple[str, int], int] = {}

        # (camera_id, track_id) → kaybolmadan önceki son bilinen bbox
        self._lost_bboxes: dict[tuple[str, int], list[int]] = {}

        # ── Velocity History (OC-SORT / Makale #2) ───────────────────────
        # (camera_id, track_id) → son N frame'deki merkez noktaları
        self._center_history: dict[tuple[str, int], deque] = {}
        self._vel_history_len = 5
        self.last_debug: dict[str, dict] = {}
        self._fallback_id_counter: dict[str, int] = {}
        self._track_motion_state: dict[tuple[str, int], dict] = {}
        self._fallback_tracks: dict[str, dict[int, dict]] = {}
        # Pipeline tarafından çağrılan predicted-track çizim limiti (kamera başına).
        # 0 → limit yok. config.debug.max_predicted_tracks_per_camera ile override edilir.
        self.max_predicted_tracks_per_camera = 0

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
        debug = {
            "tracker_input_faces": len(faces),
            "tracker_prepared_detections": 0,
            "tracker_rejected_no_bbox": 0,
            "tracker_rejected_bad_bbox": 0,
            "tracker_rejected_low_conf": 0,
            "tracker_rejected_no_embedding": 0,
            "tracker_raw_tracks": 0,
            "tracker_confirmed_tracks": 0,
            "tracker_tentative_tracks": 0,
            "tracker_output_tracks": 0,
            "tracker_stale_tracks_seen": 0,
            "tracker_stale_tracks_suppressed": 0,
            "tracker_renderable_tracks": 0,
            "tracker_time_since_update_gt0": 0,
            "tracker_backend": "deepsort",
            "tracker_embedding_cache_hits": 0,
            "tracker_embedding_cache_updates": 0,
            "tracker_embedding_match_failures": 0,
            "tracker_predicted_tracks": 0,
            "tracker_predicted_drawn": 0,
            "tracker_predicted_suppressed_by_config": 0,
            "tracker_predicted_dropped_by_age": 0,
            "tracker_predicted_dropped_by_confidence": 0,
            "tracker_predicted_dropped_by_limit": 0,
            "tracker_motion_state_dropped_by_cleanup": 0,
            "tracker_lost_tracks": 0,
            "tracker_reacquired_tracks": 0,
            "tracker_missed_tracks": 0,
            "tracker_id_switch_count": 0,
            "avg_prediction_age_sec": 0.0,
            "max_prediction_age_sec": 0.0,
            "fallback_id_reused": 0,
            "fallback_id_created": 0,
            "fallback_id_dropped": 0,
        }
        self.last_debug[camera_id] = debug

        # ── DeepSort formatına çevir ───────────────────────────────────────
        deepsort_detections = []
        deepsort_embeds = []
        deepsort_face_refs: list[FaceResult] = []
        no_embed_faces: list[FaceResult] = []
        fallback_tracks: list[Track] = []

        for face in faces:
            if face.det_score < self.min_conf_emb:
                debug["tracker_rejected_low_conf"] += 1
                continue  # Düşük güvenli tespitleri atla

            if face.bbox is None:
                debug["tracker_rejected_no_bbox"] += 1
                continue
            if not isinstance(face.bbox, (list, tuple)) or len(face.bbox) != 4:
                debug["tracker_rejected_bad_bbox"] += 1
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                debug["tracker_rejected_bad_bbox"] += 1
                continue
            if face.embedding is None:
                no_embed_faces.append(face)
                debug["tracker_rejected_no_embedding"] += 1
                continue
            deepsort_detections.append(([x1, y1, w, h], face.det_score, "face"))
            deepsort_embeds.append(face.embedding)
            deepsort_face_refs.append(face)
            debug["tracker_prepared_detections"] += 1

        # ── DeepSort güncelle ───────────────────────────────────────────
        raw_tracks = []
        if deepsort_detections:
            if len(deepsort_detections) != len(deepsort_embeds):
                debug["tracker_backend"] = "deepsort_embed_mismatch"
            else:
                try:
                    raw_tracks = tracker.update_tracks(
                        deepsort_detections,
                        frame=frame,
                        embeds=np.array(deepsort_embeds, dtype=np.float32)
                    )
                except Exception:
                    # DeepSORT hatası olsa bile fallback track akışı çalışmaya devam etsin
                    raw_tracks = []
                    debug["tracker_backend"] = "deepsort_failed_raw_fallback"
        debug["tracker_deepsort_called"] = bool(deepsort_detections)
        debug["tracker_deepsort_embed_count"] = len(deepsort_embeds)
        debug["tracker_no_embed_faces"] = len(no_embed_faces)
        debug["tracker_raw_tracks"] = len(raw_tracks)
        debug["tracker_confirmed_tracks"] = sum(1 for rt in raw_tracks if rt.is_confirmed())
        debug["tracker_tentative_tracks"] = max(0, len(raw_tracks) - debug["tracker_confirmed_tracks"])

        # ── Lost Pool Güncelle (yaşlandır) ────────────────────────────
        confirmed_ids = {(camera_id, rt.track_id) for rt in raw_tracks if rt.is_confirmed()}
        self._age_lost_pool(camera_id, confirmed_ids)

        # ── Track nesnelerine dönüştür ────────────────────────────────
        results: list[Track] = []

        seen_track_keys: set[tuple[str, int]] = set()
        for rt in raw_tracks:
            is_confirmed = rt.is_confirmed()
            if not is_confirmed and not self.draw_tentative_tracks:
                continue

            tsu = int(getattr(rt, "time_since_update", 0) or 0)
            if tsu > 0:
                debug["tracker_stale_tracks_seen"] += 1
                debug["tracker_time_since_update_gt0"] += 1
            if (not self.render_stale_deepsort_tracks) and tsu > self.max_render_time_since_update:
                debug["tracker_stale_tracks_suppressed"] += 1
                continue

            track_id = rt.track_id
            track_key = (camera_id, track_id)

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
            bbox = self._ema_smooth(track_key, raw_bbox)

            # ── Velocity Consistency (OC-SORT) ────────────────────────
            vel_ok = self._check_velocity(track_key, bbox)

            # ── İlk kez mi görülüyor? ─────────────────────────────────
            is_new = track_key not in self._seen_ids
            if is_new and is_confirmed:
                self._seen_ids.add(track_key)

            # Lost havuzundan geri döndü mü?
            if track_key in self._lost_pool:
                del self._lost_pool[track_key]
                # is_new = False tutulur (aynı track_id devam ediyor)

            # ── Embedding yakalama ────────────────────────────────────
            matched_embedding = self._match_embedding_for_track(bbox, deepsort_face_refs)
            if matched_embedding is not None:
                self._known_embeddings[track_key] = matched_embedding
                debug["tracker_embedding_cache_updates"] += 1
            elif track_key in self._known_embeddings:
                matched_embedding = self._known_embeddings.get(track_key)
                debug["tracker_embedding_cache_hits"] += 1
            else:
                matched_embedding = None
                debug["tracker_embedding_match_failures"] += 1

            track = Track(
                track_id=track_id,
                bbox=bbox,
                is_new=is_new,
                age=rt.age,
                is_confirmed=is_confirmed,
                time_since_update=tsu,
                face_embedding=matched_embedding,
                criminal_match=self._criminal_matches.get(track_key),
                camera_id=camera_id,
                velocity_ok=vel_ok,
                source="deepsort" if is_confirmed else "tracking",
                prediction_age_sec=0.0,
            )
            results.append(track)
            debug["tracker_renderable_tracks"] += 1
            seen_track_keys.add(track_key)
            self._update_motion_state(track_key, bbox, matched_embedding, face_score=1.0, source=track.source)

            # Lost pool'dan çıkar (aktif geri döndü)
            if is_confirmed:
                self._lost_pool.pop(track_key, None)

        if self.allow_no_embedding_fallback:
            now = time.time()
            cam_fallback = self._fallback_tracks.setdefault(camera_id, {})
            # Aynı update içinde aynı fallback ID'ye iki face atanmasın
            used_fallback_ids: set[int] = set()
            for face in no_embed_faces:
                if face.bbox is None or len(face.bbox) != 4:
                    continue
                fb_bbox = [int(v) for v in face.bbox]
                best_id = None
                best_score = 0.0
                for fid, state in cam_fallback.items():
                    if fid in used_fallback_ids:
                        continue
                    last_bbox = state.get("bbox")
                    if last_bbox is None:
                        continue
                    iou = self._bbox_iou([float(v) for v in fb_bbox], [float(v) for v in last_bbox])
                    cd = self._center_distance(fb_bbox, last_bbox)
                    # Aynı yüz iki çağrı arasında küçük kayma yapar; reuse koşulu:
                    # IoU > reacquire_iou_threshold (0.35) VEYA merkez mesafesi < reacquire_center_distance_px (55px)
                    if iou > self.reacquire_iou_threshold or cd < self.reacquire_center_distance_px:
                        score = iou + max(0.0, (self.reacquire_center_distance_px - cd) / max(1.0, self.reacquire_center_distance_px))
                        if score >= self.reacquire_min_score and score > best_score:
                            best_score = score
                            best_id = fid
                if best_id is None:
                    cid = camera_id
                    self._fallback_id_counter[cid] = self._fallback_id_counter.get(cid, 0) + 1
                    best_id = self._fallback_id_counter[cid]
                    debug["fallback_id_created"] += 1
                else:
                    debug["fallback_id_reused"] += 1
                used_fallback_ids.add(best_id)
                cam_fallback[best_id] = {"bbox": fb_bbox, "last_seen_ts": now}
                track_key = (camera_id, -best_id)
                fallback_tracks.append(
                    Track(
                        track_id=-best_id,
                        bbox=fb_bbox,
                        is_new=True,
                        age=1,
                        is_confirmed=False,
                        time_since_update=0,
                        face_embedding=None,
                        criminal_match=None,
                        camera_id=camera_id,
                        velocity_ok=True,
                        source="raw_fallback",
                        prediction_age_sec=0.0,
                    )
                )
                seen_track_keys.add(track_key)
                self._update_motion_state(track_key, fb_bbox, None, face_score=float(face.det_score), source="raw_fallback")
            # Eski fallback track durumlarını temizle (TTL = keep_lost_tracks_for_sec)
            for fid in list(cam_fallback.keys()):
                if (camera_id, -fid) not in seen_track_keys:
                    if now - float(cam_fallback[fid].get("last_seen_ts", 0.0)) > self.keep_lost_tracks_for_sec:
                        del cam_fallback[fid]
                        debug["fallback_id_dropped"] += 1

        # missing trackler için kısa süreli prediction
        predicted_tracks, pred_ages = self._predict_missing_tracks(camera_id, seen_track_keys, frame.shape[:2], debug)
        debug["tracker_predicted_tracks"] = len(predicted_tracks)
        if self.prediction_render_enabled:
            results.extend(predicted_tracks)
            debug["tracker_predicted_drawn"] = len(predicted_tracks)
        else:
            debug["tracker_predicted_drawn"] = 0
            debug["tracker_predicted_suppressed_by_config"] = len(predicted_tracks)
        debug["tracker_missed_tracks"] = len(predicted_tracks)
        if pred_ages:
            debug["avg_prediction_age_sec"] = round(sum(pred_ages) / len(pred_ages), 3)
            debug["max_prediction_age_sec"] = round(max(pred_ages), 3)
        debug["tracker_lost_tracks"] = len([k for k, st in self._track_motion_state.items() if k[0] == camera_id and st.get("missed_frames", 0) > 0])
        debug["tracker_reacquired_tracks"] = len([k for k in seen_track_keys if self._track_motion_state.get(k, {}).get("missed_frames", 0) == 0 and self._track_motion_state.get(k, {}).get("prev_missed", 0) > 0])

        results.extend(fallback_tracks)
        debug["tracker_fallback_tracks"] = len(fallback_tracks)

        # Agresif motion state cleanup: yaş, missed_frames, frame dışı kontrolü
        self._cleanup_motion_state(camera_id, frame.shape[:2], debug)

        debug["tracker_output_tracks"] = len(results)
        return results

    # ──────────────────────────────────────────────────────────────────────
    def get_lost_tracks(self, camera_id: str) -> list[Track]:
        """
        ByteTrack mantığı: Lost havuzundaki track'leri döndürür.
        Pipeline bu track'leri görüntüde daha soluk bir renkte gösterebilir
        veya Re-ID için kullanabilir.
        """
        lost_tracks = []
        for (cid, track_id), lost_age in self._lost_pool.items():
            if cid != camera_id:
                continue
            if lost_age > self.max_lost_age:
                continue  # Çok eskimiş, zaten silinecek

            track_key = (cid, track_id)
            bbox = self._lost_bboxes.get(track_key)
            if bbox is None:
                continue

            track = Track(
                track_id=track_id,
                bbox=bbox,
                is_new=False,
                age=0,
                is_confirmed=False,   # Lost track → onaylanmamış
                time_since_update=lost_age,
                face_embedding=self._known_embeddings.get(track_key),
                criminal_match=self._criminal_matches.get(track_key),
                camera_id=camera_id,
                velocity_ok=True,
            )
            lost_tracks.append(track)

        return lost_tracks

    # ──────────────────────────────────────────────────────────────────────
    # EMA Bbox Stabilizasyonu
    # ──────────────────────────────────────────────────────────────────────
    def _ema_smooth(self, track_key: tuple[str, int], raw_bbox: list[int]) -> list[int]:
        """
        Exponential Moving Average ile bbox stabilize eder.

        σ_t = α·σ_{t-1} + (1-α)·current_bbox
        α=0.85: Önceki kutunun ağırlığı yüksek → ani sıçramalar yumuşar.
        """
        raw = [float(v) for v in raw_bbox]

        # detection_snap_alpha=1.0 => detection bbox'a anında otur (gecikme azaltma).
        # düşük değerlerde EMA davranışı sürer.
        alpha = max(0.0, min(1.0, float(self.detection_snap_alpha)))
        if alpha >= 0.999:
            self._ema_bboxes[track_key] = raw
            return raw_bbox

        if track_key not in self._ema_bboxes:
            self._ema_bboxes[track_key] = raw
            return raw_bbox

        prev = self._ema_bboxes[track_key]
        smoothed = [
            ((1.0 - alpha) * prev[i]) + (alpha * raw[i])
            for i in range(4)
        ]
        self._ema_bboxes[track_key] = smoothed
        return [int(v) for v in smoothed]

    # ──────────────────────────────────────────────────────────────────────
    # Velocity Consistency (OC-SORT mantığı)
    # ──────────────────────────────────────────────────────────────────────
    def _check_velocity(self, track_key: tuple[str, int], bbox: list[int]) -> bool:
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

        if track_key not in self._center_history:
            self._center_history[track_key] = deque(maxlen=self._vel_history_len)

        hist = self._center_history[track_key]

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
    def _age_lost_pool(self, camera_id: str, confirmed_ids: set[tuple[str, int]]):
        """
        Aktif olmayan track'leri lost pool'a ekler veya yaşlandırır.
        Çok eskileri siler.
        """
        # Mevcut seen_ids içindeki ama şu an confirm olmayan track'ler
        seen_for_camera = {k for k in self._seen_ids if k[0] == camera_id}
        lost_for_camera = {k for k in self._lost_pool.keys() if k[0] == camera_id}
        potentially_lost = seen_for_camera - confirmed_ids - lost_for_camera

        for track_key in potentially_lost:
            # Kaybolan yeni track → lost pool'a ekle
            if track_key in self._ema_bboxes:
                self._lost_bboxes[track_key] = [int(v) for v in self._ema_bboxes[track_key]]
            self._lost_pool[track_key] = 0

        # Lost olan track'leri yaşlandır
        to_delete = []
        for track_key in [k for k in self._lost_pool.keys() if k[0] == camera_id]:
            self._lost_pool[track_key] += 1
            if self._lost_pool[track_key] > self.max_lost_age:
                to_delete.append(track_key)

        # Çok eski lost track'leri tamamen sil
        for track_key in to_delete:
            self._purge_track(track_key)

    # ──────────────────────────────────────────────────────────────────────
    def _purge_track(self, track_key: tuple[str, int]):
        """Bir track'in tüm hafıza kayıtlarını temizler."""
        self._lost_pool.pop(track_key, None)
        self._lost_bboxes.pop(track_key, None)
        self._ema_bboxes.pop(track_key, None)
        self._center_history.pop(track_key, None)
        self._known_embeddings.pop(track_key, None)
        self._criminal_matches.pop(track_key, None)
        # _seen_ids'den çıkarmıyoruz — bir kez görülen track ID tekrar atanmasın

    # ──────────────────────────────────────────────────────────────────────
    def _xyxy_to_tlwh(self, bbox: list[int]) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def _tlwh_to_xyxy(self, tlwh: list[float]) -> list[float]:
        x, y, w, h = tlwh
        return [float(x), float(y), float(x + w), float(y + h)]

    def _bbox_iou(self, a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _center_distance(self, a: list[int], b: list[int]) -> float:
        acx = (a[0] + a[2]) / 2.0
        acy = (a[1] + a[3]) / 2.0
        bcx = (b[0] + b[2]) / 2.0
        bcy = (b[1] + b[3]) / 2.0
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)

    def _match_embedding_for_track(
        self,
        track_bbox_xyxy: list[int],
        face_refs: list[FaceResult],
    ) -> np.ndarray | None:
        if not face_refs:
            return None
        best_iou = 0.0
        best_emb = None
        track_box = [float(v) for v in track_bbox_xyxy]
        for face in face_refs:
            if face.embedding is None or face.bbox is None or len(face.bbox) != 4:
                continue
            face_box = [float(v) for v in face.bbox]
            iou = self._bbox_iou(track_box, face_box)
            if iou > best_iou:
                best_iou = iou
                best_emb = face.embedding
        if best_iou > 0.3:
            return best_emb
        return None

    def _clamp_bbox(self, bbox: list[float], frame_shape: tuple[int, int]) -> list[int] | None:
        h, w = frame_shape
        x1 = max(0, min(int(round(bbox[0])), w - 1))
        y1 = max(0, min(int(round(bbox[1])), h - 1))
        x2 = max(0, min(int(round(bbox[2])), w))
        y2 = max(0, min(int(round(bbox[3])), h))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _update_motion_state(self, track_key: tuple[str, int], bbox: list[int], embedding: np.ndarray | None, face_score: float, source: str):
        now = time.time()
        state = self._track_motion_state.get(track_key)
        if state is None:
            self._track_motion_state[track_key] = {
                "bbox": list(bbox),
                "prev_bbox": None,
                "last_seen_ts": now,
                "prev_seen_ts": None,
                "velocity": [0.0, 0.0, 0.0, 0.0],
                "missed_frames": 0,
                "prev_missed": 0,
                "confidence": float(face_score),
                "source": source,
                "embedding": embedding,
            }
            return
        prev_bbox = state.get("bbox")
        prev_ts = float(state.get("last_seen_ts", now))
        dt = max(1e-3, now - prev_ts)
        new_velocity = [
            (bbox[i] - prev_bbox[i]) / dt for i in range(4)
        ] if prev_bbox is not None else [0.0, 0.0, 0.0, 0.0]
        old_velocity = state.get("velocity", [0.0, 0.0, 0.0, 0.0])
        alpha = max(0.0, min(1.0, self.velocity_smoothing_alpha))
        smoothed_velocity = [
            alpha * new_velocity[i] + (1.0 - alpha) * old_velocity[i] for i in range(4)
        ]
        state["prev_bbox"] = prev_bbox
        state["bbox"] = list(bbox)
        state["prev_seen_ts"] = prev_ts
        state["last_seen_ts"] = now
        state["velocity"] = smoothed_velocity
        state["prev_missed"] = int(state.get("missed_frames", 0))
        state["missed_frames"] = 0
        state["confidence"] = float(face_score)
        state["source"] = source
        if embedding is not None:
            state["embedding"] = embedding

    def _predict_missing_tracks(
        self,
        camera_id: str,
        seen_track_keys: set[tuple[str, int]],
        frame_shape: tuple[int, int],
        debug: dict,
    ) -> tuple[list[Track], list[float]]:
        """
        Kayıp track'ler için kısa ömürlü tahmin üretir.
        - max_prediction_gap_sec / max_prediction_missed_frames aşılırsa tahmin yok.
        - Confidence decay: conf = base * (1 - age/max_gap), düşükse atılır.
        - Aynı kamerada en fazla `max_predicted_tracks_per_camera` adet tahmin döner
          (en yüksek confidence'tan başlanarak).
        """
        if not self.bbox_prediction_enabled:
            return [], []
        now = time.time()
        # (track_key, bbox, age_sec, missed, confidence, embedding) demetlerini topla
        candidates: list[tuple[tuple[str, int], list[int], float, int, float, np.ndarray | None]] = []

        for track_key, state in list(self._track_motion_state.items()):
            if track_key[0] != camera_id:
                continue
            if track_key in seen_track_keys:
                continue
            # Yalnızca daha önce DeepSORT/Confirmed kanalından gelen track'ler için tahmin üret
            # raw_fallback kaynaklı state'ler tahmine girmez (kısa ömürlü).
            if str(state.get("source", "")) == "raw_fallback":
                continue

            last_seen_ts = float(state.get("last_seen_ts", 0.0))
            age_sec = max(0.0, now - last_seen_ts)
            state["prev_missed"] = int(state.get("missed_frames", 0))
            state["missed_frames"] = int(state.get("missed_frames", 0)) + 1
            missed = state["missed_frames"]

            if age_sec > self.max_prediction_gap_sec or missed > self.max_prediction_missed_frames:
                debug["tracker_predicted_dropped_by_age"] += 1
                # Kuyruğu agresif temizle
                if age_sec > self.keep_lost_tracks_for_sec:
                    self._track_motion_state.pop(track_key, None)
                continue

            # Confidence decay
            base_conf = float(state.get("confidence", 1.0)) if state.get("confidence", 1.0) > 0 else 1.0
            decay_ratio = max(0.0, 1.0 - (age_sec / max(1e-3, self.max_prediction_gap_sec)))
            confidence = max(0.0, base_conf * decay_ratio)
            if confidence < self.predicted_min_confidence:
                debug["tracker_predicted_dropped_by_confidence"] += 1
                continue

            bbox = [float(v) for v in state.get("bbox", [0, 0, 0, 0])]
            vel = [float(v) for v in state.get("velocity", [0.0, 0.0, 0.0, 0.0])]
            vel = [max(-self.bbox_prediction_max_velocity_px_per_sec, min(self.bbox_prediction_max_velocity_px_per_sec, v)) for v in vel]
            shift = [v * age_sec for v in vel]
            shift = [max(-self.bbox_prediction_max_shift_px, min(self.bbox_prediction_max_shift_px, s)) for s in shift]
            pred_bbox = [bbox[i] + shift[i] for i in range(4)]
            clamped = self._clamp_bbox(pred_bbox, frame_shape)
            if clamped is None:
                # Frame dışına çıktı → state'i hemen sil
                self._track_motion_state.pop(track_key, None)
                debug["tracker_predicted_dropped_by_age"] += 1
                continue
            state["bbox"] = clamped
            state["last_pred_confidence"] = confidence
            candidates.append((track_key, clamped, age_sec, missed, confidence, state.get("embedding")))

        # Confidence'a göre sırala, sadece en iyi N tanesini tut
        candidates.sort(key=lambda c: c[4], reverse=True)
        limit = int(self.max_predicted_tracks_per_camera)
        if limit > 0 and len(candidates) > limit:
            dropped = len(candidates) - limit
            debug["tracker_predicted_dropped_by_limit"] += dropped
            candidates = candidates[:limit]

        predicted_tracks: list[Track] = []
        ages: list[float] = []
        for track_key, clamped, age_sec, missed, _, embedding in candidates:
            predicted_tracks.append(
                Track(
                    track_id=track_key[1],
                    bbox=clamped,
                    is_new=False,
                    age=0,
                    is_confirmed=False,
                    time_since_update=missed,
                    face_embedding=embedding,
                    criminal_match=self._criminal_matches.get(track_key),
                    camera_id=camera_id,
                    velocity_ok=True,
                    source="predicted",
                    prediction_age_sec=age_sec,
                )
            )
            ages.append(age_sec)
        return predicted_tracks, ages

    # ──────────────────────────────────────────────────────────────────────
    def _cleanup_motion_state(self, camera_id: str, frame_shape: tuple[int, int], debug: dict):
        """
        Update sonunda kamera bazlı motion state cleanup:
          - last_seen_ts üzerinden yaş > keep_lost_tracks_for_sec → sil
          - missed_frames > max_prediction_missed_frames → sil
          - bbox tamamen frame dışındaysa → sil
        """
        now = time.time()
        h, w = frame_shape
        dropped = 0
        for track_key in [k for k in self._track_motion_state.keys() if k[0] == camera_id]:
            state = self._track_motion_state.get(track_key)
            if state is None:
                continue
            age_sec = max(0.0, now - float(state.get("last_seen_ts", 0.0)))
            missed = int(state.get("missed_frames", 0))
            bbox = state.get("bbox") or []
            out_of_frame = False
            if len(bbox) == 4:
                bx1, by1, bx2, by2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                if bx2 <= 0 or by2 <= 0 or bx1 >= w or by1 >= h:
                    out_of_frame = True
            if (
                age_sec > self.keep_lost_tracks_for_sec
                or missed > self.max_prediction_missed_frames
                or out_of_frame
            ):
                self._track_motion_state.pop(track_key, None)
                self._purge_track(track_key)
                dropped += 1
        debug["tracker_motion_state_dropped_by_cleanup"] = dropped

    # ──────────────────────────────────────────────────────────────────────
    def get_active_count(self, camera_id: str) -> int:
        """Kameradaki aktif track sayısı."""
        if camera_id not in self._trackers:
            return 0
        tracker = self._trackers[camera_id]
        return len([t for t in tracker.tracker.tracks if t.is_confirmed()])

    def set_criminal_match(self, camera_id: str, track_id: int, match: object):
        """Pipeline'dan çeşitli frameler arası eşleme sonucunu kaydet."""
        self._criminal_matches[(camera_id, track_id)] = match

    def reset(self, camera_id: str = None):
        """Tracker'ı sıfırla."""
        if camera_id:
            self._trackers.pop(camera_id, None)
            for store in (
                self._known_embeddings,
                self._criminal_matches,
                self._ema_bboxes,
                self._lost_pool,
                self._lost_bboxes,
                self._center_history,
                self._track_motion_state,
            ):
                for key in list(store.keys()):
                    if key[0] == camera_id:
                        del store[key]
            self._fallback_tracks.pop(camera_id, None)
            self._fallback_id_counter.pop(camera_id, None)
            self._seen_ids = {k for k in self._seen_ids if k[0] != camera_id}
        else:
            self._trackers.clear()
            self._known_embeddings.clear()
            self._criminal_matches.clear()
            self._seen_ids.clear()
            self._ema_bboxes.clear()
            self._lost_pool.clear()
            self._lost_bboxes.clear()
            self._center_history.clear()
            self._track_motion_state.clear()
            self._fallback_tracks.clear()
            self._fallback_id_counter.clear()
