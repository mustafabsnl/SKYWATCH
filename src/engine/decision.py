"""
SKYWATCH — Karar Motoru
Track verilerini ve DB eşleşmelerini alarak son kararı
(Sabıkalı, Temiz, Aranan vb.) ve ekranda gösterilecek rengi belirler.
"""

from core.models import Track, DecisionResult


class DecisionEngine:
    """Tüm analizlerin birleşip sonuçlandığı karar mekanizması."""

    def __init__(self, logger=None):
        self._logger = logger
        # Duruma göre BGR renk kodları (OpenCV için)
        self.colors = {
            "CLEAN": (0, 255, 0),        # Yeşil
            "CRIMINAL": (0, 204, 255),   # Sarı/Turuncu
            "WANTED": (0, 0, 255),       # Kırmızı
            "SUSPICIOUS": (255, 0, 255), # Mor (hızlı/şüpheli hareket)
            "UNKNOWN": (200, 200, 200),  # Gri
            "FACE": (0, 220, 220),
            "TENTATIVE": (180, 180, 0),
            "PREDICTED": (120, 180, 255),
            "TRACKING": (180, 180, 0),
            "TARGET_FOUND": (0, 255, 255),
            "HEDEF BULUNDU": (0, 255, 255),
            "TEMIZ": (0, 255, 0),
            "TEMİZ": (0, 255, 0),
        }
        self.current_mode = "GENERAL"
        self.target_person_id = None
        self.target_person_ids: frozenset[int] = frozenset()

    def set_mode(self, mode: str, options: dict = None):
        self.current_mode = str(mode or "GENERAL").upper()
        options = dict(options or {})
        ids_ordered: list[int] = []
        raw = options.get("target_person_ids")
        if isinstance(raw, (list, tuple)):
            for x in raw:
                try:
                    ids_ordered.append(int(x))
                except (TypeError, ValueError):
                    pass
        if not ids_ordered and options.get("target_person_id") is not None:
            try:
                ids_ordered.append(int(options["target_person_id"]))
            except (TypeError, ValueError):
                pass

        if self.current_mode == "PERSON_SEARCH":
            self.target_person_ids = frozenset(ids_ordered)
            self.target_person_id = ids_ordered[0] if ids_ordered else None
        else:
            self.target_person_ids = frozenset()
            self.target_person_id = None

    def evaluate(self, track: Track, criminal_info: dict | None = None) -> DecisionResult:
        """
        Track üzerindeki eşleşme (match) ve hareket (movement)
        verilerini değerlendirerek nihai kararı verir.
        """
        match_id = None
        try:
            if track.criminal_match is not None:
                match_id = int(track.criminal_match.criminal_id)
        except (TypeError, ValueError):
            match_id = None

        if self._logger:
            self._logger.info(
                f"[DECISION_INPUT] mode={self.current_mode} track_id={track.track_id} "
                f"match_id={match_id} target_ids={sorted(self.target_person_ids)}"
            )

        # Default state
        status = "UNKNOWN"
        danger_level = "LOW"
        color = self.colors["UNKNOWN"]

        # 1. Unconfirmed Tracks (Tracking/Predicted)
        if not track.is_confirmed:
            if track.source == "predicted":
                status = "PREDICTED"
                color = self.colors["PREDICTED"]
            elif track.source == "raw_fallback":
                status = "FACE"
                color = self.colors["FACE"]
            else:
                status = "TRACKING"
                color = self.colors["TRACKING"]

            res = DecisionResult(
                track_id=track.track_id,
                bbox=track.bbox.copy(),
                status=status,
                danger_level="LOW",
                color=color,
                criminal_id=None,
                confidence=0.0,
                behavior_label=track.movement.behavior_label,
                global_id=track.global_id,
                time_since_update=int(getattr(track, "time_since_update", 0) or 0),
            )
            if self._logger:
                self._logger.info(
                    f"[DECISION_OUTPUT] mode={self.current_mode} track_id={track.track_id} "
                    f"status={res.status} confidence={float(res.confidence):.4f}"
                )
            return res

        # 2. Confirmed Tracks
        if self.current_mode == "PERSON_SEARCH":
            if match_id is not None and match_id in self.target_person_ids:
                status = "HEDEF BULUNDU"
                color = self.colors.get("HEDEF BULUNDU", self.colors["TARGET_FOUND"])
                danger_level = "HIGH"
            else:
                status = "UNKNOWN"
                color = self.colors["UNKNOWN"]
                danger_level = "LOW"
        else:
            # GENERAL MODE: Database matching — tüm DB; durum criminal_info üzerinden
            if track.criminal_match is not None and criminal_info is not None:
                db_status = str(criminal_info.get("status", "CRIMINAL")).upper()
                danger_level = str(criminal_info.get("danger_level", "LOW")).upper()

                if db_status == "WANTED":
                    status = "WANTED"
                    color = self.colors["WANTED"]
                elif db_status == "CRIMINAL":
                    status = "CRIMINAL"
                    if danger_level in ("HIGH", "CRITICAL"):
                        color = self.colors["WANTED"]
                    else:
                        color = self.colors["CRIMINAL"]
                elif db_status in ("CLEARED", "CLEAN"):
                    status = "CLEAN"
                    color = self.colors["CLEAN"]
                else:
                    status = "UNKNOWN"
                    color = self.colors["UNKNOWN"]
            elif track.movement.behavior_score >= 0.60 or track.movement.behavior_label == "running":
                status = "SUSPICIOUS"
                color = self.colors["SUSPICIOUS"]
            else:
                status = "UNKNOWN"
                color = self.colors["UNKNOWN"]

        # 3. Quality Overrides (Velocity and Age) — onaylı hedef / aranan / sabıkalı korunur
        protected = ("HEDEF BULUNDU", "TARGET_FOUND", "WANTED", "CRIMINAL")
        if status not in protected:
            if not track.velocity_ok or track.age < 5:
                status = "UNKNOWN"
                color = self.colors["UNKNOWN"]
                danger_level = "LOW"

        conf = float(track.criminal_match.confidence) if track.criminal_match else 0.0
        res = DecisionResult(
            track_id=track.track_id,
            bbox=track.bbox.copy(),
            status=status,
            danger_level=danger_level,
            color=color,
            criminal_id=track.criminal_match.criminal_id if track.criminal_match else None,
            confidence=conf,
            behavior_label=track.movement.behavior_label,
            global_id=track.global_id,
            time_since_update=int(getattr(track, "time_since_update", 0) or 0),
        )
        if self._logger:
            self._logger.info(
                f"[DECISION_OUTPUT] mode={self.current_mode} track_id={track.track_id} "
                f"status={res.status} confidence={float(res.confidence):.4f}"
            )
        return res
