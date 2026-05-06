"""
SKYWATCH — Global Track Registry
---------------------------------
(camera_id, local_track_id) → global_id deterministik eşlemesini tutar.
Pipeline ömrü boyunca yaşar, çapraz-kamera track ID çakışmalarını engeller.

Ayrıca her track'in son Track nesnesini saklayarak Decision katmanına
global_id üzerinden erişim sağlar.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Set

from core.models import Track


@dataclass
class RegistryEntry:
    """Bir track'in son durumunu ve stabil global ID'sini saklar."""
    global_id: str
    track: Track
    stale_frames: int = 0  # Kaç frame'dir güncellenmedi


class TrackRegistry:
    """
    Kamera-spesifik Track ID'leri ile pipeline'ın geri kalanı arasında
    aracılık eden Mediator sınıfı.

    Her (camera_id, local_track_id) çifti ilk görüldüğünde
    f"{camera_id}_{track_id}" şeklinde stabil bir global_id atanır.
    """

    _MAX_STALE_FRAMES = 30  # Bu kadar frame güncellenmezse track silinir

    def __init__(self):
        self._store: Dict[Tuple[str, int], RegistryEntry] = {}
        self._active_keys_this_frame: Set[Tuple[str, int]] = set()

    # ── Public API ────────────────────────────────────────────────────────

    def register(self, track: Track, camera_id: str) -> str:
        """
        Track'i kayıt defterine ekler/günceller, stabil global_id döner.

        Args:
            track:     Güncel Track nesnesi
            camera_id: Hangi kameradan geldiği

        Returns:
            Stabil global_id  (ör: "CAM_01_7")
        """
        key = (camera_id, track.track_id)
        self._active_keys_this_frame.add(key)

        if key not in self._store:
            gid = f"{camera_id}-T{track.track_id}"
            self._store[key] = RegistryEntry(global_id=gid, track=track, stale_frames=0)
        else:
            entry = self._store[key]
            entry.track = track
            entry.stale_frames = 0

        return self._store[key].global_id

    def get(self, camera_id: str, track_id: int) -> Track | None:
        """Global registry'den track nesnesini çek."""
        entry = self._store.get((camera_id, track_id))
        return entry.track if entry is not None else None

    def begin_frame(self):
        """Her frame başında çağrılır — aktif key setini sıfırlar."""
        self._active_keys_this_frame.clear()

    def end_frame(self):
        """
        Her frame sonunda çağrılır.
        Bu frame'de güncellenmemiş entry'lerin stale sayacını artırır.
        _MAX_STALE_FRAMES'i aşanları siler.
        """
        to_delete = []
        for key, entry in self._store.items():
            if key not in self._active_keys_this_frame:
                entry.stale_frames += 1
                if entry.stale_frames > self._MAX_STALE_FRAMES:
                    to_delete.append(key)

        for key in to_delete:
            del self._store[key]

    def clear_camera(self, camera_id: str):
        """Belirli bir kameranın tüm track kayıtlarını siler."""
        to_delete = [k for k in self._store if k[0] == camera_id]
        for k in to_delete:
            del self._store[k]

    def clear_all(self):
        """Tüm kayıtları sıfırlar."""
        self._store.clear()
        self._active_keys_this_frame.clear()

    @property
    def size(self) -> int:
        return len(self._store)
