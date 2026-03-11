"""
SKYWATCH — Movement Analyzer (Hareket Analizi)
Tracking verisinden hız, yön, bekleme süresi ve davranış skoru hesaplar.
"""

import math
import time
from collections import deque

from core.models import Track, MovementReport


class MovementAnalyzer:
    """Kişilerin hareketlerini analiz eder ve şüpheli davranışları tespit eder."""

    def __init__(self, config: dict):
        self.speed_fast = config.get("speed_threshold_fast", 10)
        self.speed_running = config.get("speed_threshold_running", 20)
        self.dwell_threshold = config.get("dwell_time_threshold", 60)
        self.direction_threshold = config.get("direction_change_threshold", 90)

        # Track ID → position geçmişi (x, y, timestamp)
        self._history: dict[int, deque] = {}
        self._max_history = 90  # Max 90 frame (~3 sn @ 30fps)

        # Track ID → ilk görülme zamanı (bekleme süresi hesabı)
        self._first_seen: dict[int, float] = {}

    def analyze(self, track: Track) -> MovementReport:
        """
        Bir track'in mevcut hareket durumunu analiz eder.

        Args:
            track: Takip edilen kişi

        Returns:
            MovementReport: Hız, yön, davranış skoru
        """
        tid = track.track_id
        now = time.time()

        # Merkez nokta
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2

        # Geçmiş oluştur
        if tid not in self._history:
            self._history[tid] = deque(maxlen=self._max_history)
            self._first_seen[tid] = now

        self._history[tid].append((cx, cy, now))
        history = self._history[tid]

        # --- Hız Hesaplama ---
        speed = 0.0
        avg_speed = 0.0

        if len(history) >= 2:
            # Anlık hız (son 2 nokta arası piksel/frame)
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            speed = math.sqrt(dx * dx + dy * dy)

            # Ortalama hız (son 30 kare)
            window = min(len(history), 30)
            total_dist = 0.0
            for i in range(-window, -1):
                ddx = history[i + 1][0] - history[i][0]
                ddy = history[i + 1][1] - history[i][1]
                total_dist += math.sqrt(ddx * ddx + ddy * ddy)
            avg_speed = total_dist / window

        # --- Yön Hesaplama ---
        direction = (0.0, 0.0)
        if len(history) >= 2:
            direction = (
                history[-1][0] - history[-2][0],
                history[-1][1] - history[-2][1]
            )

        # --- Bekleme Süresi ---
        dwell_time = now - self._first_seen.get(tid, now)
        
        # Kişi hareket ediyorsa bekleme sayılmaz
        if avg_speed > self.speed_fast:
            self._first_seen[tid] = now
            dwell_time = 0.0

        # --- Rota ---
        route = [(h[0], h[1]) for h in history]

        # --- Ani Yön Değişimi Kontrolü ---
        sudden_turn = False
        if len(history) >= 10:
            # Son 5 frame'in yönü vs önceki 5 frame'in yönü
            old_dx = history[-5][0] - history[-10][0]
            old_dy = history[-5][1] - history[-10][1]
            new_dx = history[-1][0] - history[-5][0]
            new_dy = history[-1][1] - history[-5][1]

            old_angle = math.atan2(old_dy, old_dx)
            new_angle = math.atan2(new_dy, new_dx)

            angle_diff = abs(math.degrees(new_angle - old_angle))
            # Açıyı 0-180 arasına normalleştir
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            sudden_turn = angle_diff > self.direction_threshold

        # --- Davranış Skoru (0.0 - 1.0) ---
        score = 0.0
        label = "normal"

        if speed > self.speed_running:
            score += 0.35
            label = "running"
        elif speed > self.speed_fast:
            score += 0.20
            label = "fast"

        if dwell_time > self.dwell_threshold:
            score += 0.15

        if sudden_turn:
            score += 0.10

        # Skor'u limit içinde tut
        score = min(score, 1.0)

        # En yüksek durumu belirle
        if score >= 0.5:
            label = "suspicious"
        if speed > self.speed_running:
            label = "running"

        report = MovementReport(
            speed=round(speed, 2),
            avg_speed=round(avg_speed, 2),
            direction=direction,
            dwell_time=round(dwell_time, 1),
            route=route[-30:],  # Son 30 noktayı tut (hafıza dostu)
            behavior_score=round(score, 3),
            behavior_label=label
        )

        return report

    def cleanup(self, active_track_ids: set[int]):
        """Artık takip edilmeyen track'lerin geçmişini temizle."""
        dead = [tid for tid in self._history if tid not in active_track_ids]
        for tid in dead:
            self._history.pop(tid, None)
            self._first_seen.pop(tid, None)
