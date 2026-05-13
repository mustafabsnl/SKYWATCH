"""SKYWATCH — Bento Dashboard (İzleme Merkezi)"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

from gui.styles.theme import (
    SURFACE, SURFACE_2, BORDER, ACCENT, ACCENT_2,
    TEXT_1, TEXT_2, TEXT_3, RED, STATUS_COLORS,
)
from gui.widgets.card import (
    Card, SectionLabel, MetricCard, Divider, PulseRing, PageTitle
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Alert Satırı ─────────────────────────────────────────────────────────────
class AlertItem(QWidget):
    """Son tespit listesi — tek satır."""

    def __init__(self, track_id: int, status: str, name: str,
                 ts: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)

        color = STATUS_COLORS.get(status, (TEXT_2, SURFACE_2, status))[0]
        label = STATUS_COLORS.get(status, (TEXT_2, SURFACE_2, status))[2]

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(12)

        # Sol renkli tab
        tab = QWidget()
        tab.setFixedWidth(4)
        tab.setStyleSheet(f"background: {color}; border-radius: 2px;")
        lay.addWidget(tab)

        # ID badge
        # track_id bazen string gelebilir (örn. pipeline/GUI veri akışı).
        # Bu durumda formatlama hatası vermemesi için güvenli dönüştür.
        try:
            display_track_id = f"T{int(track_id):03d}"
        except (TypeError, ValueError):
            display_track_id = f"T{track_id}"
        id_lbl = QLabel(display_track_id)
        id_lbl.setFont(QFont("Segoe UI Mono", 10, QFont.Weight.Bold))
        id_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        id_lbl.setFixedWidth(48)
        id_lbl.setStyleSheet(f"""
            color: {ACCENT};
            background: {ACCENT}12;
            border-radius: 6px;
            padding: 3px 0;
        """)
        lay.addWidget(id_lbl)

        # İsim + durum
        mid = QVBoxLayout()
        mid.setSpacing(2)
        n = QLabel(name if name else "Kimlik Belirsiz")
        n.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        n.setStyleSheet(f"color: {TEXT_1};")
        s = QLabel(label)
        s.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        s.setStyleSheet(f"color: {color};")
        mid.addWidget(n)
        mid.addWidget(s)
        lay.addLayout(mid, 1)

        # Saat
        t = QLabel(ts)
        t.setFont(QFont("Segoe UI Mono", 9))
        t.setStyleSheet(f"color: {TEXT_3};")
        lay.addWidget(t, 0, Qt.AlignmentFlag.AlignVCenter)

        self.setStyleSheet(f"""
            QWidget {{ background: transparent; border-bottom: 1px solid {BORDER}; }}
            QWidget:hover {{ background: {SURFACE_2}; }}
        """)


# ── Kamera Feed ───────────────────────────────────────────────────────────────
class CameraFeed(QWidget):
    """Kamera görüntüsü + PulseRing overlay."""

    def __init__(self, cam_id: str = "CAM_0", parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img.setStyleSheet("""
            QLabel {
                background: #06080C;
                border-radius: 10px;
                color: #2A2A2A;
            }
        """)
        self._show_placeholder()

        # PulseRing overlay
        self._pulse = PulseRing(ACCENT, self)
        self._pulse.set_active(False)

        lay.addWidget(self._img)

    def resizeEvent(self, e):
        self._pulse.setGeometry(self._img.geometry())

    def _show_placeholder(self):
        self._img.setText(f"◉  Kamera Bekleniyor\n\n{self.cam_id}")
        self._img.setFont(QFont("Segoe UI Mono", 12))

    def update_frame(self, bgr: np.ndarray):
        h, w, ch = bgr.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        px  = QPixmap.fromImage(img).scaled(
            self._img.width(), self._img.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self._img.setPixmap(px)

    def set_pulse(self, v: bool):
        self._pulse.set_active(v)

    def clear_frame(self):
        self._img.setPixmap(QPixmap())
        self._show_placeholder()


# ── Dashboard Sayfası ─────────────────────────────────────────────────────────
class DashboardPage(QWidget):
    """İzleme merkezi: sol canlı kamera + metrikler, sağ yalnızca son tespitler (sabit kameralar)."""

    _MAX_ALERT_ROWS = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alerts_widgets: list[AlertItem] = []
        self._alert_count = 0
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(20)

        # ── Sol: Başlık + Metrikler + Kamera ─────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(20)

        # Başlık satırı
        hdr = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        t   = PageTitle("İzleme Merkezi")
        sub = QLabel("Canlı kamera takibi ve tespit akışı")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet(f"color: {TEXT_2};")
        title_col.addWidget(t)
        title_col.addWidget(sub)

        hdr.addLayout(title_col)
        hdr.addStretch()

        self._clock = QLabel()
        self._clock.setFont(QFont("Segoe UI Mono", 13))
        self._clock.setStyleSheet(f"color: {TEXT_3};")
        hdr.addWidget(self._clock)

        left.addLayout(hdr)

        # ── 4 Metrik Kart ────────────────────────────────────────────────
        metrics = QHBoxLayout()
        metrics.setSpacing(16)

        self._m_fps    = MetricCard("FPS",          "—",  "kare/saniye",    ACCENT,   "◎")
        self._m_active = MetricCard("Aktif Track",  "0",  "kişi izleniyor", ACCENT_2, "◉")
        self._m_total  = MetricCard("Toplam Tespit","0",  "bu oturumda",    TEXT_2,   "⊞")
        self._m_alert  = MetricCard("Uyarı",        "0",  "kritik durum",   RED,      "⚠")

        for m in [self._m_fps, self._m_active, self._m_total, self._m_alert]:
            metrics.addWidget(m)
        left.addLayout(metrics)

        # ── Kamera Kartı ─────────────────────────────────────────────────
        cam_card = Card(accent=True, accent_color=ACCENT, radius=14)
        cam_l = cam_card.layout()
        cam_l.setContentsMargins(16, 14, 16, 14)
        cam_l.setSpacing(12)

        cam_hdr = QHBoxLayout()
        cam_sec = SectionLabel("Canlı Kamera")
        self._mode_lbl = QLabel("Hazır")
        self._mode_lbl.setFont(QFont("Segoe UI Mono", 10))
        self._mode_lbl.setStyleSheet(f"color: {TEXT_3};")
        cam_hdr.addWidget(cam_sec)
        cam_hdr.addStretch()
        cam_hdr.addWidget(self._mode_lbl)
        cam_l.addLayout(cam_hdr)

        self._feed = CameraFeed()
        cam_l.addWidget(self._feed, 1)

        left.addWidget(cam_card, 1)
        root.addLayout(left, 3)

        # ── Sağ: yalnızca Son Tespitler (sabit kameralar; seçim UI yok) ──
        right_card = Card(radius=14)
        right_card.setMinimumWidth(340)
        right_card.setMaximumWidth(400)
        right_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        rc = right_card.layout()
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(0)

        al_hdr = QWidget()
        al_hdr.setFixedHeight(52)
        ahl = QHBoxLayout(al_hdr)
        ahl.setContentsMargins(20, 16, 20, 12)
        al_sec = SectionLabel("Son Tespitler")
        self._alert_count_lbl = QLabel("0")
        self._alert_count_lbl.setFont(QFont("Segoe UI Mono", 10, QFont.Weight.Bold))
        self._alert_count_lbl.setStyleSheet(f"""
            color: {ACCENT};
            background: {ACCENT}12;
            border-radius: 10px;
            padding: 4px 10px;
        """)
        ahl.addWidget(al_sec)
        ahl.addStretch()
        ahl.addWidget(self._alert_count_lbl)
        rc.addWidget(al_hdr)

        rc.addWidget(Divider())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {SURFACE_2};
                width: 8px;
                margin: 0;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {TEXT_3};
                min-height: 28px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            """
        )
        self._alert_w = QWidget()
        self._alert_w.setStyleSheet("background: transparent;")
        self._al_lay = QVBoxLayout(self._alert_w)
        self._al_lay.setSpacing(0)
        self._al_lay.setContentsMargins(8, 8, 12, 16)
        self._al_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._empty_lbl = QLabel("Henüz tespit yok")
        self._empty_lbl.setFont(QFont("Segoe UI", 11))
        self._empty_lbl.setStyleSheet(f"color: {TEXT_3}; padding: 24px 8px;")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setWordWrap(True)
        self._al_lay.addWidget(self._empty_lbl)

        scroll.setWidget(self._alert_w)
        rc.addWidget(scroll, 1)

        root.addWidget(right_card, 0)

        # Saat timer
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(1000)
        self._tick()

    def _tick(self):
        self._clock.setText(datetime.now().strftime("%d.%m.%Y  %H:%M:%S"))

    # ── Güncelleme API ────────────────────────────────────────────────────────
    def update_stats(self, fps: float, active: int, total: int, alerts: int):
        self._m_fps.set_value(f"{fps:.1f}")
        self._m_active.set_value(str(active))
        self._m_total.set_value(str(total))
        self._m_alert.set_value(str(alerts))

    def update_frame(self, frame: np.ndarray):
        self._feed.update_frame(frame)

    def clear_frame(self):
        self._feed.clear_frame()

    def add_alert(self, track_id: int, status: str, name: str = ""):
        if self._empty_lbl.isVisible():
            self._empty_lbl.setVisible(False)
        ts   = datetime.now().strftime("%H:%M:%S")
        item = AlertItem(track_id, status, name, ts)
        self._al_lay.insertWidget(0, item)
        self._alerts_widgets.insert(0, item)
        self._alert_count += 1
        self._alert_count_lbl.setText(str(self._alert_count))
        if len(self._alerts_widgets) > self._MAX_ALERT_ROWS:
            old = self._alerts_widgets.pop()
            self._al_lay.removeWidget(old)
            old.deleteLater()

    def set_mode(self, mode: str, active: bool):
        labels = {
            "GENERAL":  "Genel İzleme",
            "DATABASE": "Veritabanı",
            "PERSON":   "Kişi Arama",
            "PERSON_SEARCH": "Kişi Arama",
            "CRIMINAL": "Suçlu Takibi",
            "PREVIEW": "Kamera Önizleme",
        }
        lbl   = ("▶ " if active else "■ ") + labels.get(mode, mode)
        color = ACCENT if active else TEXT_3
        self._mode_lbl.setText(lbl)
        self._mode_lbl.setStyleSheet(f"color: {color};")
        self._feed.set_pulse(active)

    def clear_alerts(self):
        for w in self._alerts_widgets:
            self._al_lay.removeWidget(w)
            w.deleteLater()
        self._alerts_widgets.clear()
        self._alert_count = 0
        self._alert_count_lbl.setText("0")
        self._empty_lbl.setVisible(True)
