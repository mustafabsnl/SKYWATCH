"""SKYWATCH — Bento Dashboard (İzleme Merkezi)"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QSizePolicy, QCheckBox, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QBrush

from gui.styles.theme import (
    BG_APP, SURFACE, SURFACE_2, BORDER, ACCENT, ACCENT_2,
    TEXT_1, TEXT_2, TEXT_3, RED, GREEN_GLOW, AMBER,
    SEP, STATUS_COLORS, GOLD, GOLD_DIM
)
from gui.widgets.card import (
    Card, SectionLabel, MetricCard, Divider, PulseRing, PageTitle
)
from utils.config import AppConfig

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
    camera_selection_changed = pyqtSignal(list)
    _DEFAULT_TEST_CAMERAS = ["CAM_01", "CAM_03"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alerts_widgets: list[AlertItem] = []
        self._alert_count = 0
        self._camera_ids: list[str] = []
        self._camera_checks: dict[str, QCheckBox] = {}
        self._max_active_cameras = AppConfig().get_max_active_cameras()
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

        # ── Sağ: Tespit Listesi ──────────────────────────────────────────
        right_card = Card(radius=14)
        right_card.setFixedWidth(320)
        rc = right_card.layout()
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(0)

        # Kamera secim alani
        cam_sel_hdr = QWidget()
        cam_sel_hdr.setFixedHeight(56)
        csh = QHBoxLayout(cam_sel_hdr)
        csh.setContentsMargins(20, 0, 20, 0)
        cam_sel_sec = SectionLabel("Kamera Seçimi")
        csh.addWidget(cam_sel_sec)
        csh.addStretch()
        rc.addWidget(cam_sel_hdr)

        cam_sel_wrap = QWidget()
        cam_sel_l = QVBoxLayout(cam_sel_wrap)
        cam_sel_l.setContentsMargins(20, 8, 20, 14)
        cam_sel_l.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        self._btn_select_all = QPushButton("Tümünü Seç")
        self._btn_select_all.setMinimumHeight(30)
        self._btn_select_all.clicked.connect(self._select_all_cameras)
        self._btn_clear_all = QPushButton("Temizle")
        self._btn_clear_all.setMinimumHeight(30)
        self._btn_clear_all.clicked.connect(self._clear_camera_selection)
        top_row.addWidget(self._btn_select_all, 1)
        top_row.addWidget(self._btn_clear_all, 1)
        cam_sel_l.addLayout(top_row)

        self._cam_check_wrap = QWidget()
        self._cam_check_wrap.setStyleSheet("background: transparent;")
        self._cam_check_lay = QVBoxLayout(self._cam_check_wrap)
        self._cam_check_lay.setContentsMargins(0, 0, 0, 0)
        self._cam_check_lay.setSpacing(6)
        self._cam_check_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        cam_scroll = QScrollArea()
        cam_scroll.setWidgetResizable(True)
        cam_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cam_scroll.setMinimumHeight(120)
        cam_scroll.setMaximumHeight(160)
        cam_scroll.setWidget(self._cam_check_wrap)
        cam_sel_l.addWidget(cam_scroll)

        rc.addWidget(cam_sel_wrap)
        rc.addWidget(Divider())

        # Son tespitler basligi
        al_hdr = QWidget()
        al_hdr.setFixedHeight(56)
        ahl = QHBoxLayout(al_hdr)
        ahl.setContentsMargins(20, 0, 20, 0)
        al_sec = SectionLabel("Son Tespitler")
        self._alert_count_lbl = QLabel("0")
        self._alert_count_lbl.setFont(QFont("Segoe UI Mono", 10, QFont.Weight.Bold))
        self._alert_count_lbl.setStyleSheet(f"""
            color: {ACCENT};
            background: {ACCENT}12;
            border-radius: 10px;
            padding: 2px 10px;
        """)
        ahl.addWidget(al_sec)
        ahl.addStretch()
        ahl.addWidget(self._alert_count_lbl)
        rc.addWidget(al_hdr)
        rc.addWidget(Divider())

        # Scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._alert_w = QWidget()
        self._alert_w.setStyleSheet("background: transparent;")
        self._al_lay = QVBoxLayout(self._alert_w)
        self._al_lay.setSpacing(0)
        self._al_lay.setContentsMargins(0, 4, 0, 4)
        self._al_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._empty_lbl = QLabel("Henüz tespit yok")
        self._empty_lbl.setFont(QFont("Segoe UI", 11))
        self._empty_lbl.setStyleSheet(f"color: {TEXT_3};")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._al_lay.addWidget(self._empty_lbl)

        scroll.setWidget(self._alert_w)
        rc.addWidget(scroll, 1)

        root.addWidget(right_card)

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
        if len(self._alerts_widgets) > 20:
            old = self._alerts_widgets.pop()
            self._al_lay.removeWidget(old)
            old.deleteLater()

    def set_mode(self, mode: str, active: bool):
        labels = {
            "GENERAL":  "Genel İzleme",
            "DATABASE": "Veritabanı",
            "PERSON":   "Kişi Arama",
            "CRIMINAL": "Suçlu Takibi",
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

    def set_camera_options(self, camera_ids: list[str], selected_cameras: list[str] | None = None):
        self._camera_ids = list(camera_ids or [])
        while self._cam_check_lay.count():
            item = self._cam_check_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._camera_checks.clear()

        if not self._camera_ids:
            empty = QLabel("— Kamera yok —")
            empty.setStyleSheet(f"color: {TEXT_3};")
            self._cam_check_lay.addWidget(empty)
            return

        selected = set(selected_cameras or [])
        if not selected:
            selected = set(self._camera_ids[:self._max_active_cameras])

        for idx, cam_id in enumerate(self._camera_ids):
            chk = QCheckBox(f"Kamera {idx + 1} ({cam_id})")
            chk.setChecked(cam_id in selected)
            chk.toggled.connect(self._emit_camera_selection)
            self._camera_checks[cam_id] = chk
            self._cam_check_lay.addWidget(chk)

        self._emit_camera_selection()

    def _selected_camera_ids(self) -> list[str]:
        return [cam_id for cam_id, chk in self._camera_checks.items() if chk.isChecked()]

    def _emit_camera_selection(self):
        selected = self._selected_camera_ids()
        if len(selected) > self._max_active_cameras:
            keep = set(selected[:self._max_active_cameras])
            for cam_id, chk in self._camera_checks.items():
                if cam_id not in keep and chk.isChecked():
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)
            selected = self._selected_camera_ids()
        self.camera_selection_changed.emit(selected)

    def _select_all_cameras(self):
        preferred = [cid for cid in self._DEFAULT_TEST_CAMERAS if cid in self._camera_checks]
        if len(preferred) < self._max_active_cameras:
            for cid in self._camera_checks.keys():
                if cid not in preferred:
                    preferred.append(cid)
                if len(preferred) >= self._max_active_cameras:
                    break
        keep = set(preferred[:self._max_active_cameras])
        for cam_id, chk in self._camera_checks.items():
            chk.blockSignals(True)
            chk.setChecked(cam_id in keep)
            chk.blockSignals(False)
        self._emit_camera_selection()

    def _clear_camera_selection(self):
        for chk in self._camera_checks.values():
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._emit_camera_selection()