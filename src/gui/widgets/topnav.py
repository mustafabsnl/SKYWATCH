"""SKYWATCH — Üst Navigasyon Çubuğu (sidebar yerine)"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient

from gui.styles.theme import (BG_NAV, GOLD, GOLD_DIM, WHITE, GRAY_1, GRAY_2,
                               SEP, BG_CARD, RED)


class NavTab(QPushButton):
    """Üst nav sekmesi."""

    def __init__(self, label: str, index: int, parent=None):
        super().__init__(label, parent)
        self.index   = index
        self._active = False
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        self.setFixedHeight(56)
        self.setMinimumWidth(110)
        self._set_style()

    def set_active(self, v: bool):
        self._active = v
        self.setChecked(v)
        self._set_style()
        self.update()

    def _set_style(self):
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {GOLD};
                    border: none;
                    border-bottom: 2px solid {GOLD};
                    padding: 0 20px;
                    font-weight: 700;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {GRAY_1};
                    border: none;
                    border-bottom: 2px solid transparent;
                    padding: 0 20px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    color: {GOLD_DIM};
                    border-bottom: 2px solid {GRAY_2};
                }}
            """)


class AlertDot(QWidget):
    """Kırmızı uyarı noktası — aktif alarm varsa gösterilir."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._on = False

    def set_on(self, v: bool):
        self._on = v; self.update()

    def paintEvent(self, _):
        if not self._on: return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(RED)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect())


class TopNav(QWidget):
    """
    Üst navigasyon çubuğu.
    Sinyal: page_changed(int)
    """
    page_changed = pyqtSignal(int)

    TABS = [
        (0, "İzleme"),
        (1, "Mod & Kamera"),
        (2, "Kişi Ekle"),
        (3, "Veritabanı"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(f"background-color: {BG_NAV};")

        root = QHBoxLayout(self)
        root.setContentsMargins(28, 0, 28, 0)
        root.setSpacing(0)

        # ── Logo ───────────────────────────────────────────────────────
        logo_row = QWidget()
        logo_row.setFixedWidth(180)
        logo_lay = QHBoxLayout(logo_row)
        logo_lay.setContentsMargins(0, 0, 0, 0)
        logo_lay.setSpacing(8)

        diamond = QLabel("◆")
        diamond.setFont(QFont("Segoe UI", 14))
        diamond.setStyleSheet(f"color: {GOLD};")

        brand = QLabel("SKYWATCH")
        brand.setFont(QFont("Segoe UI", 13, QFont.Weight.Black))
        brand.setStyleSheet(f"color: {WHITE}; letter-spacing: 3px;")

        logo_lay.addWidget(diamond)
        logo_lay.addWidget(brand)
        root.addWidget(logo_row)

        # Dikey ayırıcı
        div = QWidget()
        div.setFixedSize(1, 28)
        div.setStyleSheet(f"background: {SEP};")
        root.addWidget(div)
        root.addSpacing(12)

        # ── Sekmeler ────────────────────────────────────────────────────
        self._tabs: list[NavTab] = []
        for idx, label in self.TABS:
            tab = NavTab(label, idx)
            tab.clicked.connect(lambda checked, i=idx: self._on_tab(i))
            self._tabs.append(tab)
            root.addWidget(tab)

        root.addStretch()

        # ── Sağ: Durum ──────────────────────────────────────────────────
        self._status_dot = QWidget()
        self._status_dot.setFixedSize(8, 8)
        self._status_dot.setStyleSheet(f"background: {GRAY_2}; border-radius: 4px;")
        self._status_lbl = QLabel("Hazır")
        self._status_lbl.setFont(QFont("Segoe UI", 10))
        self._status_lbl.setStyleSheet(f"color: {GRAY_2};")

        root.addWidget(self._status_dot)
        root.addSpacing(6)
        root.addWidget(self._status_lbl)
        root.addSpacing(4)

        # İlk sekmeyi seç
        self._select(0)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG_NAV))
        # Alt ayırıcı çizgi
        p.setPen(QPen(QColor(SEP), 1))
        p.drawLine(0, self.height()-1, self.width(), self.height()-1)

    def _on_tab(self, index: int):
        self._select(index)
        self.page_changed.emit(index)

    def _select(self, index: int):
        for tab in self._tabs:
            tab.set_active(tab.index == index)

    def set_running(self, running: bool):
        color = "#059669" if running else GRAY_2
        label = "Aktif" if running else "Hazır"
        self._status_dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        self._status_lbl.setText(label)
        self._status_lbl.setStyleSheet(f"color: {color};")
