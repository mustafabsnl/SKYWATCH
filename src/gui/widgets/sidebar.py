"""
SKYWATCH — Dual-Tone Sol Sidebar
Üst kısım (near-black #1A1A1A): Logo + marka
Alt kısım (stone #F2EFE9): Navigasyon menüsü + sistem durumu
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QBrush, QPen

from gui.styles.theme import (
    ACCENT, ACCENT_2, BG_PANEL, SURFACE, SURFACE_2,
    BORDER, TEXT_1, TEXT_2, TEXT_3, SEP,
    GREEN_GLOW, AMBER, RED
)


class SideNavItem(QWidget):
    """
    Sidebar navigasyon satırı.
    Layout: [turuncu çizgi] [ikon] [etiket] [badge?]
    """
    clicked = pyqtSignal(int)

    def __init__(self, icon: str, label: str, index: int,
                 badge: str = None, parent=None):
        super().__init__(parent)
        self.index   = index
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(50)
        self._badge_lbl = None
        self._build(icon, label, badge)
        self._update_style()

    def _build(self, icon: str, label: str, badge: str):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 16, 0)
        lay.setSpacing(14)

        self._icon_lbl = QLabel(icon)
        self._icon_lbl.setFont(QFont("Segoe UI Emoji", 15))
        self._icon_lbl.setFixedWidth(24)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label_lbl = QLabel(label)
        self._label_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))

        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._label_lbl, 1)

        if badge is not None:
            self._badge_lbl = QLabel(badge)
            self._badge_lbl.setFont(QFont("Segoe UI Mono", 9, QFont.Weight.Bold))
            self._badge_lbl.setFixedHeight(20)
            self._badge_lbl.setFixedWidth(36)
            self._badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._badge_lbl.setContentsMargins(4, 0, 4, 0)
            self._badge_lbl.setStyleSheet(f"""
                background: {ACCENT}18; color: {ACCENT};
                border-radius: 10px; padding: 0 4px;
            """)
            lay.addWidget(self._badge_lbl)

    def set_badge(self, val: str):
        if self._badge_lbl:
            self._badge_lbl.setText(val)

    def set_active(self, v: bool):
        self._active = v
        self._update_style()
        self.update()

    def _update_style(self):
        if self._active:
            self._icon_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent;")
            self._label_lbl.setStyleSheet(
                f"color: {TEXT_1}; font-weight: 700; background: transparent;"
            )
        else:
            self._icon_lbl.setStyleSheet(f"color: {TEXT_3}; background: transparent;")
            self._label_lbl.setStyleSheet(f"color: {TEXT_2}; background: transparent;")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._active:
            # Sol turuncu şerit
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(ACCENT)))
            p.drawRoundedRect(0, 10, 3, self.height() - 20, 2, 2)

            # Hafif turuncu arka plan
            bg = QColor(ACCENT)
            bg.setAlpha(10)
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(self.rect().adjusted(6, 3, -6, -3), 8, 8)

    def mousePressEvent(self, e):
        self.clicked.emit(self.index)


class _StatusDock(QWidget):
    """Alt kısım sistem durum göstergesi."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setStyleSheet(f"""
            QWidget {{
                background: {SURFACE};
                border-top: 1px solid {BORDER};
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(10)

        self._dot = QWidget()
        self._dot.setFixedSize(8, 8)
        self._dot.setStyleSheet(f"background: {TEXT_3}; border-radius: 4px;")

        self._text = QLabel("Sistem Hazır")
        self._text.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        self._text.setStyleSheet(f"color: {TEXT_3}; background: transparent;")

        lay.addWidget(self._dot)
        lay.addWidget(self._text, 1)

    def set_running(self, v: bool):
        if v:
            self._dot.setStyleSheet(
                f"background: {GREEN_GLOW}; border-radius: 4px;"
            )
            self._text.setStyleSheet(
                f"color: {GREEN_GLOW}; font-weight: 600; background: transparent;"
            )
            self._text.setText("Aktif İzleme")
        else:
            self._dot.setStyleSheet(f"background: {TEXT_3}; border-radius: 4px;")
            self._text.setStyleSheet(f"color: {TEXT_3}; background: transparent;")
            self._text.setText("Sistem Hazır")


class Sidebar(QWidget):
    """
    Dual-tone sol navigasyon.
    Sinyal: page_changed(int) → gösterilecek sayfa indeksi
    """
    page_changed = pyqtSignal(int)

    PAGES = [
        ("⬡", "İzleme",       0, None),
        ("◎", "Mod & Kamera",  1, None),
        ("＋", "Kişi Ekle",    2, None),
        ("⊞", "Veritabanı",   3, "0"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(256)
        self.setContentsMargins(0, 0, 0, 0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Üst: Near-Black Logo Bloğu ────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet(f"background: {ACCENT_2};")

        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(12)

        hex_lbl = QLabel("⬡")
        hex_lbl.setFont(QFont("Segoe UI Emoji", 28))
        hex_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent;")

        brand_col = QVBoxLayout()
        brand_col.setSpacing(1)

        brand = QLabel("SKYWATCH")
        brand.setFont(QFont("Segoe UI", 13, QFont.Weight.Black))
        brand.setStyleSheet("color: #FFFFFF; letter-spacing: 3px; background: transparent;")

        ver = QLabel("AI Surveillance  ·  v2.0")
        ver.setFont(QFont("Segoe UI", 8))
        ver.setStyleSheet("color: #505050; background: transparent; letter-spacing: 0.5px;")

        brand_col.addWidget(brand)
        brand_col.addWidget(ver)

        hl.addWidget(hex_lbl)
        hl.addLayout(brand_col)
        hl.addStretch()

        root.addWidget(header)

        # ── Orta: Stone Navigasyon ─────────────────────────────────────────
        nav_area = QWidget()
        nav_area.setStyleSheet(f"background: {BG_PANEL};")

        nl = QVBoxLayout(nav_area)
        nl.setContentsMargins(0, 20, 0, 16)
        nl.setSpacing(2)

        menu_lbl = QLabel("MENÜ")
        menu_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        menu_lbl.setStyleSheet(
            f"color: {TEXT_3}; letter-spacing: 3px;"
            f" padding-left: 20px; background: transparent;"
        )
        nl.addWidget(menu_lbl)
        nl.addSpacing(8)

        self._nav_items: list[SideNavItem] = []
        for icon, label, idx, badge in self.PAGES:
            item = SideNavItem(icon, label, idx, badge)
            item.clicked.connect(self._on_nav)
            self._nav_items.append(item)
            nl.addWidget(item)

        nl.addStretch()
        root.addWidget(nav_area, 1)

        # ── Alt: Durum Dock ────────────────────────────────────────────────
        self._status = _StatusDock()
        root.addWidget(self._status)

        # İlk sayfa seçili
        self._select(0)

    def paintEvent(self, event):
        """Sağ kenara ince border çiz."""
        p = QPainter(self)
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())

    def _on_nav(self, index: int):
        self._select(index)
        self.page_changed.emit(index)

    def _select(self, index: int):
        for item in self._nav_items:
            item.set_active(item.index == index)

    def set_running(self, running: bool):
        self._status.set_running(running)

    def set_db_count(self, count: int):
        if len(self._nav_items) > 3:
            self._nav_items[3].set_badge(str(count))
