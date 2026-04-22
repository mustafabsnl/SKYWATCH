"""SKYWATCH — Warm Stone Widget Kütüphanesi

Bileşenler:
  Card          — Sıfır border, soft shadow, hover animate
  SectionLabel  — UPPERCASE küçük bölüm etiketi
  PageTitle     — Ultra-bold büyük başlık
  StatBig       — Monospace büyük rakam
  MetricCard    — Dashboard metrik kartı (renkli alt bant)
  Divider       — İnce ayırıcı
  StatusBadge   — Pill şekli renkli rozet
  PulseRing     — Animasyonlu yayılan halka (kamera aktif)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QFont, QBrush, QPainterPath
)

from gui.styles.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, ACCENT, ACCENT_2,
    TEXT_1, TEXT_2, TEXT_3, GRAY_1, GRAY_2, SEP,
    RED, GREEN, GREEN_GLOW, AMBER, STATUS_COLORS
)


# ── Yardımcı ─────────────────────────────────────────────────────────────────
def _shadow(blur: int = 24, oy: int = 4, alpha: int = 16) -> QGraphicsDropShadowEffect:
    """Standart yumuşak gölge efekti."""
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setOffset(0, oy)
    s.setColor(QColor(0, 0, 0, alpha))
    return s


# ── Card ──────────────────────────────────────────────────────────────────────
class Card(QWidget):
    """
    Premium Warm Stone kart.
    • Sıfır border — sadece yumuşak shadow
    • hover'da SURFACE_2
    • accent=True → sol turuncu şerit
    """

    def __init__(self, parent=None, accent: bool = False,
                 accent_color: str = None, radius: int = 14,
                 shadow_blur: int = 22, shadow_alpha: int = 16):
        super().__init__(parent)
        self._accent       = accent
        self._accent_color = QColor(accent_color or ACCENT)
        self._radius       = radius
        self._hovered      = False

        self.setGraphicsEffect(_shadow(shadow_blur, 4, shadow_alpha))

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(24, 22, 24, 22)
        self._layout.setSpacing(16)

    def layout(self):
        return self._layout

    def set_accent(self, on: bool, color: str = None):
        self._accent = on
        if color:
            self._accent_color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r   = self.rect()
        rad = self._radius

        # Kart zemini
        p.setBrush(QBrush(QColor(SURFACE_2 if self._hovered else SURFACE)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, rad, rad)

        # Sol accent şeridi
        if self._accent:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(self._accent_color))
            p.drawRoundedRect(QRect(0, rad, 4, r.height() - rad * 2), 2, 2)

    def enterEvent(self, e):
        self._hovered = True
        self.update()

    def leaveEvent(self, e):
        self._hovered = False
        self.update()


# ── Etiket Bileşenleri ────────────────────────────────────────────────────────
class SectionLabel(QLabel):
    """UPPERCASE küçük bölüm etiketi."""
    def __init__(self, text: str, parent=None):
        super().__init__(text.upper(), parent)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.setStyleSheet(
            f"color: {TEXT_3}; letter-spacing: 2px; background: transparent;"
        )


class PageTitle(QLabel):
    """Ultra-bold büyük sayfa başlığı."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 28, QFont.Weight.Black))
        self.setStyleSheet(
            f"color: {TEXT_1}; background: transparent; letter-spacing: -1px;"
        )


# ── StatBig ───────────────────────────────────────────────────────────────────
class StatBig(QWidget):
    """
    Büyük monospace istatistik.
    value  : gösterilen rakam/değer
    label  : UPPERCASE alt açıklama
    color  : rakam rengi
    icon   : opsiyonel unicode ikon (üst)
    """
    def __init__(self, value: str, label: str, color: str,
                 icon: str = None, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        if icon:
            ic = QLabel(icon)
            ic.setFont(QFont("Segoe UI Emoji", 18))
            ic.setStyleSheet(f"color: {color}; background: transparent;")
            lay.addWidget(ic)

        self._val = QLabel(value)
        self._val.setFont(QFont("Segoe UI Mono", 44, QFont.Weight.Black))
        self._val.setStyleSheet(
            f"color: {color}; background: transparent; letter-spacing: -2px;"
        )

        lbl = QLabel(label.upper())
        lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {TEXT_3}; background: transparent; letter-spacing: 2px;")

        lay.addWidget(self._val)
        lay.addWidget(lbl)

    def set_value(self, v: str):
        self._val.setText(v)


# ── MetricCard ────────────────────────────────────────────────────────────────
class MetricCard(QWidget):
    """
    Dashboard büyük metrik kartı.
    Üst: küçük başlık | Alt bant: ince renkli çizgi
    Büyük monospace rakam ortada.
    """

    def __init__(self, title: str, value: str, sublabel: str,
                 color: str = None, icon: str = None, parent=None):
        super().__init__(parent)
        self._color  = color or ACCENT
        self._radius = 14
        self.setGraphicsEffect(_shadow(20, 3, 14))

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # İçerik alanı
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(22, 20, 22, 18)
        cl.setSpacing(6)

        # Üst: ikon + başlık
        top = QHBoxLayout()
        if icon:
            ic = QLabel(icon)
            ic.setFont(QFont("Segoe UI Emoji", 15))
            ic.setStyleSheet(f"color: {self._color}; background: transparent;")
            top.addWidget(ic)
            top.addSpacing(6)

        t = QLabel(title.upper())
        t.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        t.setStyleSheet(f"color: {TEXT_3}; letter-spacing: 1.5px; background: transparent;")
        top.addWidget(t)
        top.addStretch()
        cl.addLayout(top)

        # Büyük değer
        self._val_lbl = QLabel(value)
        self._val_lbl.setFont(QFont("Segoe UI Mono", 38, QFont.Weight.Black))
        self._val_lbl.setStyleSheet(
            f"color: {TEXT_1}; letter-spacing: -2px; background: transparent;"
        )
        cl.addWidget(self._val_lbl)

        # Alt açıklama
        sub = QLabel(sublabel)
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet(f"color: {TEXT_3}; background: transparent;")
        cl.addWidget(sub)

        root.addWidget(content)

        # Renkli alt bant
        band = QWidget()
        band.setFixedHeight(3)
        band.setStyleSheet(
            f"background: {self._color}; border-bottom-left-radius: {self._radius}px;"
            f" border-bottom-right-radius: {self._radius}px;"
        )
        root.addWidget(band)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(SURFACE)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), self._radius, self._radius)

    def set_value(self, v: str):
        self._val_lbl.setText(v)


# ── Divider ───────────────────────────────────────────────────────────────────
class Divider(QWidget):
    """İnce yatay/dikey ayırıcı çizgi."""
    def __init__(self, parent=None, vertical: bool = False):
        super().__init__(parent)
        self._color = QColor(BORDER)
        if vertical:
            self.setFixedWidth(1)
        else:
            self.setFixedHeight(1)

    def paintEvent(self, _):
        QPainter(self).fillRect(self.rect(), self._color)


# ── StatusBadge ───────────────────────────────────────────────────────────────
class StatusBadge(QLabel):
    """Renkli pill rozet."""
    def __init__(self, status: str, parent=None):
        super().__init__(parent)
        color, bg, label = STATUS_COLORS.get(status, (GRAY_1, "#F1F5F9", status))
        self.setText(label)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {color};
                border-radius: 10px;
                padding: 3px 12px;
            }}
        """)
        self.setFixedHeight(22)


# ── PulseRing ─────────────────────────────────────────────────────────────────
class PulseRing(QWidget):
    """
    Animasyonlu yayılan halka efekti.
    Kamera feed aktif olduğunda overlay olarak kullanılır.
    """
    def __init__(self, color: str = None, parent=None):
        super().__init__(parent)
        self._color  = QColor(color or ACCENT)
        self._phase  = 0.0
        self._active = False
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._step)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def set_active(self, v: bool):
        self._active = v
        if v:
            self._timer.start(28)
        else:
            self._timer.stop()
            self.update()

    def _step(self):
        self._phase = (self._phase + 0.022) % 1.0
        self.update()

    def paintEvent(self, event):
        if not self._active:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self.rect().center().x()
        cy = self.rect().center().y()
        max_r = min(self.width(), self.height()) // 2 - 8

        for i in range(3):
            phase = (self._phase + i / 3) % 1.0
            r     = int(max_r * 0.25 + max_r * 0.75 * phase)
            alpha = int(160 * (1.0 - phase))
            c = QColor(self._color)
            c.setAlpha(alpha)
            pen = QPen(c, 2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
