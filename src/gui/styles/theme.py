"""
SKYWATCH — Warm Stone Design System
Konsept: "Bento Board" — sıcak, asimetrik, premium
Renk dili: Toprak tonu zemin + yakıcı turuncu accent + near-black header
"""

# ── Temel Yüzeyler ───────────────────────────────────────────────────────────
BG_APP     = "#FAF9F7"   # Ana zemin — sıcak kırık beyaz
BG_PANEL   = "#F2EFE9"   # Panel / sidebar stone zemini
BG_NAV     = "#F2EFE9"   # Compat alias
SURFACE    = "#FFFFFF"   # Kart yüzeyi
SURFACE_2  = "#F7F5F1"   # İkincil yüzey / hover
SURFACE_3  = "#EDEBE5"   # Üçüncül yüzey / pressed

# ── Accent Renkler ───────────────────────────────────────────────────────────
ACCENT     = "#E8572A"   # Birincil vurgu — yakıcı turuncu
ACCENT_DIM = "#F07850"   # Açık turuncu
ACCENT_2   = "#1A1A1A"   # İkincil vurgu — near black (sidebar header)

# Compat aliases (eski kod uyumluluğu)
GOLD       = ACCENT
GOLD_DIM   = ACCENT_DIM
GOLD_GLOW  = ACCENT
BG_CARD    = SURFACE
BG_CARD_H  = SURFACE_2
BG_INPUT   = "#F2EFE9"
BG_ROW_ALT = "#F7F5F1"

# ── Durum Renkleri ───────────────────────────────────────────────────────────
RED        = "#DC2626"
RED_DIM    = "#FEE2E2"
RED_GLOW   = "#EF4444"

GREEN      = "#16A34A"
GREEN_GLOW = "#22C55E"

AMBER      = "#D97706"
AMBER_DIM  = "#FEF3C7"

PURPLE     = "#7C3AED"
PURPLE_DIM = "#EDE9FE"

# ── Metin ────────────────────────────────────────────────────────────────────
WHITE      = "#0C0C0C"   # Açık temada "white" = koyu metin (compat)
TEXT_1     = "#0C0C0C"   # Başlık
TEXT_2     = "#6B6B6B"   # İkincil
TEXT_3     = "#ABABAB"   # Muted
GRAY_1     = "#6B6B6B"   # Compat
GRAY_2     = "#ABABAB"   # Compat

# ── Yapısal ──────────────────────────────────────────────────────────────────
BORDER     = "#E5E2DC"
SEP        = "#E5E2DC"

# ── Status Renk Haritası ─────────────────────────────────────────────────────
STATUS_COLORS = {
    "WANTED":     (RED,    RED_DIM,    "ARANIYOR"),
    "CRIMINAL":   (AMBER,  AMBER_DIM,  "SABIKALI"),
    "HEDEF BULUNDU": (GOLD, GOLD_DIM, "HEDEF BULUNDU"),
    "TARGET_FOUND":  (GOLD, GOLD_DIM, "HEDEF BULUNDU"),
    "CLEARED":    (GREEN,  "#D1FAE5",  "TEMİZ"),
    "SUSPICIOUS": (PURPLE, PURPLE_DIM, "ŞÜPHELİ"),
    "UNKNOWN":    (GRAY_1, BG_ROW_ALT, "BİLİNMİYOR"),
}

DANGER_COLORS = {
    "LOW":      (GREEN_GLOW, "DÜŞÜK"),
    "MEDIUM":   (AMBER,      "ORTA"),
    "HIGH":     ("#EA580C",  "YÜKSEK"),
    "CRITICAL": (RED,        "KRİTİK"),
}

# ── Global QSS — Warm Stone Edition ──────────────────────────────────────────
GLOBAL_STYLE = f"""
/* ── Temel ── */
QMainWindow, QWidget, QDialog {{
    background-color: {BG_APP};
    color: {TEXT_1};
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 13px;
}}

/* ── Scroll ── */
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{
    background: transparent; width: 4px; margin: 4px 0;
    border-radius: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 2px; min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_3}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    height: 4px; background: transparent; border-radius: 2px;
}}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 2px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Butonlar — Default ── */
QPushButton {{
    background-color: {SURFACE};
    color: {TEXT_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 9px 22px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
    background: {SURFACE_2};
}}
QPushButton:pressed {{ background: {SURFACE_3}; }}
QPushButton:disabled {{ color: {TEXT_3}; border-color: {BORDER}; background: {SURFACE_2}; }}

/* ── Accent Butonu — Turuncu ── */
QPushButton#btn_gold, QPushButton#btn_accent {{
    background: {ACCENT};
    color: #FFFFFF;
    border: none;
    font-size: 14px;
    font-weight: 700;
    border-radius: 12px;
    padding: 14px 32px;
    letter-spacing: 0.3px;
    min-height: 52px;
}}
QPushButton#btn_gold:hover, QPushButton#btn_accent:hover {{
    background: {ACCENT_DIM};
}}
QPushButton#btn_gold:disabled, QPushButton#btn_accent:disabled {{
    background: #D8D2CC;
    color: #777777;
    border: 1px solid #C5BEB6;
}}

/* ── Tehlike Butonu ── */
QPushButton#btn_red {{
    background: {RED_DIM};
    color: {RED};
    border: 1px solid {RED}30;
    font-weight: 600;
    border-radius: 12px;
    padding: 10px 18px;
    min-height: 52px;
}}
QPushButton#btn_red:hover {{
    background: {RED};
    color: white;
    border-color: {RED};
}}
QPushButton#btn_red:disabled {{
    background: #F0EDEA;
    color: #999999;
    border: 1px solid #D8D2CC;
}}

/* ── Mod ekranı — footer + Başlat/Durdur (ModePage / mode_footer) ── */
QWidget#mode_footer {{
    background-color: {BG_PANEL};
    border-top: 1px solid {BORDER};
}}

QPushButton#btn_start {{
    background-color: #e85d2a;
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 14px;
    padding: 0 18px;
    min-height: 52px;
    max-height: 52px;
}}
QPushButton#btn_start:hover:enabled {{
    background-color: #f06a36;
}}
QPushButton#btn_start:disabled {{
    background-color: #d8d2cc;
    color: #777777;
    border: 1px solid #c8c0b8;
}}

QPushButton#btn_stop {{
    background-color: #fff5f2;
    color: #c0392b;
    border: 1px solid #e0b4aa;
    border-radius: 12px;
    font-weight: 700;
    font-size: 14px;
    padding: 0 18px;
    min-height: 52px;
    max-height: 52px;
}}
QPushButton#btn_stop:hover:enabled {{
    background-color: #ffe8e2;
}}
QPushButton#btn_stop:disabled {{
    background-color: #f0ede8;
    color: #999999;
    border: 1px solid #ddd6ce;
}}

/* ── Dark Butonu (sidebar içi) ── */
QPushButton#btn_dark {{
    background: #2A2A2A;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 9px 22px;
    font-weight: 600;
}}
QPushButton#btn_dark:hover {{ background: #3A3A3A; }}

/* ── Input ── */
QLineEdit, QPlainTextEdit {{
    background: {BG_INPUT};
    color: {TEXT_1};
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    padding: 10px 16px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QLineEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
    background: {SURFACE};
}}

/* ── ComboBox ── */
QComboBox {{
    background: {BG_INPUT};
    color: {TEXT_1};
    border: 1.5px solid {BORDER};
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 120px;
}}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_2};
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    color: {TEXT_1};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {SURFACE_2};
    selection-color: {ACCENT};
    outline: none;
    padding: 4px;
}}

/* ── Tablo ── */
QTableWidget {{
    background: {BG_APP};
    alternate-background-color: {SURFACE_2};
    color: {TEXT_1};
    border: none;
    gridline-color: transparent;
    outline: none;
}}
QTableWidget::item {{
    padding: 0 16px;
    border-bottom: 1px solid {BORDER};
}}
QTableWidget::item:selected {{
    background: #FFF0EB;
    color: {ACCENT};
}}
QHeaderView::section {{
    background: {SURFACE};
    color: {TEXT_3};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 14px 16px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}}
QHeaderView {{ background: transparent; }}

/* ── CheckBox / Radio ── */
QCheckBox, QRadioButton {{ color: {TEXT_2}; spacing: 10px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px; height: 18px;
    border: 1.5px solid {BORDER};
    border-radius: 5px;
    background: {SURFACE};
}}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}
QCheckBox:hover, QRadioButton:hover {{ color: {ACCENT}; }}

/* ── Label ── */
QLabel {{ background: transparent; color: {TEXT_1}; }}

/* ── ToolTip ── */
QToolTip {{
    background: {ACCENT_2};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 500;
}}

/* ── MessageBox ── */
QMessageBox {{ background: {SURFACE}; }}
QMessageBox QPushButton {{ min-width: 80px; }}

/* ── Status Bar ── */
QStatusBar {{
    background: {SURFACE};
    color: {TEXT_3};
    border-top: 1px solid {BORDER};
    font-size: 11px;
    padding: 0 12px;
}}

/* ── GroupBox ── */
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    margin-top: 8px;
    padding: 16px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    color: {ACCENT};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}}
"""
