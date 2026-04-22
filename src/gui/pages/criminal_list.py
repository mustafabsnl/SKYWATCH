"""SKYWATCH — Veritabanı Listesi Sayfası"""

import sys
import io
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QComboBox, QMessageBox, QAbstractItemView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from gui.styles.theme import (
    BG_APP, SURFACE, SURFACE_2, BG_PANEL, BORDER, SEP,
    ACCENT, ACCENT_DIM, ACCENT_2,
    TEXT_1, TEXT_2, TEXT_3,
    GREEN_GLOW, AMBER, RED, RED_DIM, RED_GLOW,
    GOLD, GOLD_DIM, WHITE, GRAY_1, GRAY_2
)
from gui.widgets.card import SectionLabel, PageTitle, Divider

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load(sf: str = "ALL") -> list:
    try:
        import sqlite3
        db = PROJECT_ROOT / "database" / "skywatch.db"
        if not db.exists():
            return []
        conn = sqlite3.connect(str(db), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if sf == "ALL":
            r = conn.execute(
                "SELECT id,name,crime_type,danger_level,status,created_at"
                " FROM criminals ORDER BY id DESC"
            ).fetchall()
        else:
            r = conn.execute(
                "SELECT id,name,crime_type,danger_level,status,created_at"
                " FROM criminals WHERE status=? ORDER BY id DESC",
                (sf,)
            ).fetchall()
        conn.close()
        return [dict(x) for x in r]
    except Exception:
        return []


def _delete(cid: int) -> bool:
    try:
        import sqlite3
        db = PROJECT_ROOT / "database" / "skywatch.db"
        conn = sqlite3.connect(str(db), check_same_thread=False)
        conn.execute("DELETE FROM embeddings WHERE criminal_id=?", (cid,))
        conn.execute("DELETE FROM criminals WHERE id=?", (cid,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ── Pill Filtre Butonu ────────────────────────────────────────────────────────
class FilterPill(QPushButton):
    """Seçilebilir pill filtre butonu."""

    def __init__(self, label: str, value: str, parent=None):
        super().__init__(label, parent)
        self.value    = value
        self._active  = False
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(34)
        self._update_style()
        self.toggled.connect(lambda v: self._update_style())

    def _update_style(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {ACCENT};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 17px;
                    padding: 0 18px;
                    font-size: 12px;
                    font-weight: 700;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {SURFACE};
                    color: {TEXT_2};
                    border: 1px solid {BORDER};
                    border-radius: 17px;
                    padding: 0 18px;
                    font-size: 12px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    border-color: {ACCENT};
                    color: {ACCENT};
                }}
            """)


class CriminalListPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._active_filter = "ALL"
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(84)
        toolbar.setStyleSheet(f"background: {SURFACE}; border-bottom: 1px solid {BORDER};")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(36, 0, 36, 0)
        tl.setSpacing(14)

        # Başlık
        title = QLabel("Veritabanı")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Black))
        title.setStyleSheet(f"color: {TEXT_1};")
        tl.addWidget(title)

        # Kayıt sayısı badge
        self._count = QLabel("0 kayıt")
        self._count.setFont(QFont("Segoe UI Mono", 10, QFont.Weight.Bold))
        self._count.setStyleSheet(f"""
            color: {ACCENT};
            background: {ACCENT}12;
            border-radius: 12px;
            padding: 4px 14px;
        """)
        tl.addWidget(self._count)

        tl.addStretch()

        # Pill filtreler
        self._pills: list[FilterPill] = []
        filter_defs = [
            ("Tümü",       "ALL"),
            ("Arananlar",  "WANTED"),
            ("Sabıkalılar","CRIMINAL"),
            ("Temizler",   "CLEARED"),
        ]
        pill_row = QHBoxLayout()
        pill_row.setSpacing(6)
        for label, value in filter_defs:
            p = FilterPill(label, value)
            p.toggled.connect(lambda checked, v=value, px=p: self._on_pill(v, px) if checked else None)
            self._pills.append(p)
            pill_row.addWidget(p)
        self._pills[0].blockSignals(True)
        self._pills[0].setChecked(True)
        self._pills[0].blockSignals(False)
        tl.addLayout(pill_row)

        # Arama kutusu
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  İsim, ID veya suç türü ara...")
        self._search.setMinimumHeight(38)
        self._search.setFixedWidth(260)
        self._search.textChanged.connect(self._filter)
        tl.addWidget(self._search)

        # Yenile
        btn_ref = QPushButton("↻")
        btn_ref.setFixedSize(38, 38)
        btn_ref.setFont(QFont("Segoe UI", 14))
        btn_ref.setToolTip("Yenile")
        btn_ref.clicked.connect(self.refresh)
        tl.addWidget(btn_ref)

        root.addWidget(toolbar)

        # ── Tablo ───────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "ID", "AD SOYAD", "SUÇ TÜRÜ", "DURUM", "TEHLİKE", "İŞLEM"
        ])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 64)
        self._table.setColumnWidth(5, 88)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {BG_APP};
                alternate-background-color: {SURFACE_2};
                border: none;
                color: {TEXT_1};
            }}
            QTableWidget::item {{ padding: 0 16px; border: none; }}
            QTableWidget::item:selected {{ background: #FFF0EB; color: {ACCENT}; }}
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
        """)
        root.addWidget(self._table, 1)

        # ── Alt Bar ──────────────────────────────────────────────────────
        foot = QWidget()
        foot.setFixedHeight(56)
        foot.setStyleSheet(f"background: {SURFACE}; border-top: 1px solid {BORDER};")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(36, 0, 36, 0)

        self._summary = QLabel("")
        self._summary.setFont(QFont("Segoe UI", 10))
        self._summary.setStyleSheet(f"color: {TEXT_3};")
        fl.addWidget(self._summary)
        fl.addStretch()

        btn_del = QPushButton("Seçili Kaydı Sil")
        btn_del.setObjectName("btn_red")
        btn_del.setFixedHeight(38)
        btn_del.clicked.connect(self._del_sel)
        fl.addWidget(btn_del)

        root.addWidget(foot)

    # ── Veri ─────────────────────────────────────────────────────────────────
    def refresh(self):
        self._data = _load(self._active_filter)
        self._populate(self._data)

    def _on_pill(self, value: str, active_pill: FilterPill):
        # Diğer pilleri devre dışı bırak
        for p in self._pills:
            if p is not active_pill:
                p.blockSignals(True)
                p.setChecked(False)
                p.blockSignals(False)
                p._update_style()
        self._active_filter = value
        self.refresh()

    def _filter(self):
        q = self._search.text().strip().lower()
        if not q:
            self._populate(self._data)
            return
        self._populate([
            r for r in self._data
            if q in r.get("name", "").lower()
            or q in r.get("crime_type", "").lower()
            or q in str(r.get("id", ""))
        ])

    def _populate(self, data: list):
        self._table.setRowCount(0)

        st_col = {
            "WANTED":   (RED_GLOW,   "ARANIYOR"),
            "CRIMINAL": (AMBER,      "SABIKALI"),
            "CLEARED":  (GREEN_GLOW, "TEMİZ"),
        }
        dg_col = {
            "LOW":      "#27AE60",
            "MEDIUM":   AMBER,
            "HIGH":     "#E67E22",
            "CRITICAL": RED_GLOW,
        }
        dg_lbl = {
            "LOW": "DÜŞÜK", "MEDIUM": "ORTA",
            "HIGH": "YÜKSEK", "CRITICAL": "KRİTİK",
        }

        for i, row in enumerate(data):
            self._table.insertRow(i)
            self._table.setRowHeight(i, 56)

            def item(txt, col=TEXT_1, bold=False, center=False):
                it = QTableWidgetItem(str(txt))
                it.setForeground(QColor(col))
                it.setFont(QFont(
                    "Segoe UI", 11,
                    QFont.Weight.Bold if bold else QFont.Weight.Normal
                ))
                if center:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                return it

            self._table.setItem(i, 0, item(str(row["id"]), ACCENT_DIM, center=True))
            self._table.setItem(i, 1, item(row["name"], TEXT_1, bold=True))
            self._table.setItem(i, 2, item(row["crime_type"], TEXT_2))

            sc, sl = st_col.get(row["status"], (GRAY_1, "?"))
            self._table.setItem(i, 3, item(sl, sc, bold=True, center=True))

            dc = dg_col.get(row["danger_level"], GRAY_1)
            dl = dg_lbl.get(row["danger_level"], row["danger_level"])
            self._table.setItem(i, 4, item(dl, dc, center=True))

            # Sil butonu
            btn = QPushButton("Sil")
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {RED_DIM};
                    color: {RED_GLOW};
                    border: none;
                    border-radius: 8px;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: {RED};
                    color: white;
                }}
            """)
            btn.clicked.connect(
                lambda _, cid=row["id"], name=row["name"]: self._confirm(cid, name)
            )
            self._table.setCellWidget(i, 5, btn)

        tot = len(data)
        self._count.setText(f"{tot} kayıt")
        w  = sum(1 for r in data if r["status"] == "WANTED")
        cr = sum(1 for r in data if r["status"] == "CRIMINAL")
        self._summary.setText(f"Toplam {tot}  ·  Aranan {w}  ·  Sabıkalı {cr}")

    def _confirm(self, cid: int, name: str):
        r = QMessageBox.question(
            self, "Sil",
            f"ID {cid} - '{name}' kaydı silinecek. Onaylıyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if r == QMessageBox.StandardButton.Yes:
            if _delete(cid):
                self.refresh()

    def _del_sel(self):
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Bilgi", "Satır seçin.")
            return
        cid  = int(self._table.item(row, 0).text())
        name = self._table.item(row, 1).text()
        self._confirm(cid, name)
