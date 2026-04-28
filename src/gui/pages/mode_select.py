"""SKYWATCH — Mod & Kamera Seçim Sayfası"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QScrollArea, QFrame, QComboBox, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QBrush, QPen

from gui.styles.theme import (
    BG_APP, SURFACE, SURFACE_2, SURFACE_3, BG_PANEL, BORDER, SEP,
    ACCENT, ACCENT_DIM, ACCENT_2,
    TEXT_1, TEXT_2, TEXT_3,
    GREEN_GLOW, AMBER, RED, RED_DIM,
    GOLD, GOLD_DIM, WHITE, GRAY_1, GRAY_2, GREEN
)
from gui.widgets.card import Card, SectionLabel, PageTitle, Divider

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class ModeOption(QWidget):
    """
    Mod seçim kartı — büyük tıklanabilir blok.
    Aktif halde çerçevesi ve arka planı renkli olur.
    """
    selected = pyqtSignal(int)

    def __init__(self, idx, icon, title, desc, color, parent=None):
        super().__init__(parent)
        self.idx     = idx
        self._color  = QColor(color)
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(100)
        self._build(icon, title, desc, color)
        self._update_style()

    def _build(self, icon, title, desc, color):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(22, 0, 22, 0)
        lay.setSpacing(18)

        # Numara dairesi
        self._num = QLabel(str(self.idx + 1))
        self._num.setFixedSize(44, 44)
        self._num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._num.setFont(QFont("Segoe UI Mono", 14, QFont.Weight.Bold))
        lay.addWidget(self._num)

        # Metin
        txt = QVBoxLayout()
        txt.setSpacing(5)
        t = QLabel(title)
        t.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        t.setStyleSheet(f"color: {TEXT_1}; background: transparent;")
        d = QLabel(desc)
        d.setFont(QFont("Segoe UI", 10))
        d.setStyleSheet(f"color: {TEXT_2}; background: transparent;")
        txt.addWidget(t)
        txt.addWidget(d)
        lay.addLayout(txt, 1)

        # Ok
        self._arrow = QLabel("→")
        self._arrow.setFont(QFont("Segoe UI", 18))
        self._arrow.setStyleSheet(f"color: {TEXT_3}; background: transparent;")
        lay.addWidget(self._arrow)

    def _update_style(self):
        c = self._color.name()
        if self._active:
            self.setStyleSheet(f"""
                QWidget {{
                    background: {c}0F;
                    border: 1.5px solid {c};
                    border-radius: 12px;
                }}
            """)
            self._num.setStyleSheet(f"""
                color: {c};
                background: {c}20;
                border: 1.5px solid {c}60;
                border-radius: 22px;
            """)
            self._arrow.setStyleSheet(f"color: {c}; background: transparent;")
        else:
            self.setStyleSheet(f"""
                QWidget {{
                    background: {SURFACE};
                    border: 1px solid {BORDER};
                    border-radius: 12px;
                }}
                QWidget:hover {{
                    background: {SURFACE_2};
                    border-color: {c}60;
                }}
            """)
            self._num.setStyleSheet(f"""
                color: {TEXT_3};
                background: {SURFACE_2};
                border: 1px solid {BORDER};
                border-radius: 22px;
            """)
            self._arrow.setStyleSheet(f"color: {TEXT_3}; background: transparent;")

    def set_active(self, v: bool):
        self._active = v
        self._update_style()

    def mousePressEvent(self, e):
        self.selected.emit(self.idx)


class CamRow(QWidget):
    """Tek kamera satırı."""

    def __init__(self, cam_id, name, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.setFixedHeight(58)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        self._chk = QCheckBox()
        self._chk.setChecked(True)

        ic = QLabel("▣")
        ic.setFont(QFont("Arial", 14))
        ic.setStyleSheet(f"color: {ACCENT_DIM};")

        n = QLabel(name)
        n.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        n.setStyleSheet(f"color: {TEXT_1};")

        dot = QLabel("● Bağlı")
        dot.setFont(QFont("Segoe UI", 10))
        dot.setStyleSheet(f"color: {GREEN_GLOW};")

        lay.addWidget(self._chk)
        lay.addWidget(ic)
        lay.addWidget(n, 1)
        lay.addWidget(dot)

        self.setStyleSheet(
            f"border-bottom: 1px solid {BORDER}; background: transparent;"
        )

    def is_checked(self):
        return self._chk.isChecked()


class ModePage(QWidget):
    system_start = pyqtSignal(str, list)
    system_stop  = pyqtSignal()

    MODES = [
        (0, "◎", "Genel İzleme",    "Tüm kişileri takip et, DB ile karşılaştır", ACCENT),
        (1, "◈", "Veritabanı Modu", "Sadece kayıtlı kişiler karşılaştırılsın",   GREEN_GLOW),
        (2, "◉", "Kişi Ara",        "Seçilen belirli kişiyi aktif olarak ara",   AMBER),
        (3, "⊗", "Suçlu Takibi",    "Yalnızca ARANIYOR ve SABIKALI kişileri işaretle", RED),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sel     = 0
        self._cams:   list[CamRow] = []
        self._running = False
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sol Panel: Mod Seçimi ────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(500)
        left.setStyleSheet(f"background: {SURFACE}; border-right: 1px solid {BORDER};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(40, 40, 40, 40)
        ll.setSpacing(14)

        ll.addWidget(PageTitle("Mod Seçimi"))
        sub = QLabel("Sistemin nasıl çalışacağını belirle")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet(f"color: {TEXT_2};")
        ll.addWidget(sub)
        ll.addSpacing(12)

        self._mode_opts: list[ModeOption] = []
        for idx, icon, title, desc, color in self.MODES:
            opt = ModeOption(idx, icon, title, desc, color)
            opt.selected.connect(self._select_mode)
            self._mode_opts.append(opt)
            ll.addWidget(opt)

        ll.addSpacing(16)
        ll.addWidget(Divider())
        ll.addSpacing(12)

        # Kişi arama combo
        self._person_sec = SectionLabel("ARANACAK KİŞİ")
        ll.addWidget(self._person_sec)
        self._person_combo = QComboBox()
        self._person_combo.setMinimumHeight(44)
        self._person_combo.setEnabled(False)
        self._person_combo.addItem("— Kişi seçin —", None)
        ll.addWidget(self._person_combo)
        ll.addStretch()

        self._mode_opts[0].set_active(True)
        root.addWidget(left)

        # ── Sağ Panel: Kameralar ─────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(40, 40, 40, 40)
        rl.setSpacing(16)

        rl.addWidget(PageTitle("Kameralar"))
        sub2 = QLabel("Aktif olmasını istediğin kameraları seç")
        sub2.setFont(QFont("Segoe UI", 11))
        sub2.setStyleSheet(f"color: {TEXT_2};")
        rl.addWidget(sub2)
        rl.addWidget(Divider())

        # Kamera scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cam_w = QWidget()
        cam_w.setStyleSheet("background: transparent;")
        self._cam_lay = QVBoxLayout(cam_w)
        self._cam_lay.setSpacing(0)
        self._cam_lay.setContentsMargins(0, 0, 0, 0)
        self._cam_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(cam_w)
        rl.addWidget(scroll, 1)

        self._load_cams()

        btn_add = QPushButton("＋  Kamera Ekle")
        btn_add.setMinimumHeight(42)
        btn_add.clicked.connect(self._add_cam)
        rl.addWidget(btn_add)

        rl.addWidget(Divider())
        rl.addSpacing(4)

        self._summary = QLabel("Sistemi başlatmak için mod ve kamera seç")
        self._summary.setFont(QFont("Segoe UI", 10))
        self._summary.setStyleSheet(f"color: {TEXT_3};")
        self._summary.setWordWrap(True)
        rl.addWidget(self._summary)
        rl.addSpacing(8)

        # Başlat / Durdur
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._btn_stop = QPushButton("■  Durdur")
        self._btn_stop.setObjectName("btn_red")
        self._btn_stop.setMinimumHeight(56)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)

        self._btn_start = QPushButton("▶  Sistemi Başlat")
        self._btn_start.setObjectName("btn_gold")
        self._btn_start.setMinimumHeight(56)
        self._btn_start.clicked.connect(self._start)

        btn_row.addWidget(self._btn_stop, 1)
        btn_row.addWidget(self._btn_start, 2)
        rl.addLayout(btn_row)

        root.addWidget(right, 1)
        self._load_persons()

    def _select_mode(self, idx: int):
        self._sel = idx
        for o in self._mode_opts:
            o.set_active(o.idx == idx)
        self._person_combo.setEnabled(idx == 2)

    def _load_cams(self):
        while self._cam_lay.count():
            i = self._cam_lay.takeAt(0)
            if i.widget():
                i.widget().deleteLater()
        self._cams.clear()
        cameras = []

        # 1) Test video kaynak dosyasi varsa onu kullan
        try:
            from video_sources import VIDEO_SOURCES, CAMERA_LABELS
            for cam_id in VIDEO_SOURCES.keys():
                cameras.append({
                    "id": cam_id,
                    "name": CAMERA_LABELS.get(cam_id, cam_id),
                })
        except Exception:
            cameras = []

        # 2) Yoksa config.yaml kameralarina dus
        if not cameras:
            try:
                import yaml
                with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                cameras = cfg.get("cameras", []) or [{"id": "CAM_0", "name": "Kamera 1"}]
            except Exception:
                cameras = [{"id": "CAM_0", "name": "Kamera 1"}]

        for c in cameras:
            row = CamRow(c.get("id", "CAM"), c.get("name", "Kamera"))
            self._cams.append(row)
            self._cam_lay.addWidget(row)

    def _add_cam(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Kamera Ekle", "Kamera adı:")
        if ok and name:
            row = CamRow(f"CAM_{len(self._cams)}", name)
            self._cams.append(row)
            self._cam_lay.addWidget(row)

    def _load_persons(self):
        try:
            import sqlite3
            db = PROJECT_ROOT / "database" / "skywatch.db"
            if not db.exists():
                return
            conn = sqlite3.connect(str(db), check_same_thread=False)
            rows = conn.execute(
                "SELECT id,name FROM criminals WHERE status!='CLEARED'"
            ).fetchall()
            conn.close()
            self._person_combo.clear()
            self._person_combo.addItem("— Kişi seçin —", None)
            for rid, name in rows:
                self._person_combo.addItem(name, rid)
        except Exception:
            pass

    def _start(self):
        cams = [c.cam_id for c in self._cams if c.is_checked()]
        if not cams:
            QMessageBox.warning(self, "Uyarı", "En az bir kamera seçin.")
            return
        if self._sel == 2 and self._person_combo.currentData() is None:
            QMessageBox.warning(self, "Uyarı", "Aranacak kişiyi seçin.")
            return
        mode_map = {0: "GENERAL", 1: "DATABASE", 2: "PERSON", 3: "CRIMINAL"}
        mode = mode_map[self._sel]
        self._running = True
        self._btn_start.setEnabled(False)
        self._btn_start.setText("▶  Aktif...")
        self._btn_stop.setEnabled(True)
        self._summary.setText(f"▶  Aktif — {', '.join(cams)}")
        self._summary.setStyleSheet(f"color: {GREEN_GLOW};")
        self.system_start.emit(mode, cams)

    def _stop(self):
        self._running = False
        self._btn_start.setEnabled(True)
        self._btn_start.setText("▶  Sistemi Başlat")
        self._btn_stop.setEnabled(False)
        self._summary.setText("■  Sistem durduruldu")
        self._summary.setStyleSheet(f"color: {AMBER};")
        self.system_stop.emit()

    def refresh_persons(self):
        self._load_persons()

    def get_camera_ids(self, only_checked: bool = False) -> list[str]:
        if only_checked:
            return [c.cam_id for c in self._cams if c.is_checked()]
        return [c.cam_id for c in self._cams]
