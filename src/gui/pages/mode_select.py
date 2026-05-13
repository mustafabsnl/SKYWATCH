"""SKYWATCH — Mod Seçimi (sabit kameralar, minimal düzen)"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QMessageBox,
    QScrollArea,
    QFrame,
    QToolButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import logging

from database.db import Database
from gui.styles.theme import (
    SURFACE,
    SURFACE_2,
    BG_PANEL,
    BORDER,
    ACCENT,
    TEXT_1,
    TEXT_2,
    TEXT_3,
    GREEN_GLOW,
    AMBER,
)
from gui.widgets.card import SectionLabel, PageTitle, Divider
from utils.config import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_mode_log = logging.getLogger("SKYWATCH")
_MODESELECT_FILE = str(Path(__file__).resolve())

if not _mode_log.handlers:
    _fallback = logging.StreamHandler()
    _fallback.setLevel(logging.INFO)
    _fallback.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | SKYWATCH | %(message)s"))
    _mode_log.addHandler(_fallback)
    _mode_log.setLevel(logging.INFO)


_FALLBACK_FIXED_CAMERAS = ["CAM_01", "CAM_03"]
_ROW_H = 48
_CHIP_H = 32

# Mod & Kamera — alt aksiyon butonları (Qt disabled yerine manuel stiller; GLOBAL QSS ile çakışmayı keser)
_SS_MODE_BTN_BASE = """
    font-weight: 700;
    font-size: 14px;
    border-radius: 12px;
    padding: 0 16px;
"""

_SS_START_PROMINENT = f"""
QPushButton {{
    background: #e85d2a;
    color: #ffffff;
    border: none;
    {_SS_MODE_BTN_BASE}
}}
QPushButton:hover {{ background: #f06a36; color: #ffffff; }}
QPushButton:pressed {{ background: #d14e22; color: #ffffff; }}
"""

_SS_START_NEUTRAL = f"""
QPushButton {{
    background: #f2eee8;
    color: #8a8178;
    border: 1px solid #d5cdc4;
    {_SS_MODE_BTN_BASE}
}}
QPushButton:hover {{ background: #eae4dc; color: #6f6a64; }}
QPushButton:pressed {{ background: #ddd8cf; }}
"""

_SS_STOP_PROMINENT = f"""
QPushButton {{
    background: #fff1ed;
    color: #c0392b;
    border: 1px solid #e0b4aa;
    {_SS_MODE_BTN_BASE}
}}
QPushButton:hover {{ background: #ffe4dc; color: #a93226; }}
QPushButton:pressed {{ background: #ffd4cc; }}
"""

_SS_STOP_NEUTRAL = f"""
QPushButton {{
    background: #f2eee8;
    color: #8a8178;
    border: 1px solid #d5cdc4;
    {_SS_MODE_BTN_BASE}
}}
QPushButton:hover {{ background: #eae4dc; color: #6f6a64; }}
QPushButton:pressed {{ background: #ddd8cf; }}
"""


class _PersonPickRow(QWidget):
    """Beyaz liste satırı: yuvarlak seçim göstergesi + “İsim • DURUM” metni."""

    row_clicked = pyqtSignal(int)

    def __init__(self, person_id: int, name: str, status: str, has_embedding: bool, parent=None):
        super().__init__(parent)
        self.person_id = int(person_id)
        self.person_name = str(name or "")
        self.person_status = str(status or "").upper()
        self.has_embedding = bool(has_embedding)
        self._selected = False
        self.setObjectName("person_row")
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if self.has_embedding else Qt.CursorShape.ArrowCursor
        )
        self.setFixedHeight(_ROW_H)
        self._build()
        self._apply_style()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 0, 12, 0)
        outer.setSpacing(14)

        self._dot = QLabel()
        self._dot.setFixedSize(22, 22)
        self._refresh_dot_style()

        self._text = QLabel()
        emb_note = "" if self.has_embedding else "  · yüz kaydı yok"
        self._text.setText(f"{self.person_name} • {self.person_status}{emb_note}")
        self._text.setFont(QFont("Segoe UI", 11))
        self._text.setWordWrap(False)

        outer.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addWidget(self._text, 1, Qt.AlignmentFlag.AlignVCenter)

    def _refresh_dot_style(self):
        if not self.has_embedding:
            self._dot.setStyleSheet(
                """
                QLabel {
                    background: #eceff3;
                    border: 2px solid #d5d9e0;
                    border-radius: 11px;
                }
                """
            )
            return
        if self._selected:
            self._dot.setStyleSheet(
                f"""
                QLabel {{
                    background: {ACCENT};
                    border: 2px solid {ACCENT};
                    border-radius: 11px;
                }}
                """
            )
        else:
            self._dot.setStyleSheet(
                """
                QLabel {
                    background: #ffffff;
                    border: 2px solid #c5cad3;
                    border-radius: 11px;
                }
                """
            )

    def set_selected(self, v: bool):
        self._selected = bool(v)
        self._refresh_dot_style()

    def _apply_style(self):
        if not self.has_embedding:
            self._text.setStyleSheet("color: #9aa3ad; background: transparent;")
            self.setStyleSheet(
                """
                QWidget#person_row {
                    background: #fafbfc;
                    border: none;
                    border-radius: 8px;
                }
                """
            )
            return
        self._text.setStyleSheet("color: #1a2332; background: transparent;")
        sel_bg = f"{ACCENT}14"
        if self._selected:
            self.setStyleSheet(
                f"""
                QWidget#person_row {{
                    background: {sel_bg};
                    border: none;
                    border-radius: 8px;
                }}
                """
            )
        else:
            self.setStyleSheet(
                """
                QWidget#person_row {
                    background: transparent;
                    border: none;
                    border-radius: 8px;
                }
                QWidget#person_row:hover {
                    background: #f4f6f9;
                }
                """
            )

    def refresh_look(self):
        self._refresh_dot_style()
        self._apply_style()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.has_embedding:
            self.row_clicked.emit(self.person_id)
        super().mouseReleaseEvent(e)


class ModeOption(QWidget):
    """İki mod için eş boyutlu kart."""

    selected = pyqtSignal(int)

    def __init__(self, idx, icon, title, desc, color, parent=None):
        super().__init__(parent)
        self.idx = idx
        self._color = QColor(color)
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(128)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build(icon, title, desc)
        self._update_style()

    def _build(self, icon, title, desc):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(14)

        num = QLabel(str(self.idx + 1))
        num.setFixedSize(40, 40)
        num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num.setFont(QFont("Segoe UI Mono", 13, QFont.Weight.Bold))
        self._num = num
        lay.addWidget(num)

        txt = QVBoxLayout()
        txt.setSpacing(4)
        ic = QLabel(icon)
        ic.setFont(QFont("Segoe UI", 15))
        ic.setStyleSheet("background: transparent;")
        self._icon_lbl = ic
        t = QLabel(title)
        t.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        t.setStyleSheet(f"color: {TEXT_1}; background: transparent;")
        d = QLabel(desc)
        d.setFont(QFont("Segoe UI", 10))
        d.setStyleSheet(f"color: {TEXT_2}; background: transparent;")
        d.setWordWrap(True)
        txt.addWidget(ic)
        txt.addWidget(t)
        txt.addWidget(d)
        lay.addLayout(txt, 1)

    def _update_style(self):
        c = self._color.name()
        if self._active:
            self.setStyleSheet(f"""
                QWidget {{
                    background: {c}12;
                    border: 2px solid {c};
                    border-radius: 14px;
                }}
            """)
            self._num.setStyleSheet(f"""
                color: {c};
                background: {c}22;
                border: 1.5px solid {c};
                border-radius: 20px;
            """)
            self._icon_lbl.setStyleSheet(f"color: {c}; background: transparent;")
        else:
            self.setStyleSheet(f"""
                QWidget {{
                    background: {SURFACE};
                    border: 1px solid {BORDER};
                    border-radius: 14px;
                }}
                QWidget:hover {{
                    background: {SURFACE_2};
                    border-color: {c}55;
                }}
            """)
            self._num.setStyleSheet(f"""
                color: {TEXT_3};
                background: {SURFACE_2};
                border: 1px solid {BORDER};
                border-radius: 20px;
            """)
            self._icon_lbl.setStyleSheet(f"color: {TEXT_3}; background: transparent;")

    def set_active(self, v: bool):
        self._active = v
        self._update_style()

    def mousePressEvent(self, e):
        self.selected.emit(self.idx)


class ModePage(QWidget):
    system_start = pyqtSignal(str, list, dict)
    system_stop = pyqtSignal()

    MODES = [
        (0, "◎", "Genel İzleme", "Tüm yüzleri takip et ve veritabanındaki kayıtlarla karşılaştır", ACCENT),
        (
            1,
            "◉",
            "Kişi Ara",
            "Seçtiğiniz bir veya birden fazla kişiyi aynı anda aktif olarak ara",
            AMBER,
        ),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sel = 0
        self._running = False
        self._cfg = AppConfig()

        self._persons_loaded = False
        self._persons_by_id: dict[int, dict] = {}
        self._selection_order: list[int] = []
        self._person_rows: dict[int, _PersonPickRow] = {}

        self._event_logger = None
        self._run_logger = None

        inst_msg = (
            f"[MODEPAGE_INSTANCE_CREATED] file=\"{_MODESELECT_FILE}\" "
            f"class=ModePage object_id={id(self)}"
        )
        print(inst_msg, flush=True)
        _mode_log.info(inst_msg)

        fixed = self._get_fixed_camera_ids()
        fc_msg = f"[MODEPAGE_FIXED_CAMERAS] cameras={fixed}"
        print(fc_msg, flush=True)
        _mode_log.info(fc_msg)

        self._build()

    def attach_runtime_logger(self, event_logger=None, run_logger=None):
        self._event_logger = event_logger
        self._run_logger = run_logger
        msg = (
            f"[MODEPAGE_LOGGER_ATTACHED] object_id={id(self)} "
            f"event_logger={'yes' if event_logger else 'no'} "
            f"run_logger={'yes' if run_logger else 'no'}"
        )
        print(msg, flush=True)
        _mode_log.info(msg)
        if self._event_logger:
            self._event_logger.info(msg)
        if self._run_logger:
            self._run_logger.log_event("MODEPAGE_LOGGER_ATTACHED", msg, object_id=id(self))

    def _emit_lifecycle(self, event_type: str, message: str, **fields):
        print(message, flush=True)
        try:
            _mode_log.info(message)
        except Exception:
            pass
        try:
            if self._event_logger:
                self._event_logger.info(message)
        except Exception:
            pass
        try:
            if self._run_logger:
                self._run_logger.log_event(event_type, message, **fields)
        except Exception:
            pass

    def _get_fixed_camera_ids(self) -> list[str]:
        cameras_cfg = self._cfg._cameras_cfg if isinstance(self._cfg._cameras_cfg, dict) else {}
        explicit = cameras_cfg.get("runtime_fixed_camera_ids") or cameras_cfg.get("fixed_camera_ids")
        known = {str(c.get("id")) for c in self._cfg.cameras if c.get("id")}

        if isinstance(explicit, list) and explicit:
            out = []
            for x in explicit:
                cid = str(x).strip()
                if cid and cid in known:
                    out.append(cid)
            max_n = self._cfg.get_max_active_cameras()
            if out:
                return out[:max_n]

        active = self._cfg.get_active_cameras()
        ids = [str(c["id"]) for c in active if c.get("id")]
        if ids:
            return ids

        return list(_FALLBACK_FIXED_CAMERAS)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QWidget()
        content.setStyleSheet(f"background: {BG_PANEL};")
        cv = QVBoxLayout(content)
        cv.setContentsMargins(56, 40, 56, 32)
        cv.setSpacing(22)

        row_center = QHBoxLayout()
        row_center.addStretch(1)
        column = QWidget()
        column.setStyleSheet("background: transparent;")
        column.setMaximumWidth(920)
        column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ll = QVBoxLayout(column)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(22)

        ll.addWidget(PageTitle("Mod Seçimi"))
        sub = QLabel("Sistemin nasıl çalışacağını belirle")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet(f"color: {TEXT_2};")
        ll.addWidget(sub)

        modes_row = QHBoxLayout()
        modes_row.setSpacing(16)
        self._mode_opts: list[ModeOption] = []
        for idx, icon, title, desc, color in self.MODES:
            opt = ModeOption(idx, icon, title, desc, color)
            opt.selected.connect(self._select_mode)
            self._mode_opts.append(opt)
            modes_row.addWidget(opt, 1)
        ll.addLayout(modes_row)

        ll.addWidget(Divider())

        person_header_lay = QHBoxLayout()
        self._person_sec = SectionLabel("ARANACAK KİŞİ")
        person_header_lay.addWidget(self._person_sec)
        person_header_lay.addStretch()
        self._btn_refresh_persons = QPushButton("↻")
        self._btn_refresh_persons.setFixedSize(30, 30)
        self._btn_refresh_persons.setStyleSheet(
            f"background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px; color: {TEXT_1};"
        )
        self._btn_refresh_persons.clicked.connect(self._load_persons)
        person_header_lay.addWidget(self._btn_refresh_persons)
        ll.addLayout(person_header_lay)

        self._person_wrap = QFrame()
        self._person_wrap.setObjectName("mode_person_panel")
        self._person_wrap.setStyleSheet(
            f"""
            QFrame#mode_person_panel {{
                background: #ffffff;
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
            """
        )
        pw = QVBoxLayout(self._person_wrap)
        pw.setContentsMargins(14, 14, 14, 14)
        pw.setSpacing(12)

        self._chip_scroll = QScrollArea()
        self._chip_scroll.setObjectName("mode_person_chip_scroll")
        self._chip_scroll.setWidgetResizable(True)
        self._chip_scroll.setFixedHeight(44)
        self._chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chip_scroll.setStyleSheet(
            """
            QScrollArea#mode_person_chip_scroll {
                background: transparent;
                border: none;
            }
            """
        )
        self._chip_inner = QWidget()
        self._chip_inner.setStyleSheet("background: transparent;")
        self._chips_hbox = QHBoxLayout(self._chip_inner)
        self._chips_hbox.setContentsMargins(0, 0, 0, 0)
        self._chips_hbox.setSpacing(8)
        self._chip_scroll.setWidget(self._chip_inner)

        self._person_scroll = QScrollArea()
        self._person_scroll.setObjectName("mode_person_scroll")
        self._person_scroll.setWidgetResizable(True)
        self._person_scroll.setMinimumHeight(200)
        self._person_scroll.setMaximumHeight(292)
        self._person_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._person_scroll.setStyleSheet(
            """
            QScrollArea#mode_person_scroll {
                background: #ffffff;
                border: none;
            }
            """
        )
        self._person_scroll.viewport().setStyleSheet("background: #ffffff;")

        self._person_list_inner = QWidget()
        self._person_list_inner.setStyleSheet("background: #ffffff;")
        self._person_list_layout = QVBoxLayout(self._person_list_inner)
        self._person_list_layout.setContentsMargins(4, 4, 4, 4)
        self._person_list_layout.setSpacing(4)
        self._person_scroll.setWidget(self._person_list_inner)

        pw.addWidget(self._chip_scroll)
        pw.addWidget(self._person_scroll)
        ll.addWidget(self._person_wrap)

        ll.addStretch(1)

        row_center.addWidget(column, 0)
        row_center.addStretch(1)
        cv.addLayout(row_center)

        root.addWidget(content, 1)

        footer = QWidget()
        footer.setObjectName("mode_footer")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(56, 8, 56, 12)
        fl.setSpacing(4)

        self._summary = QLabel()
        self._summary.setFont(QFont("Segoe UI", 9))
        self._summary.setWordWrap(True)
        self._summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._summary.setStyleSheet(f"color: {TEXT_2}; background: transparent; padding: 0;")
        fl.addWidget(self._summary)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 2, 0, 0)
        btn_row.setSpacing(14)

        self._btn_stop = QPushButton("■  Durdur")
        self._btn_stop.setObjectName("mode_btn_stop")
        self._btn_stop.setFixedHeight(52)
        self._btn_stop.clicked.connect(self._stop)

        self._btn_start = QPushButton("▶  Sistemi Başlat")
        self._btn_start.setObjectName("mode_btn_start")
        self._btn_start.setFixedHeight(52)
        try:
            self._btn_start.clicked.disconnect()
        except TypeError:
            pass
        self._btn_start.clicked.connect(self._start)

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(True)

        try:
            nrecv = self._btn_start.receivers(self._btn_start.clicked)
        except Exception:
            nrecv = -1
        connect_msg = (
            f"[MODEPAGE_START_BUTTON_CONNECTED] object_id={id(self._btn_start)} "
            f"enabled={self._btn_start.isEnabled()} receivers={nrecv} modepage_object_id={id(self)}"
        )
        print(connect_msg, flush=True)
        _mode_log.info(connect_msg)

        btn_row.addWidget(self._btn_stop, 1)
        btn_row.addWidget(self._btn_start, 1)
        fl.addLayout(btn_row)

        root.addWidget(footer)

        self._mode_opts[0].set_active(True)
        self._load_persons()
        self._update_start_button_state()

    def _clear_person_list_widgets(self):
        while self._person_list_layout.count():
            item = self._person_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._person_rows.clear()

    def _can_start_system(self) -> bool:
        if self._sel == 0:
            return True
        return len(self._collect_search_targets()[0]) >= 1

    def _apply_button_styles(self):
        """Her iki buton her zaman görünür; vurguyu işletim durumuna göre verir (Qt disabled kullanılmaz)."""
        if self._running:
            self._btn_start.setStyleSheet(_SS_START_NEUTRAL)
            self._btn_stop.setStyleSheet(_SS_STOP_PROMINENT)
        else:
            if self._can_start_system():
                self._btn_start.setStyleSheet(_SS_START_PROMINENT)
            else:
                self._btn_start.setStyleSheet(_SS_START_NEUTRAL)
            self._btn_stop.setStyleSheet(_SS_STOP_NEUTRAL)
        self._btn_start.update()
        self._btn_stop.update()

    def _apply_person_area_visual_state(self):
        is_ps = self._sel == 1
        interact = bool(is_ps and not self._running)
        self._person_wrap.setEnabled(interact)
        self._btn_refresh_persons.setEnabled(not self._running)
        if is_ps:
            self._person_sec.setStyleSheet("")
        else:
            self._person_sec.setStyleSheet(f"color: {TEXT_3};")

    def _update_start_button_state(self):
        can_go = self._can_start_system()

        if self._running:
            self._btn_start.setText("● Aktif")
            self._btn_stop.setText("■  Durdur")
            self._summary.setText("Sistem çalışıyor")
            self._summary.setStyleSheet(f"color: {GREEN_GLOW}; background: transparent; padding: 0;")
            reason = "running"
            state_msg = (
                f"[MODEPAGE_START_BUTTON_STATE] running=true can_start_hint={str(can_go).lower()} "
                f"mode={'GENERAL' if self._sel == 0 else 'PERSON_SEARCH'} cameras={self._get_fixed_camera_ids()}"
            )
        else:
            self._btn_start.setText("▶  Sistemi Başlat")
            self._btn_stop.setText("■  Durdur")
            reason = ""
            if self._sel == 0:
                reason = "general_ready"
                self._summary.setText("Genel İzleme hazır")
                self._summary.setStyleSheet(f"color: {TEXT_2}; background: transparent; padding: 0;")
            else:
                ids, names = self._collect_search_targets()
                if len(ids) >= 1:
                    reason = "person_search_ready"
                    if len(names) <= 3:
                        label = ", ".join(names)
                    else:
                        label = ", ".join(names[:3]) + f" +{len(names) - 3}"
                    self._summary.setText(f"Kişi Ara hazır — {label}")
                    self._summary.setStyleSheet(f"color: {TEXT_2}; background: transparent; padding: 0;")
                else:
                    reason = "person_required"
                    self._summary.setText("Kişi Ara için en az bir kişi seçin (yüz kaydı olan)")
                    self._summary.setStyleSheet(f"color: {TEXT_3}; background: transparent; padding: 0;")

            state_msg = (
                f"[MODEPAGE_START_BUTTON_STATE] running=false can_start={str(can_go).lower()} "
                f"reason=\"{reason}\" mode={'GENERAL' if self._sel == 0 else 'PERSON_SEARCH'} "
                f"cameras={self._get_fixed_camera_ids()}"
            )

        print(state_msg, flush=True)
        _mode_log.info(state_msg)

        self._apply_person_area_visual_state()
        self._apply_button_styles()

    def _select_mode(self, idx: int):
        self._sel = idx
        for o in self._mode_opts:
            o.set_active(o.idx == idx)
        if idx == 1:
            self._load_persons(preserve_selection=True)
        self._update_start_button_state()

    def _on_pick_row_clicked(self, pid: int):
        row = self._person_rows.get(pid)
        if row is None or not row.has_embedding:
            return
        if pid in self._selection_order:
            self._selection_order.remove(pid)
        else:
            self._selection_order.append(int(pid))
        self._refresh_row_visuals()
        self._rebuild_chips()
        self._update_start_button_state()

    def _remove_selected(self, pid: int):
        if pid in self._selection_order:
            self._selection_order.remove(int(pid))
        self._refresh_row_visuals()
        self._rebuild_chips()
        self._update_start_button_state()

    def _refresh_row_visuals(self):
        sel = set(self._selection_order)
        for pid, row in self._person_rows.items():
            row.set_selected(pid in sel)
            row.refresh_look()

    def _rebuild_chips(self):
        while self._chips_hbox.count():
            item = self._chips_hbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not self._selection_order:
            hint = QLabel("Henüz kişi seçilmedi — alttaki listeden ekleyin.")
            hint.setFont(QFont("Segoe UI", 10))
            hint.setStyleSheet("color: #7d8a98; border: none; background: transparent;")
            self._chips_hbox.addWidget(hint)
            self._chips_hbox.addStretch(1)
            return
        for pid in self._selection_order:
            self._chips_hbox.addWidget(self._make_chip(pid))
        self._chips_hbox.addStretch(1)

    def _make_chip(self, pid: int) -> QWidget:
        rec = self._persons_by_id.get(pid) or {}
        name = str(rec.get("name") or f"#{pid}")
        box = QFrame()
        box.setStyleSheet(
            """
            QFrame {
                background: #eef2f7;
                border: 1px solid #dbe3ed;
                border-radius: 16px;
            }
            """
        )
        h = QHBoxLayout(box)
        h.setContentsMargins(10, 4, 4, 4)
        h.setSpacing(4)
        lbl = QLabel(name)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        lbl.setStyleSheet("color: #1a2332; border: none; background: transparent;")
        xb = QToolButton()
        xb.setText("×")
        xb.setFixedSize(24, 24)
        xb.setToolTip("Seçimi kaldır")
        xb.setStyleSheet(
            """
            QToolButton {
                border: none;
                color: #627080;
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border-radius: 12px;
            }
            QToolButton:hover {
                background: #dde5ef;
                color: #1a2332;
            }
            """
        )
        xb.clicked.connect(lambda _=False, p=pid: self._remove_selected(p))
        h.addWidget(lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(xb, 0, Qt.AlignmentFlag.AlignVCenter)
        box.setFixedHeight(_CHIP_H)
        return box

    def _collect_search_targets(self) -> tuple[list[int], list[str]]:
        ids: list[int] = []
        names: list[str] = []
        for pid in self._selection_order:
            rec = self._persons_by_id.get(pid)
            if rec and bool(rec.get("has_embedding")):
                ids.append(int(pid))
                names.append(str(rec.get("name") or ""))
        return ids, names

    def _load_persons(self, preserve_selection: bool = True):
        try:
            prev_sel = list(self._selection_order) if preserve_selection else []
            db = Database(self._cfg, None)
            persons = db.list_persons_for_search()

            self._clear_person_list_widgets()
            self._persons_by_id = {}

            if not persons:
                self._persons_loaded = True
                lbl = QLabel("Veritabanında kayıtlı kişi yok.")
                lbl.setFont(QFont("Segoe UI", 11))
                lbl.setStyleSheet("color: #5c6b7a; background: transparent;")
                lbl.setWordWrap(True)
                self._person_list_layout.addWidget(lbl)
                self._selection_order = []
                self._rebuild_chips()
                self._update_start_button_state()
                return

            for p in persons:
                pid = int(p["id"])
                self._persons_by_id[pid] = dict(p)

            alive = set(self._persons_by_id.keys())
            self._selection_order = [
                int(x)
                for x in prev_sel
                if int(x) in alive
                and bool((self._persons_by_id.get(int(x)) or {}).get("has_embedding"))
            ]

            for p in persons:
                pid = int(p["id"])
                row = _PersonPickRow(pid, p["name"], p["status"], bool(p["has_embedding"]))
                row.row_clicked.connect(self._on_pick_row_clicked)
                self._person_list_layout.addWidget(row)
                self._person_rows[pid] = row

            self._person_list_layout.addStretch(1)

            self._persons_loaded = True
            self._refresh_row_visuals()
            self._rebuild_chips()
            self._update_start_button_state()

        except Exception as e:
            print(f"Error loading persons: {e}")
            self._update_start_button_state()

    def _start(self):
        if self._running:
            QMessageBox.information(self, "SKYWATCH", "Sistem zaten aktif.")
            return

        mode = "GENERAL" if self._sel == 0 else "PERSON_SEARCH"
        cams = list(self._get_fixed_camera_ids())
        options: dict = {}

        if mode == "PERSON_SEARCH":
            ids, names = self._collect_search_targets()
            if not ids:
                QMessageBox.warning(
                    self,
                    "Uyarı",
                    "Kişi Ara modunda başlamak için en az bir kişi seçmelisiniz (yüz kaydı olan).",
                )
                return
            options = {
                "target_person_ids": list(ids),
                "target_person_names": list(names),
            }
            if len(ids) == 1:
                options["target_person_id"] = int(ids[0])
                options["target_person_name"] = str(names[0] if names else "")

        print("[MODEPAGE_START_CLICKED]", self._sel, flush=True)
        self._emit_lifecycle(
            "MODEPAGE_START_CLICKED",
            f"[MODEPAGE_START_CLICKED] selected_mode_index={self._sel} mode={mode} cameras={cams}",
            selected_mode_index=self._sel,
            mode=mode,
            cameras=cams,
        )

        emit_msg = f"[MODEPAGE_START_EMIT] mode={mode} cameras={cams} options={options}"
        self._emit_lifecycle("MODEPAGE_START_EMIT", emit_msg, mode=mode, cameras=cams, options=options)

        self.system_start.emit(mode, cams, options)

        self._emit_lifecycle(
            "MODEPAGE_START_EMITTED",
            f"[MODEPAGE_START_EMITTED] mode={mode} cameras={cams}",
            mode=mode,
            cameras=cams,
        )

        self._running = True
        self._update_start_button_state()

    def _stop(self):
        if not self._running:
            return
        self._running = False
        self.system_stop.emit()
        self._update_start_button_state()
        self._summary.setText("■  Sistem durduruldu")
        self._summary.setStyleSheet(f"color: {AMBER}; background: transparent; padding: 0;")

    def release_start_after_main_failure(self):
        self._running = False
        self._update_start_button_state()

    def refresh_persons(self):
        self._load_persons(preserve_selection=True)

    def get_camera_ids(self, only_checked: bool = False) -> list[str]:
        return list(self._get_fixed_camera_ids())
