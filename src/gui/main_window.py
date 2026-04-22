"""SKYWATCH — Ana Pencere (Sidebar Mimarisi)"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from gui.styles.theme import GLOBAL_STYLE, BG_APP
from gui.widgets.sidebar import Sidebar
from gui.pages.dashboard    import DashboardPage
from gui.pages.mode_select  import ModePage
from gui.pages.add_criminal import AddCriminalPage
from gui.pages.criminal_list import CriminalListPage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SKYWATCH")
        self.setMinimumSize(1280, 760)
        self.resize(1440, 900)
        self.setStyleSheet(GLOBAL_STYLE)
        self.statusBar().hide()

        # ── Merkezi widget ──────────────────────────────────────────────────
        central = QWidget()
        central.setStyleSheet(f"background: {BG_APP};")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sol Sidebar ─────────────────────────────────────────────────────
        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self._goto)
        root.addWidget(self.sidebar)

        # ── Sayfa Yığını ────────────────────────────────────────────────────
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {BG_APP};")
        root.addWidget(self.stack, 1)

        self.pg_dash = DashboardPage()
        self.pg_mode = ModePage()
        self.pg_add  = AddCriminalPage()
        self.pg_list = CriminalListPage()

        self.stack.addWidget(self.pg_dash)   # 0
        self.stack.addWidget(self.pg_mode)   # 1
        self.stack.addWidget(self.pg_add)    # 2
        self.stack.addWidget(self.pg_list)   # 3

        # ── Bağlantılar ─────────────────────────────────────────────────────
        self.pg_mode.system_start.connect(self._on_start)
        self.pg_mode.system_stop.connect(self._on_stop)
        self.pg_add.person_added.connect(self._on_person_added)

        # DB sayısını ilk göster
        self._update_db_count()

    # ── Navigasyon ────────────────────────────────────────────────────────────
    def _goto(self, index: int):
        self.stack.setCurrentIndex(index)
        if index == 3:
            self.pg_list.refresh()
            self._update_db_count()

    # ── Pipeline sinyalleri ──────────────────────────────────────────────────
    def _on_start(self, mode: str, cameras: list):
        self.sidebar.set_running(True)
        self.pg_dash.set_mode(mode, True)
        self.pg_dash.clear_alerts()
        self._goto(0)
        self.sidebar._select(0)

    def _on_stop(self):
        self.sidebar.set_running(False)
        self.pg_dash.set_mode("", False)

    def _on_person_added(self, name: str):
        self.pg_list.refresh()
        self.pg_mode.refresh_persons()
        self._update_db_count()

    def _update_db_count(self):
        try:
            import sqlite3
            db = PROJECT_ROOT / "database" / "skywatch.db"
            if db.exists():
                conn  = sqlite3.connect(str(db), check_same_thread=False)
                count = conn.execute("SELECT COUNT(*) FROM criminals").fetchone()[0]
                conn.close()
                self.sidebar.set_db_count(count)
        except Exception:
            pass

    # ── Pipeline Entegrasyon API ─────────────────────────────────────────────
    def push_frame(self, bgr_frame):
        self.pg_dash.update_frame(bgr_frame)

    def push_stats(self, fps, active, total, alerts):
        self.pg_dash.update_stats(fps, active, total, alerts)

    def push_alert(self, tid, status, name=""):
        self.pg_dash.add_alert(tid, status, name)


def launch_gui():
    app = QApplication(sys.argv)
    app.setApplicationName("SKYWATCH")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
