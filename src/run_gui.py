"""
SKYWATCH — GUI Başlatıcı
Çalıştır: python src/run_gui.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent   # = …/SKYWATCH/src
REPO_ROOT    = PROJECT_ROOT.parent                 # = …/SKYWATCH
sys.path.insert(0, str(PROJECT_ROOT))              # gui/… doğrudan bulunur
sys.path.insert(0, str(REPO_ROOT))

# GPU DLL ayarı
import os
_venv = Path(sys.executable).parent.parent
for _sub in ("cudnn", "cublas"):
    _d = _venv / "Lib" / "site-packages" / "nvidia" / _sub / "bin"
    if _d.exists():
        os.add_dll_directory(str(_d))

from gui.main_window import launch_gui

if __name__ == "__main__":
    launch_gui()
