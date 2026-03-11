"""
SKYWATCH — GPU Kurulum Modülü
cuDNN DLL'lerini PATH'e ekler. Her zaman İLK import edilmeli.
"""
import os
import sys
from pathlib import Path


def setup_gpu():
    """
    nvidia-cudnn-cu12 ve cublas DLL'lerini Windows PATH'e ekler.
    Bu olmadan InsightFace CPU modunda çalışır.
    """
    venv_root = Path(sys.executable).parent.parent
    dll_dirs = [
        venv_root / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        venv_root / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        venv_root / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        venv_root / "Lib" / "site-packages" / "nvidia" / "cufft" / "bin",
    ]
    added = []
    for d in dll_dirs:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
            os.environ["PATH"] = str(d) + ";" + os.environ.get("PATH", "")
            added.append(d.name)
    return added


# Modül import edildiğinde otomatik çalıştır
_added = setup_gpu()
