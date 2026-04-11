"""
SKYWATCH-Det Kaggle Setup Script
==================================
1. Sistem ultralytics'i kurar (pip install ultralytics)
2. Bizim custom dosyalarimizi uzerine kopyalar (patch)
3. C2f_CAM, FRM ve skywatch-det.yaml aktif olur

Kullanim:
    !python /kaggle/working/SKYWATCH/kaggle_setup.py
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path("/kaggle/working/SKYWATCH")

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def find_ultralytics_install():
    """Site-packages altinda ultralytics kurulum yerini bul."""
    try:
        import ultralytics
        return Path(ultralytics.__file__).parent
    except ImportError:
        return None

def main():
    print("\n" + "="*55)
    print("  SKYWATCH-Det Kaggle Kurulumu")
    print("="*55)

    # 1) Resmi ultralytics kur (tam paket)
    print("\n[1] ultralytics kuruluyor...")
    run("pip install ultralytics==8.3.145 -q")

    # 2) Kurulum yerini bul
    import importlib
    importlib.invalidate_caches()

    import ultralytics
    ult_dir = Path(ultralytics.__file__).parent
    print(f"\n[2] ultralytics kurulum yeri: {ult_dir}")

    # 3) Custom dosyalari uzerine kopyala (patch)
    print("\n[3] Custom dosyalar kopyalaniyor (patch)...")

    src_custom = REPO / "src" / "ultralytics_patch"
    patches = [
        # (kaynak, hedef)
        (src_custom / "nn" / "modules" / "__init__.py",      ult_dir / "nn" / "modules" / "__init__.py"),
        (src_custom / "nn" / "modules" / "skywatch_modules.py", ult_dir / "nn" / "modules" / "skywatch_modules.py"),
        (src_custom / "nn" / "tasks.py",                     ult_dir / "nn" / "tasks.py"),
    ]

    for src, dst in patches:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  OK: {src.name} -> {dst}")
        else:
            print(f"  EKSIK: {src}")

    # 4) skywatch-det.yaml'i kopyala
    yaml_src = src_custom / "cfg" / "models" / "skywatch"
    yaml_dst = ult_dir / "cfg" / "models" / "skywatch"
    if yaml_src.exists():
        shutil.copytree(yaml_src, yaml_dst, dirs_exist_ok=True)
        print(f"  OK: skywatch YAML'lar kopyalandi -> {yaml_dst}")
    else:
        print(f"  EKSIK: {yaml_src}")

    # 5) sys.modules temizle ve yeniden import et
    for key in list(sys.modules.keys()):
        if 'ultralytics' in key:
            del sys.modules[key]

    # 6) Dogrulama
    print("\n[4] Dogrulama...")
    sys.path.insert(0, str(REPO))

    from ultralytics.nn.modules import C2f_CAM, FRM
    print(f"  C2f_CAM: OK")
    print(f"  FRM: OK")

    from ultralytics import YOLO
    print(f"  YOLO: OK")

    # data.yaml yolunu guncelle
    data_yaml = Path("/kaggle/working/skywatch_data/data.yaml")
    if data_yaml.exists():
        print(f"  data.yaml: {data_yaml} - OK")
    else:
        print(f"  data.yaml bulunamadi - veri donusumunu calistir")

    print("\n" + "="*55)
    print("  Kurulum tamamlandi! Egitim baslatilabilir.")
    print("="*55)

if __name__ == "__main__":
    main()
