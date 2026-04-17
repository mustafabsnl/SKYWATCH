"""
SKYWATCH-Det Kaggle Setup Script
==================================
Strateji:
1. Mevcut ultralytics'i kullan (versiyon bagimsiz)
2. Sadece skywatch_modules.py kopyala
3. __init__.py'nin sonuna 2 satir EKLE (degistirme)
4. tasks.py'ye sadece custom modulleri kaydet
5. skywatch-det.yaml kopyala

Bu yaklasim versiyon bagimliligini ortadan kaldirir.
"""

import sys
import os
import re
import shutil
import subprocess
import importlib
from pathlib import Path
from typing import Optional

REPO = Path("/kaggle/working/SKYWATCH")
SRC  = REPO / "src" / "ultralytics_patch"


def run(cmd, check=True):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def get_ult_dir():
    """Kurulu ultralytics dizinini bul."""
    importlib.invalidate_caches()
    # Cache temizle
    for key in list(sys.modules.keys()):
        if "ultralytics" in key:
            del sys.modules[key]
    import ultralytics
    return Path(ultralytics.__file__).parent


def patch_init(ult_dir: Path):
    """
    __init__.py'ye C2f_CAM ve FRM satirlarini EKLE.
    Dosyayi degistirmez, sadece append eder.
    """
    init_path = ult_dir / "nn" / "modules" / "__init__.py"
    content = init_path.read_text(encoding="utf-8")

    inject = "\n# ── SKYWATCH-Det Custom Modules ──────────────────────\nfrom .skywatch_modules import C2f_CAM, FRM\n"

    if "skywatch_modules" in content:
        print("  __init__.py zaten patch'li, atlaniyor.")
        return

    with open(init_path, "a", encoding="utf-8") as f:
        f.write(inject)
    print(f"  OK: __init__.py -> C2f_CAM, FRM eklendi")


def patch_tasks(ult_dir: Path):
    """
    tasks.py'ye C2f_CAM ve FRM'yi parse_model'e ekle.

    Yapılanlar:
      1) Import satırı ekle
      2) C2f_CAM'i base_modules frozenset'ine ekle   (c1, c2 = ch[f], args[0])
      3) C2f_CAM'i repeat_modules frozenset'ine ekle  (repeat desteği)
      4) C2f_CAM elif — args.insert(0, c1)            (C2f ile aynı)
      5) FRM elif — c1 = ch[f]; c2 = c1; args = [c1, c2]
    """
    tasks_path = ult_dir / "nn" / "tasks.py"
    content = tasks_path.read_text(encoding="utf-8")
    modified = False

    # ── 1) Import ──
    inject_import = "from ultralytics.nn.modules.skywatch_modules import C2f_CAM, FRM"
    if inject_import not in content:
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                lines.insert(i, inject_import)
                break
        content = "\n".join(lines)
        modified = True
        print("  Import eklendi")

    # ── 2) C2f_CAM'i base_modules frozenset'ine ekle ──
    bm = re.search(r'base_modules\s*=\s*frozenset\(\s*\{.*?\}\s*\)', content, re.DOTALL)
    if bm and "C2f_CAM" not in bm.group(0):
        content, n = re.subn(
            r'(base_modules\s*=\s*frozenset\(\s*\{.*?)(\s*\}\s*\))',
            r'\g<1>\n            C2f_CAM,\g<2>',
            content, count=1, flags=re.DOTALL,
        )
        if n:
            modified = True
            print("  C2f_CAM base_modules'a eklendi")

    # ── 3) C2f_CAM'i repeat_modules frozenset'ine ekle ──
    rm = re.search(r'repeat_modules\s*=\s*frozenset\(\s*\{.*?\}\s*\)', content, re.DOTALL)
    if rm and "C2f_CAM" not in rm.group(0):
        content, n = re.subn(
            r'(repeat_modules\s*=\s*frozenset\(\s*\{.*?)(\s*\}\s*\))',
            r'\g<1>\n            C2f_CAM,\g<2>',
            content, count=1, flags=re.DOTALL,
        )
        if n:
            modified = True
            print("  C2f_CAM repeat_modules'a eklendi")

    # ── 4) C2f_CAM args.insert(0, c1) elif ──
    if "elif m is C2f_CAM:" not in content:
        old_c2f = "        elif m is C2f:\n            args.insert(0, c1)"
        if old_c2f in content:
            c2f_block = (
                "        elif m is C2f:\n"
                "            args.insert(0, c1)\n"
                "        elif m is C2f_CAM:\n"
                "            args.insert(0, c1)"
            )
            content = content.replace(old_c2f, c2f_block)
            modified = True
            print("  parse_model'e C2f_CAM elif eklendi")

    # ── 5) FRM elif (c1 = ch[f] ile) ──
    old_frm_wrong = (
        "        elif m is FRM:\n"
        "            c2 = c1\n"
        "            args = [c1, c2]\n"
    )
    frm_correct = (
        "        elif m is FRM:\n"
        "            c1 = ch[f]\n"
        "            c2 = c1\n"
        "            args = [c1, c2]\n"
    )
    if old_frm_wrong in content:
        content = content.replace(old_frm_wrong, frm_correct)
        modified = True
        print("  FRM elif duzeltildi (c1 = ch[f] eklendi)")
    elif "elif m is FRM:" not in content:
        content, n_frm = re.subn(
            r"(elif m is Concat:[\s\S]{0,100}c2 = sum\(ch\[x\] for x in f\)\n)",
            r"\1" + frm_correct,
            content, count=1,
        )
        if n_frm:
            modified = True
            print("  parse_model'e FRM elif eklendi")

    if modified:
        tasks_path.write_text(content, encoding="utf-8")
        print("  OK: tasks.py guncellendi")
    else:
        print("  tasks.py zaten guncel.")


def copy_yaml(ult_dir: Path):
    """skywatch-det.yaml'i ultralytics cfg dizinine kopyala."""
    yaml_src = SRC / "cfg" / "models" / "skywatch"
    yaml_dst = ult_dir / "cfg" / "models" / "skywatch"

    if yaml_src.exists():
        shutil.copytree(yaml_src, yaml_dst, dirs_exist_ok=True)
        yamls = list(yaml_dst.glob("*.yaml"))
        print(f"  OK: {len(yamls)} YAML kopyalandi -> {yaml_dst}")
    else:
        # Yoksa REPO icinde ara
        alt = REPO / "ultralytics_skywatch" / "ultralytics" / "cfg" / "models" / "skywatch"
        if alt.exists():
            shutil.copytree(alt, yaml_dst, dirs_exist_ok=True)
            print(f"  OK: YAML (alt yol) -> {yaml_dst}")
        else:
            print(f"  UYARI: skywatch-det.yaml bulunamadi! {yaml_src}")
            print("         Egitim baslatmadan once YAML'i yukleyin.")


def verify(ult_dir: Path):
    """Kurulumu dogrula."""
    print("\n[5] Dogrulama...")

    # Cache temizle
    for key in list(sys.modules.keys()):
        if "ultralytics" in key:
            del sys.modules[key]

    try:
        from ultralytics.nn.modules import C2f_CAM, FRM
        print("  C2f_CAM: OK")
        print("  FRM: OK")
    except ImportError as e:
        print(f"  HATA: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("  YOLO: OK")
    except Exception as e:
        print(f"  YOLO HATA: {e}")
        return False

    # YAML kontrol
    yaml = ult_dir / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"
    if yaml.exists():
        print(f"  skywatch-det.yaml: OK")
    else:
        print(f"  skywatch-det.yaml: EKSIK")

    return True


def main():
    print("\n" + "="*55)
    print("  SKYWATCH-Det Kaggle Kurulumu (v2 - patch modu)")
    print("="*55)

    # 1) ultralytics kurulu mu kontrol et
    print("\n[1] ultralytics kontrol...")
    try:
        import ultralytics
        ver = ultralytics.__version__
        print(f"  Mevcut: ultralytics {ver} - kullanilacak")
        ult_dir = Path(ultralytics.__file__).parent
    except ImportError:
        print("  Yuklu degil, kuruluyor...")
        run("pip install ultralytics -q")
        ult_dir = get_ult_dir()

    print(f"\n[2] Kurulum yeri: {ult_dir}")

    # 3) skywatch_modules.py kopyala
    print("\n[3] skywatch_modules.py kopyalaniyor...")
    src_mod = SRC / "nn" / "modules" / "skywatch_modules.py"
    dst_mod = ult_dir / "nn" / "modules" / "skywatch_modules.py"
    if src_mod.exists():
        shutil.copy2(src_mod, dst_mod)
        print(f"  OK: {dst_mod}")
    else:
        print(f"  EKSIK: {src_mod}")

    # 4) __init__.py'ye append
    print("\n[4] __init__.py patch...")
    patch_init(ult_dir)

    # 5) tasks.py patch
    print("\n[5] tasks.py patch...")
    patch_tasks(ult_dir)

    # 6) YAML kopyala
    print("\n[6] skywatch-det.yaml...")
    copy_yaml(ult_dir)

    # 7) Dogrula
    ok = verify(ult_dir)

    print("\n" + "="*55)
    if ok:
        print("  Kurulum TAMAMLANDI! Egitim baslatilabilir.")
    else:
        print("  Hatalar var! Yukaridaki mesajlari kontrol edin.")
    print("="*55)


if __name__ == "__main__":
    main()
