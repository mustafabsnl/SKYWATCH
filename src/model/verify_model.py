"""
SKYWATCH-Det Model Doğrulama Scripti
=====================================
Modelin YAML'ı doğru parse edilip edilmediğini,
özel modüllerin (C2f_CAM, FRM) yüklenip yüklenmediğini
ve forward pass'ın düzgün çalışıp çalışmadığını test eder.

Çalıştır:
    cd SKYWATCH
    python src/model/verify_model.py
"""

import sys
from pathlib import Path

# Proje root'unu ekle
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ultralytics_skywatch"))


def test_module_imports():
    """Özel modüllerin import edildiğini doğrula."""
    print("\n[1] Modül import testi...")
    try:
        from ultralytics.nn.modules.skywatch_modules import C2f_CAM, FRM
        print("  ✓ C2f_CAM import OK")
        print("  ✓ FRM import OK")

        from ultralytics.nn.modules import C2f_CAM as C2f_CAM2, FRM as FRM2
        print("  ✓ C2f_CAM __init__.py'de kayıtlı")
        print("  ✓ FRM __init__.py'de kayıtlı")
        return True
    except ImportError as e:
        print(f"  ✗ Import hatası: {e}")
        return False


def test_module_forward():
    """Modüllerin forward pass'larını test et."""
    print("\n[2] Forward pass testi...")
    import torch
    from ultralytics.nn.modules.skywatch_modules import C2f_CAM, FRM

    # Test tensörü: (batch=2, channels=128, H=80, W=80)
    x = torch.randn(2, 128, 80, 80)

    # C2f_CAM testi
    try:
        m = C2f_CAM(c1=128, c2=128, n=1)
        out = m(x)
        assert out.shape == x.shape, f"Beklenen {x.shape}, alınan {out.shape}"
        print(f"  ✓ C2f_CAM: {tuple(x.shape)} → {tuple(out.shape)}")
    except Exception as e:
        print(f"  ✗ C2f_CAM forward hatası: {e}")
        return False

    # FRM testi
    try:
        x5 = torch.randn(2, 512, 20, 20)
        m2 = FRM(c1=512, c2=512)
        out2 = m2(x5)
        assert out2.shape == x5.shape
        print(f"  ✓ FRM: {tuple(x5.shape)} → {tuple(out2.shape)}")
    except Exception as e:
        print(f"  ✗ FRM forward hatası: {e}")
        return False

    return True


def test_yaml_parse():
    """YAML'ı parse et ve model oluşturulabildiğini doğrula."""
    print("\n[3] YAML parse testi...")
    try:
        from ultralytics import YOLO

        yaml_path = ROOT / "ultralytics_skywatch" / "ultralytics" / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"

        if not yaml_path.exists():
            print(f"  ✗ YAML bulunamadı: {yaml_path}")
            return False

        print(f"  Yükleniyor: {yaml_path.name}")
        model = YOLO(str(yaml_path))
        print(f"  ✓ Model oluşturuldu")
        print(f"  ✓ Parametre sayısı: {sum(p.numel() for p in model.model.parameters()):,}")

        # Katman özetini göster
        print("\n  Katman özeti:")
        for i, m in enumerate(model.model.model):
            print(f"    [{i:2d}] {m.__class__.__name__} — çıkış:{getattr(m, '_np', '?')} params")

        return True
    except Exception as e:
        print(f"  ✗ YAML parse hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Model üzerinde dummy forward pass testi."""
    print("\n[4] Tam model forward pass testi...")
    try:
        import torch
        from ultralytics import YOLO

        yaml_path = ROOT / "ultralytics_skywatch" / "ultralytics" / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"
        model = YOLO(str(yaml_path))
        model.model.eval()

        # Dummy giriş: (1, 3, 640, 640)
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model.model(x)

        print(f"  ✓ Forward pass başarılı")
        if isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                if hasattr(o, 'shape'):
                    print(f"  ✓ Çıkış [{i}]: {tuple(o.shape)}")
        return True
    except Exception as e:
        print(f"  ✗ Forward pass hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 55)
    print("  SKYWATCH-Det Model Doğrulama")
    print("=" * 55)

    results = []
    results.append(("Modül Import", test_module_imports()))
    results.append(("Forward Pass (modüller)", test_module_forward()))
    results.append(("YAML Parse", test_yaml_parse()))
    results.append(("Tam Model Forward", test_forward_pass()))

    print("\n" + "=" * 55)
    print("  SONUÇLAR")
    print("=" * 55)
    for name, ok in results:
        status = "✓ GEÇTI" if ok else "✗ BAŞARISIZ"
        print(f"  {status:<14} {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} test geçti")
    if passed == len(results):
        print("  Model eğitime hazır! 🚀")
    else:
        print("  Hataları düzeltip tekrar deneyin.")
    print("=" * 55)


if __name__ == "__main__":
    main()
