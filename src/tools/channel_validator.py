"""
SKYWATCH-Det — Kanal Doğrulama Aracı
======================================
Eğitim başlamadan önce çalıştırılmalı.

Yapılanlar:
  1. skywatch-det.yaml'den model inşa eder
  2. Sahte bir tensor (640x640) gönderir
  3. Her katmanda kanal sayısını raporlar
  4. C2f_CAM ve FRM'de kanal uyuşmazlığı varsa → açık hata
  5. Rebuild tetiklendiyse → açık hata (_rebuilt flag kontrolü)
  6. FRM giriş==çıkış, C2f_CAM beklenen desene uyuyor mu → strict assert

Kullanım:
  python src/tools/channel_validator.py

Beklenen çıktı (doğru kurulumda):
  ✅ [Layer 11 — FRM ] giriş: 512  çıkış: 512
  ✅ [Layer 20 — C2f_CAM] giriş: 256  çıkış: 128
  ✅ Rebuild tetiklenmedi — mimari init'te doğru kurulmuş.
  ✅ Tüm kanallar doğru! Eğitimi başlatabilirsiniz.
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_YAML = PROJECT_ROOT / "src" / "ultralytics_patch" / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"


def _ensure_skywatch_modules():
    """C2f_CAM ve FRM ultralytics'e kayıtlı değilse repo'dan enjekte et.

    Kaggle'da kaggle_setup.py zaten patch uyguladıysa bu fonksiyon hiçbir şey yapmaz.
    Lokal ortamda veya patch uygulanmamışsa repo kaynak dosyasından yükler.
    """
    try:
        from ultralytics.nn.modules import C2f_CAM, FRM  # noqa: F401
        return
    except ImportError:
        pass

    import importlib.util
    src = PROJECT_ROOT / "src" / "ultralytics_patch" / "nn" / "modules" / "skywatch_modules.py"
    if not src.exists():
        raise RuntimeError(f"skywatch_modules.py bulunamadı: {src}")

    spec = importlib.util.spec_from_file_location("ultralytics.nn.modules.skywatch_modules", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"skywatch_modules.py yüklenemedi: {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import ultralytics.nn.modules as ult_modules
    ult_modules.skywatch_modules = mod
    ult_modules.C2f_CAM = mod.C2f_CAM
    ult_modules.FRM = mod.FRM
    sys.modules["ultralytics.nn.modules.skywatch_modules"] = mod
    print("  [validator] skywatch modülleri repo'dan enjekte edildi")


_ensure_skywatch_modules()

from ultralytics import YOLO  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
# Veri yapıları
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelRow:
    layer_idx: int
    module_name: str
    in_ch: int
    out_ch: int


@dataclass
class ValidationResult:
    ok: bool = True
    yaml_path: Path | None = None
    error: str = ""
    rows: list[ChannelRow] = field(default_factory=list)
    cam_rows: list[ChannelRow] = field(default_factory=list)
    frm_rows: list[ChannelRow] = field(default_factory=list)
    input_channels: int = 3
    imgsz: int = 640
    rebuild_detected: bool = False
    rebuild_modules: list[str] = field(default_factory=list)
    channel_violations: list[str] = field(default_factory=list)


class ChannelValidator:
    """Model kanal doğrulayıcı — instance bazlı, global state yok."""

    def __init__(self):
        self._log: list[ChannelRow] = []

    def _make_hook(self, layer_idx: int, module_name: str):
        log = self._log

        def _get_channels(t) -> int | None:
            """Tensor veya tensor listesinden kanal sayısını çıkar."""
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                return t.shape[1]
            if isinstance(t, (list, tuple)):
                for item in t:
                    if isinstance(item, torch.Tensor) and item.ndim >= 2:
                        return item.shape[1]
            return None

        def hook(module, inp, output):
            try:
                first_inp = inp[0] if isinstance(inp, tuple) else inp
                in_ch = _get_channels(first_inp)
                out_ch = _get_channels(output)
                if in_ch is not None and out_ch is not None:
                    log.append(ChannelRow(layer_idx, module_name, in_ch, out_ch))
            except Exception:
                pass

        return hook

    @staticmethod
    def resolve_yaml(model_yaml: str | Path | None) -> Path:
        if model_yaml is not None:
            candidate = Path(model_yaml)
            if candidate.exists():
                return candidate
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
                if candidate.exists():
                    return candidate

        try:
            import ultralytics as _ult
            ult_yaml = Path(_ult.__file__).parent / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"
            if ult_yaml.exists():
                return ult_yaml
        except Exception:
            pass

        return DEFAULT_YAML

    @staticmethod
    def _check_rebuild_flags(model) -> tuple[bool, list[str]]:
        rebuilt = False
        modules_rebuilt: list[str] = []
        for i, layer in enumerate(model.model.model):
            name = type(layer).__name__
            if hasattr(layer, "_rebuilt") and layer._rebuilt:
                rebuilt = True
                modules_rebuilt.append(f"Layer {i} ({name})")
        return rebuilt, modules_rebuilt

    @staticmethod
    def _check_channel_patterns(
        cam_rows: list[ChannelRow],
        frm_rows: list[ChannelRow],
    ) -> list[str]:
        violations: list[str] = []
        for row in frm_rows:
            if row.in_ch != row.out_ch:
                violations.append(
                    f"FRM (Layer {row.layer_idx}): giriş ({row.in_ch}) != çıkış ({row.out_ch}) — "
                    f"FRM pass-through olmalı (in_ch == out_ch)"
                )
        for row in cam_rows:
            if row.out_ch > row.in_ch:
                violations.append(
                    f"C2f_CAM (Layer {row.layer_idx}): çıkış ({row.out_ch}) > giriş ({row.in_ch}) — "
                    f"C2f_CAM kanal daraltmalı (out_ch <= in_ch)"
                )
        return violations

    @staticmethod
    def _check_init_channel_match(model) -> list[str]:
        issues: list[str] = []
        for i, layer in enumerate(model.model.model):
            name = type(layer).__name__
            if name == "C2f_CAM":
                init_c1 = getattr(layer, "_init_c1", None)
                actual_c1 = layer.cv1.conv.in_channels
                if init_c1 is not None and init_c1 != actual_c1:
                    issues.append(
                        f"C2f_CAM (Layer {i}): init c1={init_c1} ama cv1.in_channels={actual_c1}"
                    )
            elif name == "FRM":
                init_c1 = getattr(layer, "_init_c1", None)
                actual_c1 = layer.branch1[0].conv.in_channels
                if init_c1 is not None and init_c1 != actual_c1:
                    issues.append(
                        f"FRM (Layer {i}): init c1={init_c1} ama branch1.in_channels={actual_c1}"
                    )
        return issues

    # ── Ana doğrulama ────────────────────────────────────────────────
    def validate(
        self,
        model_yaml: str | Path | None = None,
        input_channels: int = 3,
        imgsz: int = 640,
        verbose: bool = True,
    ) -> ValidationResult:
        self._log = []
        yaml_path = self.resolve_yaml(model_yaml)
        result = ValidationResult(yaml_path=yaml_path, input_channels=input_channels, imgsz=imgsz)

        if verbose:
            print("\n" + "=" * 60)
            print("  SKYWATCH-Det Kanal Doğrulayıcı")
            print("=" * 60)

        if not yaml_path.exists():
            result.ok = False
            result.error = f"YAML bulunamadı: {yaml_path}"
            if verbose:
                self._print_report(result)
            return result

        if verbose:
            print(f"\n[1] Model inşa ediliyor: {yaml_path.name}")

        try:
            model = YOLO(str(yaml_path))
        except Exception as e:
            result.ok = False
            err = str(e)
            if "FRM" in err or "C2f_CAM" in err:
                err = (
                    f"Model inşa hatası: {e}\n"
                    "İpucu: Ultralytics patch'i aktif görünmüyor olabilir. "
                    "Önce kaggle_setup.py / ensure_patch() adımını çalıştırın."
                )
            else:
                err = f"Model inşa hatası: {e}"
            result.error = err
            if verbose:
                self._print_report(result)
            return result

        # ── Hook'ları kaydet ──
        if verbose:
            print("\n[2] Kanal hook'ları ekleniyor...")
        handles = []
        for i, layer in enumerate(model.model.model):
            name = type(layer).__name__
            h = layer.register_forward_hook(self._make_hook(i, name))
            handles.append(h)

        # ── Dummy forward ──
        if verbose:
            print(f"\n[3] Sahte forward geçişi yapılıyor ({imgsz}x{imgsz}, ch={input_channels})...")
        dummy = torch.zeros(1, input_channels, imgsz, imgsz)
        try:
            with torch.no_grad():
                model.model(dummy)
        except Exception as e:
            result.error = str(e)
        finally:
            for h in handles:
                h.remove()

        # ── Sonuçları topla ──
        result.rows = list(self._log)
        result.cam_rows = [r for r in self._log if r.module_name == "C2f_CAM"]
        result.frm_rows = [r for r in self._log if r.module_name == "FRM"]

        rebuilt, rebuilt_modules = self._check_rebuild_flags(model)
        result.rebuild_detected = rebuilt
        result.rebuild_modules = rebuilt_modules

        pattern_violations = self._check_channel_patterns(result.cam_rows, result.frm_rows)
        init_violations = self._check_init_channel_match(model)
        result.channel_violations = pattern_violations + init_violations

        has_critical = bool(result.cam_rows) and bool(result.frm_rows)
        result.ok = (
            not result.error
            and has_critical
            and not result.rebuild_detected
            and not result.channel_violations
        )

        if verbose:
            self._print_report(result)
        return result

    @staticmethod
    def _print_report(result: ValidationResult) -> None:
        if result.error:
            print(f"\n{'=' * 60}")
            print(f"❌ HATA:\n{result.error}")
            print(f"{'=' * 60}")

        if result.rows:
            print("\n[4] Katman Kanal Raporu:")
            print(f"  {'Layer':<8} {'Modül':<16} {'Giriş CH':>10} {'Çıkış CH':>10}")
            print(f"  {'-' * 48}")
            critical = {"C2f_CAM", "FRM"}
            for row in result.rows:
                marker = "★" if row.module_name in critical else " "
                print(f"  {row.layer_idx:<8} {row.module_name:<16} {row.in_ch:>10} {row.out_ch:>10}  {marker}")

        print()
        if result.cam_rows:
            for r in result.cam_rows:
                print(f"  ✅ [Layer {r.layer_idx:>2} — C2f_CAM] giriş: {r.in_ch:<6} çıkış: {r.out_ch}")
        else:
            print("  ⚠️  C2f_CAM katmanı bulunamadı — YAML kontrolü yapın")

        if result.frm_rows:
            for r in result.frm_rows:
                print(f"  ✅ [Layer {r.layer_idx:>2} — FRM     ] giriş: {r.in_ch:<6} çıkış: {r.out_ch}")
        else:
            print("  ⚠️  FRM katmanı bulunamadı — YAML kontrolü yapın")

        print()
        if result.rebuild_detected:
            print("  ❌ REBUILD TESPİT EDİLDİ!")
            for m in result.rebuild_modules:
                print(f"     → {m}")
        else:
            print("  ✅ Rebuild tetiklenmedi — mimari init'te doğru kurulmuş.")

        if result.channel_violations:
            print()
            print("  ❌ KANAL DESENİ İHLALLERİ:")
            for v in result.channel_violations:
                print(f"     → {v}")
        else:
            print("  ✅ Kanal desenleri doğru (FRM: in==out, C2f_CAM: out<=in).")

        print()
        if result.ok:
            print("  ✅ Tüm kontroller geçti! Eğitimi başlatabilirsiniz.")
        else:
            print("  ❌ Sorun var — yukarıdaki uyarıları inceleyin.")
        print("=" * 60)


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════

def validate_model_channels(
    model_yaml: str | Path | None = None,
    input_channels: int = 3,
    imgsz: int = 640,
    verbose: bool = True,
) -> dict[str, Any]:
    validator = ChannelValidator()
    r = validator.validate(model_yaml=model_yaml, input_channels=input_channels, imgsz=imgsz, verbose=verbose)
    return {
        "ok": r.ok,
        "yaml_path": r.yaml_path,
        "error": r.error,
        "rows": [(row.layer_idx, row.module_name, row.in_ch, row.out_ch) for row in r.rows],
        "cam_rows": [(row.layer_idx, row.module_name, row.in_ch, row.out_ch) for row in r.cam_rows],
        "frm_rows": [(row.layer_idx, row.module_name, row.in_ch, row.out_ch) for row in r.frm_rows],
        "input_channels": r.input_channels,
        "imgsz": r.imgsz,
        "rebuild_detected": r.rebuild_detected,
        "rebuild_modules": r.rebuild_modules,
        "channel_violations": r.channel_violations,
    }


def main():
    result = validate_model_channels()
    if not result["ok"]:
        print("\n→ Yukarıdaki çözüm adımlarını uygulayın, ardından tekrar çalıştırın.")
        sys.exit(1)


if __name__ == "__main__":
    main()
