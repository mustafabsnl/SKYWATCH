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
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# skywatch_modules'ü doğrudan dosyadan yükle ve sys.modules'e enjekte et.
SKYWATCH_MODULE_PATH = PROJECT_ROOT / "src" / "ultralytics_patch" / "nn" / "modules" / "skywatch_modules.py"
spec = importlib.util.spec_from_file_location("ultralytics.nn.modules.skywatch_modules", SKYWATCH_MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"skywatch_modules.py yüklenemedi: {SKYWATCH_MODULE_PATH}")
skywatch_modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skywatch_modules)

import ultralytics.nn.modules as ult_modules
ult_modules.skywatch_modules = skywatch_modules
ult_modules.C2f_CAM = skywatch_modules.C2f_CAM
ult_modules.FRM = skywatch_modules.FRM
sys.modules["ultralytics.nn.modules.skywatch_modules"] = skywatch_modules

from ultralytics import YOLO

DEFAULT_YAML = PROJECT_ROOT / "src" / "ultralytics_patch" / "cfg" / "models" / "skywatch" / "skywatch-det.yaml"


# ════════════════════════════════════════════════════════════════════════
# Veri yapıları — global state yerine instance bazlı
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

        def hook(module, inp, output):
            in_ch = inp[0].shape[1] if isinstance(inp, tuple) else inp.shape[1]
            out_ch = output.shape[1]
            log.append(ChannelRow(layer_idx, module_name, in_ch, out_ch))

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

    # ── Rebuild flag kontrolü ────────────────────────────────────────
    @staticmethod
    def _check_rebuild_flags(model) -> tuple[bool, list[str]]:
        """Modeldeki C2f_CAM ve FRM modüllerinin _rebuilt flag'ini kontrol et."""
        rebuilt = False
        modules_rebuilt: list[str] = []
        for i, layer in enumerate(model.model.model):
            name = type(layer).__name__
            if hasattr(layer, "_rebuilt") and layer._rebuilt:
                rebuilt = True
                modules_rebuilt.append(f"Layer {i} ({name})")
        return rebuilt, modules_rebuilt

    # ── Strict kanal deseni doğrulama ────────────────────────────────
    @staticmethod
    def _check_channel_patterns(
        cam_rows: list[ChannelRow],
        frm_rows: list[ChannelRow],
    ) -> list[str]:
        """Beklenen kanal desenlerini strict doğrula."""
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
            if row.in_ch < 1:
                violations.append(
                    f"C2f_CAM (Layer {row.layer_idx}): giriş kanalı 0 — model inşa hatası"
                )

        return violations

    # ── init kanal uyumu kontrolü ────────────────────────────────────
    @staticmethod
    def _check_init_channel_match(model) -> list[str]:
        """Modüllerin __init__'te aldığı c1 ile forward'da gördüğü c1 aynı mı?"""
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
        """Modeli kur, forward çalıştır, rebuild + kanal deseni + rapor."""
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
            result.error = err
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

        # ── Rebuild flag kontrolü ──
        rebuilt, rebuilt_modules = self._check_rebuild_flags(model)
        result.rebuild_detected = rebuilt
        result.rebuild_modules = rebuilt_modules

        # ── Strict kanal deseni kontrolü ──
        pattern_violations = self._check_channel_patterns(result.cam_rows, result.frm_rows)
        init_violations = self._check_init_channel_match(model)
        result.channel_violations = pattern_violations + init_violations

        # ── ok kararı ──
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

    # ── Rapor yazdır ─────────────────────────────────────────────────
    @staticmethod
    def _print_report(result: ValidationResult) -> None:
        if result.error:
            print(f"\n{'=' * 60}")
            print(f"❌ KANAL UYUŞMAZLIĞI YAKALANDI:\n{result.error}")
            print(f"{'=' * 60}")

        print("\n[4] Katman Kanal Raporu:")
        print(f"  {'Layer':<8} {'Modül':<16} {'Giriş CH':>10} {'Çıkış CH':>10}")
        print(f"  {'-' * 48}")

        critical = {"C2f_CAM", "FRM"}
        for row in result.rows:
            marker = "★" if row.module_name in critical else " "
            print(f"  {row.layer_idx:<8} {row.module_name:<16} {row.in_ch:>10} {row.out_ch:>10}  {marker}")

        # ── C2f_CAM detayı ──
        print()
        if result.cam_rows:
            for r in result.cam_rows:
                print(f"  ✅ [Layer {r.layer_idx:>2} — C2f_CAM] giriş: {r.in_ch:<6} çıkış: {r.out_ch}")
        else:
            print("  ⚠️  C2f_CAM katmanı bulunamadı — YAML kontrolü yapın")

        # ── FRM detayı ──
        if result.frm_rows:
            for r in result.frm_rows:
                print(f"  ✅ [Layer {r.layer_idx:>2} — FRM     ] giriş: {r.in_ch:<6} çıkış: {r.out_ch}")
        else:
            print("  ⚠️  FRM katmanı bulunamadı — YAML kontrolü yapın")

        # ── Rebuild kontrolü ──
        print()
        if result.rebuild_detected:
            print("  ❌ REBUILD TESPİT EDİLDİ — model forward sırasında kendini yeniden inşa etti!")
            for m in result.rebuild_modules:
                print(f"     → {m}")
            print("     Bu, parse_model'ın yanlış kanal gönderdiği anlamına gelir.")
            print("     Çözüm: tasks.py patch veya YAML args düzeltilmeli.")
        else:
            print("  ✅ Rebuild tetiklenmedi — mimari init'te doğru kurulmuş.")

        # ── Strict kanal deseni ──
        if result.channel_violations:
            print()
            print("  ❌ KANAL DESENİ İHLALLERİ:")
            for v in result.channel_violations:
                print(f"     → {v}")
        else:
            print("  ✅ Kanal desenleri doğru (FRM: in==out, C2f_CAM: out<=in).")

        # ── Genel sonuç ──
        print()
        if result.ok:
            print("  ✅ Tüm kontroller geçti! Eğitimi başlatabilirsiniz.")
        else:
            print("  ❌ Sorun var — yukarıdaki uyarıları inceleyin.")
        print("=" * 60)


# ════════════════════════════════════════════════════════════════════════
# Public API — eğitim scriptlerinden çağrılır
# ════════════════════════════════════════════════════════════════════════

def validate_model_channels(
    model_yaml: str | Path | None = None,
    input_channels: int = 3,
    imgsz: int = 640,
    verbose: bool = True,
) -> dict[str, Any]:
    """Eski dict-based API — geriye uyumluluk için korunuyor."""
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
