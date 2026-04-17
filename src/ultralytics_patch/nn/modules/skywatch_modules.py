"""
SKYWATCH Custom Neural Network Modules

Bu dosya iki ana özel modül içerir:

1. C2f_CAM — Contextual Attention Module ile güçlendirilmiş C2f bloğu
   - Dilate edilmiş conv ile geniş bağlam yakalama
   - Channel attention (SE-like) ile önemli özellikleri öne çıkarma
   - P2 detection head'de küçük yüzler için kullanılır
   Kaynak: "Contextual Feature Enhancement for Small Face Detection in Crowded Scenes"

2. FRM — Feature Refinement Module
   - Bulanık görüntülerde kaybolan kenar bilgisini kurtarma
   - Çift kollu dilated convolution (dilation=2 ve dilation=4)
   - Residual connection ile orijinal bilgiyi koruma
   Kaynak: "Face Detection in Low-quality Images via Deep Feature Enhancement"

Kullanım (YAML):
    - [-1, 2, C2f_CAM, [128, False]]   # P2 head'de
    - [-1, 1, FRM, []]                 # Backbone sonrasında (pass-through: c2=c1, tasks.py otomatik çözer)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Ultralytics temel blokları
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck

__all__ = ("C2f_CAM", "FRM")


# ════════════════════════════════════════════════════════════════════════
# C2f_CAM — Contextual Attention Module
# ════════════════════════════════════════════════════════════════════════

class C2f_CAM(nn.Module):
    """C2f with Contextual Attention Module (CAM) for P2 small face detection.

    Standard C2f akışını iki katmanlı bir dikkat mekanizmasıyla güçlendirir:
    1. Dilated convolution branch: 7x7 efektif alanda bağlamsal bilgi toplar
       (Yüzün etrafındaki omuz/kafa yapısını anlama — küçük yüz için kritik)
    2. Channel Attention (SE-block): Hangi özellik kanallarının önemli olduğunu öğrenir
       (Yüz dokuları vs arka plan dokularını ayırt etme)

    Kaynak: "Contextual Feature Enhancement for Small Face Detection in Crowded Scenes"
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """C2f_CAM'ı başlat."""
        super().__init__()

        # ── Savunmaci tip kontrolu ────────────────────────────────────
        if isinstance(c2, bool):
            shortcut = bool(c2)
            c2 = int(c1)
        if isinstance(n, bool):
            shortcut = bool(n)
            n = 1

        c1 = int(c1)
        c2 = int(c2)
        n  = max(int(n), 0)
        e  = float(e)

        self._init_c1 = c1
        self._c2 = c2
        self._n  = n
        self._sc = shortcut
        self._g  = g
        self._e  = e
        self._rebuilt = False

        self.c = int(c2 * e)  # gizli kanal sayisi

        # ─── Standart C2f Bileşenleri ───────────────────────────────
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        # ─── Contextual Attention Branch ────────────────────────────
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=2, dilation=2, groups=c2, bias=False),
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # ─── Channel Attention (SE-block) ───────────────────────────
        c_squeeze = max(c2 // 4, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, c_squeeze),
            nn.SiLU(inplace=True),
            nn.Linear(c_squeeze, c2),
            nn.Sigmoid(),
        )

    # ─── Kanal doğrulama (Fail-Fast) ────────────────────────────────
    def _validate_channels(self, x: torch.Tensor) -> None:
        """Giriş kanalı beklenenle uyuşmuyorsa açık hata fırlat.

        Sessiz auto-rebuild DEĞİL — Fail-Fast prensibi:
          Eğitim başlamadan önce kanal uyumsuzluğu açığa çıksın,
          tasks.py patch veya YAML args düzeltilerek çözülsün.

        Nasıl okunur:
          Hata gelirse:
          1. kaggle_setup.py → patch_tasks() içinde elif m is C2f_CAM: bloğunu kontrol et
          2. skywatch-det.yaml → C2f_CAM satırındaki args'ı kontrol et
          3. channel_validator.py çalıştır, tekrar dene
        """
        actual_c1 = x.shape[1]
        expected_c1 = self.cv1.conv.in_channels
        if actual_c1 != expected_c1:
            raise ValueError(
                f"\n{'='*60}\n"
                f"  [C2f_CAM] KANAL UYUŞMAZLIĞI — EĞİTİMİ DURDURUN!\n"
                f"{'='*60}\n"
                f"  Beklenen giriş kanalı : {expected_c1}\n"
                f"  Gelen giriş kanalı    : {actual_c1}\n"
                f"  Çözüm:\n"
                f"    1. kaggle_setup.py → patch_tasks() → elif m is C2f_CAM: bloğunu kontrol et\n"
                f"    2. skywatch-det.yaml → C2f_CAM satırındaki args'ı düzelt\n"
                f"    3. channel_validator.py çalıştır, tekrar dene\n"
                f"{'='*60}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş: C2f akışı + contextual attention."""
        self._validate_channels(x)

        # ── Standart C2f akışı ──
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))  # (B, c2, H, W)

        # ── Contextual Attention ──
        ctx = self.ctx_conv(out)
        w = self.channel_att(ctx).view(ctx.shape[0], ctx.shape[1], 1, 1)

        return out + ctx * w

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """split() kullanan alternatif forward (chunk yerine)."""
        self._validate_channels(x)
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        ctx = self.ctx_conv(out)
        w = self.channel_att(ctx).view(ctx.shape[0], ctx.shape[1], 1, 1)
        return out + ctx * w


# ════════════════════════════════════════════════════════════════════════
# FRM — Feature Refinement Module
# ════════════════════════════════════════════════════════════════════════

class FRM(nn.Module):
    """Feature Refinement Module (FRM) for blurry/low-quality surveillance images.

    Bulanık güvenlik kamerası görüntülerinde kaybolan kenar ve doku bilgisini
    backbone'dan neck'e geçmeden önce kurtarır.

    İki paralel dilated convolution kolu:
    - Branch 1 (dilation=2): Yakın bağlam (yüz kenarları)
    - Branch 2 (dilation=4): Uzak bağlam (kafa/omuz yapısı)
    Her iki kol birleştirilip 1x1 conv ile yoğunlaştırılır ve residual eklenir.

    Kaynak: "Face Detection in Low-quality Images via Deep Feature Enhancement"
    """

    def __init__(self, c1: int, c2: int | None = None):
        """FRM'yi başlat.

        Args:
            c1 (int): Giriş kanal sayısı.
            c2 (int | None): Çıkış kanal sayısı. None ise c1 ile aynı.
        """
        super().__init__()
        c2 = c2 or c1
        c_ = max(c1 // 2, 32)  # gizli kanal sayısı (en az 32)

        self._init_c1 = c1
        self._c1 = c1
        self._c2 = c2
        self._rebuilt = False

        # ─── Branch 1: Yakın bağlam (dilation=2) ───────────────────
        self.branch1 = nn.Sequential(
            Conv(c1, c_, 1),
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # ─── Branch 2: Uzak bağlam (dilation=4) ────────────────────
        self.branch2 = nn.Sequential(
            Conv(c1, c_, 1),
            nn.Conv2d(c_, c_, kernel_size=3, padding=4, dilation=4, groups=c_, bias=False),
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # ─── Füzyon ─────────────────────────────────────────────────
        self.fuse = Conv(c_ * 2, c2, 1)

        # ─── Residual ───────────────────────────────────────────────
        self.shortcut = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def _validate_channels(self, x: torch.Tensor) -> None:
        """Giriş kanalı beklenenle uyuşmuyorsa açık hata fırlat.

        FRM pass-through modülüdür: giriş kaç kanal ise çıkış da o olur.
        YAML'da args=[] (boş) olmalı — c1/c2 tasks.py parse_model tarafından
        bir önceki katmanın çıkışından otomatik belirlenir.
        """
        actual_c1 = x.shape[1]
        expected_c1 = self.branch1[0].conv.in_channels
        if actual_c1 != expected_c1:
            raise ValueError(
                f"\n{'='*60}\n"
                f"  [FRM] KANAL UYUŞMAZLIĞI — EĞİTİMİ DURDURUN!\n"
                f"{'='*60}\n"
                f"  Beklenen giriş kanalı : {expected_c1}\n"
                f"  Gelen giriş kanalı    : {actual_c1}\n"
                f"  FRM pass-through'dur (c2=c1), YAML'da args=[] olmalıdır.\n"
                f"  Çözüm:\n"
                f"    1. skywatch-det.yaml → FRM satırı: [-1, 1, FRM, []] olmalı\n"
                f"    2. tasks.py patch → elif m is FRM: bloğunun c1=ch[f] verdiğini kontrol et\n"
                f"    3. channel_validator.py çalıştır, tekrar dene\n"
                f"{'='*60}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş: çift dilated kol + residual."""
        self._validate_channels(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        refined = self.fuse(torch.cat([b1, b2], dim=1))  # (B, c2, H, W)
        return refined + self.shortcut(x)  # residual bağlantı
