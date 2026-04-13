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
    - [-1, 1, FRM, [1024]]             # Backbone sonrasında
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

        # ── Parametreleri sakla (forward-time auto-rebuild icin) ──────
        # parse_model ultra versiyonuna gore yanlis c1 gonderebilir.
        # forward() gercek input kanaline gore gerekirse rebuild yapar.
        self._c2 = c2
        self._n  = n
        self._sc = shortcut
        self._g  = g
        self._e  = e

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

    # ─── Forward-time kanal düzeltmesi ──────────────────────────────
    def _ensure_channels(self, x: torch.Tensor) -> None:
        """parse_model yanlis c1 gonderdiyse ilk forward'da otomatik rebuild yap.

        ultralytics parse_model, C2f_CAM'i tanimadiginda YAML args'i (128, False)
        direkt gecebilir: c1=128, c2=False -> defensive check -> c1=c2=128.
        Ama gercek input 256 kanal ise burada yakalanip duzeltilir.
        Bu fix tasks.py patch'e BAGIMLI DEGILDIR.
        """
        actual_c1 = x.shape[1]
        if actual_c1 == self.cv1.conv.in_channels:
            return  # Kanal sayisi dogru, islem yok

        # Yanlis c1 tespit edildi: rebuild
        c2, n, e = self._c2, self._n, self._e
        print(f"  [C2f_CAM] Auto-fix: c1 {self.cv1.conv.in_channels} -> {actual_c1} "
              f"(c2={c2}, n={n})")

        self.c   = int(c2 * e)
        self.cv1 = Conv(actual_c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, self._sc, self._g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        # ctx_conv ve channel_att c2 kullanir — c2 degismedi, rebuild gereksiz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş: C2f akışı + contextual attention."""
        # parse_model yanlis c1 gonderdiyse otomatik duzelt
        self._ensure_channels(x)

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
        self._ensure_channels(x)
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

        # ─── Branch 1: Yakın bağlam (dilation=2) ───────────────────
        # Yüz kenarlarını ve ince dokuları yakalama
        self.branch1 = nn.Sequential(
            Conv(c1, c_, 1),  # kanal azaltma
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),  # DW dilated
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),  # PW
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # ─── Branch 2: Uzak bağlam (dilation=4) ────────────────────
        # Kafa/omuz yapısını ve geniş yüz hatlarını yakalama
        self.branch2 = nn.Sequential(
            Conv(c1, c_, 1),  # kanal azaltma
            nn.Conv2d(c_, c_, kernel_size=3, padding=4, dilation=4, groups=c_, bias=False),  # DW dilated
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),  # PW
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # ─── Füzyon ─────────────────────────────────────────────────
        # İki branşı birleştir ve çıkış kanalına sıkıştır
        self.fuse = Conv(c_ * 2, c2, 1)

        # ─── Residual ───────────────────────────────────────────────
        # Kanal sayısı değişirse projeksiyon, değişmezse identity
        self.shortcut = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş: çift dilated kol + residual."""
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        refined = self.fuse(torch.cat([b1, b2], dim=1))  # (B, c2, H, W)
        return refined + self.shortcut(x)  # residual bağlantı
