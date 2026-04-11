"""
SKYWATCH — Dataset Analiz Scripti
Veri setinin detaylı istatistiklerini çıkarır:
- Görüntü sayısı, boyutları
- Label sayısı ve dağılımı
- Küçük yüz oranı (surveillance açısından kritik)
- Yüz başına bbox boyutu analizi

Çalıştır: python src/model/analyze_dataset.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────
ARCHIVE = Path("C:/Users/musta/OneDrive/Desktop/SKYWATCH/archive")
IMAGES_TRAIN = ARCHIVE / "images" / "train"
IMAGES_VAL   = ARCHIVE / "images" / "val"
LABELS_TRAIN = ARCHIVE / "labels" / "train"
LABELS_VAL   = ARCHIVE / "labels" / "val"

def analyze_split(img_dir: Path, lbl_dir: Path, split_name: str):
    print(f"\n{'='*55}")
    print(f"  {split_name.upper()} SET")
    print(f"{'='*55}")

    img_files = list(img_dir.glob("*"))
    img_files = [f for f in img_files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    lbl_files = list(lbl_dir.glob("*.txt"))

    print(f"  Görüntü sayısı : {len(img_files)}")
    print(f"  Label dosyası  : {len(lbl_files)}")

    # ── Label analizi ─────────────────────────────────────────────────
    face_counts = []       # Her görüntüdeki yüz sayısı
    bbox_widths  = []      # Normalize bbox genişlikleri
    bbox_heights = []      # Normalize bbox yükseklikleri
    missing_labels = 0
    empty_labels   = 0

    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_labels += 1
            continue

        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            empty_labels += 1
            face_counts.append(0)
            continue

        count = 0
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                _, xc, yc, w, h = parts[:5]
                bbox_widths.append(float(w))
                bbox_heights.append(float(h))
                count += 1
        face_counts.append(count)

    # ── Boyut analizi (örnek görüntüler) ──────────────────────────────
    widths, heights = [], []
    sample = img_files[:200]  # İlk 200 görüntüyü ölç
    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    # ── İstatistikler ─────────────────────────────────────────────────
    fc = np.array(face_counts)
    bw = np.array(bbox_widths)
    bh = np.array(bbox_heights)

    print(f"\n  [YÜZLER]")
    print(f"  Toplam yüz      : {int(fc.sum())}")
    print(f"  Görüntü başına  : ort={fc.mean():.2f}, max={fc.max()}, min={fc.min()}")
    print(f"  Etiket eksik    : {missing_labels}")
    print(f"  Boş etiket      : {empty_labels}")

    if len(bw) > 0:
        # Küçük yüz analizi (surveillance açısından kritik)
        # Normalize genişlik < 0.05 → görüntünün %5inden küçük → çok küçük yüz
        small_face_mask = (bw < 0.05) | (bh < 0.05)
        medium_face_mask = ((bw >= 0.05) & (bw < 0.15)) | ((bh >= 0.05) & (bh < 0.15))
        large_face_mask = (bw >= 0.15) & (bh >= 0.15)

        print(f"\n  [BBOX BOYUTU (normalize)]")
        print(f"  Genişlik  → ort:{bw.mean():.3f}, min:{bw.min():.3f}, max:{bw.max():.3f}")
        print(f"  Yükseklik → ort:{bh.mean():.3f}, min:{bh.min():.3f}, max:{bh.max():.3f}")

        total = len(bw)
        print(f"\n  [YÜZ BOYUT DAĞILIMI]")
        print(f"  🔴 Küçük  (< %5 )  : {small_face_mask.sum():5d}  ({100*small_face_mask.sum()/total:.1f}%)")
        print(f"  🟡 Orta   (%5-%15) : {medium_face_mask.sum():5d}  ({100*medium_face_mask.sum()/total:.1f}%)")
        print(f"  🟢 Büyük  (> %15)  : {large_face_mask.sum():5d}  ({100*large_face_mask.sum()/total:.1f}%)")

    if widths:
        print(f"\n  [GÖRÜNTÜ BOYUTU (ilk 200 örnek)]")
        print(f"  Genişlik  → ort:{np.mean(widths):.0f}px, min:{min(widths)}px, max:{max(widths)}px")
        print(f"  Yükseklik → ort:{np.mean(heights):.0f}px, min:{min(heights)}px, max:{max(heights)}px")

    return {
        "total_images": len(img_files),
        "total_faces": int(fc.sum()) if len(fc) > 0 else 0,
        "small_face_ratio": float(small_face_mask.sum() / len(bw)) if len(bw) > 0 else 0
    }


def check_label_format(lbl_dir: Path):
    """İlk 3 label dosyasının içeriğini göster."""
    print(f"\n{'='*55}")
    print(f"  LABEL FORMAT KONTROLÜ")
    print(f"{'='*55}")
    files = list(lbl_dir.glob("*.txt"))[:3]
    for f in files:
        print(f"\n  📄 {f.name}")
        lines = f.read_text().splitlines()[:5]
        for line in lines:
            print(f"     {line}")


if __name__ == "__main__":
    print("\n🔍 SKYWATCH — Dataset Analizi")
    print("─" * 55)

    # Label format kontrolü
    check_label_format(LABELS_TRAIN)

    # Train ve Val analizi
    train_stats = analyze_split(IMAGES_TRAIN, LABELS_TRAIN, "train")
    val_stats   = analyze_split(IMAGES_VAL,   LABELS_VAL,   "val")

    print(f"\n{'='*55}")
    print(f"  ÖZET")
    print(f"{'='*55}")
    print(f"  Toplam görüntü  : {train_stats['total_images'] + val_stats['total_images']}")
    print(f"  Toplam yüz      : {train_stats['total_faces'] + val_stats['total_faces']}")
    ratio = train_stats['small_face_ratio']
    print(f"  Küçük yüz oranı : {ratio*100:.1f}%  ← {'⚠️ Yüksek, iyileştirme kritik!' if ratio > 0.2 else '✅ Normal'}")
    print(f"\n  ✅ Analiz tamamlandı.\n")
