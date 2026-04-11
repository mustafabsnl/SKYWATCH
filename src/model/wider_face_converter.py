"""
WIDER FACE -> YOLO Detection Format Donusturucu

WIDER FACE formatindan YOLO Detection formatina donusum yapar.
YOLO Detection formati: class cx cy w h  (5 sutun)

WIDER FACE annotation formati:
    # image_path
    N_faces
    x1 y1 w h blur expression illumination invalid occlusion pose

YOLO Detection ciktisi:
    0 cx cy w h


WIDER FACE annotation formati:
    # image_path
    N_faces
    x1 y1 w h blur expression illumination invalid occlusion pose

YOLO Pose ciktisi (landmark yok, visibility=0):
    0 cx cy w h 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Kullanim:
    python wider_face_converter.py \
        --wider_root /path/to/WIDER_FACE \
        --output_dir /path/to/yolo_dataset

Kaggle'da kullanim:
    python wider_face_converter.py \
        --wider_root /kaggle/input/wider-face \
        --output_dir /kaggle/working/skywatch_data
"""

import sys
import os
# Windows'ta UTF-8 stdout zorla
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import shutil
from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════════════════
# Sabitler
# ════════════════════════════════════════════════════════════════════════

# YOLO sınıf indeksi — sadece 1 sınıf var: yüz
FACE_CLASS = 0


# Küçük yüzleri filtrele (min piksel boyutu)
MIN_FACE_SIZE = 5  # 5x5 pikselden küçük yüzleri atla

# Geçersiz (invalid=1) yüzleri atlayıp atlamama
SKIP_INVALID = False  # True: geçersiz yüzleri atla, False: hepsini dahil et


# ════════════════════════════════════════════════════════════════════════
# WIDER FACE Annotation Parser
# ════════════════════════════════════════════════════════════════════════

def parse_wider_face_annotation(ann_file: Path) -> dict[str, list[list[float]]]:
    """WIDER FACE annotation dosyasını parse et.

    Args:
        ann_file (Path): wider_face_train_bbx_gt.txt veya val versiyonu.

    Returns:
        dict: {image_relative_path: [[x1, y1, w, h, blur, ...], ...]}
    """
    annotations = {}

    with open(ann_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        # Görüntü yolu
        img_path = lines[i].strip()
        i += 1

        if i >= len(lines):
            break

        # Yüz sayısı
        n_faces = int(lines[i])
        i += 1

        faces = []
        for _ in range(max(n_faces, 1)):  # En az 1 satır oku (n=0 durumu)
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1

            if len(parts) >= 8:
                x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                blur = int(parts[4])
                expression = int(parts[5])
                illumination = int(parts[6])
                invalid = int(parts[7]) if len(parts) > 7 else 0
                occlusion = int(parts[8]) if len(parts) > 8 else 0
                pose = int(parts[9]) if len(parts) > 9 else 0

                if n_faces > 0:  # n_faces=0 ise dummy satırı atla
                    faces.append([x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose])

        annotations[img_path] = faces

    print(f"  Parse edildi: {len(annotations)} görüntü")
    return annotations


# ════════════════════════════════════════════════════════════════════════
# Yardımcı Fonksiyonlar
# ════════════════════════════════════════════════════════════════════════

def xywh_to_normalized_cxcywh(x1: float, y1: float, w: float, h: float,
                                img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """WIDER FACE [x,y,w,h] → YOLO normalize [cx,cy,w,h].

    Args:
        x1, y1, w, h: WIDER FACE bbox (piksel, sol üst köşe + genişlik/yükseklik).
        img_w, img_h: Görüntü boyutları.

    Returns:
        (cx, cy, w_norm, h_norm) normalize edilmiş veya None (geçersiz kutu).
    """
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + w / 2) / img_w
    cy = (y1 + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Sınır kontrolü
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    w_norm = np.clip(w_norm, 0, 1)
    h_norm = np.clip(h_norm, 0, 1)

    if w_norm < 1e-6 or h_norm < 1e-6:
        return None

    return cx, cy, w_norm, h_norm


def get_image_size(img_path: Path) -> tuple[int, int] | None:
    """Görüntü boyutunu oku (sadece header, hızlı).

    Args:
        img_path (Path): Görüntü dosyası yolu.

    Returns:
        (width, height) veya None (dosya yoksa).
    """
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════
# Ana Dönüştürücü
# ════════════════════════════════════════════════════════════════════════

def convert_split(
    split: str,
    wider_root: Path,
    output_dir: Path,
    skip_invalid: bool = SKIP_INVALID,
    min_face_size: int = MIN_FACE_SIZE,
) -> dict[str, int]:
    """Bir split'i (train/val) dönüştür."""

    # ── Esnek yol algılaması (iç içe veya düz yapı) ─────────────────
    # Düz yapı (kullanıcı düzeltti): WIDER_train/images/
    # İç içe yapı (orijinal zip):    WIDER_train/WIDER_train/images/
    def _find_img_root(split_name):
        candidates = [
            wider_root / f"WIDER_{split_name}" / "images",
            wider_root / f"WIDER_{split_name}" / f"WIDER_{split_name}" / "images",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"Görüntü dizini bulunamadı! Denenen:\n" +
            "\n".join(f"  {c}" for c in candidates)
        )

    def _find_ann_file(split_name):
        candidates = [
            wider_root / "wider_face_split" / f"wider_face_{split_name}_bbx_gt.txt",
            wider_root / "wider_face_split" / "wider_face_split" / f"wider_face_{split_name}_bbx_gt.txt",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"Annotation dosyası bulunamadı! Denenen:\n" +
            "\n".join(f"  {c}" for c in candidates)
        )
    # ─────────────────────────────────────────────────────────────────

    ann_file = _find_ann_file(split)
    img_root = _find_img_root(split)


    # Çıktı dizinleri
    out_img_dir = output_dir / "images" / split
    out_lbl_dir = output_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  {split.upper()} split dönüştürülüyor...")
    print(f"{'='*55}")

    annotations = parse_wider_face_annotation(ann_file)

    stats = {
        "total_images": 0,
        "converted_images": 0,
        "total_faces": 0,
        "kept_faces": 0,
        "skipped_invalid": 0,
        "skipped_small": 0,
        "missing_images": 0,
    }

    for rel_path, faces in annotations.items():
        stats["total_images"] += 1
        img_src = img_root / rel_path

        if not img_src.exists():
            stats["missing_images"] += 1
            continue

        # Görüntü boyutunu al
        size = get_image_size(img_src)
        if size is None:
            stats["missing_images"] += 1
            continue
        img_w, img_h = size

        # Label içeriği oluştur
        label_lines = []
        for face in faces:
            stats["total_faces"] += 1
            x1, y1, w, h = face[0], face[1], face[2], face[3]
            invalid = int(face[7]) if len(face) > 7 else 0

            # Geçersiz yüzü atla
            if skip_invalid and invalid == 1:
                stats["skipped_invalid"] += 1
                continue

            # Çok küçük yüzü atla
            if w < min_face_size or h < min_face_size:
                stats["skipped_small"] += 1
                continue

            # Koordinat dönüşümü
            result = xywh_to_normalized_cxcywh(x1, y1, w, h, img_w, img_h)
            if result is None:
                continue

            cx, cy, w_n, h_n = result
            stats["kept_faces"] += 1

            # YOLO Detection formati: class cx cy w h  (5 sutun)
            label_lines.append(f"{FACE_CLASS} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}")

        # Label dosyasını kaydet (yüz yoksa bile boş dosya oluştur)
        # Relative path'ten güvenli dosya adı oluştur
        safe_name = rel_path.replace("/", "_").replace("\\", "_")
        stem = Path(safe_name).stem

        lbl_dst = out_lbl_dir / f"{stem}.txt"
        with open(lbl_dst, "w") as f:
            f.write("\n".join(label_lines))

        # Görüntüyü kopyala
        img_dst = out_img_dir / f"{stem}{img_src.suffix}"
        shutil.copy2(img_src, img_dst)
        stats["converted_images"] += 1

    # Rapor
    print(f"  Toplam görüntü:    {stats['total_images']:,}")
    print(f"  Dönüştürülen:      {stats['converted_images']:,}")
    print(f"  Eksik görüntü:     {stats['missing_images']:,}")
    print(f"  Toplam yüz:        {stats['total_faces']:,}")
    print(f"  Saklanan yüz:      {stats['kept_faces']:,}")
    print(f"  Atlanan (küçük):   {stats['skipped_small']:,}")
    print(f"  Atlanan (invalid): {stats['skipped_invalid']:,}")

    return stats


def write_data_yaml(output_dir: Path, nc: int = 1) -> None:
    """YOLO data.yaml dosyasi olustur. Val yoksa train'e fallback yapar."""
    val_img_dir = output_dir / "images" / "val"
    # Val goruntuleri var mi? Yoksa train'e yonlendir
    if val_img_dir.exists() and any(val_img_dir.iterdir()):
        val_path = "images/val"
        print("  Val split: images/val")
    else:
        val_path = "images/train"
        print("  [!] Val split bulunamadi, val -> images/train olarak ayarlandi")
        print("      Kaggle'da tam WIDER_val dataseti ile dogru calisacak.")

    yaml_content = f"""path: {str(output_dir.resolve()).replace(chr(92), '/')}
train: images/train
val: {val_path}

nc: {nc}
names: ['face']
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  data.yaml olusturuldu: {yaml_path}")



# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WIDER FACE → YOLO Pose Format Dönüştürücü"
    )
    parser.add_argument(
        "--wider_root", type=Path, required=True,
        help="WIDER_FACE kök dizini (wider_face_split/, WIDER_train/, WIDER_val/ içermeli)"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="YOLO format çıktı dizini"
    )
    parser.add_argument(
        "--skip_invalid", action="store_true",
        help="Geçersiz (invalid=1) yüzleri atla"
    )
    parser.add_argument(
        "--min_face_size", type=int, default=MIN_FACE_SIZE,
        help=f"Minimum yüz boyutu piksel (varsayılan: {MIN_FACE_SIZE})"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Dönüştürülecek split'ler (varsayılan: train val)"
    )
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  WIDER FACE → YOLO Pose Dönüştürücü")
    print("  SKYWATCH-Det Veri Hazırlama")
    print("="*55)
    print(f"  Kaynak:   {args.wider_root}")
    print(f"  Çıktı:    {args.output_dir}")
    print(f"  Min yüz:  {args.min_face_size}px")

    total_stats = {}
    for split in args.splits:
        try:
            stats = convert_split(
                split=split,
                wider_root=args.wider_root,
                output_dir=args.output_dir,
                skip_invalid=args.skip_invalid,
                min_face_size=args.min_face_size,
            )
            total_stats[split] = stats
        except FileNotFoundError as e:
            print(f"\n[HATA] {e}")
            print(f"  '{split}' split'i atlandı.")

    write_data_yaml(args.output_dir)

    print("\n" + "="*55)
    print("  DÖNÜŞÜM TAMAMLANDI")
    print("="*55)
    for split, stats in total_stats.items():
        print(f"  [{split}] {stats['converted_images']:,} görüntü, {stats['kept_faces']:,} yüz")


if __name__ == "__main__":
    main()
