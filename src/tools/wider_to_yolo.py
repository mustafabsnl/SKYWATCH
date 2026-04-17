"""
WIDER FACE → YOLO Format Dönüştürücü
=====================================
WIDER FACE dataset'ini YOLO eğitim formatına dönüştürür.

Kaynak yapı:
  Kalabalık_yuz_veri_seti/
  ├── WIDER_train/WIDER_train/images/0--Parade/*.jpg
  ├── WIDER_val/WIDER_val/images/0--Parade/*.jpg
  └── wider_face_split/wider_face_split/
      ├── wider_face_train_bbx_gt.txt
      └── wider_face_val_bbx_gt.txt

Hedef yapı:
  skywatch_wider/
  ├── images/train/  (düz, alt klasörsüz)
  ├── images/val/
  ├── labels/train/  (YOLO formatı: 0 cx cy w h)
  ├── labels/val/
  └── data.yaml

WIDER formatı:  x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
YOLO formatı:   class_id  cx  cy  w  h   (normalize edilmiş, 0-1 arası)

Kullanım:
  python src/tools/wider_to_yolo.py
  python src/tools/wider_to_yolo.py --src "C:/path/to/Kalabalik_yuz_veri_seti" --dst "C:/path/to/output"
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SRC = PROJECT_ROOT / "Kalabalık_yuz_veri_seti"
DEFAULT_DST = PROJECT_ROOT / "skywatch_wider"

MIN_FACE_PX = 1


def parse_wider_gt(gt_path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """WIDER ground-truth txt dosyasını parse et.

    Returns:
        {relative_image_path: [(x1, y1, w, h), ...]}
        invalid=1 olan yüzler filtrelenir.
    """
    annotations: dict[str, list[tuple[int, int, int, int]]] = {}
    lines = gt_path.read_text(encoding="utf-8").strip().split("\n")

    i = 0
    skipped_invalid = 0
    skipped_tiny = 0

    while i < len(lines):
        img_rel = lines[i].strip()
        i += 1

        n_faces = int(lines[i].strip())
        i += 1

        boxes: list[tuple[int, int, int, int]] = []
        for _ in range(max(n_faces, 1)):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            i += 1

            if n_faces == 0:
                continue

            if len(parts) < 4:
                continue

            x1, y1, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

            invalid = int(parts[7]) if len(parts) > 7 else 0
            if invalid == 1:
                skipped_invalid += 1
                continue

            if w < MIN_FACE_PX or h < MIN_FACE_PX:
                skipped_tiny += 1
                continue

            boxes.append((x1, y1, w, h))

        annotations[img_rel] = boxes

    print(f"  Parse: {gt_path.name}")
    print(f"    Resim: {len(annotations)}")
    total_faces = sum(len(b) for b in annotations.values())
    print(f"    Yuz : {total_faces}")
    print(f"    Filtrelenen: invalid={skipped_invalid}, tiny(<{MIN_FACE_PX}px)={skipped_tiny}")

    return annotations


def wider_to_yolo_label(
    boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> list[str]:
    """WIDER bbox'ları YOLO formatına dönüştür.

    WIDER: x1, y1, w, h (piksel, sol-üst köşe)
    YOLO:  class cx cy w h (normalize 0-1, merkez)
    """
    yolo_lines: list[str] = []
    for x1, y1, bw, bh in boxes:
        cx = (x1 + bw / 2) / img_w
        cy = (y1 + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        if nw > 0 and nh > 0:
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def convert_split(
    annotations: dict[str, list[tuple[int, int, int, int]]],
    images_root: Path,
    dst_images: Path,
    dst_labels: Path,
    split_name: str,
) -> dict[str, int]:
    """Tek bir split'i (train veya val) dönüştür."""
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "faces": 0, "skipped_missing": 0, "skipped_read_err": 0}

    for img_rel, boxes in annotations.items():
        src_img = images_root / img_rel

        if not src_img.exists():
            stats["skipped_missing"] += 1
            continue

        flat_name = img_rel.replace("/", "_").replace("\\", "_")
        stem = Path(flat_name).stem

        try:
            with Image.open(src_img) as im:
                img_w, img_h = im.size
        except Exception:
            stats["skipped_read_err"] += 1
            continue

        yolo_lines = wider_to_yolo_label(boxes, img_w, img_h)

        dst_img_path = dst_images / flat_name
        dst_lbl_path = dst_labels / f"{stem}.txt"

        shutil.copy2(src_img, dst_img_path)
        dst_lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        stats["images"] += 1
        stats["faces"] += len(yolo_lines)

    print(f"\n  [{split_name}] Sonuc:")
    print(f"    Resim     : {stats['images']}")
    print(f"    Yuz       : {stats['faces']}")
    if stats["skipped_missing"]:
        print(f"    Bulunamayan: {stats['skipped_missing']}")
    if stats["skipped_read_err"]:
        print(f"    Okunamayan : {stats['skipped_read_err']}")

    return stats


def create_data_yaml(dst: Path, stats_train: dict, stats_val: dict):
    """YOLO data.yaml oluştur."""
    yaml_content = (
        f"path: {dst.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: face\n"
        f"\n"
        f"# WIDER FACE -> YOLO conversion\n"
        f"# Train: {stats_train['images']} images, {stats_train['faces']} faces\n"
        f"# Val  : {stats_val['images']} images, {stats_val['faces']} faces\n"
    )
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n  data.yaml: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="WIDER FACE -> YOLO converter")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="Source WIDER FACE folder")
    parser.add_argument("--dst", type=str, default=str(DEFAULT_DST), help="Target YOLO folder")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print("=" * 60)
    print("  WIDER FACE -> YOLO Donusturucu")
    print("=" * 60)
    print(f"  Kaynak : {src}")
    print(f"  Hedef  : {dst}")

    gt_dir = src / "wider_face_split" / "wider_face_split"
    train_gt = gt_dir / "wider_face_train_bbx_gt.txt"
    val_gt = gt_dir / "wider_face_val_bbx_gt.txt"

    train_images = src / "WIDER_train" / "WIDER_train" / "images"
    val_images = src / "WIDER_val" / "WIDER_val" / "images"

    for p, desc in [(train_gt, "Train GT"), (val_gt, "Val GT"), (train_images, "Train images"), (val_images, "Val images")]:
        if not p.exists():
            print(f"\n  HATA: {desc} bulunamadi: {p}")
            return

    print("\n[1] Ground-truth parse ediliyor...")
    train_ann = parse_wider_gt(train_gt)
    val_ann = parse_wider_gt(val_gt)

    print("\n[2] Train donusturuluyor...")
    stats_train = convert_split(train_ann, train_images, dst / "images" / "train", dst / "labels" / "train", "TRAIN")

    print("\n[3] Val donusturuluyor...")
    stats_val = convert_split(val_ann, val_images, dst / "images" / "val", dst / "labels" / "val", "VAL")

    print("\n[4] data.yaml olusturuluyor...")
    create_data_yaml(dst, stats_train, stats_val)

    print("\n" + "=" * 60)
    print("  DONUSUM TAMAMLANDI!")
    print(f"  Train : {stats_train['images']} resim, {stats_train['faces']} yuz")
    print(f"  Val   : {stats_val['images']} resim, {stats_val['faces']} yuz")
    print(f"  Toplam: {stats_train['images'] + stats_val['images']} resim, {stats_train['faces'] + stats_val['faces']} yuz")
    print(f"  Cikti : {dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
