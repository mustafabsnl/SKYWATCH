"""
SKYWATCH WIDER -> YOLO Dataset Dogrulama
=========================================
Donusum sonrasi kapsamli kontrol:
  1. Label format: class_id 0-1 arasi, 5 kolon, normalize degerler
  2. Resim-label eslesmesi: eksik/fazla dosya
  3. Bos label: yuz icermeyen resimler
  4. Bbox sinirlari: cx,cy,w,h mantikli mi
  5. Resim boyutlari: okunabirlik, minimum boyut
  6. data.yaml path kontrolu
"""

import sys
from pathlib import Path
from collections import Counter

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "skywatch_wider"


def check_labels(split: str) -> dict:
    labels_dir = DATASET / "labels" / split
    label_files = sorted(labels_dir.glob("*.txt"))

    stats = {
        "total_files": len(label_files),
        "total_boxes": 0,
        "empty_labels": 0,
        "bad_format": [],
        "bad_class_id": [],
        "out_of_range": [],
        "negative_size": [],
        "tiny_boxes": 0,
        "box_counts": [],
    }

    for lf in label_files:
        content = lf.read_text(encoding="utf-8").strip()
        if not content:
            stats["empty_labels"] += 1
            stats["box_counts"].append(0)
            continue

        lines = content.split("\n")
        stats["box_counts"].append(len(lines))

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()

            if len(parts) != 5:
                stats["bad_format"].append(f"{lf.name}:{line_num} ({len(parts)} cols)")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                stats["bad_format"].append(f"{lf.name}:{line_num} (parse error)")
                continue

            stats["total_boxes"] += 1

            if cls_id != 0:
                stats["bad_class_id"].append(f"{lf.name}:{line_num} class={cls_id}")

            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0.0 or val > 1.0:
                    stats["out_of_range"].append(f"{lf.name}:{line_num} {name}={val:.4f}")

            if w <= 0 or h <= 0:
                stats["negative_size"].append(f"{lf.name}:{line_num} w={w:.6f} h={h:.6f}")

            if w < 0.002 and h < 0.002:
                stats["tiny_boxes"] += 1

    return stats


def check_image_label_match(split: str) -> dict:
    images_dir = DATASET / "images" / split
    labels_dir = DATASET / "labels" / split

    img_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    lbl_stems = {p.stem for p in labels_dir.glob("*.txt")}

    return {
        "images": len(img_stems),
        "labels": len(lbl_stems),
        "missing_labels": sorted(img_stems - lbl_stems)[:10],
        "missing_images": sorted(lbl_stems - img_stems)[:10],
        "missing_labels_count": len(img_stems - lbl_stems),
        "missing_images_count": len(lbl_stems - img_stems),
    }


def check_sample_images(split: str, n: int = 50) -> dict:
    images_dir = DATASET / "images" / split
    imgs = sorted(images_dir.glob("*.jpg"))[:n]

    stats = {"checked": 0, "unreadable": [], "sizes": []}

    for img_path in imgs:
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                stats["sizes"].append(im.size)
            stats["checked"] += 1
        except Exception as e:
            stats["unreadable"].append(f"{img_path.name}: {e}")

    return stats


def check_data_yaml() -> dict:
    yaml_path = DATASET / "data.yaml"
    result = {"exists": yaml_path.exists(), "issues": []}

    if not yaml_path.exists():
        result["issues"].append("data.yaml bulunamadi!")
        return result

    content = yaml_path.read_text(encoding="utf-8")

    if "nc: 1" not in content:
        result["issues"].append("nc: 1 bulunamadi")
    if "face" not in content:
        result["issues"].append("'face' sinif adi bulunamadi")
    if "train: images/train" not in content:
        result["issues"].append("train path hatali")
    if "val: images/val" not in content:
        result["issues"].append("val path hatali")

    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        if not (DATASET / d).exists():
            result["issues"].append(f"Klasor eksik: {d}")

    return result


def main():
    print("=" * 60)
    print("  SKYWATCH Dataset Dogrulama")
    print("=" * 60)
    print(f"  Dataset: {DATASET}")

    all_ok = True

    # 1) data.yaml
    print("\n[1] data.yaml kontrolu...")
    yaml_check = check_data_yaml()
    if yaml_check["issues"]:
        for issue in yaml_check["issues"]:
            print(f"  HATA: {issue}")
        all_ok = False
    else:
        print("  OK: data.yaml dogru")

    for split in ["train", "val"]:
        print(f"\n{'='*60}")
        print(f"  [{split.upper()}] Kontrol")
        print(f"{'='*60}")

        # 2) Resim-label eslesmesi
        print(f"\n[2] Resim-label eslesmesi...")
        match = check_image_label_match(split)
        print(f"  Resim: {match['images']}, Label: {match['labels']}")
        if match["missing_labels_count"] > 0:
            print(f"  HATA: {match['missing_labels_count']} resmin etiketi yok!")
            print(f"    Ornek: {match['missing_labels'][:5]}")
            all_ok = False
        elif match["missing_images_count"] > 0:
            print(f"  HATA: {match['missing_images_count']} etiketin resmi yok!")
            print(f"    Ornek: {match['missing_images'][:5]}")
            all_ok = False
        else:
            print("  OK: Tum resim-label eslesmeleri tam")

        # 3) Label format
        print(f"\n[3] Label format kontrolu...")
        label_stats = check_labels(split)
        print(f"  Dosya: {label_stats['total_files']}, Toplam bbox: {label_stats['total_boxes']}")
        print(f"  Bos label (yuz yok): {label_stats['empty_labels']}")

        if label_stats["bad_format"]:
            print(f"  HATA: {len(label_stats['bad_format'])} satirda format hatasi!")
            for bf in label_stats["bad_format"][:5]:
                print(f"    -> {bf}")
            all_ok = False
        else:
            print("  OK: Tum satirlar 5 kolonlu")

        if label_stats["bad_class_id"]:
            print(f"  HATA: {len(label_stats['bad_class_id'])} satirda class_id != 0!")
            for bc in label_stats["bad_class_id"][:5]:
                print(f"    -> {bc}")
            all_ok = False
        else:
            print("  OK: Tum class_id = 0")

        if label_stats["out_of_range"]:
            print(f"  HATA: {len(label_stats['out_of_range'])} deger 0-1 araligi disinda!")
            for oor in label_stats["out_of_range"][:5]:
                print(f"    -> {oor}")
            all_ok = False
        else:
            print("  OK: Tum degerler [0, 1] araliginda")

        if label_stats["negative_size"]:
            print(f"  HATA: {len(label_stats['negative_size'])} bbox w<=0 veya h<=0!")
            all_ok = False
        else:
            print("  OK: Tum bbox boyutlari pozitif")

        if label_stats["tiny_boxes"] > 0:
            print(f"  BILGI: {label_stats['tiny_boxes']} cok kucuk bbox (w<0.002 ve h<0.002)")

        # Box dagilimi
        counts = label_stats["box_counts"]
        if counts:
            avg = sum(counts) / len(counts)
            mx = max(counts)
            print(f"  Ortalama yuz/resim: {avg:.1f}, Maksimum: {mx}")

        # 4) Resim okuma testi
        if HAS_PIL:
            print(f"\n[4] Resim okuma testi (ilk 50)...")
            img_stats = check_sample_images(split, n=50)
            if img_stats["unreadable"]:
                print(f"  HATA: {len(img_stats['unreadable'])} resim okunamadi!")
                for ur in img_stats["unreadable"][:5]:
                    print(f"    -> {ur}")
                all_ok = False
            else:
                print(f"  OK: {img_stats['checked']}/{img_stats['checked']} resim okunabilir")
                if img_stats["sizes"]:
                    widths = [s[0] for s in img_stats["sizes"]]
                    heights = [s[1] for s in img_stats["sizes"]]
                    print(f"  Boyut araligi: {min(widths)}x{min(heights)} ~ {max(widths)}x{max(heights)}")

    # Genel sonuc
    print("\n" + "=" * 60)
    if all_ok:
        print("  TAMAMLANDI: Tum kontroller gecti! Dataset egitim icin hazir.")
    else:
        print("  HATALAR VAR: Yukaridaki sorunlari duzelt ve tekrar calistir.")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
