"""
SKYWATCH — Video Dosyası Test Aracı
=====================================
Eğitilmiş best.pt modelini bir video dosyası üzerinden test eder.
Sonuçları ekranda gösterir ve isteğe bağlı olarak yeni video olarak kaydeder.

Kullanım:
    python src/tools/video_test.py
    python src/tools/video_test.py --video Video1.webm --conf 0.35
    python src/tools/video_test.py --video Video1.webm --save

Kontroller:
    q / ESC  → Çıkış
    SPACE    → Duraklat / Devam et
    s        → Ekran görüntüsü kaydet
    +/-      → Confidence eşiğini artır/azalt
    r        → Başa sar
    f        → 10 saniye ileri
    b        → 10 saniye geri

Gereksinimler: C:\\Users\\musta\\OneDrive\\Desktop\\SKYWATCH\\venv\\Scripts\\activate.bat
    pip install ultralytics opencv-python
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# ── Proje kökünü bul ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("[HATA] ultralytics kütüphanesi bulunamadı.")
    print("       Kurulum: pip install ultralytics")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
# Yardımcı Fonksiyonlar
# ════════════════════════════════════════════════════════════

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0,   0, 90, color, thickness)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def draw_progress_bar(frame, current_frame, total_frames, fps_video):
    """Alt kısma ilerleme çubuğu çizer."""
    h, w = frame.shape[:2]
    bar_h = 28
    y_start = h - bar_h

    # Arka plan
    cv2.rectangle(frame, (0, y_start), (w, h), (20, 20, 20), -1)

    # İlerleme
    if total_frames > 0:
        progress = current_frame / total_frames
        bar_w = int((w - 120) * progress)
        # Gri arka plan çubuğu
        cv2.rectangle(frame, (10, y_start + 8), (w - 110, y_start + 18), (60, 60, 60), -1)
        # Renkli ilerleme
        cv2.rectangle(frame, (10, y_start + 8), (10 + bar_w, y_start + 18), (0, 200, 255), -1)

        # Zaman gösterimi
        current_sec = current_frame / max(fps_video, 1)
        total_sec   = total_frames  / max(fps_video, 1)
        time_text   = f"{format_time(current_sec)} / {format_time(total_sec)}"
        cv2.putText(frame, time_text, (w - 105, y_start + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def draw_detections(frame, results, conf_threshold):
    det_count = 0
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        det_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w_box = x2 - x1
        h_box = y2 - y1

        ratio  = min((conf - conf_threshold) / max(1.0 - conf_threshold, 0.01), 1.0)
        b      = int(200 * (1 - ratio))
        g      = int(255 * ratio)
        r      = int(50 + 100 * (1 - ratio))
        color  = (b, g, r)

        draw_rounded_rect(frame, (x1, y1), (x2, y2), color, 2, radius=8)

        corner_len = min(20, w_box // 3, h_box // 3)
        t = 3
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

        # Etiket
        label = f"Yuz {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 5, lh + 10)
        cv2.rectangle(frame, (x1, ly - lh - 6), (x1 + lw + 8, ly), color, -1)
        cv2.putText(frame, label, (x1 + 4, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"{w_box}x{h_box}px", (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    return det_count


def draw_hud(frame, fps, det_count, conf_threshold, total_detections,
             current_frame, total_frames, inference_ms, paused, saving):
    h, w = frame.shape[:2]

    # Sol üst panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 215), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Başlık
    title_color = (0, 80, 255) if paused else (0, 200, 255)
    cv2.putText(frame, "SKYWATCH - Video Test",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2, cv2.LINE_AA)

    # PAUSED göstergesi
    if paused:
        cv2.putText(frame, "|| DURAKLATILDI", (150, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (20, 48), (320, 48), title_color, 1)

    fps_color = (0, 255, 100) if fps >= 20 else ((0, 220, 255) if fps >= 10 else (0, 80, 255))

    progress_pct = (current_frame / max(total_frames, 1)) * 100
    info = [
        (f"FPS: {fps:.1f}",                         fps_color),
        (f"Inference: {inference_ms:.1f} ms",        (200, 200, 200)),
        (f"Tespit: {det_count} yuz",                 (255, 255, 255)),
        (f"Conf Esik: {conf_threshold:.2f}",         (180, 180, 180)),
        (f"Toplam Tespit: {total_detections}",       (180, 180, 180)),
        (f"Kare: {current_frame}/{total_frames} ({progress_pct:.1f}%)", (140, 140, 140)),
    ]
    y = 72
    for text, color in info:
        cv2.putText(frame, text, (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
        y += 22

    # Kayıt göstergesi (sağ üst)
    if saving:
        cv2.circle(frame, (w - 24, 24), 8, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 60, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)

    # Alt kısım — kontrol bilgisi
    bar_h = 28
    hints = "[SPACE] Duraklat   [S] Screenshot   [+/-] Conf   [F/B] ileri/geri   [R] Basa sar   [Q] Cikis"
    cv2.putText(frame, hints, (10, h - bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════
# Ana Çalıştırıcı
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SKYWATCH — Video Dosyası Detection Test"
    )
    parser.add_argument(
        "--video", type=str,
        default=str(PROJECT_ROOT / "Video1.webm"),
        help="Video dosyası yolu (default: Video1.webm)"
    )
    parser.add_argument(
        "--model", type=str,
        default=str(PROJECT_ROOT / "best.pt"),
        help="Model dosyası yolu (default: best.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Başlangıç confidence eşiği (default: 0.35)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLO inference boyutu (default: 640)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Tespit sonuçlarını video olarak kaydet"
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=str(PROJECT_ROOT / "logs" / "video_results"),
        help="Çıktı kayıt dizini"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Oynatma hızı çarpanı (default: 1.0, örn: 0.5 yavaş, 2.0 hızlı)"
    )
    args = parser.parse_args()

    # ── Doğrulamalar ──
    model_path = Path(args.model)
    video_path = Path(args.video)

    if not model_path.exists():
        print(f"[HATA] Model bulunamadı: {model_path}")
        sys.exit(1)
    if not video_path.exists():
        print(f"[HATA] Video bulunamadı: {video_path}")
        sys.exit(1)

    print(f"\n{'='*58}")
    print(f"  SKYWATCH — Video Test Aracı")
    print(f"{'='*58}")
    print(f"  Video   : {video_path.name}")
    print(f"  Model   : {model_path.name} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Conf    : {args.conf}")
    print(f"  Kayıt   : {'Evet' if args.save else 'Hayır'}")
    print(f"{'='*58}\n")

    # ── Model yükle ──
    print("[*] Model yükleniyor...")
    model = YOLO(str(model_path))
    print("[✓] Model hazır!\n")

    # ── Video aç ──
    print(f"[*] Video açılıyor: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps_video

    print(f"[✓] Video: {vid_w}x{vid_h} | {fps_video:.1f} FPS | {format_time(duration_sec)} | {total_frames} kare")

    # ── Video yazıcı ──
    writer = None
    save_dir = Path(args.save_dir)
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = save_dir / f"skywatch_result_{ts}.mp4"
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        writer    = cv2.VideoWriter(str(out_path), fourcc, fps_video, (vid_w, vid_h))
        print(f"[✓] Video kaydı: {out_path}")

    print(f"\n[*] Oynatma başlıyor... (SPACE: duraklat, Q: çıkış)\n")

    # ── Pencere ve Ekran Boyutu Ayarı ──
    # Pencerenin ekrana sığması için (WINDOW_NORMAL) boyutlandırılabilir ayar.
    cv2.namedWindow("SKYWATCH — Video Test", cv2.WINDOW_NORMAL)
    disp_w, disp_h = vid_w, vid_h
    if disp_h > 700:
        ratio = 700.0 / disp_h
        disp_h = 700
        disp_w = int(disp_w * ratio)
    cv2.resizeWindow("SKYWATCH — Video Test", disp_w, disp_h)

    # ── Durum Değişkenleri ──
    conf_threshold   = args.conf
    total_detections = 0
    frame_count      = 0
    paused           = False
    fps_display      = fps_video
    fps_start        = time.time()
    fps_frame_count  = 0
    inference_ms     = 0.0
    delay_ms         = max(1, int(1000 / (fps_video * args.speed)))

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Video bitti → başa sar ve duraklat
                    print("\n[✓] Video bitti. Başa sarılıyor...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    paused      = True
                    continue

                frame_count += 1
                fps_frame_count += 1

                # FPS hesapla
                if fps_frame_count >= 15:
                    elapsed = time.time() - fps_start
                    fps_display = fps_frame_count / max(elapsed, 0.001)
                    fps_start = time.time()
                    fps_frame_count = 0

                # Inference
                t0 = time.time()
                results = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=max(conf_threshold - 0.05, 0.05),
                    iou=0.85,
                    verbose=False,
                    device="cpu"
                )
                inference_ms = (time.time() - t0) * 1000

                det_count         = draw_detections(frame, results, conf_threshold)
                total_detections += det_count

            # HUD & progress bar çiz (duraklı hâlde de göster)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            draw_hud(frame, fps_display, det_count if not paused else 0,
                     conf_threshold, total_detections,
                     current_pos, total_frames, inference_ms,
                     paused, args.save)
            draw_progress_bar(frame, current_pos, total_frames, fps_video)

            # Video yaz
            if writer and not paused:
                writer.write(frame)

            cv2.imshow("SKYWATCH — Video Test", frame)

            # ── Klavye ──
            key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

            if key in (ord('q'), 27):
                print("\n[*] Çıkış yapılıyor...")
                break

            elif key == ord(' '):
                paused = not paused
                state  = "DURAKLATILDI" if paused else "DEVAM EDİYOR"
                print(f"[~] {state}")

            elif key == ord('s'):
                save_dir.mkdir(parents=True, exist_ok=True)
                ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn  = save_dir / f"skywatch_frame_{ts}.jpg"
                cv2.imwrite(str(fn), frame)
                print(f"[✓] Ekran görüntüsü: {fn}")

            elif key in (ord('+'), ord('=')):
                conf_threshold = min(conf_threshold + 0.05, 0.95)
                print(f"[~] Confidence: {conf_threshold:.2f}")

            elif key == ord('-'):
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                print(f"[~] Confidence: {conf_threshold:.2f}")

            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count      = 0
                total_detections = 0
                paused           = False
                print("[~] Başa sarıldı, istatistikler sıfırlandı.")

            elif key == ord('f'):  # 10 sn ileri
                new_pos = min(current_pos + int(fps_video * 10), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                print(f"[~] 10 sn ileri → {format_time(new_pos / fps_video)}")

            elif key == ord('b'):  # 10 sn geri
                new_pos = max(current_pos - int(fps_video * 10), 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                print(f"[~] 10 sn geri → {format_time(new_pos / fps_video)}")

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C — Çıkış yapılıyor...")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\n[✓] Sonuç videosu kaydedildi: {out_path}")
        cv2.destroyAllWindows()

        print(f"\n{'='*58}")
        print(f"  OTURUM ÖZETİ")
        print(f"{'='*58}")
        print(f"  İşlenen Kare   : {frame_count} / {total_frames}")
        print(f"  Toplam Tespit  : {total_detections}")
        print(f"  Son Conf Eşiği : {conf_threshold:.2f}")
        if frame_count > 0:
            print(f"  Kare Başına    : {total_detections / frame_count:.2f} tespit")
        print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
