"""
SKYWATCH — Webcam Gerçek Zamanlı Test Aracı
=============================================
Eğitilmiş best.pt modelini bilgisayar kamerası üzerinden test eder.

Kullanım:
    python src/tools/webcam_test.py
    python src/tools/webcam_test.py --model best.pt --conf 0.35 --camera 0

Kontroller:
    q / ESC  → Çıkış
    s        → Ekran görüntüsü kaydet
    +/-      → Confidence eşiğini artır/azalt
    r        → İstatistikleri sıfırla

Gereksinimler:
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
    """Köşeleri yuvarlatılmış dikdörtgen çizer."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Düz kenarlar
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    # Köşe yayları
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_label(img, text, org, font_scale=0.55, color=(255, 255, 255),
               bg_color=(30, 30, 30), thickness=1, padding=5):
    """Arka planlı etiket çizer."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    y = max(y, h + padding * 2)

    cv2.rectangle(img,
                  (x, y - h - padding * 2),
                  (x + w + padding * 2, y),
                  bg_color, -1)
    cv2.putText(img, text,
                (x + padding, y - padding),
                font, font_scale, color, thickness, cv2.LINE_AA)


def create_hud_overlay(frame, fps, det_count, conf_threshold, total_detections,
                       session_time, inference_ms):
    """Ekranın sol üstüne HUD (Head-Up Display) paneli çizer."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Panel arka planı (yarı saydam)
    panel_w, panel_h = 310, 200
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Başlık
    cv2.putText(frame, "SKYWATCH - Live Test",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 200, 255), 2, cv2.LINE_AA)

    # Ayırıcı çizgi
    cv2.line(frame, (20, 48), (300, 48), (0, 200, 255), 1)

    # FPS rengi (yeşil > sarı > kırmızı)
    if fps >= 25:
        fps_color = (0, 255, 100)
    elif fps >= 15:
        fps_color = (0, 220, 255)
    else:
        fps_color = (0, 80, 255)

    info_lines = [
        (f"FPS: {fps:.1f}", fps_color),
        (f"Inference: {inference_ms:.1f} ms", (200, 200, 200)),
        (f"Tespit: {det_count} yuz", (255, 255, 255)),
        (f"Conf Esik: {conf_threshold:.2f}  (+/- ile ayarla)", (180, 180, 180)),
        (f"Toplam Tespit: {total_detections}", (180, 180, 180)),
        (f"Sure: {session_time}", (140, 140, 140)),
    ]

    y_offset = 72
    for text, color in info_lines:
        cv2.putText(frame, text, (25, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y_offset += 22

    # Alt bilgi çubuğu
    bar_h = 30
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, "[Q] Cikis   [S] Screenshot   [+/-] Conf   [R] Reset",
                (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 120, 120), 1, cv2.LINE_AA)


def draw_detections(frame, results, conf_threshold):
    """Tespit sonuçlarını frame üzerine çizer."""
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

        # Confidence'a göre renk gradyanı (düşük=kırmızı, yüksek=yeşil)
        ratio = min((conf - conf_threshold) / max(1.0 - conf_threshold, 0.01), 1.0)
        b = int(200 * (1 - ratio))
        g = int(255 * ratio)
        r = int(50 + 100 * (1 - ratio))
        color = (b, g, r)

        # Bbox çizimi
        draw_rounded_rect(frame, (x1, y1), (x2, y2), color, 2, radius=8)

        # Köşe vurgusu (4 köşeye kısa çizgiler)
        corner_len = min(20, w_box // 3, h_box // 3)
        t = 3
        # Sol üst
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
        # Sağ üst
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
        # Sol alt
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
        # Sağ alt
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

        # Etiket
        label = f"Yuz {conf:.0%}"
        draw_label(frame, label, (x1, y1 - 5), font_scale=0.5,
                   bg_color=color, color=(255, 255, 255))

        # Boyut bilgisi (küçük font)
        size_text = f"{w_box}x{h_box}px"
        cv2.putText(frame, size_text, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150),
                    1, cv2.LINE_AA)

    return det_count


# ════════════════════════════════════════════════════════════
# Ana Çalıştırıcı
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SKYWATCH — Webcam Real-Time Detection Test"
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
        "--camera", type=int, default=0,
        help="Kamera indeksi (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Kamera çözünürlük genişliği (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Kamera çözünürlük yüksekliği (default: 720)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLO inference boyutu (default: 640)"
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=str(PROJECT_ROOT / "logs" / "webcam_captures"),
        help="Ekran görüntüsü kayıt dizini"
    )
    args = parser.parse_args()

    # ── Model Yükleme ──
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[HATA] Model bulunamadı: {model_path}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  SKYWATCH — Webcam Test Aracı")
    print(f"{'='*55}")
    print(f"  Model   : {model_path.name} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Kamera  : {args.camera}")
    print(f"  Conf    : {args.conf}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"{'='*55}\n")

    print("[*] Model yükleniyor...")
    model = YOLO(str(model_path))
    print("[✓] Model hazır!\n")

    # ── Kamera Açma ──
    print(f"[*] Kamera {args.camera} açılıyor...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[HATA] Kamera {args.camera} açılamadı!")
        print("       Farklı bir kamera deneyin: --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[✓] Kamera açıldı: {actual_w}x{actual_h}")
    print(f"\n[*] Gerçek zamanlı tespit başlıyor...")
    print(f"    Çıkış için 'q' veya ESC tuşuna basın.\n")

    # ── Durum Değişkenleri ──
    conf_threshold = args.conf
    total_detections = 0
    frame_count = 0
    start_time = time.time()
    fps_time = time.time()
    fps = 0.0
    save_dir = Path(args.save_dir)

    # ── Ana Döngü ──
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[UYARI] Kameradan frame okunamadı, tekrar deniyor...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Inference
            t_inf = time.time()
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=max(conf_threshold - 0.05, 0.1),  # Biraz düşük gönder, sonra filtrele
                verbose=False,
                device="cpu"  # GPU varsa "0" yapılabilir
            )
            inference_ms = (time.time() - t_inf) * 1000

            # Çizimler
            det_count = draw_detections(frame, results, conf_threshold)
            total_detections += det_count

            # FPS hesapla (son 10 frame ortalaması)
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10.0 / max(now - fps_time, 0.001)
                fps_time = now

            # Oturum süresi
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            session_time = f"{minutes:02d}:{seconds:02d}"

            # HUD
            create_hud_overlay(frame, fps, det_count, conf_threshold,
                               total_detections, session_time, inference_ms)

            # Tespit yoksa bilgi mesajı
            if det_count == 0:
                h = frame.shape[0]
                cv2.putText(frame, "Tespit yok — Kameraya bakın",
                            (frame.shape[1] // 2 - 160, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (100, 100, 100), 1, cv2.LINE_AA)

            # Göster
            cv2.imshow("SKYWATCH — Live Detection", frame)

            # ── Klavye Kontrolleri ──
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # q veya ESC
                print("\n[*] Çıkış yapılıyor...")
                break

            elif key == ord('s'):  # Screenshot
                save_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = save_dir / f"skywatch_capture_{ts}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"[✓] Ekran görüntüsü kaydedildi: {filename}")

            elif key in (ord('+'), ord('=')):  # Conf artır
                conf_threshold = min(conf_threshold + 0.05, 0.95)
                print(f"[~] Confidence eşiği: {conf_threshold:.2f}")

            elif key == ord('-'):  # Conf azalt
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                print(f"[~] Confidence eşiği: {conf_threshold:.2f}")

            elif key == ord('r'):  # Reset
                total_detections = 0
                frame_count = 0
                start_time = time.time()
                print("[~] İstatistikler sıfırlandı.")

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C — Çıkış yapılıyor...")

    finally:
        # ── Temizlik ──
        cap.release()
        cv2.destroyAllWindows()

        # Oturum özeti
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"\n{'='*55}")
        print(f"  OTURUM ÖZETİ")
        print(f"{'='*55}")
        print(f"  Süre            : {minutes:02d}:{seconds:02d}")
        print(f"  Toplam Frame    : {frame_count}")
        print(f"  Toplam Tespit   : {total_detections}")
        print(f"  Ort. FPS        : {frame_count / max(elapsed, 0.001):.1f}")
        print(f"  Son Conf Eşiği  : {conf_threshold:.2f}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
