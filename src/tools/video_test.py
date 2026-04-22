"""
SKYWATCH — Video Test Aracı (YOLO + Track + Otomatik DB Kontrolü)
=================================================================
Her algılanan yüz otomatik olarak track'e kilitlenir ve
veritabanında suçlu kontrolü yapılır. Manuel kilitleme gerekmez.

Detection : best.pt (YOLO)
Tracking  : DeepSORT + EMA + Lost Pool + Velocity Consistency
DB Check  : InsightFace (buffalo_l) ile embedding → DB karşılaştırma
Movement  : MovementAnalyzer → davranış skoru

Kullanım:
    python src/tools/video_test.py
    python src/tools/video_test.py --video Video1.webm --conf 0.35

Kontroller:
    q / ESC  → Çıkış
    SPACE    → Duraklat / Devam
    s        → Ekran görüntüsü kaydet
    +/-      → Confidence eşiğini artır/azalt
    r        → Başa sar + Tracker sıfırla
    f / b    → 10 saniye ileri / geri
    d        → Debug paneli aç/kapat
"""

import argparse
import sys
import time
import math
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

# ── Proje kökü ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT     = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

# ── GPU DLL (onnxruntime için) ────────────────────────────────────────────────
import os
_venv = Path(sys.executable).parent.parent
for _sub in ("cudnn", "cublas"):
    _d = _venv / "Lib" / "site-packages" / "nvidia" / _sub / "bin"
    if _d.exists():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + ";" + os.environ.get("PATH", "")

# ── YOLO ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[HATA] ultralytics eksik: pip install ultralytics")
    sys.exit(1)

# ── DeepSORT ──────────────────────────────────────────────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("[HATA] deep_sort_realtime eksik: pip install deep-sort-realtime")
    sys.exit(1)

# ── SKYWATCH core modüller ────────────────────────────────────────────────────
try:
    from core.movement import MovementAnalyzer
    from core.models import Track
    _CORE_OK = True
except Exception as e:
    print(f"[!] core modüller: {e}")
    _CORE_OK = False

# ── DB ve InsightFace (opsiyonel) ─────────────────────────────────────────────
_DB_OK = False
_FACE_OK = False
_db_embeddings = []   # [(criminal_id, np.ndarray)]
_face_app      = None
_SIMILARITY_THRESHOLD = 0.40

def _try_load_db_and_face():
    """InsightFace + DB'yi yüklemeyi dener. Başarısız olursa sadece tracking çalışır."""
    global _DB_OK, _FACE_OK, _db_embeddings, _face_app
    import sqlite3, io

    # InsightFace
    try:
        from insightface.app import FaceAnalysis
        models_root = str(PROJECT_ROOT / "models")
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root=models_root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        _FACE_OK = True
        print("[✓] InsightFace (buffalo_l) hazır — DB kontrolü aktif")
    except Exception as e:
        print(f"[!] InsightFace yüklenemedi → DB kontrolü kapalı ({e})")
        return

    # DB bağlantısı
    db_path = PROJECT_ROOT / "database" / "skywatch.db"
    if not db_path.exists():
        print(f"[!] DB bulunamadı: {db_path}")
        return
    try:
        def convert_array(b):
            return np.load(io.BytesIO(b))
        sqlite3.register_converter("array", convert_array)

        conn = sqlite3.connect(str(db_path),
                               detect_types=sqlite3.PARSE_DECLTYPES,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT e.criminal_id, e.embedding
            FROM embeddings e
            JOIN criminals c ON e.criminal_id = c.id
            WHERE c.status IN ('WANTED','CRIMINAL')
        """).fetchall()
        _db_embeddings = [(r[0], r[1]) for r in rows]
        conn.close()
        _DB_OK = True
        print(f"[✓] DB: {len(_db_embeddings)} embedding yüklendi")
    except Exception as e:
        print(f"[!] DB okunamadı: {e}")


def _get_criminal_info(criminal_id: int) -> dict | None:
    """criminal_id'ye göre DB'den kişi bilgisini çeker."""
    db_path = PROJECT_ROOT / "database" / "skywatch.db"
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM criminals WHERE id=?",
                           (criminal_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def _extract_embedding(face_crop: np.ndarray) -> np.ndarray | None:
    """Yüz kırpığından 512-d embedding çıkarır."""
    if not _FACE_OK or _face_app is None:
        return None
    try:
        faces = _face_app.get(face_crop)
        if faces:
            return faces[0].normed_embedding
    except Exception:
        pass
    return None


def _search_db(embedding: np.ndarray) -> tuple | None:
    """DB'deki en yakın eşleşmeyi döndürür. (criminal_id, score) veya None."""
    if not _DB_OK or embedding is None or len(_db_embeddings) == 0:
        return None
    best_score = 0.0
    best_id    = None
    for cid, db_emb in _db_embeddings:
        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(db_emb)
        if norm_a < 1e-6 or norm_b < 1e-6:
            continue
        score = float(np.dot(embedding, db_emb) / (norm_a * norm_b))
        if score > best_score:
            best_score = score
            best_id    = cid
    if best_score >= _SIMILARITY_THRESHOLD:
        return (best_id, best_score)
    return None


# ════════════════════════════════════════════════════════════════════════════
# LightTracker  (embedder=None + normalize unit vektör  →  hızlı, NaN yok)
# ════════════════════════════════════════════════════════════════════════════
_RNG = np.random.default_rng(seed=0)

class LightTracker:
    """YOLO bbox'larını DeepSORT + EMA + Lost Pool + Velocity ile takip eder."""

    _EMA     = 0.80   # α — büyük = daha fazla yumuşatma
    _VEL_EPS = 18.0   # px/frame eşiği — velocity consistency

    def __init__(self, max_age=10, min_hits=2, iou_thr=0.4, max_lost=30):
        self.max_age  = max_age
        self.min_hits = min_hits
        self.iou_thr  = iou_thr
        self.max_lost = max_lost
        self._build_ds()

        self._ema:        dict[int, list]    = {}
        self._lost_pool:  dict[int, int]     = {}
        self._lost_bboxes:dict[int, list]    = {}
        self._centers:    dict[int, deque]   = {}
        self._seen:       set[int]           = set()
        self._emb_dim:    int                = 64   # placeholder embedding boyutu

    def _build_ds(self):
        # embedder=None + küçük normalize vektörler → NaN olmaz, hızlı
        self._ds = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            max_iou_distance=self.iou_thr,
            embedder=None,
        )

    def _make_embed(self, n: int) -> np.ndarray:
        """n adet rastgele birim vektör üret (NaN'sız placeholder)."""
        vecs  = _RNG.standard_normal((n, self._emb_dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-6)

    def update(self, dets: list, frame: np.ndarray) -> list:
        """
        dets: [(x1,y1,x2,y2,conf), ...]
        returns: [{"track_id","bbox","is_new","age","time_since_update","vel_ok"}, ...]
        """
        ds_inp = [([x1, y1, x2-x1, y2-y1], c, "face")
                  for x1, y1, x2, y2, c in dets]

        embeds = self._make_embed(len(ds_inp)) if ds_inp else np.zeros((0, self._emb_dim))
        raw    = self._ds.update_tracks(ds_inp, frame=frame, embeds=embeds)

        confirmed = {rt.track_id for rt in raw if rt.is_confirmed()}
        self._age_lost(confirmed)

        out = []
        for rt in raw:
            if not rt.is_confirmed():
                continue
            tid   = rt.track_id
            ltrb  = rt.to_ltrb()
            raw_b = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            bbox  = self._ema_smooth(tid, raw_b)
            vel_ok= self._vel_ok(tid, bbox)
            is_new= tid not in self._seen
            if is_new:
                self._seen.add(tid)
            self._lost_pool.pop(tid, None)
            out.append({"track_id": tid, "bbox": bbox, "is_new": is_new,
                        "age": rt.age, "time_since_update": rt.time_since_update,
                        "vel_ok": vel_ok})
        return out

    def get_lost(self):
        return [{"track_id": tid, "bbox": self._lost_bboxes[tid], "lost_age": age}
                for tid, age in self._lost_pool.items()
                if age <= self.max_lost and tid in self._lost_bboxes]

    # ── EMA ─────────────────────────────────────────────────────────────────
    def _ema_smooth(self, tid, raw):
        r = [float(v) for v in raw]
        if tid not in self._ema:
            self._ema[tid] = r; return raw
        p = self._ema[tid]
        s = [self._EMA * p[i] + (1 - self._EMA) * r[i] for i in range(4)]
        self._ema[tid] = s
        return [int(v) for v in s]

    # ── Velocity Consistency ─────────────────────────────────────────────────
    def _vel_ok(self, tid, bbox):
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        if tid not in self._centers:
            self._centers[tid] = deque(maxlen=5)
        h = self._centers[tid]
        ok = True
        if len(h) >= 2:
            adx = (h[-1][0]-h[0][0])/len(h)
            ady = (h[-1][1]-h[0][1])/len(h)
            diff = math.sqrt((cx-h[-1][0]-adx)**2 + (cy-h[-1][1]-ady)**2)
            ok   = diff < self._VEL_EPS
        h.append((cx, cy))
        return ok

    # ── Lost Pool ────────────────────────────────────────────────────────────
    def _age_lost(self, confirmed):
        # Yeni lost track'leri ekle
        for tid in (self._seen - confirmed - set(self._lost_pool)):
            self._lost_bboxes[tid] = [int(v) for v in self._ema.get(tid, [0,0,0,0])]
            self._lost_pool[tid]   = 0
        # Yaşlıları sil
        to_del = [t for t, a in self._lost_pool.items() if a > self.max_lost]
        for tid in to_del:
            self._ema.pop(tid, None); self._lost_bboxes.pop(tid, None)
            self._centers.pop(tid, None); del self._lost_pool[tid]
        # Kalan'ları yaşlandır
        for tid in list(self._lost_pool):
            self._lost_pool[tid] += 1
        # _seen setini sınırla (bellek tasarrufu — kalabalık sahnede şişer)
        if len(self._seen) > 400:
            # En eski 200 ID'yi at (set sırasız ama pratikte yeterli)
            excess = len(self._seen) - 200
            self._seen -= set(list(self._seen)[:excess])
        # Lost pool'u sınırla (sonsuz büyüme engeli)
        if len(self._lost_pool) > 80:
            oldest = sorted(self._lost_pool, key=lambda t: self._lost_pool[t], reverse=True)
            for tid in oldest[80:]:
                self._ema.pop(tid, None); self._lost_bboxes.pop(tid, None)
                self._centers.pop(tid, None); self._lost_pool.pop(tid, None)

    def reset(self):
        self._build_ds()
        self._ema.clear(); self._lost_pool.clear(); self._lost_bboxes.clear()
        self._centers.clear(); self._seen.clear()


# ════════════════════════════════════════════════════════════════════════════
# Renk / Durum
# ════════════════════════════════════════════════════════════════════════════
_COLORS = {
    "CLEAN":      (0,  210,  0),
    "CRIMINAL":   (0,  165, 255),
    "WANTED":     (0,    0, 255),
    "SUSPICIOUS": (190,  0, 190),
    "UNKNOWN":    (130, 130, 130),
}
_LABELS = {
    "CLEAN":      "TEMIZ",
    "CRIMINAL":   "! SABIKALI !",
    "WANTED":     "!! ARANIYOR !!",
    "SUSPICIOUS": "SUPHELI",
    "UNKNOWN":    "?",
}
MOVE_COLORS = {
    "normal":     (0,  200,  0),
    "fast":       (0,  180, 255),
    "running":    (0,    0, 255),
    "suspicious": (190,  0, 190),
    "unknown":    (130, 130, 130),
}


# ════════════════════════════════════════════════════════════════════════════
# Çizim Yardımcıları
# ════════════════════════════════════════════════════════════════════════════
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def fmt_time(s):
    return f"{int(s//60):02d}:{int(s%60):02d}"


def put_label(frame, text, x, y, color, scale=0.46, bold=False):
    th = 2 if bold else 1
    (tw, lh), _ = cv2.getTextSize(text, _FONT, scale, th)
    cv2.rectangle(frame, (x, y - lh - 5), (x + tw + 6, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - 1), _FONT, scale, (255, 255, 255), th, cv2.LINE_AA)


def draw_corner_box(frame, x1, y1, x2, y2, color, thick=2, clen=16):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cl = min(clen, (x2-x1)//3, (y2-y1)//3)
    for cx, cy, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + sx*cl, cy), color, thick)
        cv2.line(frame, (cx, cy), (cx, cy + sy*cl), color, thick)


def draw_progress(frame, cur, total, fps_v):
    h, w = frame.shape[:2]
    y0   = h - 24
    cv2.rectangle(frame, (0, y0), (w, h), (18, 18, 18), -1)
    if total > 0:
        bw = int((w - 110) * cur / total)
        cv2.rectangle(frame, (8, y0+6), (w-102, y0+16), (50,50,50), -1)
        cv2.rectangle(frame, (8, y0+6), (8+bw,  y0+16), (0,200,255), -1)
        ts = f"{fmt_time(cur/max(fps_v,1))} / {fmt_time(total/max(fps_v,1))}"
        cv2.putText(frame, ts, (w-100, y0+16), _FONT, 0.38, (160,160,160), 1, cv2.LINE_AA)


def draw_hud(frame, fps, active, conf, total_det, cur, total, inf_ms,
             paused, saving, db_ok, face_ok, vel_rej, lost_cnt):
    h, w = frame.shape[:2]
    ov  = frame.copy()
    cv2.rectangle(ov, (8, 8), (285, 215), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

    db_str = "DB+FACE" if (db_ok and face_ok) else ("FACE-ONLY" if face_ok else "TRACK-ONLY")
    tc = (0, 80, 255) if paused else (0, 200, 255)
    cv2.putText(frame, f"SKYWATCH [{db_str}]", (18, 34), _FONT, 0.58, tc, 2, cv2.LINE_AA)
    cv2.line(frame, (18, 42), (278, 42), tc, 1)

    fc = (0,255,100) if fps>=20 else ((0,220,255) if fps>=10 else (0,80,255))
    pct = cur / max(total, 1) * 100
    rows = [
        (f"FPS        : {fps:.1f}",                fc),
        (f"Inference  : {inf_ms:.1f} ms",          (200,200,200)),
        (f"Aktif Track: {active}",                 (255,255,255)),
        (f"Tespit     : {total_det}",               (180,180,180)),
        (f"Conf Esik  : {conf:.2f}",                (180,180,180)),
        (f"Kare       : {cur}/{total} ({pct:.1f}%)",(140,140,140)),
        (f"Vel.Reject : {vel_rej}",                 (140,140,140)),
        (f"Lost Pool  : {lost_cnt}",                (140,140,140)),
    ]
    y = 58
    for txt, col in rows:
        cv2.putText(frame, txt, (20, y), _FONT, 0.41, col, 1, cv2.LINE_AA)
        y += 19

    if paused:
        cv2.putText(frame, "|| DURAKLATILDI", (20, y+4), _FONT, 0.48, (0,80,255), 1, cv2.LINE_AA)
    if saving:
        cv2.circle(frame, (w-22, 22), 7, (0,0,220), -1)
        cv2.putText(frame, "REC", (w-58, 27), _FONT, 0.48, (0,0,220), 1, cv2.LINE_AA)

    hints = "[SPC]Duraklat  [+/-]Conf  [D]Debug  [S]Shot  [F/B]±10sn  [R]Sifirla  [Q]Cikis"
    cv2.putText(frame, hints, (8, h-28), _FONT, 0.31, (85,85,85), 1, cv2.LINE_AA)


def draw_debug_panel(frame, tracker, move_data):
    h, w = frame.shape[:2]
    pw   = 230
    px   = w - pw - 6
    ov   = frame.copy()
    cv2.rectangle(ov, (px-4, 4), (w-4, 380), (10,10,10), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

    y = 24
    def t(txt, col=(200,200,200), sc=0.40, bold=False):
        nonlocal y
        cv2.putText(frame, txt, (px, y), _FONT, sc, col, 2 if bold else 1, cv2.LINE_AA)
        y += 20

    t("── TRACKER ──", (0,200,255), 0.44, True)
    t(f"Seen IDs  : {len(tracker._seen)}")
    t(f"EMA bboxes: {len(tracker._ema)}")
    t(f"Lost Pool : {len(tracker._lost_pool)}")

    y += 4; cv2.line(frame, (px, y), (w-6, y), (40,40,40), 1); y += 8
    t("── HAREKET ──", (0,200,255), 0.44, True)
    for tid, info in list(move_data.items())[:6]:
        col = MOVE_COLORS.get(info.get("label","unknown"), (140,140,140))
        t(f"T{tid}: {info.get('label','?')} ({info.get('score',0):.2f})", col)

    y += 4; cv2.line(frame, (px, y), (w-6, y), (40,40,40), 1); y += 8
    t("── DB ──", (0,200,255), 0.44, True)
    t(f"Embeddings: {len(_db_embeddings)}")


# ════════════════════════════════════════════════════════════════════════════
# Ana Çalıştırıcı
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SKYWATCH — Video Test")
    parser.add_argument("--video",    default=str(PROJECT_ROOT/"Video.mp4"))
    parser.add_argument("--model",    default=str(PROJECT_ROOT/"best.pt"))
    parser.add_argument("--conf",     type=float, default=0.35)
    parser.add_argument("--imgsz",    type=int,   default=320)  # 640→320: ~3x hızlı
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--save",     action="store_true")
    parser.add_argument("--save-dir", default=str(PROJECT_ROOT/"logs"/"video_results"))
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    for p in [video_path, model_path]:
        if not p.exists():
            print(f"[HATA] Bulunamadı: {p}"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SKYWATCH — Video Test")
    print(f"  Model : {model_path.name}  ({model_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Video : {video_path.name}")
    print(f"{'='*60}\n")

    # ── DB + InsightFace ─────────────────────────────────────────────────────
    print("[*] DB ve InsightFace yükleniyor...")
    _try_load_db_and_face()

    # ── YOLO ─────────────────────────────────────────────────────────────────
    print("[*] YOLO modeli yükleniyor...")
    model = YOLO(str(model_path))
    print("[✓] YOLO hazır!\n")

    # ── Tracker + Movement ───────────────────────────────────────────────────
    tracker  = LightTracker(max_age=8, min_hits=2, iou_thr=0.4, max_lost=15)
    movement = MovementAnalyzer({
        "speed_threshold_fast":       50,
        "speed_threshold_running":   100,
        "dwell_time_threshold":      120,
        "direction_change_threshold": 90,
    }) if _CORE_OK else None

    # ── Video ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı"); sys.exit(1)

    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[✓] Video: {vid_w}x{vid_h} | {fps_v:.1f}fps | {fmt_time(total_fr/fps_v)} | {total_fr} kare\n")
    print("[*] Oynatma başlıyor...\n")

    # ── Pencere boyutu (portrait-aware, max 650px yüksek) ────────────────────
    MAX_H = 650
    MAX_W = 960
    if vid_h > MAX_H:
        dh = MAX_H; dw = int(vid_w * MAX_H / vid_h)
    else:
        dh = vid_h; dw = vid_w
    if dw > MAX_W:
        dw = MAX_W; dh = int(vid_h * MAX_W / vid_w)

    cv2.namedWindow("SKYWATCH", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SKYWATCH", dw, dh)

    # ── Video yazıcı ─────────────────────────────────────────────────────────
    writer   = None
    save_dir = Path(args.save_dir)
    out_path = None
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"skywatch_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer   = cv2.VideoWriter(str(out_path),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (vid_w, vid_h))
        print(f"[✓] Kayıt: {out_path}")

    # ── Durum ─────────────────────────────────────────────────────────────────
    conf        = args.conf
    total_det   = 0
    frame_cnt   = 0
    paused      = False
    show_debug  = True
    fps_disp    = fps_v
    fps_start   = time.time()
    fps_fc      = 0
    inf_ms      = 0.0
    active_cnt  = 0
    vel_rej     = 0
    display     = None
    move_data: dict[int, dict] = {}

    # Track ID → DB eşleşme sonucu  {status, criminal_id, confidence, name}
    track_status: dict[int, dict] = {}

    # Track ID → embedding çıkarılıp çıkarılmadığı
    track_emb_done: set[int] = set()

    # Görüntü oynatma: delay=1ms, frame-sync ile gerçek hıza eş
    # video fps'i gerçek zamanlı tutmak için işlem sonrası kalan süreyi ölç
    frame_interval = 1.0 / fps_v   # her kare için hedef süre (sn)

    # Frame skip: her N karede bir InsightFace çalıştır (ağır)
    FACE_EVERY = 5
    frame_skip_cnt = 0

    try:
        while True:
            if not paused:
                ret, raw = cap.read()
                if not ret:
                    print("\n[✓] Video bitti. Başa sarılıyor...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_cnt = 0; paused = True
                    if display is None: continue
                else:
                    frame_cnt      += 1
                    fps_fc         += 1
                    frame_skip_cnt += 1
                    if fps_fc >= 20:
                        elapsed   = time.time() - fps_start
                        fps_disp  = fps_fc / max(elapsed, 0.001)
                        fps_start = time.time()
                        fps_fc    = 0

                    t0 = time.time()

                    # ── 1. YOLO Detection ─────────────────────────────────
                    yolo_res = model.predict(
                        raw, imgsz=args.imgsz,
                        conf=max(conf - 0.05, 0.05),
                        iou=0.85, verbose=False, device=args.device
                    )
                    dets = []
                    boxes = yolo_res[0].boxes
                    if boxes is not None:
                        for b in boxes:
                            c = float(b.conf[0])
                            if c >= conf:
                                x1,y1,x2,y2 = map(int, b.xyxy[0])
                                dets.append((x1,y1,x2,y2,c))
                    total_det += len(dets)

                    # ── 2. Track ──────────────────────────────────────────
                    tracks     = tracker.update(dets, raw)
                    active_cnt = len(tracks)

                    # ── 3. InsightFace + DB (her N karede, her track için 1 kez) ──
                    if frame_skip_cnt >= FACE_EVERY:
                        frame_skip_cnt = 0
                        if _FACE_OK:
                            for tr in tracks:
                                tid = tr["track_id"]
                                if tid in track_emb_done:
                                    continue
                                x1,y1,x2,y2 = tr["bbox"]
                                # Sınır kontrolü
                                x1 = max(0, x1); y1 = max(0, y1)
                                x2 = min(raw.shape[1], x2); y2 = min(raw.shape[0], y2)
                                if x2 - x1 < 20 or y2 - y1 < 20:
                                    continue
                                crop = raw[y1:y2, x1:x2]
                                emb  = _extract_embedding(crop)
                                if emb is None:
                                    continue
                                track_emb_done.add(tid)
                                result = _search_db(emb)
                                if result:
                                    cid, score = result
                                    info = _get_criminal_info(cid) or {}
                                    status = info.get("status", "CRIMINAL").upper()
                                    track_status[tid] = {
                                        "status": status,
                                        "criminal_id": cid,
                                        "confidence": score,
                                        "name": info.get("name", "?"),
                                    }
                                    print(f"[!] T{tid}: {info.get('name','?')} "
                                          f"({status}, %{score*100:.0f})")
                                else:
                                    track_status[tid] = {"status": "CLEAN"}

                    inf_ms = (time.time() - t0) * 1000

                    # ── Frame-Sync: işlem uzun sürdüyse kareleri atla ──────
                    # Örn: inference=150ms, video=33ms → 4 kare atla → hız korunur
                    if inf_ms > frame_interval * 1000:
                        skip = int(inf_ms / (frame_interval * 1000)) - 1
                        if skip > 0:
                            new_pos = min(
                                int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + skip,
                                total_fr - 1
                            )
                            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

                    # ── 4. Çizim ──────────────────────────────────────────
                    display = raw.copy()
                    active_ids = set()

                    for tr in tracks:
                        tid = tr["track_id"]
                        x1, y1, x2, y2 = tr["bbox"]
                        active_ids.add(tid)

                        # Durum
                        st = track_status.get(tid, {})
                        if not st:
                            db_status = "UNKNOWN"   # henüz kontrol edilmedi
                        else:
                            db_status = st.get("status", "CLEAN")

                        # Movement
                        move_label = "normal"
                        move_score = 0.0
                        if movement and _CORE_OK:
                            t_obj = Track(
                                track_id=tid, bbox=[x1,y1,x2,y2],
                                is_new=tr["is_new"], age=tr["age"],
                                is_confirmed=True,
                                time_since_update=tr["time_since_update"],
                                velocity_ok=tr["vel_ok"],
                            )
                            rep        = movement.analyze(t_obj)
                            move_label = rep.behavior_label
                            move_score = rep.behavior_score
                            move_data[tid] = {"label": move_label, "score": move_score}
                            if not tr["vel_ok"]: vel_rej += 1

                        # SUSPICIOUS override
                        if db_status == "CLEAN" and move_score >= 0.60:
                            db_status = "SUSPICIOUS"

                        # Yeni track / velocity tutarsız → UNKNOWN renk
                        if tr["age"] < 5 or not tr["vel_ok"]:
                            color = _COLORS["UNKNOWN"]
                        else:
                            color = _COLORS.get(db_status, _COLORS["UNKNOWN"])

                        # Çizim
                        draw_corner_box(display, x1, y1, x2, y2, color, thick=2)

                        # Yanıp sönen WANTED uyarısı
                        is_wanted = db_status == "WANTED"
                        if is_wanted and (frame_cnt // 8) % 2 == 0:
                            cv2.rectangle(display, (x1-3,y1-3), (x2+3,y2+3), (0,0,255), 3)

                        # Üst etiket: ID + durum
                        label_str = (f"T{tid}  {_LABELS.get(db_status, db_status)}"
                                     if tid in track_status
                                     else f"T{tid}  ?")
                        put_label(display, label_str, x1, y1, color,
                                  scale=0.50, bold=(db_status in ("WANTED","CRIMINAL")))

                        # Güven skoru (eşleşme varsa)
                        if db_status in ("WANTED", "CRIMINAL") and "confidence" in st:
                            name_str = st.get("name", "")
                            conf_str = f"{name_str}  %{st['confidence']*100:.0f}"
                            cv2.putText(display, conf_str, (x1, y2+15),
                                        _FONT, 0.44, color, 1, cv2.LINE_AA)

                        # Hareket durumu (alt)
                        if move_label not in ("normal",) and db_status == "CLEAN":
                            cv2.putText(display, move_label.upper(), (x1, y2+15),
                                        _FONT, 0.38, MOVE_COLORS.get(move_label,(140,140,140)),
                                        1, cv2.LINE_AA)

                    if movement:
                        movement.cleanup(active_ids)

            # ── UI ────────────────────────────────────────────────────────────
            if display is None:
                display = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

            frame = display.copy()
            cur_p = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            draw_hud(frame, fps_disp, active_cnt, conf, total_det,
                     cur_p, total_fr, inf_ms, paused, args.save,
                     _DB_OK, _FACE_OK, vel_rej, len(tracker._lost_pool))

            if show_debug:
                draw_debug_panel(frame, tracker, move_data)

            draw_progress(frame, cur_p, total_fr, fps_v)

            # WANTED banner
            if any(v.get("status") == "WANTED" for v in track_status.values()):
                if (frame_cnt // 8) % 2 == 0:
                    h, w = frame.shape[:2]
                    ov = frame.copy()
                    cv2.rectangle(ov, (0, h-42), (w, h-24), (0,0,160), -1)
                    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, "!!! ARANAN KISI TESPIT EDILDI !!!",
                                (w//2-190, h-28), _FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)

            if writer and not paused:
                writer.write(frame)

            # Ekrana göster (yeniden boyutlandır)
            show = cv2.resize(frame, (dw, dh))
            cv2.imshow("SKYWATCH", show)

            key = cv2.waitKey(1) & 0xFF  # 1ms — gecikme yok, hız inference'a bağlı

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"[~] {'DURAKLATILDI' if paused else 'DEVAM'}")
            elif key == ord('d'):
                show_debug = not show_debug
            elif key == ord('s'):
                save_dir.mkdir(parents=True, exist_ok=True)
                fn = save_dir / f"frame_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(fn), frame); print(f"[✓] {fn}")
            elif key in (ord('+'), ord('=')):
                conf = min(conf+0.05, 0.95); print(f"[~] Conf: {conf:.2f}")
            elif key == ord('-'):
                conf = max(conf-0.05, 0.05); print(f"[~] Conf: {conf:.2f}")
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_cnt = 0; total_det = 0; vel_rej = 0
                tracker.reset(); track_status.clear(); track_emb_done.clear()
                move_data.clear()
                if movement: movement._history.clear()
                paused = False; print("[~] Sıfırlandı.")
            elif key == ord('f'):
                p2 = min(cur_p + int(fps_v*10), total_fr-1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, p2)
                print(f"[~] +10sn → {fmt_time(p2/fps_v)}")
            elif key == ord('b'):
                p2 = max(cur_p - int(fps_v*10), 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, p2)
                print(f"[~] -10sn → {fmt_time(p2/fps_v)}")

    except KeyboardInterrupt:
        print("\n[*] Durduruldu.")
    finally:
        cap.release()
        if writer: writer.release(); print(f"\n[✓] Video: {out_path}")
        cv2.destroyAllWindows()
        print(f"\n{'='*55}")
        print(f"  İşlenen Kare  : {frame_cnt}/{total_fr}")
        print(f"  Toplam Tespit : {total_det}")
        print(f"  Unique Track  : {len(tracker._seen)}")
        print(f"  Vel. Rejected : {vel_rej}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
