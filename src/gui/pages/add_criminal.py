"""SKYWATCH — Kişi Ekleme Sayfası (YOLO best.pt tabanlı)"""

import sys
import io
import re
from pathlib import Path

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QFileDialog, QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QColor, QPainter, QBrush

from gui.styles.theme import (
    BG_APP, SURFACE, SURFACE_2, BG_PANEL, BORDER, SEP,
    ACCENT, ACCENT_DIM, ACCENT_2,
    TEXT_1, TEXT_2, TEXT_3,
    GREEN_GLOW, AMBER, RED, RED_DIM,
    GOLD, GOLD_DIM, WHITE, GRAY_1, GRAY_2
)
from gui.widgets.card import Card, SectionLabel, PageTitle, Divider

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_MODEL_PATH = PROJECT_ROOT / "best.pt"
_EMBED_SIZE = 64
_EMBED_DIM  = 512

_yolo_model  = None
_yolo_loaded = False


def _get_yolo():
    """YOLO modelini tek seferlik yükle."""
    global _yolo_model, _yolo_loaded
    if _yolo_loaded:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model  = YOLO(str(_MODEL_PATH))
        _yolo_model.overrides['verbose'] = False
        _yolo_loaded = True
        return _yolo_model
    except Exception as e:
        print(f"[!] YOLO yüklenemedi: {e}")
        return None


def _make_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Kırpılmış yüz görüntüsünden 512-d L2-normalize embedding üret.
    CLAHE + DCT tabanlı, aydınlatmadan bağımsız.
    """
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (_EMBED_SIZE, _EMBED_SIZE),
                         interpolation=cv2.INTER_AREA)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq      = clahe.apply(resized).astype(np.float32) / 255.0
    dct     = cv2.dct(eq)
    flat    = dct.flatten()[:_EMBED_DIM]
    if len(flat) < _EMBED_DIM:
        flat = np.pad(flat, (0, _EMBED_DIM - len(flat)))
    norm = np.linalg.norm(flat)
    if norm > 1e-6:
        flat = flat / norm
    return flat.astype(np.float32)


def _detect_and_embed(img_bgr: np.ndarray):
    """
    YOLO ile yüz algıla → en büyük tespiti kırp → embedding üret.
    Returns: (embedding, error_str)
    """
    model = _get_yolo()
    if model is None:
        return None, "YOLO modeli yüklenemedi."

    results = model(img_bgr, conf=0.25, verbose=False)
    best_box  = None
    best_area = 0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)

    if best_box is None:
        # YOLO hiçbir şey tespit etmediyse tüm resmi kullan
        # (portre fotoğrafi gibi durumlarda yüz zaten ortada)
        crop = img_bgr
    else:
        x1, y1, x2, y2 = best_box
        pad = 12
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(img_bgr.shape[1], x2 + pad)
        cy2 = min(img_bgr.shape[0], y2 + pad)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = img_bgr

    emb = _make_embedding(crop)
    return emb, ""


def _read_image(path: str) -> np.ndarray | None:
    """
    Windows/Unicode path sorunlarına dayanıklı görsel okuma.
    cv2.imread bazı OneDrive/TR karakterli yollarda None döndürebiliyor.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _detect_faces_with_ids(img_bgr: np.ndarray):
    """
    Fotoğraftaki tüm yüzleri tespit eder, soldan-sağa sıralayıp ID atar.
    Returns:
        faces: [{"face_id": int, "bbox": (x1,y1,x2,y2), "embedding": np.ndarray}]
        err: str
    """
    model = _get_yolo()
    if model is None:
        return [], "YOLO modeli yüklenemedi."

    h, w = img_bgr.shape[:2]
    results = model(img_bgr, conf=0.25, verbose=False)
    boxes = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))

    if not boxes:
        return [], "Fotoğrafta yüz tespit edilemedi."

    # Soldan-sağa, sonra yukarıdan-aşağıya sıralama ile stabil ID ataması
    boxes.sort(key=lambda b: (b[0], b[1]))

    faces = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        pad = 12
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = _make_embedding(crop)
        faces.append({
            "face_id": idx,
            "bbox": (x1, y1, x2, y2),
            "embedding": emb
        })

    if not faces:
        return [], "Yüz kırpma başarısız oldu."
    return faces, ""


class EmbedWorker(QThread):
    """Arka planda YOLO + DCT embedding üret."""
    finished = pyqtSignal(object, object, str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        img = _read_image(self.path)
        if img is None:
            self.finished.emit(None, None, "Fotoğraf okunamadı.")
            return
        faces, err = _detect_faces_with_ids(img)
        if err:
            self.finished.emit(None, None, err)
            return

        preview = img.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            fid = face["face_id"]

            # Daha belirgin gorunum: dis siyah + ic kirmizi cift cerceve
            cv2.rectangle(preview, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 0), 4)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 6)

            # ID etiketi: cok daha buyuk ve net gorunur badge
            label = f"ID {fid}"
            font_scale = 2.0
            font_thickness = 6
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            pad_x = 22
            pad_y = 18
            min_w = 150
            box_w = max(tw + (pad_x * 2), min_w)
            box_h = th + (pad_y * 2)
            ly1 = max(0, y1 - box_h - 8)
            ly2 = max(0, y1 - 4)
            lx2 = min(preview.shape[1] - 1, x1 + box_w)
            cv2.rectangle(preview, (x1, ly1), (lx2, ly2), (0, 0, 255), -1)
            cv2.rectangle(preview, (x1, ly1), (lx2, ly2), (0, 0, 0), 3)
            cv2.putText(
                preview,
                label,
                (x1 + pad_x, ly2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        self.finished.emit(faces, preview, "")


class SaveWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, path, emb, name, crime, danger, status, update_criminal_id=None):
        super().__init__()
        self.path   = path
        self.emb    = emb
        self.name   = name
        self.crime  = crime
        self.danger = danger
        self.status = status
        self.update_criminal_id = update_criminal_id

    def run(self):
        try:
            import sqlite3
            import shutil
            import datetime as dt
            db   = PROJECT_ROOT / "database" / "skywatch.db"
            pdir = PROJECT_ROOT / "database" / "photos"
            pdir.mkdir(parents=True, exist_ok=True)
            ext  = Path(self.path).suffix
            ts   = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            dst  = pdir / f"{self.name.replace(' ', '_')}_{ts}{ext}"
            shutil.copy2(self.path, dst)
            buf  = io.BytesIO()
            np.save(buf, self.emb)
            blob = buf.getvalue()
            conn = sqlite3.connect(str(db), check_same_thread=False)
            c    = conn.cursor()
            if self.update_criminal_id is not None:
                cid = int(self.update_criminal_id)
                c.execute(
                    "UPDATE criminals SET name=?,crime_type=?,danger_level=?,status=?,photo_path=? WHERE id=?",
                    (self.name, self.crime, self.danger, self.status, str(dst), cid)
                )
                c.execute("DELETE FROM embeddings WHERE criminal_id=?", (cid,))
                c.execute(
                    "INSERT INTO embeddings(criminal_id,embedding) VALUES(?,?)",
                    (cid, blob)
                )
            else:
                c.execute(
                    "INSERT INTO criminals(name,crime_type,danger_level,status,photo_path)"
                    " VALUES(?,?,?,?,?)",
                    (self.name, self.crime, self.danger, self.status, str(dst))
                )
                cid = c.lastrowid
                c.execute(
                    "INSERT INTO embeddings(criminal_id,embedding) VALUES(?,?)",
                    (cid, blob)
                )
            conn.commit()
            conn.close()
            if self.update_criminal_id is not None:
                self.finished.emit(True, f"'{self.name}' güncellendi (ID: {cid})")
            else:
                self.finished.emit(True, f"'{self.name}' eklendi (ID: {cid})")
        except Exception as e:
            self.finished.emit(False, str(e))


class PhotoZone(QLabel):
    """Fotoğraf yükleme alanı — sürükle/bırak veya tıkla."""
    photo_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._path = None
        self._empty()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _empty(self):
        self.setPixmap(QPixmap())
        self.setText("◆\n\nFotoğraf Yükle\n\nSürükle & Bırak  —  veya tıkla")
        self.setFont(QFont("Segoe UI", 12))
        self.setStyleSheet(f"""
            QLabel {{
                background: {SURFACE};
                border: 1.5px dashed {BORDER};
                border-radius: 16px;
                color: {ACCENT};
            }}
            QLabel:hover {{
                border-color: {ACCENT};
                background: {SURFACE_2};
            }}
        """)

    def mousePressEvent(self, e):
        path, _ = QFileDialog.getOpenFileName(
            self, "Fotoğraf Seç", str(Path.home()),
            "Görseller (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self._load(path)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(f"""
                QLabel {{
                    background: {ACCENT}10;
                    border: 2px solid {ACCENT};
                    border-radius: 16px;
                    color: {ACCENT};
                }}
            """)

    def dragLeaveEvent(self, e):
        if self._path is None:
            self._empty()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            if Path(p).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self._load(p)

    def _load(self, path: str):
        self._path = path
        px = QPixmap(path)
        if not px.isNull():
            px = px.scaled(
                self.width() - 4, self.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            off_x = max(0, (px.width()  - self.width()  + 4) // 2)
            off_y = max(0, (px.height() - self.height() + 4) // 2)
            px = px.copy(off_x, off_y, self.width() - 4, self.height() - 4)
            self.setPixmap(px)
            self.setStyleSheet(f"""
                QLabel {{
                    border: 1.5px solid {ACCENT};
                    border-radius: 16px;
                    background: {SURFACE};
                }}
            """)
        self.photo_selected.emit(path)

    def get_path(self):
        return self._path

    def clear(self):
        self._path = None
        self._empty()


def _field(label: str, widget: QWidget) -> QWidget:
    """Label + Widget dikey grubu."""
    w   = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(7)
    lbl = QLabel(label)
    lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {TEXT_3}; letter-spacing: 1.5px;")
    lay.addWidget(lbl)
    lay.addWidget(widget)
    return w


class AddCriminalPage(QWidget):
    person_added = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emb = None
        self._faces = []
        self._added_face_ids = set()
        self._batch_criminal_to_face: dict[int, int] = {}
        self._selected_face_id = None
        self._pending_face_id = None
        self._pending_name = ""
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sol: Fotoğraf Paneli ─────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(380)
        left.setStyleSheet(f"background: {BG_PANEL}; border-right: 1px solid {BORDER};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(36, 40, 36, 40)
        ll.setSpacing(20)

        # Başlık
        p_title = QLabel("Fotoğraf")
        p_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Black))
        p_title.setStyleSheet(f"color: {TEXT_1};")
        ll.addWidget(p_title)

        sub = QLabel("Yüz analizi için net bir fotoğraf ekle")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet(f"color: {TEXT_2};")
        ll.addWidget(sub)

        # Drop zone
        self._photo = PhotoZone()
        self._photo.setMinimumHeight(300)
        self._photo.photo_selected.connect(self._on_photo)
        ll.addWidget(self._photo, 1)

        # Embedding durumu
        self._emb_lbl = QLabel("Yüz verisi bekleniyor")
        self._emb_lbl.setFont(QFont("Segoe UI", 10))
        self._emb_lbl.setStyleSheet(f"color: {TEXT_3};")
        self._emb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ll.addWidget(self._emb_lbl)

        btn_clr = QPushButton("Temizle")
        btn_clr.clicked.connect(self._clear)
        ll.addWidget(btn_clr)

        root.addWidget(left)

        # ── Sağ: Form ────────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(48, 40, 48, 40)
        rl.setSpacing(0)

        r_title = PageTitle("Kişi Ekle")
        sub2 = QLabel("Yeni bir sabıkalı veya aranan kişiyi kayıt altına al")
        sub2.setFont(QFont("Segoe UI", 11))
        sub2.setStyleSheet(f"color: {TEXT_2};")
        rl.addWidget(r_title)
        rl.addSpacing(6)
        rl.addWidget(sub2)
        rl.addSpacing(32)
        rl.addWidget(Divider())
        rl.addSpacing(28)

        # Form alanları
        grid = QGridLayout()
        grid.setSpacing(20)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._name = QLineEdit()
        self._name.setPlaceholderText("İsim girin veya sistem otomatik atama yapar")
        self._name.setMinimumHeight(46)
        grid.addWidget(_field("AD SOYAD", self._name), 0, 0, 1, 2)

        self._crime = QComboBox()
        self._crime.setMinimumHeight(46)
        self._crime.addItems([
            "Uyuşturucu Kaçakçılığı", "Silahlı Soygun",
            "Terör", "Dolandırıcılık", "Adam Kaçırma",
            "Cinayet", "Hırsızlık", "Diğer"
        ])
        grid.addWidget(_field("SUÇ TÜRÜ *", self._crime), 1, 0)

        self._danger = QComboBox()
        self._danger.setMinimumHeight(46)
        self._danger.addItems(["DÜŞÜK", "ORTA", "YÜKSEK", "KRİTİK"])
        grid.addWidget(_field("TEHLİKE SEVİYESİ *", self._danger), 1, 1)

        self._status = QComboBox()
        self._status.setMinimumHeight(46)
        self._status.addItems(["ARANIYOR", "SABIKALI", "TEMİZE ÇIKMIŞ"])
        grid.addWidget(_field("DURUM *", self._status), 2, 0)

        self._gender = QComboBox()
        self._gender.setMinimumHeight(46)
        self._gender.addItems(["Erkek", "Kadın", "Belirtilmemiş"])
        grid.addWidget(_field("CİNSİYET", self._gender), 2, 1)

        self._face_select = QComboBox()
        self._face_select.setMinimumHeight(46)
        self._face_select.addItem("— Önce fotoğraf analiz edilsin —", None)
        self._face_select.currentIndexChanged.connect(self._on_face_selected)
        grid.addWidget(_field("YÜZ ID *", self._face_select), 3, 0, 1, 2)

        rl.addLayout(grid)
        rl.addSpacing(18)

        # Ekle butonu (Yuz ID seciminin hemen altinda, her zaman gorunur)
        self._btn = QPushButton("Seçili Yüzü Veritabanına Ekle")
        # Global tema disinda, sabit ve belirgin gorunum
        self._btn.setStyleSheet("""
            QPushButton {
                background: #C62828;
                color: #FFFFFF;
                border: 1px solid #8E0000;
                border-radius: 12px;
                font-size: 13px;
                font-weight: 700;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background: #D32F2F;
            }
            QPushButton:pressed {
                background: #B71C1C;
            }
            QPushButton:disabled {
                background: #8D6E63;
                color: #F5F5F5;
                border: 1px solid #6D4C41;
            }
        """)
        self._btn.setMinimumHeight(56)
        self._btn.clicked.connect(self._save)
        rl.addWidget(self._btn)

        rl.addStretch()
        rl.addWidget(Divider())
        rl.addSpacing(10)

        note = QLabel("* Fotoğraf ve Yüz ID zorunlu. İsim boşsa sistem otomatik üretir.")
        note.setFont(QFont("Segoe UI", 9))
        note.setStyleSheet(f"color: {TEXT_3};")
        rl.addSpacing(10)
        rl.addWidget(note)

        root.addWidget(right, 1)

    # ── Olaylar ───────────────────────────────────────────────────────────────
    def _on_photo(self, path: str):
        self._faces = []
        self._added_face_ids = set()
        self._batch_criminal_to_face = {}
        self._selected_face_id = None
        self._pending_face_id = None
        self._emb = None
        self._face_select.clear()
        self._face_select.addItem("— Analiz ediliyor... —", None)
        self._emb_lbl.setText("Yüz verisi analiz ediliyor...")
        self._emb_lbl.setStyleSheet(f"color: {AMBER};")
        w = EmbedWorker(path)
        w.finished.connect(self._on_emb)
        w.start()
        self._worker = w

    def _on_emb(self, faces, preview, err: str):
        if err or not faces:
            self._emb_lbl.setText(f"✕  {err}")
            self._emb_lbl.setStyleSheet(f"color: {RED};")
            self._face_select.clear()
            self._face_select.addItem("— ID bulunamadı —", None)
        else:
            self._faces = faces
            self._added_face_ids = set()
            self._batch_criminal_to_face = {}
            self._refresh_face_select()

            self._emb_lbl.setText(f"✓  {len(self._faces)} yüz tespit edildi, ID seçin")
            self._emb_lbl.setStyleSheet(f"color: {GREEN_GLOW};")

            if preview is not None:
                self._set_preview(preview)

    def _refresh_face_select(self, preferred_face_id=None):
        remaining = [f for f in self._faces if f["face_id"] not in self._added_face_ids]
        self._face_select.blockSignals(True)
        self._face_select.clear()

        if not remaining:
            self._face_select.addItem("— Tüm yüzler eklendi —", None)
            self._face_select.setCurrentIndex(0)
            self._face_select.blockSignals(False)
            self._selected_face_id = None
            self._emb = None
            return

        target_idx = 0
        for idx, face in enumerate(remaining):
            self._face_select.addItem(f"ID {face['face_id']}", face["face_id"])
            if preferred_face_id is not None and face["face_id"] == preferred_face_id:
                target_idx = idx

        self._face_select.setCurrentIndex(target_idx)
        self._face_select.blockSignals(False)
        self._on_face_selected(target_idx)

    def _set_preview(self, img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        px = QPixmap.fromImage(qimg)
        px = px.scaled(
            self._photo.width() - 4, self._photo.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        off_x = max(0, (px.width() - self._photo.width() + 4) // 2)
        off_y = max(0, (px.height() - self._photo.height() + 4) // 2)
        px = px.copy(off_x, off_y, self._photo.width() - 4, self._photo.height() - 4)
        self._photo.setPixmap(px)
        self._photo.setStyleSheet(f"""
            QLabel {{
                border: 1.5px solid {ACCENT};
                border-radius: 16px;
                background: {SURFACE};
            }}
        """)

    def _on_face_selected(self, _idx: int):
        self._emb = None
        self._selected_face_id = self._face_select.currentData()
        if self._selected_face_id is None:
            return
        selected = next((f for f in self._faces if f["face_id"] == self._selected_face_id), None)
        if selected is None:
            return
        self._emb = selected["embedding"]

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.clip(np.dot(emb1, emb2) / (n1 * n2), -1.0, 1.0))

    def _duplicate_threshold(self) -> float:
        try:
            import yaml
            with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            face_cfg = (cfg.get("face", {}) or {})
            # Mükerrer kayıt kontrolü, canlı tanıma eşiğinden daha sıkı olmalı.
            # duplicate_similarity_threshold varsa onu kullan; yoksa güvenli taban uygula.
            base = float(face_cfg.get("duplicate_similarity_threshold", face_cfg.get("similarity_threshold", 0.45)))
            return max(0.78, base)
        except Exception:
            return 0.78

    def _find_existing_criminal(self, emb: np.ndarray):
        import sqlite3
        db = PROJECT_ROOT / "database" / "skywatch.db"
        if not db.exists():
            return None

        threshold = self._duplicate_threshold()
        best = None
        best_score = threshold

        conn = sqlite3.connect(str(db), check_same_thread=False)
        c = conn.cursor()
        c.execute("""
            SELECT c.id, c.name, c.crime_type, c.danger_level, c.status, e.embedding
            FROM criminals c
            JOIN embeddings e ON e.criminal_id = c.id
        """)
        rows = c.fetchall()
        conn.close()

        for rid, name, crime_type, danger_level, status, emb_blob in rows:
            try:
                vec = np.load(io.BytesIO(emb_blob))
            except Exception:
                continue
            score = self._cosine_similarity(emb, vec)
            if score >= best_score:
                best_score = score
                best = {
                    "id": int(rid),
                    "name": name,
                    "crime_type": crime_type,
                    "danger_level": danger_level,
                    "status": status,
                    "score": score,
                }
        return best

    def _save(self):
        name = self._name.text().strip()
        # Kaydetme anında embedding'i mutlaka seçili ID'den yeniden al
        selected = next((f for f in self._faces if f["face_id"] == self._selected_face_id), None)
        if selected is None:
            QMessageBox.warning(self, "Hata", "Önce fotoğraf ekleyin ve bir Yüz ID seçin.")
            return
        current_emb = selected["embedding"]
        self._emb = current_emb
        if self._selected_face_id in self._added_face_ids:
            QMessageBox.warning(self, "Uyarı", "Bu Yüz ID zaten eklendi. Kalan ID'lerden birini seçin.")
            return
        if not name:
            name = "Bilinmiyor"
            self._name.setText(name)

        self._pending_face_id = self._selected_face_id
        self._pending_name = name
        dm = {"DÜŞÜK": "LOW", "ORTA": "MEDIUM", "YÜKSEK": "HIGH", "KRİTİK": "CRITICAL"}
        sm = {"ARANIYOR": "WANTED", "SABIKALI": "CRIMINAL", "TEMİZE ÇIKMIŞ": "CLEARED"}

        existing = self._find_existing_criminal(current_emb)
        update_id = None
        if existing is not None:
            # Aynı fotoğraf batch'inde farklı bir Yüz ID az önce eklendiyse,
            # embedding benzerliği yüksek çıksa bile yanlış mükerrer olabilir.
            batch_face = self._batch_criminal_to_face.get(existing["id"])
            if (batch_face is not None
                    and self._selected_face_id is not None
                    and batch_face != self._selected_face_id
                    and existing["score"] < 0.92):
                existing = None

        if existing is not None:
            status_tr = {
                "WANTED": "ARANIYOR",
                "CRIMINAL": "SABIKALI",
                "CLEARED": "TEMİZE ÇIKMIŞ",
            }.get(existing["status"], existing["status"])
            line = (
                f"ID {existing['id']} zaten kayıtlı: {existing['name']} | "
                f"Suç: {existing['crime_type']} | Tehlike: {existing['danger_level']} | "
                f"Durum: {status_tr} | Benzerlik: %{existing['score'] * 100:.1f}. "
                "Son ayarlarla güncellemek ister misiniz?"
            )
            ans = QMessageBox.question(
                self,
                "Mükerrer Kayıt Uyarısı",
                line,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if ans == QMessageBox.StandardButton.Yes:
                update_id = existing["id"]
            else:
                return

        self._btn.setEnabled(False)
        self._btn.setText("Kaydediliyor...")
        w = SaveWorker(
            self._photo.get_path(), current_emb, name,
            self._crime.currentText(),
            dm.get(self._danger.currentText(), "LOW"),
            sm.get(self._status.currentText(), "CRIMINAL"),
            update_criminal_id=update_id
        )
        w.finished.connect(self._on_save)
        w.start()
        self._sworker = w

    def _on_save(self, ok: bool, msg: str):
        self._btn.setText("Seçili Yüzü Veritabanına Ekle")
        if ok:
            if self._pending_face_id is not None:
                self._added_face_ids.add(self._pending_face_id)
            m = re.search(r"ID:\s*(\d+)", msg)
            if m and self._pending_face_id is not None:
                self._batch_criminal_to_face[int(m.group(1))] = int(self._pending_face_id)
            QMessageBox.information(self, "Başarılı", msg)
            self.person_added.emit(self._pending_name)
            if len(self._added_face_ids) >= len(self._faces):
                self._clear()
            else:
                remaining = len(self._faces) - len(self._added_face_ids)
                self._name.clear()
                self._pending_name = ""
                self._pending_face_id = None
                self._emb_lbl.setText(f"✓  Kayıt alındı. Kalan yüz sayısı: {remaining}")
                self._emb_lbl.setStyleSheet(f"color: {GREEN_GLOW};")
                self._refresh_face_select()
                self._btn.setEnabled(True)
        else:
            QMessageBox.critical(self, "Hata", msg)
            self._btn.setEnabled(True)

    def _clear(self):
        self._photo.clear()
        self._name.clear()
        self._emb = None
        self._faces = []
        self._added_face_ids = set()
        self._batch_criminal_to_face = {}
        self._selected_face_id = None
        self._pending_face_id = None
        self._pending_name = ""
        self._btn.setEnabled(True)
        self._btn.setText("Seçili Yüzü Veritabanına Ekle")
        self._emb_lbl.setText("Yüz verisi bekleniyor")
        self._emb_lbl.setStyleSheet(f"color: {TEXT_3};")
        self._face_select.clear()
        self._face_select.addItem("— Önce fotoğraf analiz edilsin —", None)
        self._crime.setCurrentIndex(0)
        self._danger.setCurrentIndex(0)
        self._status.setCurrentIndex(0)
        self._gender.setCurrentIndex(0)
