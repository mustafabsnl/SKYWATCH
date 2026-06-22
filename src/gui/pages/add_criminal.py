"""SKYWATCH — Kişi Ekleme Sayfası (YOLO best.pt tabanlı)"""

import sys
import io
import re
from pathlib import Path

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QFileDialog, QMessageBox, QGridLayout, QSizePolicy,
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


def _fit_pixmap_contain(src: QPixmap, box_w: int, box_h: int, bg: str = SURFACE) -> QPixmap:
    """Görseli kutuya oran koruyarak sığdırır (letterbox); kırpma yok."""
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))
    if src.isNull():
        out = QPixmap(box_w, box_h)
        out.fill(QColor(bg))
        return out
    scaled = src.scaled(
        box_w,
        box_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    canvas = QPixmap(box_w, box_h)
    canvas.fill(QColor(bg))
    painter = QPainter(canvas)
    x = (box_w - scaled.width()) // 2
    y = (box_h - scaled.height()) // 2
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return canvas


def _bgr_to_fit_pixmap(img_bgr: np.ndarray, box_w: int, box_h: int) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return _fit_pixmap_contain(QPixmap.fromImage(qimg), box_w, box_h)


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


def _detect_faces_with_ids(img_bgr: np.ndarray, logger=None):
    """
    Fotoğraftaki tüm yüzleri tespit eder, soldan-sağa sıralayıp ID atar.
    Returns:
        faces: [{"face_id": int, "bbox": (x1,y1,x2,y2), "embedding": np.ndarray}]
        err: str
    """
    try:
        from utils.config import AppConfig
        from core.face_analyzer import FaceAnalyzer
        
        cfg = AppConfig()
        analyzer = FaceAnalyzer(cfg)
        
        dup_cfg = cfg.get("face", {}).get("duplicate_check", {})
        min_conf = float(dup_cfg.get("min_face_confidence", 0.60))
        
        if logger:
            logger.info("[FACE_DETECT_ADD] Starting face detection for new criminal.")

        results = analyzer.detect_faces(img_bgr)
        
        if not results:
            if logger: logger.warning("[FACE_DETECT_ADD] No faces found.")
            return [], "Fotoğrafta yüz tespit edilemedi."
            
        faces = []
        for r in results:
            # bbox var mı?
            if not r.bbox or len(r.bbox) != 4:
                continue
            
            # Confidence kontrolü
            if r.det_score < min_conf:
                if logger: logger.warning(f"[FACE_DETECT_ADD] Face rejected, low confidence: {r.det_score:.2f} < {min_conf}")
                continue
                
            # Embedding var mı?
            if r.embedding is None:
                if logger: logger.warning("[FACE_DETECT_ADD] Face rejected, could not extract embedding.")
                continue
                
            # Embedding doğrulaması (512-d, float32, normalized)
            emb = np.asarray(r.embedding, dtype=np.float32).reshape(-1)
            if emb.shape[0] != _EMBED_DIM:
                if logger: logger.warning(f"[FACE_DETECT_ADD] Invalid embedding shape: {emb.shape}")
                continue
                
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                if logger: logger.warning("[FACE_DETECT_ADD] Face rejected, zero embedding vector.")
                continue
                
            emb = emb / norm
            
            if logger:
                logger.info(f"[EMBEDDING_NEW] shape={emb.shape} norm={np.linalg.norm(emb):.3f} valid=true conf={r.det_score:.2f}")

            faces.append({
                "bbox": tuple(map(int, r.bbox)),
                "embedding": emb,
                "det_score": r.det_score
            })
            
        if not faces:
            return [], "Yüz net algılanamadı. Daha net bir fotoğraf seçin."
            
        # Soldan-sağa sıralama
        faces.sort(key=lambda f: f["bbox"][0])
        
        # ID atama
        for idx, f in enumerate(faces, start=1):
            f["face_id"] = idx
            
        if logger:
            logger.info(f"[FACE_DETECT_ADD] Successfully extracted {len(faces)} faces.")
            
        return faces, ""
        
    except Exception as e:
        if logger: logger.error(f"[FACE_DETECT_ADD] Error: {e}")
        return [], f"Yüz analizi sırasında hata oluştu: {e}"


class EmbedWorker(QThread):
    """Arka planda YOLO + DCT embedding üret."""
    finished = pyqtSignal(object, object, str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        from utils.logger import EventLogger
        from utils.config import AppConfig
        
        cfg = AppConfig()
        logger = EventLogger(cfg)
        
        img = _read_image(self.path)
        if img is None:
            self.finished.emit(None, None, "Fotoğraf okunamadı.")
            return
            
        faces, err = _detect_faces_with_ids(img, logger)
        if err:
            self.finished.emit(None, None, err)
            return
            
        if len(faces) > 1:
            logger.warning("[FACE_DETECT_ADD] Multiple faces detected.")
            # İsteğe bağlı olarak burada uyarı döndürülebilir, ama şimdilik devam edip kullanıcıya seçtiriyoruz.

        preview = img.copy()
        ph, pw = preview.shape[:2]
        ref = max(1, min(pw, ph))
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            fid = face["face_id"]
            thick = max(2, int(round(ref / 280)))
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), thick)

            label = f"ID {fid}"
            fs = max(0.45, min(0.85, ref / 900.0))
            th = max(1, thick - 1)
            (tw, th_txt), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            pad_x, pad_y = 6, 5
            lx1 = x1
            ly2 = max(bl + 2, y1 - 4)
            ly1 = max(0, ly2 - th_txt - pad_y * 2)
            lx2 = min(pw - 1, lx1 + tw + pad_x * 2)
            cv2.rectangle(preview, (lx1, ly1), (lx2, ly2), (30, 30, 34), -1)
            cv2.rectangle(preview, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)
            cv2.putText(
                preview,
                label,
                (lx1 + pad_x, ly2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (255, 255, 255),
                th,
                cv2.LINE_AA,
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
            from utils.config import AppConfig
            from utils.logger import EventLogger
            import sqlite3
            import shutil
            import datetime as dt
            
            cfg = AppConfig()
            logger = EventLogger(cfg)
            
            # 1. Normalize and check embedding before saving
            try:
                emb = np.asarray(self.emb, dtype=np.float32).reshape(-1)
                if emb.shape != (512,):
                    raise ValueError(f"Invalid shape: {emb.shape}")
                
                norm = np.linalg.norm(emb)
                if norm < 1e-8:
                    raise ValueError("Zero norm embedding")
                
                emb = emb / norm
                logger.info(f"[PERSON_SAVE_EMBEDDING] shape={emb.shape} norm={np.linalg.norm(emb):.3f} dtype={emb.dtype}")
            except Exception as e:
                logger.error(f"[PERSON_SAVE_ERROR] Failed to prepare embedding: {e}")
                self.finished.emit(False, f"Yüz verisi geçersiz: {e}")
                return

            db   = PROJECT_ROOT / "database" / "skywatch.db"
            pdir = PROJECT_ROOT / "database" / "photos"
            pdir.mkdir(parents=True, exist_ok=True)
            ext  = Path(self.path).suffix
            ts   = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            dst  = pdir / f"{self.name.replace(' ', '_')}_{ts}{ext}"
            shutil.copy2(self.path, dst)
            buf  = io.BytesIO()
            np.save(buf, emb)
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
        self._source_px: QPixmap | None = None
        self._preview_bgr: np.ndarray | None = None
        self._empty()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(sp)

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

    def _display_box_size(self) -> tuple[int, int]:
        return max(120, self.width() - 8), max(120, self.height() - 8)

    def _refresh_display(self):
        bw, bh = self._display_box_size()
        if self._preview_bgr is not None:
            self.setPixmap(_bgr_to_fit_pixmap(self._preview_bgr, bw, bh))
            return
        if self._source_px is not None and not self._source_px.isNull():
            self.setPixmap(_fit_pixmap_contain(self._source_px, bw, bh))
            return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._path is not None or self._preview_bgr is not None:
            self._refresh_display()

    def set_preview_image(self, img_bgr: np.ndarray | None):
        """Yüz tespiti sonrası önizleme (oran korunur)."""
        self._preview_bgr = img_bgr
        if img_bgr is not None:
            self._refresh_display()
            self.setStyleSheet(f"""
                QLabel {{
                    border: 1.5px solid {ACCENT};
                    border-radius: 16px;
                    background: {SURFACE};
                }}
            """)

    def _load(self, path: str):
        self._path = path
        self._preview_bgr = None
        px = QPixmap(path)
        if not px.isNull():
            self._source_px = px
            self._refresh_display()
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
        self._source_px = None
        self._preview_bgr = None
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

        # ── Sol: Fotoğraf Paneli (geniş) ─────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(440)
        left.setMaximumWidth(920)
        left.setStyleSheet(f"background: {BG_PANEL}; border-right: 1px solid {BORDER};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(28, 36, 28, 36)
        ll.setSpacing(16)

        # Başlık
        p_title = QLabel("Fotoğraf")
        p_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Black))
        p_title.setStyleSheet(f"color: {TEXT_1};")
        ll.addWidget(p_title)

        sub = QLabel("Yüz analizi için net bir fotoğraf ekle")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet(f"color: {TEXT_2};")
        ll.addWidget(sub)

        # Drop zone — dikey alanda geniş, görsel oran korunarak sığar
        self._photo = PhotoZone()
        self._photo.setMinimumHeight(340)
        self._photo.setMinimumWidth(360)
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

        root.addWidget(left, 3)

        # ── Sağ: Form (daha dar) ─────────────────────────────────────────
        right = QWidget()
        right.setMaximumWidth(520)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(32, 36, 36, 36)
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
        self._photo.set_preview_image(img_bgr)

    def _on_face_selected(self, _idx: int):
        self._emb = None
        self._selected_face_id = self._face_select.currentData()
        if self._selected_face_id is None:
            return
        selected = next((f for f in self._faces if f["face_id"] == self._selected_face_id), None)
        if selected is None:
            return
        self._emb = selected["embedding"]

    def _save(self):
        from utils.config import AppConfig
        from utils.logger import EventLogger
        from database.db import Database
        
        cfg = AppConfig()
        logger = EventLogger(cfg)
        db = Database(cfg, logger)

        name = self._name.text().strip()
        
        if not name:
            name = "Bilinmiyor"
            self._name.setText(name)
            
        logger.info(f"[PERSON_ADD_START] name='{name}'")

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

        self._pending_face_id = self._selected_face_id
        self._pending_name = name
        dm = {"DÜŞÜK": "LOW", "ORTA": "MEDIUM", "YÜKSEK": "HIGH", "KRİTİK": "CRITICAL"}
        sm = {"ARANIYOR": "WANTED", "SABIKALI": "CRIMINAL", "TEMİZE ÇIKMIŞ": "CLEARED"}

        dup_cfg = cfg.get("face", {}).get("duplicate_check", {})
        is_dup_enabled = bool(dup_cfg.get("enabled", True))
        threshold = float(dup_cfg.get("cosine_threshold", 0.62))
        uncertain_low = float(dup_cfg.get("uncertain_low", 0.52))

        update_id = None
        
        if is_dup_enabled:
            logger.info(f"[DUP_CHECK_START] threshold={threshold} uncertain_low={uncertain_low}")
            existing = db.find_duplicate_person(current_emb, threshold, uncertain_low)
            
            if existing is not None:
                # Aynı fotoğraf batch'inde farklı bir Yüz ID az önce eklendiyse
                batch_face = self._batch_criminal_to_face.get(existing["person_id"])
                if (batch_face is not None
                        and self._selected_face_id is not None
                        and batch_face != self._selected_face_id
                        and existing["similarity"] < 0.92):
                    existing = None
                    
            if existing is not None:
                logger.info(f"[DUP_CHECK_SCORE] candidate_id={existing['person_id']} name='{existing['name']}' similarity={existing['similarity']:.2f} percent={existing['percent']:.1f}")

                status_tr = {
                    "WANTED": "ARANIYOR",
                    "CRIMINAL": "SABIKALI",
                    "CLEARED": "TEMİZE ÇIKMIŞ",
                }.get(existing["status"], existing["status"])
                
                if existing["level"] == "duplicate":
                    title = "Kesin Mükerrer Kayıt Uyarısı"
                    line = (
                        f"ID {existing['person_id']} zaten kayıtlı: {existing['name']} | "
                        f"Suç: {existing['crime_type']} | Tehlike: {existing['danger_level']} | "
                        f"Durum: {status_tr}\n\n"
                        f"Benzerlik: %{existing['percent']:.1f}\n\n"
                        "Bu kişinin zaten sistemde olduğu tespit edildi.\n"
                        "Mevcut kişiyi güncelleyelim mi yoksa yine de yeni kayıt olarak mı eklensin?"
                    )
                else:
                    title = "Şüpheli Mükerrer Kayıt"
                    line = (
                        f"Bu kişi ID {existing['person_id']} ({existing['name']}) ile orta seviyede benziyor.\n"
                        f"Benzerlik: %{existing['percent']:.1f}\n\n"
                        "Mevcut kişiyi güncelleyelim mi yoksa yine de yeni kayıt olarak mı eklensin?"
                    )

                msg_box = QMessageBox(self)
                msg_box.setWindowTitle(title)
                msg_box.setText(line)
                
                btn_update = msg_box.addButton("Mevcut Kişiyi Güncelle", QMessageBox.ButtonRole.AcceptRole)
                btn_new = msg_box.addButton("Yeni Kişi Olarak Ekle", QMessageBox.ButtonRole.DestructiveRole)
                btn_cancel = msg_box.addButton("İptal", QMessageBox.ButtonRole.RejectRole)
                
                msg_box.exec()
                
                clicked_btn = msg_box.clickedButton()
                
                if clicked_btn == btn_update:
                    update_id = existing["person_id"]
                elif clicked_btn == btn_new:
                    update_id = None
                else:
                    logger.info("[PERSON_ADD_CANCELLED] reason='user_cancelled_duplicate_dialog'")
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
        from utils.config import AppConfig
        from utils.logger import EventLogger
        
        cfg = AppConfig()
        logger = EventLogger(cfg)
        
        self._btn.setText("Seçili Yüzü Veritabanına Ekle")
        if ok:
            if self._pending_face_id is not None:
                self._added_face_ids.add(self._pending_face_id)
            m = re.search(r"ID:\s*(\d+)", msg)
            if m and self._pending_face_id is not None:
                pid = int(m.group(1))
                self._batch_criminal_to_face[pid] = int(self._pending_face_id)
                logger.info(f"[PERSON_ADD_SUCCESS] person_id={pid}")
                
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
