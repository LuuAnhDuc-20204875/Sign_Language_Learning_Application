from __future__ import annotations

import time
import random
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

from PyQt5.QtCore import Qt, QTimer, QSettings, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox,
    QPushButton, QFrame, QGridLayout, QSplitter, QSizePolicy, QProgressBar,
    QToolButton, QStyle, QSpacerItem, QScrollArea, QButtonGroup,
    QComboBox, QInputDialog
)
from PyQt5.QtWidgets import QApplication

from core.config import AppConfig
from core.progress import ProgressManager
from core.model import ASLModel
from core.user_store import UserStore
from core.telemetry import Telemetry
from games.snake import SnakeGame
from ui.theme import APP_QSS
from ui.dialogs import SettingsDialog


LABELS = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "del","nothing","space"
]
LEARN_LETTERS = LABELS[:26]


class ASLLearningApp(QMainWindow):
    def __init__(self, cfg: AppConfig, username: Optional[str] = None, user_store: Optional[UserStore] = None):
        super().__init__()

        # user (local-only)
        self.username = username  # None = Guest
        self.user_store = user_store if user_store is not None else UserStore("users.json")
        self.high_scores = {
            "quiz": self.user_store.get_high_score(self.username, "quiz"),
            "snake": self.user_store.get_high_score(self.username, "snake"),
            "spelling": self.user_store.get_high_score(self.username, "spelling"),
            "mcq": self.user_store.get_high_score(self.username, "mcq"),
        }

        self.cfg = cfg
        self.settings = QSettings("ASLExample", "ASLLearningSuiteV2")

        # per-user progress file
        self.progress = ProgressManager.for_user(self.username)

        # per-user telemetry (events + aggregated stats for thesis)
        self.telemetry = Telemetry.for_user(self.username)
        try:
            self.telemetry.events.log("app_start", {"user": self.telemetry.user_key})
        except Exception:
            pass

        # timers for evaluation metrics
        self._mode_entered_at: float = time.time()
        self._target_started_at: float = time.time()
        self._mcq_question_started_at: float = 0.0
        self._spelling_word_started_at: float = 0.0
        self._snake_session_started_at: float = 0.0
        self._snake_last_score: int = 0
        self._snake_last_gameover: bool = False

        # ----- State -----
        self.mode = "menu"  # menu | learn | quiz | mcq | snake | spelling
        self.learn_idx = 0
        self.quiz_target: Optional[str] = None
        self.score = 0

        # multiple choice (MCQ)
        self.mcq_target: Optional[str] = None
        self.mcq_options = {}  # {'A': 'F', 'B': 'K', ...}
        self.mcq_correct: Optional[str] = None  # option key: 'A'/'B'/'C'/'D'
        self.mcq_score = 0
        self.mcq_rects = []  # list[(opt_key, (x1,y1,x2,y2))] in frame coords
        self._mcq_last_choice_at = 0.0
        # MCQ gesture-selection state
        self._mcq_hover_key: Optional[str] = None
        self._mcq_prev_pinched: bool = False
        self._mcq_is_pinched: bool = False

        # suggestion popup throttling
        self._last_suggest_check = 0.0
        self._last_popup_at = 0.0

        # spelling
        self.spell_target: Optional[str] = None
        self.spell_current = ""
        self.spell_index = 0
        self.spell_score = 0

        # prediction state
        self.last_pred_time = 0.0
        self.correct_streak = 0
        self.stable_label = "—"
        self.last_conf = 0.0
        self.recent_preds = deque(maxlen=5)

        # to avoid logging the same frame repeatedly:
        self._last_logged_target: Optional[str] = None
        self._last_logged_pred: Optional[str] = None
        self._last_logged_at: float = 0.0

        # toggles
        self.flip_camera = self.settings.value("flip_camera", "true") == "true"
        self.teacher_mode = False
        self.prev_frame_time = time.time()
        self.current_fps = 0.0

        # correct overlay
        self.correct_effect_active = False
        self.correct_effect_end_time = 0.0
        self.last_correct_target: Optional[str] = None

        # update weak letters throttling
        self._last_weak_ui_update = 0.0

        # ----- Camera + Detector + Model + Snake -----
        cam_index = int(self.settings.value("camera_index", cfg.camera_index))
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

        self.detector = HandDetector(maxHands=1)

        try:
            self.asl_model = ASLModel(cfg.model_path, LABELS)
        except Exception as e:
            QMessageBox.critical(self, "Model load failed", f"Cannot load model:\n{e}")
            raise

        ret, tmp_frame = self.cap.read()
        if not ret or tmp_frame is None:
            self.game_h, self.game_w = cfg.frame_height, cfg.frame_width
        else:
            self.game_h, self.game_w = tmp_frame.shape[:2]

        self.snake_game = SnakeGame(cfg.food_image, self.game_w, self.game_h)

        # UI
        self.init_ui()
        self.apply_theme()

        # ----- Frame timer (must start AFTER UI is created) -----
        # If the camera LED is ON but the UI stays blank, the usual reason is
        # update_frame() is never being called. Starting a QTimer fixes that.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

        # Render first frame immediately (best-effort)
        try:
            self.update_frame()
        except Exception:
            pass

        # initial telemetry
        try:
            self.telemetry.mode_enter(self.mode)
        except Exception:
            pass

    # =========================
    # Telemetry helpers
    # =========================
    def _mark_mode_enter(self, mode: str) -> None:
        self._mode_entered_at = time.time()
        try:
            self.telemetry.mode_enter(mode)
        except Exception:
            pass

    def _mark_target_start(self) -> None:
        self._target_started_at = time.time()

        # NOTE: Frame timer is started once in __init__. Do not start it here.

    # =========================
    # Multiple Choice (MCQ)
    # =========================
    # =========================
    def next_mcq_question(self):
        # Prefer weak letter sometimes
        target = random.choice(LEARN_LETTERS)
        try:
            weak, acc, total = self.progress.suggestion_candidate(min_attempts=6, acc_threshold=0.70)
            if weak:
                # 60% chance focus weak letter
                if random.random() < 0.6:
                    target = weak
        except Exception:
            pass
        distractors = [l for l in LEARN_LETTERS if l != target]
        distractors = random.sample(distractors, 3)
        letters = [target] + distractors
        random.shuffle(letters)
        keys = ["A", "B", "C", "D"]
        self.mcq_options = {keys[i]: letters[i] for i in range(4)}
        self.mcq_target = target
        self.mcq_correct = next((k for k, v in self.mcq_options.items() if v == target), None)
        self.mcq_rects = []
        self._mcq_last_choice_at = 0.0
        # evaluation timer + log
        self._mcq_question_started_at = time.time()
        try:
            self.telemetry.mcq_question(target=str(target), options=dict(self.mcq_options))
        except Exception:
            pass
        # pre-load option images
        self._mcq_imgs = {}
        for k, lt in self.mcq_options.items():
            img = self._load_letter_image(lt)
            self._mcq_imgs[k] = img

    def _load_letter_image(self, letter: str):
        """Load one random image for a given letter.

        Priority:
        1) cfg.multiplechoice_dir
           - supports a single file:   MCQs/A.jpg|png|...
           - or a folder of files:     MCQs/A/*.jpg|png|...
        2) cfg.reference_dir (EXAMPLE_IMG/Sample_A.*)
        """
        lt = (letter or "").strip().upper()
        # 1) Multiplechoice directory (accept F.jpg/png or any file inside subfolder F)
        try:
            base = Path(getattr(self.cfg, "multiplechoice_dir", "Multiplechoice"))
            cand: list[Path] = []
            # A single file: MCQs/A.jpg, MCQs/A.png, ...
            for ext in ("png", "jpg", "jpeg", "webp"):
                cand.append(base / f"{lt}.{ext}")

            # A folder of files: MCQs/A/*.jpg|png|...
            sub = base / lt
            if sub.exists() and sub.is_dir():
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        cand.append(f)

            # Choose random among existing candidates
            cand = [p for p in cand if p.exists()]
            if cand:
                p = random.choice(cand)
                img = cv2.imread(str(p))
                if img is not None:
                    return img
        except Exception:
            pass
        # 2) Reference images
        try:
            ref = Path(getattr(self.cfg, "reference_dir", "EXAMPLE_IMG"))
            for ext in ("jpeg", "jpg", "png"):
                p = ref / f"Sample_{lt}.{ext}"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        return img
        except Exception:
            pass
        return None

    def _mcq_calc_rects(self, w: int, h: int):
        """Compute MCQ option rectangles (A-D) in frame coordinates (centered)."""
        size = int(min(w, h) * 0.22)
        size = max(120, min(size, 220))
        margin = 16

        grid_w = 2 * size + margin
        grid_h = 2 * size + margin

        # Center the 2x2 grid
        #MQCS UI, UI MQCS
        x0 = (w - grid_w) // 2
        y0 = (h - grid_h) // 2 - 60  # +20 để chừa chỗ header/text phía trên

        # clamp tránh sát mép
        x0 = max(margin, min(x0, w - grid_w - margin))
        y0 = max(60,   min(y0, h - grid_h - margin))  # 60 để không đè chữ câu hỏi

        rects = []
        keys = ["A", "B", "C", "D"]
        for i, k in enumerate(keys):
            row = i // 2
            col = i % 2
            x1 = int(x0 + col * (size + margin))
            y1 = int(y0 + row * (size + margin))
            x2 = x1 + size
            y2 = y1 + size
            rects.append((k, (x1, y1, x2, y2)))
        return rects


    def _draw_mcq_overlay(self, display):
        if self.mcq_target is None or not self.mcq_options:
            return display

        h, w = display.shape[:2]

        rects = self._mcq_calc_rects(w, h)  # ✅ dùng chung vị trí
        self.mcq_rects = rects

        for k, (x1, y1, x2, y2) in rects:
            size = x2 - x1

            img = getattr(self, "_mcq_imgs", {}).get(k)
            if img is not None:
                thumb = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                display[y1:y2, x1:x2] = thumb
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), (40, 40, 40), -1)

            hover = (getattr(self, "_mcq_hover_key", None) == k)
            border_col = (0, 255, 255) if hover else (255, 255, 255)
            border_th = 4 if hover else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), border_col, border_th)

            cv2.putText(display, k, (x1 + 8, y1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(display, k, (x1 + 8, y1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # question + gesture hint
        cv2.putText(display, f"Dau la chu {self.mcq_target}?", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        hint = "Pinch (tro + giua) de chon" if not getattr(self, "_mcq_is_pinched", False) else "PINCHED"
        cv2.putText(display, hint, (12, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        return display
    # =========================
    # UI
    # =========================
    def apply_theme(self):
        self.setStyleSheet(APP_QSS)

    def _make_card(self, title: str, subtitle: str = "") -> Tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        ttl = QLabel(title)
        ttl.setObjectName("H1")
        lay.addWidget(ttl)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("Muted")
            sub.setWordWrap(True)
            lay.addWidget(sub)

        return card, lay

    def _make_mode_btn(self, text: str, icon_sp: int) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("ModeBtn")
        btn.setCheckable(True)
        btn.setMinimumHeight(44)
        btn.setIcon(self.style().standardIcon(icon_sp))
        btn.setIconSize(QSize(18, 18))
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return btn

    def init_ui(self):
        self.setWindowTitle("ASL Learning Suite")
        self.setMinimumSize(1280, 760)

        # ===== Root central =====
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ===== Top Bar =====
        top = QFrame()
        top.setObjectName("TopBar")
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(14, 12, 14, 12)
        top_l.setSpacing(10)

        title = QLabel("ASL Learning Suite")
        title.setObjectName("H1")
        subtitle = QLabel("Real-time hand tracking • Learn / Quiz / Spelling • Snake Game")
        subtitle.setObjectName("Muted")

        left_title = QVBoxLayout()
        left_title.setSpacing(2)
        left_title.addWidget(title)
        left_title.addWidget(subtitle)
        top_l.addLayout(left_title)

        top_l.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # status chips
        self.chip_mirror = QLabel("Mirror: ON")
        self.chip_mirror.setObjectName("Pill")
        self.chip_teacher = QLabel("Teacher: OFF")
        self.chip_teacher.setObjectName("Pill")
        self.chip_user = QLabel(f"User: {self.username or 'Guest'}")
        self.chip_user.setObjectName("Pill")

        top_l.addWidget(self.chip_mirror)
        top_l.addWidget(self.chip_teacher)
        top_l.addWidget(self.chip_user)

        # tools
        self.btn_mirror = QToolButton()
        self.btn_mirror.setObjectName("IconBtn")
        self.btn_mirror.setCheckable(True)
        self.btn_mirror.setChecked(self.flip_camera)
        self.btn_mirror.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.btn_mirror.setToolTip("Mirror camera (F)")
        top_l.addWidget(self.btn_mirror)

        self.btn_settings = QToolButton()
        self.btn_settings.setObjectName("IconBtn")
        self.btn_settings.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.btn_settings.setToolTip("Settings")
        top_l.addWidget(self.btn_settings)

        self.btn_logout = QToolButton()
        self.btn_logout.setObjectName("IconBtn")
        self.btn_logout.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.btn_logout.setToolTip("Đổi tài khoản / Logout")
        top_l.addWidget(self.btn_logout)

        root.addWidget(top)

        # ===== Splitter body =====
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        # ===== Left: Video Card =====
        video_card, video_lay = self._make_card("Camera", "Giữ tay trong khung • F: mirror • ESC: thoát")
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setObjectName("VideoFrame")
        self.camera_label.setMinimumSize(840, 540)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # allow mouse click selection for MCQ
        self.camera_label.mousePressEvent = self.on_camera_clicked
        video_lay.addWidget(self.camera_label, stretch=1)

        # Footer row under video
        video_footer = QHBoxLayout()
        self.lbl_mode = QLabel("Mode: MENU")
        self.lbl_mode.setObjectName("Muted")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setObjectName("Muted")
        video_footer.addWidget(self.lbl_mode)
        video_footer.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        video_footer.addWidget(self.lbl_fps)
        video_lay.addLayout(video_footer)

        splitter.addWidget(video_card)

        # ===== Right panel: scrollable dashboard =====
        scroll = QScrollArea()
        scroll.setObjectName("RightScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right = QWidget()
        scroll.setWidget(right)

        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(12)

        # Modes card
        modes_card, modes_lay = self._make_card("Modes", "Phím: 0 Menu • 1 Learn • 2 Quiz • 3 Snake • 4 Spelling • 5 MCQ")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.btn_menu = self._make_mode_btn("Menu", QStyle.SP_DesktopIcon)
        self.btn_learn = self._make_mode_btn("Learn A–Z", QStyle.SP_DirHomeIcon)
        self.btn_quiz = self._make_mode_btn("Quiz", QStyle.SP_MessageBoxQuestion)
        self.btn_snake = self._make_mode_btn("Snake", QStyle.SP_MediaPlay)
        self.btn_spelling = self._make_mode_btn("Spelling", QStyle.SP_FileDialogContentsView)
        self.btn_mcq = self._make_mode_btn("MCQ", QStyle.SP_DialogYesButton)

        grid.addWidget(self.btn_menu, 0, 0)
        grid.addWidget(self.btn_learn, 0, 1)
        grid.addWidget(self.btn_quiz, 1, 0)
        grid.addWidget(self.btn_snake, 1, 1)
        grid.addWidget(self.btn_spelling, 2, 0)
        grid.addWidget(self.btn_mcq, 2, 1)

        modes_lay.addLayout(grid)
        right_l.addWidget(modes_card)

        # Current target card
        target_card, target_lay = self._make_card("Current Target", "Chữ / từ cần thực hiện")
        self.lbl_big_target = QLabel("—")
        self.lbl_big_target.setAlignment(Qt.AlignCenter)
        self.lbl_big_target.setFont(QFont("Arial", int(getattr(self.cfg, "target_font_pt", 56)), QFont.Bold))
        self.lbl_big_target.setObjectName("BigTarget")
        self.lbl_big_target.setMinimumHeight(120)
        self.lbl_big_target.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        target_lay.addWidget(self.lbl_big_target)

        self.lbl_target = QLabel("Target: —")
        self.lbl_target.setObjectName("Muted")
        target_lay.addWidget(self.lbl_target)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Chọn chữ (A–Z):"))
        self.cmb_learn_letter = QComboBox()
        self.cmb_learn_letter.addItems(LEARN_LETTERS)
        self.cmb_learn_letter.setCurrentText(LEARN_LETTERS[self.learn_idx])
        self.cmb_learn_letter.setEnabled(True)
        sel_row.addWidget(self.cmb_learn_letter)
        self.btn_set_letter = QPushButton("Học chữ này")
        self.btn_set_letter.setMinimumHeight(32)
        sel_row.addWidget(self.btn_set_letter)
        target_lay.addLayout(sel_row)

        right_l.addWidget(target_card)

        # Recognition card
        rec_card, rec_lay = self._make_card("Recognition", "Prediction + độ tin cậy + streak")
        self.lbl_prediction = QLabel("Prediction: —")
        self.lbl_prediction.setObjectName("PredLabel")
        rec_lay.addWidget(self.lbl_prediction)

        self.pb_conf = QProgressBar()
        self.pb_conf.setRange(0, 100)
        self.pb_conf.setValue(0)
        self.pb_conf.setFormat("Confidence: %p%")
        rec_lay.addWidget(self.pb_conf)

        self.pb_streak = QProgressBar()
        self.pb_streak.setRange(0, int(self.cfg.streak_required))
        self.pb_streak.setFormat("Streak: %v / %m")
        self.pb_streak.setValue(0)
        rec_lay.addWidget(self.pb_streak)

        self.lbl_score = QLabel("Score: —")
        self.lbl_score.setObjectName("Muted")
        rec_lay.addWidget(self.lbl_score)

        self.lbl_high_score = QLabel("High Score: —")
        self.lbl_high_score.setObjectName("Muted")
        rec_lay.addWidget(self.lbl_high_score)
        right_l.addWidget(rec_card)

        # Progress + Reference row
        row = QHBoxLayout()
        row.setSpacing(12)

        prog_card, prog_lay = self._make_card("Weak Letters", "Top chữ cần luyện thêm")
        self.lbl_weak_letters = QLabel("Chưa có dữ liệu.")
        self.lbl_weak_letters.setWordWrap(True)
        self.lbl_weak_letters.setObjectName("Muted")
        prog_lay.addWidget(self.lbl_weak_letters)
        row.addWidget(prog_card, 1)

        ref_card, ref_lay = self._make_card("Reference", "Ảnh minh hoạ ký hiệu")
        self.ref_label = QLabel("No image")
        self.ref_label.setAlignment(Qt.AlignCenter)
        self.ref_label.setObjectName("RefFrame")
        self.ref_label.setMinimumSize(220, 220)
        self.ref_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ref_lay.addWidget(self.ref_label, 1)
        row.addWidget(ref_card, 1)

        right_l.addLayout(row)

        # Controls card
        bottom_card, bottom_lay = self._make_card("Controls", "R: reset snake • T: teacher overlay • F: mirror")
        self.btn_teacher = QPushButton("Teacher Mode: OFF")
        self.btn_teacher.setCheckable(True)
        self.btn_teacher.setMinimumHeight(42)
        bottom_lay.addWidget(self.btn_teacher)

        self.hint_label = QLabel("Tip: giữ tay ổn định vài giây để đủ streak.")
        self.hint_label.setObjectName("Muted")
        self.hint_label.setWordWrap(True)
        bottom_lay.addWidget(self.hint_label)
        right_l.addWidget(bottom_card)

        right_l.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # add scroll to splitter
        splitter.addWidget(scroll)
        splitter.setSizes([900, 420])

        # ===== Connect signals =====
        self.btn_menu.clicked.connect(self.on_menu_clicked)
        self.btn_learn.clicked.connect(self.on_learn_clicked)
        self.btn_quiz.clicked.connect(self.on_quiz_clicked)
        self.btn_snake.clicked.connect(self.on_snake_clicked)
        self.btn_spelling.clicked.connect(self.on_spelling_clicked)
        self.btn_mcq.clicked.connect(self.on_mcq_clicked)

        self.btn_teacher.clicked.connect(self.on_teacher_toggled)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_mirror.clicked.connect(self.toggle_flip)
        self.cmb_learn_letter.currentTextChanged.connect(self.on_learn_letter_changed)
        self.btn_set_letter.clicked.connect(self.on_set_learn_letter)
        self.btn_logout.clicked.connect(self.on_logout_clicked)

        # Exclusive button group
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for b in [self.btn_menu, self.btn_learn, self.btn_quiz, self.btn_snake, self.btn_spelling]:
            self.mode_group.addButton(b)

        # init state
        self._set_checked_button(self.btn_menu)
        self.update_ui_text()
        self.update_progress_quick_view()
        self._maybe_show_study_suggestion(time.time())  # init-time suggestion check
        # High score label (by current mode)
        if hasattr(self, "lbl_high_score"):
            if self.mode in ("quiz", "snake", "spelling", "mcq"):
                hs = int(self.high_scores.get(self.mode, 0) or 0)
                self.lbl_high_score.setText(f"High Score: {hs}")
            else:
                self.lbl_high_score.setText("High Score: —")

        self._update_topbar_chips()
        self._update_settings_access()

    def _update_topbar_chips(self):
        self.chip_mirror.setText(f"Mirror: {'ON' if self.flip_camera else 'OFF'}")
        self.chip_teacher.setText(f"Teacher: {'ON' if self.teacher_mode else 'OFF'}")
        if hasattr(self, "chip_user"):
            self.chip_user.setText(f"User: {self.username or 'Guest'}")
        self.btn_mirror.setChecked(self.flip_camera)
    def _update_settings_access(self):
        """Lock Settings unless Teacher mode is ON."""
        try:
            if hasattr(self, 'btn_settings'):
                enabled = bool(getattr(self, 'teacher_mode', False))
                self.btn_settings.setEnabled(enabled)
                if enabled:
                    self.btn_settings.setToolTip('Settings (Teacher mode)')
                else:
                    self.btn_settings.setToolTip('Settings (Teacher mode required)')
        except Exception:
            pass


    # =========================
    # Modes
    # =========================
    def _set_checked_button(self, btn: QPushButton):
        btn.setChecked(True)

    def _telemetry_enter_mode(self, mode: str) -> None:
        """Best-effort mode enter marker for evaluation logs."""
        self._mode_entered_at = time.time()
        try:
            self.telemetry.mode_enter(str(mode))
        except Exception:
            pass

    def _telemetry_start_target_timer(self) -> None:
        self._target_started_at = time.time()

    def on_menu_clicked(self):
        self.mode = "menu"
        self._telemetry_enter_mode("menu")
        self.correct_streak = 0
        self.stable_label = "—"
        self._set_checked_button(self.btn_menu)
        self.update_ui_text()

    def on_learn_clicked(self):
        self.mode = "learn"
        self._telemetry_enter_mode("learn")
        try:
            if hasattr(self, "cmb_learn_letter"):
                self.learn_idx = LEARN_LETTERS.index(self.cmb_learn_letter.currentText())
        except Exception:
            pass
        self._telemetry_start_target_timer()
        self.correct_streak = 0
        self.stable_label = "—"
        self._set_checked_button(self.btn_learn)
        self.update_ui_text()

    def on_quiz_clicked(self):
        self.mode = "quiz"
        self._telemetry_enter_mode("quiz")
        self.score = 0
        self.correct_streak = 0
        self.stable_label = "—"
        self.quiz_target = random.choice(LEARN_LETTERS)
        self._telemetry_start_target_timer()
        self._set_checked_button(self.btn_quiz)
        self.update_ui_text()

    def on_snake_clicked(self):
        self.mode = "snake"
        self._telemetry_enter_mode("snake")
        self._snake_session_started_at = time.time()
        self._snake_last_score = 0
        self._snake_last_gameover = False
        try:
            self.telemetry.snake_session_start()
        except Exception:
            pass
        self.snake_game.reset()
        self.correct_streak = 0
        self.stable_label = "—"
        self._set_checked_button(self.btn_snake)
        self.update_ui_text()

    def on_spelling_clicked(self):
        self.mode = "spelling"
        self._telemetry_enter_mode("spelling")
        self.correct_streak = 0
        self.stable_label = "—"
        self.spell_score = 0
        self.spell_index = 0
        self.next_spelling_word()
        self._set_checked_button(self.btn_spelling)
        self.update_ui_text()

    def on_mcq_clicked(self):
        self.mode = "mcq"
        self._telemetry_enter_mode("mcq")
        self.correct_streak = 0
        self.stable_label = "—"
        self.mcq_score = 0
        self.next_mcq_question()
        self._set_checked_button(self.btn_mcq)
        self.update_ui_text()

    def on_learn_letter_changed(self, letter: str):
        if not letter:
            return
        letter = letter.strip().upper()
        if letter in LEARN_LETTERS:
            self.learn_idx = LEARN_LETTERS.index(letter)
            if self.mode == "learn":
                self._telemetry_start_target_timer()
                self.correct_streak = 0
                self.recent_preds.clear()
                self.update_ui_text()

    def on_set_learn_letter(self):
        try:
            letter = self.cmb_learn_letter.currentText()
        except Exception:
            letter = LEARN_LETTERS[self.learn_idx]
        self.on_learn_letter_changed(letter)
        self.on_learn_clicked()

    def on_logout_clicked(self):
        if QMessageBox.question(self, "Logout", "Bạn muốn đổi tài khoản / logout?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self.settings.setValue("last_username", "")
        except Exception:
            pass
        try:
            from app.landing import LandingWindow
            w = LandingWindow(self.cfg)
            w.show()
            self._landing_ref = w
        except Exception:
            pass
        self.close()

    def on_camera_clicked(self, event):
        # Mouse click selection for MCQ (click on option image)
        if self.mode != "mcq" or not self.mcq_rects:
            return
        try:
            label_w = self.camera_label.width()
            label_h = self.camera_label.height()
            pix = self.camera_label.pixmap()
            if pix is None:
                return
            pix_w = pix.width()
            pix_h = pix.height()
            off_x = (label_w - pix_w) / 2.0
            off_y = (label_h - pix_h) / 2.0
            x = event.pos().x() - off_x
            y = event.pos().y() - off_y
            if x < 0 or y < 0 or x > pix_w or y > pix_h:
                return
            # map to frame coords
            fw = getattr(self, "_last_frame_w", None)
            fh = getattr(self, "_last_frame_h", None)
            if not fw or not fh:
                return
            fx = int(x * (fw / pix_w))
            fy = int(y * (fh / pix_h))
            for opt_key, (x1, y1, x2, y2) in self.mcq_rects:
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    self._choose_mcq_option(opt_key, time.time())
                    return
        except Exception:
            return

    def next_spelling_word(self):
        words = self.cfg.spelling_words or ["HELLO", "YES", "NO", "LOVE", "NAME"]
        self.spell_target = words[self.spell_index % len(words)]
        self.spell_index += 1
        self.spell_current = ""
        # timers for evaluation
        self._spelling_word_started_at = time.time()
        self._telemetry_start_target_timer()

    # =========================
    # Multiple Choice (MCQ)
    # =========================
    def next_mcq_question(self):
        # Prefer weak letter sometimes
        target = random.choice(LEARN_LETTERS)
        try:
            weak, acc, total = self.progress.suggestion_candidate(min_attempts=6, acc_threshold=0.70)
            if weak:
                # 60% chance focus weak letter
                if random.random() < 0.6:
                    target = weak
        except Exception:
            pass
        distractors = [l for l in LEARN_LETTERS if l != target]
        distractors = random.sample(distractors, 3)
        letters = [target] + distractors
        random.shuffle(letters)
        keys = ["A", "B", "C", "D"]
        self.mcq_options = {keys[i]: letters[i] for i in range(4)}
        self.mcq_target = target
        self.mcq_correct = next((k for k, v in self.mcq_options.items() if v == target), None)
        self.mcq_rects = []
        self._mcq_last_choice_at = 0.0
        self._mcq_question_started_at = time.time()
        # pre-load option images
        self._mcq_imgs = {}
        for k, lt in self.mcq_options.items():
            img = self._load_letter_image(lt)
            self._mcq_imgs[k] = img

        # telemetry: store question meta
        try:
            self.telemetry.mcq_question(target=str(self.mcq_target or ""), options=dict(self.mcq_options or {}))
        except Exception:
            pass

    def _load_letter_image(self, letter: str):
        """Load one random image for a given letter.

        Priority:
        1) cfg.multiplechoice_dir (supports either MCQs/A/*.jpg.. OR MCQs/A.jpg..)
        2) cfg.reference_dir (EXAMPLE_IMG/Sample_A.*)
        """
        lt = (letter or "").strip().upper()
        # 1) Multiplechoice directory (accept F.jpg/png or any file inside subfolder F)
        try:
            base = Path(getattr(self.cfg, "multiplechoice_dir", "Multiplechoice"))
            cand: list[Path] = []

            # (a) Single image file: MCQs/A.jpg|png|...
            for ext in ("png", "jpg", "jpeg", "webp"):
                p = base / f"{lt}.{ext}"
                if p.exists():
                    cand.append(p)

            # (b) Folder: MCQs/A/*.jpg|png|...
            sub = base / lt
            if sub.exists() and sub.is_dir():
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        cand.append(f)

            if cand:
                p = random.choice(cand)
                img = cv2.imread(str(p))
                if img is not None:
                    return img
        except Exception:
            pass
        # 2) Reference images
        try:
            ref = Path(getattr(self.cfg, "reference_dir", "EXAMPLE_IMG"))
            for ext in ("jpeg", "jpg", "png"):
                p = ref / f"Sample_{lt}.{ext}"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        return img
        except Exception:
            pass
        return None


    def _choose_mcq_option(self, opt_key: str, now: float):
        opt_key = (opt_key or "").strip().upper()
        if self.mode != "mcq" or opt_key not in ("A", "B", "C", "D"):
            return
        # cooldown
        if now - float(self._mcq_last_choice_at or 0.0) < 1.0:
            return
        self._mcq_last_choice_at = now
        is_correct = (opt_key == self.mcq_correct)

        # telemetry: store chosen letter + reaction time
        try:
            chosen_letter = str(self.mcq_options.get(opt_key, ""))
            rt = max(0.0, float(now) - float(getattr(self, "_mcq_question_started_at", 0.0) or 0.0))
            if self.mcq_target:
                self.telemetry.mcq_answer(
                    target=str(self.mcq_target),
                    chosen=chosen_letter,
                    correct=bool(is_correct),
                    reaction_time_sec=float(rt),
                )
        except Exception:
            pass
        # record: target letter correctness
        if self.mcq_target:
            self.progress.record_letter(self.mcq_target, bool(is_correct))
        if is_correct:
            self.mcq_score += 1
            self.hint_label.setText(f"✅ Correct! ({opt_key})")
        else:
            corr = self.mcq_correct or "?"
            self.hint_label.setText(f"❌ Sai. Đáp án đúng: {corr}")
        self.lbl_score.setText(f"Score: {self.mcq_score}")
        self._maybe_update_high_score("mcq", self.mcq_score)
        # maybe suggest practice for weak letter
        self._maybe_show_study_suggestion(time.time())  # init-time suggestion check
        # next question
        self.next_mcq_question()
        self.update_ui_text()


    # =========================
    # Settings / toggles
    # =========================
    def open_settings(self):
        if not bool(getattr(self, 'teacher_mode', False)):
            QMessageBox.information(
                self, 'Locked',
                'Bật Teacher Mode (phím T hoặc nút Teacher Mode) để mở Settings.'
            )
            return

        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec_() and getattr(dlg, "ok", False):
            self.settings.setValue("camera_index", int(self.cfg.camera_index))

            # Apply base UI font (DPI-friendly)
            try:
                QApplication.instance().setFont(QFont("Segoe UI", int(getattr(self.cfg, "base_font_pt", 11))))
            except Exception:
                pass

            # Apply target font immediately
            try:
                f = self.lbl_big_target.font()
                f.setPointSize(int(getattr(self.cfg, "target_font_pt", 56)))
                f.setBold(True)
                self.lbl_big_target.setFont(f)
            except Exception:
                pass

            # Persist to config.json so it stays after restart
            try:
                self.cfg.save("config.json")
            except Exception:
                pass

            QMessageBox.information(
                self, "Saved",
                "Settings saved. Nếu đổi model/camera, em nên restart app để ổn định nhất."
            )

    def toggle_flip(self):
        self.flip_camera = not self.flip_camera
        self.settings.setValue("flip_camera", "true" if self.flip_camera else "false")
        # High score label (by current mode)
        if hasattr(self, "lbl_high_score"):
            if self.mode in ("quiz", "snake", "spelling", "mcq"):
                hs = int(self.high_scores.get(self.mode, 0) or 0)
                self.lbl_high_score.setText(f"High Score: {hs}")
            else:
                self.lbl_high_score.setText("High Score: —")

        self._update_topbar_chips()

    def on_teacher_toggled(self, checked: bool):
        self.teacher_mode = bool(checked)
        self.btn_teacher.setText("Teacher Mode: ON" if self.teacher_mode else "Teacher Mode: OFF")
        # High score label (by current mode)
        if hasattr(self, "lbl_high_score"):
            if self.mode in ("quiz", "snake", "spelling", "mcq"):
                hs = int(self.high_scores.get(self.mode, 0) or 0)
                self.lbl_high_score.setText(f"High Score: {hs}")
            else:
                self.lbl_high_score.setText("High Score: —")

        self._update_topbar_chips()
        self._update_settings_access()

    def closeEvent(self, event):
        try:
            self.telemetry.events.log("app_exit", {"user": self.telemetry.user_key, "mode": str(getattr(self, "mode", ""))})
        except Exception:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.settings.setValue("flip_camera", "true" if self.flip_camera else "false")
        event.accept()

    # =========================
    # Reference images
    # =========================
    def update_reference_image(self, letter: Optional[str]):
        if not letter:
            self.ref_label.setText("No image")
            self.ref_label.setPixmap(QPixmap())
            return

        path = Path(self.cfg.reference_dir) / f"Sample_{letter}.jpeg"
        img = cv2.imread(str(path))
        if img is None:
            self.ref_label.setText(f"No image for {letter}")
            self.ref_label.setPixmap(QPixmap())
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.ref_label.width(), self.ref_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.ref_label.setPixmap(pixmap)
        self.ref_label.setText("")

    # =========================
    # Progress view
    # =========================
    def _maybe_show_study_suggestion(self, now: float):
        # Global toggle
        if not bool(getattr(self.cfg, "suggestion_enabled", True)):
            return
        # throttle checks
        if now - float(getattr(self, "_last_suggest_check", 0.0)) < 3.0:
            return
        self._last_suggest_check = now
        # per-user snooze
        try:
            user_key = ProgressManager._safe_user_key(self.username)
        except Exception:
            user_key = "guest"
        try:
            snoozed_until = float(self.settings.value(f"suggest_snoozed_until_{user_key}", 0.0) or 0.0)
        except Exception:
            snoozed_until = 0.0
        if now < snoozed_until:
            return
        # avoid rapid popups
        if now - float(getattr(self, "_last_popup_at", 0.0)) < 10.0:
            return
        try:
            letter, acc, total = self.progress.suggestion_candidate(min_attempts=6, acc_threshold=0.70)
        except Exception:
            return
        if not letter:
            return
        self._last_popup_at = now
        pct = int(round(float(acc) * 100)) if total else 0
        msg = f"Kiến thức chữ {letter} của bạn chưa vững cần luyện thêm (đúng {pct}% / {total} lần). Luyện ngay?"
        res = QMessageBox.question(self, "Gợi ý học tập", msg, QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            # switch to learn and jump to that letter
            try:
                if hasattr(self, "cmb_learn_letter"):
                    self.cmb_learn_letter.setCurrentText(letter)
            except Exception:
                pass
            try:
                self.learn_idx = LEARN_LETTERS.index(letter)
            except Exception:
                self.learn_idx = 0
            self.on_learn_clicked()
        else:
            snooze_min = int(getattr(self.cfg, "suggestion_snooze_minutes", 10) or 10)
            try:
                self.settings.setValue(f"suggest_snoozed_until_{user_key}", float(now + snooze_min * 60))
            except Exception:
                pass

    def update_progress_quick_view(self):
        weak = self.progress.get_weak_letters(top_k=3)
        if not weak:
            self.lbl_weak_letters.setText("Chưa có dữ liệu. Hãy luyện tập để xem chữ nào còn yếu nhé.")
            return
        parts = [f"{letter}: {acc*100:.0f}% (n={total})" for letter, acc, total in weak]
        self.lbl_weak_letters.setText("Cần luyện thêm: " + ", ".join(parts))

    # =========================
    # UI text
    # =========================
    def update_ui_text(self):
        self.lbl_mode.setText(f"Mode: {self.mode.upper()}")

        if self.mode == "learn":
            tgt = LEARN_LETTERS[self.learn_idx]
            self.lbl_target.setText(f"Target: {tgt}")
            self.lbl_big_target.setText(tgt)
            self.lbl_score.setText("Score: —")
            self.update_reference_image(tgt)

        elif self.mode == "quiz":
            tgt = self.quiz_target or "—"
            self.lbl_target.setText(f"Target: {tgt}")
            self.lbl_big_target.setText(tgt)
            self.lbl_score.setText(f"Score: {self.score}")
            self.update_reference_image(self.quiz_target)

        elif self.mode == "mcq":
            tgt = self.mcq_target or "—"
            self.lbl_target.setText(f"MCQ: Đâu là chữ {tgt}?")
            self.lbl_big_target.setText(tgt)
            self.lbl_score.setText(f"Score: {self.mcq_score}")
            self._maybe_update_high_score("mcq", self.mcq_score)
            self.update_reference_image(None)

        elif self.mode == "snake":
            self.lbl_target.setText("Snake Game")
            self.lbl_big_target.setText("🐍")
            self.lbl_score.setText(f"Score: {self.snake_game.score}")
            self._maybe_update_high_score("snake", self.snake_game.score)
            self.update_reference_image(None)

        elif self.mode == "spelling":
            tgt = self.spell_target or "—"
            self.lbl_target.setText(f"Word: {tgt}")
            self.lbl_big_target.setText(tgt)
            self.lbl_score.setText(f"Score: {self.spell_score}")
            self.update_reference_image(None)

        else:
            self.lbl_target.setText("Target: —")
            self.lbl_big_target.setText("—")
            self.lbl_score.setText("Score: —")
            self.update_reference_image(None)

        self.lbl_prediction.setText(f"Prediction: {self.stable_label}")

        conf_pct = int(max(0.0, min(1.0, float(self.last_conf))) * 100)
        self.pb_conf.setValue(conf_pct)

        self.pb_streak.setRange(0, int(self.cfg.streak_required))
        self.pb_streak.setValue(int(max(0, min(self.correct_streak, self.cfg.streak_required))))

        # High score label (by current mode)
        if hasattr(self, "lbl_high_score"):
            if self.mode in ("quiz", "snake", "spelling", "mcq"):
                hs = int(self.high_scores.get(self.mode, 0) or 0)
                self.lbl_high_score.setText(f"High Score: {hs}")
            else:
                self.lbl_high_score.setText("High Score: —")

        self._update_topbar_chips()

    # =========================
    # ROI
    # =========================
    def get_hand_roi(self, frame) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        hands, _ = self.detector.findHands(frame, draw=False)
        if not hands:
            self._last_hands = []
            return None, None
        self._last_hands = hands

        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = hands[0]["bbox"]

        cx, cy = x + w // 2, y + h // 2
        side = int(max(w, h) * self.cfg.offset_scale)

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_frame, cx + side // 2)
        y2 = min(h_frame, cy + side // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        roi = cv2.resize(crop, self.cfg.img_size)
        return roi, (x1, y1, x2, y2)

    # =========================
    # logging gate
    # =========================
    def _maybe_log_attempt(self, target: str, pred: str, correct: bool, now: float):
        if (pred != self._last_logged_pred) or (target != self._last_logged_target) or (now - self._last_logged_at > 0.8):
            self.progress.record_letter(target, correct)
            try:
                self.telemetry.gesture_attempt(
                    mode=str(self.mode),
                    target=str(target),
                    pred=str(pred),
                    correct=bool(correct),
                    conf=float(getattr(self, "last_conf", 0.0) or 0.0),
                    dt_sec=0.0,
                )
            except Exception:
                pass
            self._last_logged_pred = pred
            self._last_logged_target = target
            self._last_logged_at = now

    # =========================
    # Frame loop
    # =========================
    
    def _maybe_update_high_score(self, mode: str, score: int) -> None:
        """Update high score (local file) only when improved."""
        if mode not in ("quiz", "snake", "spelling", "mcq"):
            return
        try:
            score_i = int(score)
        except Exception:
            return
        prev = int(self.high_scores.get(mode, 0) or 0)
        if score_i > prev:
            self.high_scores[mode] = self.user_store.set_high_score(self.username, mode, score_i)
            if hasattr(self, "lbl_high_score"):
                self.lbl_high_score.setText(f"High Score: {self.high_scores[mode]}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        now = time.time()
        dt = now - self.prev_frame_time
        self.prev_frame_time = now
        self.current_fps = (1.0 / dt) if dt > 0 else 0.0

        # update footer fps
        self.lbl_fps.setText(f"FPS: {self.current_fps:.1f}")

        if self.flip_camera:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        h_frame, w_frame = display.shape[:2]
        # for mouse click mapping (MCQ)
        self._last_frame_w = w_frame
        self._last_frame_h = h_frame

        # top header inside video
        cv2.rectangle(display, (0, 0), (w_frame, 44), (0, 0, 0), -1)
        cv2.putText(display, "ASL Learning Suite",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 210, 255), 2)

        # -------- SNAKE --------
        if self.mode == "snake":
            hands, _ = self.detector.findHands(frame, draw=False)
            pointIndex = None
            if hands:
                lmList = hands[0]["lmList"]
                pointIndex = tuple(lmList[8][0:2])
            display = self.snake_game.update(display, pointIndex)
            if not hands:
                cv2.putText(display, "No Hand in frame (Keep Hand in frame please!)",
                            (12, h_frame - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.stable_label = "—"
            self.correct_streak = 0
            self.last_conf = 0.0
            self.lbl_score.setText(f"Score: {self.snake_game.score}")
            self._maybe_update_high_score("snake", self.snake_game.score)

            # telemetry: food eaten + game over
            try:
                cur = int(getattr(self.snake_game, "score", 0) or 0)
                if cur > int(getattr(self, "_snake_last_score", 0) or 0):
                    diff = cur - int(getattr(self, "_snake_last_score", 0) or 0)
                    for _ in range(max(0, diff)):
                        self.telemetry.snake_food_eaten(score=cur)
                    self._snake_last_score = cur

                is_over = bool(getattr(self.snake_game, "gameOver", False))
                if is_over and not bool(getattr(self, "_snake_last_gameover", False)):
                    dur = max(0.0, float(now) - float(getattr(self, "_snake_session_started_at", now) or now))
                    self.telemetry.snake_game_over(score=cur, duration_sec=float(dur))
                    self._snake_last_gameover = True
            except Exception:
                pass

        # -------- LEARN/QUIZ/SPELLING --------
        elif self.mode in ("learn", "quiz", "spelling", "mcq"):
            roi, bbox = self.get_hand_roi(frame)
            stable_pred: Optional[str] = None

            if roi is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (180, 80, 255), 2)

                if now - self.last_pred_time >= self.cfg.pred_interval:
                    self.last_pred_time = now
                    pred_label, conf = self.asl_model.predict_letter(roi)
                    self.last_conf = conf

                    if pred_label is not None and conf >= self.cfg.conf_thresh:
                        self.recent_preds.append(pred_label)
                        values, counts = np.unique(list(self.recent_preds), return_counts=True)
                        maj = values[int(np.argmax(counts))]
                        maj_count = int(np.max(counts))
                        if maj_count >= 3 or len(self.recent_preds) < 3:
                            stable_pred = str(maj)

            else:
                cv2.putText(display, "No Hand in frame (Keep Hand in frame please!)",
                            (12, h_frame - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # decide current target
            target: Optional[str] = None
            if self.mode == "learn":
                target = LEARN_LETTERS[self.learn_idx]
            elif self.mode == "quiz":
                target = self.quiz_target
            elif self.mode == "spelling":
                if self.spell_target and len(self.spell_current) < len(self.spell_target):
                    target = self.spell_target[len(self.spell_current)]

            # update streak & progress
            if stable_pred is not None and target is not None:
                self.stable_label = f"{stable_pred} ({self.last_conf:.2f})"

                is_correct = (stable_pred == target)
                self._maybe_log_attempt(target, stable_pred, is_correct, now)

                if is_correct:
                    self.correct_streak += 1
                else:
                    self.correct_streak = max(0, self.correct_streak - 1)

            # achieved streak
            if self.correct_streak >= self.cfg.streak_required and target is not None:
                completed = target

                # telemetry: completion time for this target
                try:
                    ttc = max(0.0, float(now) - float(getattr(self, "_target_started_at", now) or now))
                    self.telemetry.gesture_completion(mode=str(self.mode), target=str(completed), time_to_complete_sec=float(ttc))
                except Exception:
                    pass

                if self.mode == "learn":
                    self.learn_idx = (self.learn_idx + 1) % len(LEARN_LETTERS)
                    self.update_reference_image(LEARN_LETTERS[self.learn_idx])
                    self._telemetry_start_target_timer()
                elif self.mode == "quiz":
                    self.score += 1
                    self._maybe_update_high_score("quiz", self.score)
                    self.quiz_target = random.choice(LEARN_LETTERS)
                    self._telemetry_start_target_timer()
                elif self.mode == "spelling":
                    if self.spell_target:
                        self.spell_current += target
                        if self.spell_current == self.spell_target:
                            self.spell_score += 1
                            self._maybe_update_high_score("spelling", self.spell_score)
                            self.progress.record_word(self.spell_target, True)

                            # telemetry: full word completion time
                            try:
                                wt = max(0.0, float(now) - float(getattr(self, "_spelling_word_started_at", now) or now))
                                self.telemetry.spelling_word_complete(word=str(self.spell_target), time_sec=float(wt))
                            except Exception:
                                pass
                            self.next_spelling_word()
                        else:
                            # move to next letter in the same word
                            self._telemetry_start_target_timer()

                self.correct_streak = 0
                self.stable_label = "—"
                self.correct_effect_active = True
                self.correct_effect_end_time = now + 1.15
                self.last_correct_target = completed

            # MCQ selection logic
            if self.mode == "mcq" and self.mcq_target is not None:
                # We select answer by: point (index tip) + pinch (index tip touches middle tip)
                # Option rectangles are computed in frame coordinates.
                self.mcq_rects = self._mcq_calc_rects(w_frame, h_frame)
                self._mcq_hover_key = None
                self._mcq_is_pinched = False

                try:
                    hands = getattr(self, "_last_hands", []) or []
                    if hands:
                        lm = hands[0].get("lmList", [])
                        if len(lm) >= 13:
                            x8, y8 = int(lm[8][0]), int(lm[8][1])
                            x12, y12 = int(lm[12][0]), int(lm[12][1])
                            dist = ((x8 - x12) ** 2 + (y8 - y12) ** 2) ** 0.5
                            pinch_thr = max(18.0, min(40.0, min(w_frame, h_frame) * 0.07)) # Tăng từ 0.035 -> 0.07 để dễ chọn MCQs
                            pinched = (dist < pinch_thr)
                            self._mcq_is_pinched = bool(pinched)

                            # Hover = index tip inside an option box
                            for opt_key, (x1, y1, x2, y2) in self.mcq_rects:
                                if x1 <= x8 <= x2 and y1 <= y8 <= y2:
                                    self._mcq_hover_key = opt_key
                                    break

                            # Trigger only on pinch edge (not while holding pinch)
                            if pinched and (not self._mcq_prev_pinched) and self._mcq_hover_key:
                                self._choose_mcq_option(self._mcq_hover_key, now)

                            self._mcq_prev_pinched = bool(pinched)
                        else:
                            self._mcq_prev_pinched = False
                    else:
                        self._mcq_prev_pinched = False
                except Exception:
                    self._mcq_prev_pinched = False

                # Do not show streak bar in MCQ (selection uses pinch / click)
                self.correct_streak = 0

            # refresh UI (right panel)
            self.lbl_prediction.setText(f"Prediction: {self.stable_label}")
            self.pb_conf.setValue(int(max(0.0, min(1.0, float(self.last_conf))) * 100))
            self.pb_streak.setRange(0, int(self.cfg.streak_required))
            self.pb_streak.setValue(int(max(0, min(self.correct_streak, self.cfg.streak_required))))

            if self.mode == "learn":
                tgt = LEARN_LETTERS[self.learn_idx]
                self.lbl_target.setText(f"Target: {tgt}")
                self.lbl_big_target.setText(tgt)
                self.lbl_score.setText("Score: —")
            elif self.mode == "quiz":
                tgt = self.quiz_target or "—"
                self.lbl_target.setText(f"Target: {tgt}")
                self.lbl_big_target.setText(tgt)
                self.lbl_score.setText(f"Score: {self.score}")
            elif self.mode == "spelling":
                tgt = self.spell_target or "—"
                self.lbl_target.setText(f"Word: {tgt}")
                self.lbl_big_target.setText(tgt)
                self.lbl_score.setText(f"Score: {self.spell_score}")

        # -------- MENU --------
        else:
            cv2.putText(display, "MENU: Select the mode in the right-hand panel",
                        (12, h_frame - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 140), 2)
            self.last_conf = 0.0
            self.correct_streak = 0
            self.stable_label = "—"
            self.lbl_prediction.setText("Prediction: —")
            self.pb_conf.setValue(0)
            self.pb_streak.setValue(0)

        # -------- MCQ overlay (option images + gesture hint) --------
        if self.mode == "mcq" and self.mcq_target is not None:
            display = self._draw_mcq_overlay(display)

        # -------- correct overlay --------
        if self.correct_effect_active and now <= self.correct_effect_end_time:
            overlay = display.copy()
            box_w = int(w_frame * 0.72)
            box_h = int(h_frame * 0.28)
            x1 = (w_frame - box_w) // 2
            y1 = (h_frame - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 220, 120), -1)
            display = cv2.addWeighted(overlay, 0.32, display, 0.68, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, "CORRECT!", (x1 + 50, y1 + int(box_h * 0.55)),
                        font, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
            if self.last_correct_target:
                cv2.putText(display, f"Letter: {self.last_correct_target}", (x1 + 50, y1 + int(box_h * 0.80)),
                            font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.correct_effect_active and now > self.correct_effect_end_time:
            self.correct_effect_active = False
            self.last_correct_target = None

        # -------- teacher overlay --------
        if self.teacher_mode:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, f"FPS: {self.current_fps:.1f}", (12, 78), font, 0.7, (255, 255, 255), 2)
            bar_w, bar_h = 220, 18
            bx, by = 12, h_frame - 56
            cv2.rectangle(display, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 1)
            conf_norm = max(0.0, min(1.0, float(self.last_conf)))
            cv2.rectangle(display, (bx, by), (bx + int(bar_w * conf_norm), by + bar_h), (30, 220, 120), -1)
            cv2.putText(display, f"Conf: {self.last_conf:.2f}", (bx, by - 6), font, 0.55, (255, 255, 255), 1)

        # render to Qt
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

        # update weak letters throttled (1.0s)
        if now - self._last_weak_ui_update >= 1.0:
            self._last_weak_ui_update = now
            self.update_progress_quick_view()
        self._maybe_show_study_suggestion(time.time())  # init-time suggestion check

    # =========================
    # Keyboard shortcuts
    # =========================
    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_1:
            self.on_learn_clicked()
        elif k == Qt.Key_2:
            self.on_quiz_clicked()
        elif k == Qt.Key_3:
            self.on_snake_clicked()
        elif k == Qt.Key_4:
            self.on_spelling_clicked()
        elif k == Qt.Key_0:
            self.on_menu_clicked()
        elif k == Qt.Key_R:
            if self.mode == "snake":
                self.snake_game.reset()
        elif k == Qt.Key_F:
            self.toggle_flip()
        elif k == Qt.Key_T:
            self.teacher_mode = not self.teacher_mode
            self.btn_teacher.setChecked(self.teacher_mode)
            self.btn_teacher.setText("Teacher Mode: ON" if self.teacher_mode else "Teacher Mode: OFF")
            # High score label (by current mode)
        if hasattr(self, "lbl_high_score"):
            if self.mode in ("quiz", "snake", "spelling", "mcq"):
                hs = int(self.high_scores.get(self.mode, 0) or 0)
                self.lbl_high_score.setText(f"High Score: {hs}")
            else:
                self.lbl_high_score.setText("High Score: —")

        self._update_topbar_chips()
        self._update_settings_access()
