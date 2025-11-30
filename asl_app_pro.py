import sys
import time
import random
from collections import deque
import math
import json
import os

import cv2
import cvzone
import numpy as np

from cvzone.HandTrackingModule import HandDetector

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model

from PyQt5.QtCore import Qt, QTimer, QSettings
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
)

# ========== 0. CẤU HÌNH CƠ BẢN ==========

IMG_SIZE = (224, 224)

LABELS = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "del","nothing","space"
]
NUM_CLASSES = len(LABELS)
LEARN_LETTERS = LABELS[:26]  # A-Z

MODEL_WEIGHTS_PATH = r"D:\Year 6\Graduation_Project\Hand_Tracking\Traning_and_model\results\asl_model_MobileNetV2_finetuned.h5"

PRED_INTERVAL = 0.30
CONF_THRESH = 0.7
STREAK_REQUIRED = 5
OFFSET_SCALE = 1.2

REFERENCE_DIR = "EXAMPLE_IMG"  # chứa Sample_A.jpeg ... Sample_Z.jpeg

# DANH SÁCH TỪ DÙNG CHO SPELLING MODE
SPELLING_WORDS = ["HELLO", "YES", "NO", "LOVE", "NAME"]


# ========== PROGRESS MANAGER ==========

class ProgressManager:
    """
    Lưu tiến độ học vào file JSON:
    {
      "letters": {
         "A": {"total": 10, "correct": 8},
         ...
      },
      "words": {
         "HELLO": {"total": 3, "correct": 2},
         ...
      }
    }
    """
    def __init__(self, path="progress.json"):
        self.path = path
        self.data = {"letters": {}, "words": {}}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"letters": {}, "words": {}}
        else:
            self.data = {"letters": {}, "words": {}}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            # Không để app crash chỉ vì lỗi save log
            pass

    def record_letter(self, letter, correct: bool):
        if letter is None:
            return
        letters = self.data.setdefault("letters", {})
        stat = letters.setdefault(letter, {"total": 0, "correct": 0})
        stat["total"] += 1
        if correct:
            stat["correct"] += 1
        self.save()

    def record_word(self, word, correct: bool):
        if word is None:
            return
        words = self.data.setdefault("words", {})
        stat = words.setdefault(word, {"total": 0, "correct": 0})
        stat["total"] += 1
        if correct:
            stat["correct"] += 1
        self.save()

    def get_weak_letters(self, top_k=3):
        """
        Trả về list [(letter, acc, total), ...] sort theo acc tăng dần.
        """
        letters = self.data.get("letters", {})
        if not letters:
            return []
        stats = []
        for lt, v in letters.items():
            total = max(1, v.get("total", 0))
            correct = v.get("correct", 0)
            acc = correct / total
            stats.append((lt, acc, v.get("total", 0)))
        stats.sort(key=lambda x: x[1])  # sort by acc tăng dần
        return stats[:top_k]


# ========== 1. MODEL ==========

def build_mobilenet_model(num_classes=NUM_CLASSES):
    base = MobileNetV2(weights=None, include_top=False,
                       input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class ASLModel:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        print("✅ Loaded ASL MobileNetV2 model from:", model_path)

        out_units = self.model.output_shape[-1]
        if out_units != len(LABELS):
            raise ValueError(
                f"Model output has {out_units} units, "
                f"but LABELS has {len(LABELS)} classes!"
            )

    def predict_letter(self, img_roi_bgr):
        x = cv2.cvtColor(img_roi_bgr, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x, verbose=0)
        index = int(np.argmax(preds))
        conf = float(np.max(preds))

        if 0 <= index < len(LABELS):
            return LABELS[index], conf
        return None, conf


# ========== 2. SNAKE GAME CLASS ==========

class SnakeGameClass:
    def __init__(self, pathFood, game_width, game_height):
        self.game_w = game_width
        self.game_h = game_height
        self.margin = 80

        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = (0, 0)
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        x = random.randint(self.margin, self.game_w - self.margin)
        y = random.randint(self.margin, self.game_h - self.margin)
        self.foodPoint = (x, y)

    def reset(self):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)
        self.score = 0
        self.gameOver = False
        self.randomFoodLocation()

    def update(self, imgMain, currentHead):
        if self.gameOver:
            overlay = imgMain.copy()
            cv2.rectangle(overlay, (150, 200), (self.game_w - 150, self.game_h - 150),
                          (0, 0, 0), -1)
            alpha = 0.6
            imgMain = cv2.addWeighted(overlay, alpha, imgMain, 1 - alpha, 0)

            cvzone.putTextRect(imgMain, "GAME OVER", [260, 260],
                               scale=4, thickness=4, offset=20, colorT=(255, 255, 255),
                               colorR=(0, 0, 255))
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [260, 340],
                               scale=3, thickness=3, offset=15)
            cvzone.putTextRect(imgMain, "Press R to Restart", [260, 420],
                               scale=2, thickness=2, offset=10)
            return imgMain

        px, py = self.previousHead
        cx, cy = currentHead

        if px == 0 and py == 0:
            self.previousHead = (cx, cy)
            px, py = cx, cy

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = (cx, cy)

        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break

        rx, ry = self.foodPoint
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
           ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
            print("Score:", self.score)

        if len(self.points) > 1:
            num_points = len(self.points)
            for i in range(1, num_points):
                x1, y1 = self.points[i - 1]
                x2, y2 = self.points[i]

                t = i / num_points
                r = int(255 * (1 - t))
                g = int(200 * t)
                b = 255

                thickness = 18
                cv2.line(imgMain, (x1, y1), (x2, y2), (b, g, r), thickness)

        if self.points:
            hx, hy = self.points[-1]
            cv2.circle(imgMain, (hx, hy), 22, (0, 0, 0), cv2.FILLED)
            cv2.circle(imgMain, (hx, hy), 18, (0, 255, 0), cv2.FILLED)

        imgMain = cvzone.overlayPNG(
            imgMain, self.imgFood,
            (rx - self.wFood // 2, ry - self.hFood // 2)
        )

        cvzone.putTextRect(imgMain, f'Score: {self.score}', [40, 60],
                           scale=2, thickness=2, offset=8,
                           colorR=(50, 50, 50), colorT=(255, 255, 255))

        if len(self.points) > 4:
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1 <= minDist <= 1:
                print("Hit")
                self.gameOver = True
                self.points = []
                self.lengths = []
                self.currentLength = 0
                self.allowedLength = 150
                self.previousHead = (0, 0)
                self.randomFoodLocation()

        return imgMain


# ========== 3. APP GUI CHÍNH ==========

class ASLLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # ----- Settings & Progress -----
        self.settings = QSettings("ASLExample", "ASLLearningApp")
        self.progress = ProgressManager()

        # --- State ---
        self.mode = "menu"              # "menu" | "learn" | "quiz" | "snake" | "spelling"
        self.learn_idx = 0
        self.quiz_target = None
        self.score = 0

        # Spelling mode
        self.spell_target = None
        self.spell_current = ""
        self.spell_index = 0
        self.spell_score = 0

        self.last_pred_time = 0.0
        self.correct_streak = 0
        self.stable_label = "..."
        self.last_conf = 0.0

        self.recent_preds = deque(maxlen=5)

        # Flip camera (remember bằng QSettings)
        self.flip_camera = self.settings.value("flip_camera", "true") == "true"

        # Teacher mode
        self.teacher_mode = False
        self.prev_frame_time = time.time()
        self.current_fps = 0.0

        # Hiệu ứng CORRECT
        self.correct_effect_active = False
        self.correct_effect_end_time = 0
        self.last_correct_target = None

        # --- Camera + Detector + Model + Snake ---
        cam_index = int(self.settings.value("camera_index", 0))
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.detector = HandDetector(maxHands=1)
        self.asl_model = ASLModel(MODEL_WEIGHTS_PATH)

        ret, tmp_frame = self.cap.read()
        if not ret:
            self.game_h, self.game_w = 720, 1280
        else:
            self.game_h, self.game_w = tmp_frame.shape[:2]

        self.snake_game = SnakeGameClass("Donut.png", self.game_w, self.game_h)

        # --- UI ---
        self.init_ui()

        # --- Timer cập nhật frame ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ---------- UI ----------
    def init_ui(self):
        self.setWindowTitle("ASL Learning App + Snake Game")
        self.setMinimumSize(1200, 720)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #151515;
            }
            QLabel {
                color: #f5f5f5;
            }
            QGroupBox {
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 10px;
                color: #dddddd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2d8cff;
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1f6ad6;
            }
            QPushButton:pressed {
                background-color: #174c99;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # --------- LEFT: Camera ---------
        camera_box = QGroupBox("Camera / Game")
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(6, 6, 6, 6)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            "background-color: #000000; border-radius: 12px; border: 1px solid #444;"
        )
        self.camera_label.setMinimumSize(720, 480)

        camera_layout.addWidget(self.camera_label)
        camera_box.setLayout(camera_layout)

        # --------- RIGHT: Control Panel ---------
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Mode buttons
        mode_box = QGroupBox("Chế độ")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(8)

        self.btn_menu = QPushButton("Menu")
        self.btn_learn = QPushButton("Học A–Z")
        self.btn_quiz = QPushButton("Quiz")
        self.btn_snake = QPushButton("Snake Game")
        self.btn_spelling = QPushButton("Spelling")

        mode_layout.addWidget(self.btn_menu)
        mode_layout.addWidget(self.btn_learn)
        mode_layout.addWidget(self.btn_quiz)
        mode_layout.addWidget(self.btn_snake)
        mode_layout.addWidget(self.btn_spelling)
        mode_box.setLayout(mode_layout)

        self.btn_menu.clicked.connect(self.on_menu_clicked)
        self.btn_learn.clicked.connect(self.on_learn_clicked)
        self.btn_quiz.clicked.connect(self.on_quiz_clicked)
        self.btn_snake.clicked.connect(self.on_snake_clicked)
        self.btn_spelling.clicked.connect(self.on_spelling_clicked)

        # BIG TARGET LETTER / WORD
        big_target_box = QGroupBox("Chữ/Từ hiện tại")
        big_target_layout = QVBoxLayout()
        self.lbl_big_target = QLabel("-")
        self.lbl_big_target.setAlignment(Qt.AlignCenter)
        self.lbl_big_target.setFont(QFont("Arial", 48, QFont.Bold))
        self.lbl_big_target.setStyleSheet("color: #2d8cff;")
        big_target_layout.addWidget(self.lbl_big_target)
        big_target_box.setLayout(big_target_layout)

        # Info Box
        info_box = QGroupBox("Thông tin")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self.lbl_mode = QLabel("Mode: MENU")
        self.lbl_target = QLabel("Target: -")
        self.lbl_prediction = QLabel("Prediction: -")
        self.lbl_streak = QLabel(f"Progress: 0 / {STREAK_REQUIRED}")
        self.lbl_score = QLabel("Score: 0")

        for lbl in [self.lbl_mode, self.lbl_target, self.lbl_prediction, self.lbl_streak, self.lbl_score]:
            lbl.setFont(QFont("Arial", 11))
            info_layout.addWidget(lbl)

        info_box.setLayout(info_layout)

        # Progress quick view
        prog_box = QGroupBox("Tiến độ học (quick view)")
        prog_layout = QVBoxLayout()
        self.lbl_weak_letters = QLabel("Chưa có dữ liệu.")
        self.lbl_weak_letters.setWordWrap(True)
        self.lbl_weak_letters.setFont(QFont("Arial", 10))
        prog_layout.addWidget(self.lbl_weak_letters)
        prog_box.setLayout(prog_layout)

        # Reference image box (ảnh mẫu A-Z)
        ref_box = QGroupBox("Hình minh họa ký hiệu")
        ref_layout = QVBoxLayout()
        self.ref_label = QLabel("Chưa có hình")
        self.ref_label.setAlignment(Qt.AlignCenter)
        self.ref_label.setStyleSheet(
            "background-color: #222; border-radius: 10px; border: 1px solid #444;"
        )
        self.ref_label.setMinimumSize(220, 220)
        ref_layout.addWidget(self.ref_label)
        ref_box.setLayout(ref_layout)

        # Teacher mode button
        self.btn_teacher = QPushButton("Teacher Mode: OFF")
        self.btn_teacher.setCheckable(True)
        self.btn_teacher.clicked.connect(self.on_teacher_toggled)

        # Hint / instructions
        self.hint_label = QLabel(
            "Phím tắt: 1=Học | 2=Quiz | 3=Snake | 4=Spelling | 0=Menu | R=Reset Snake | F=Bật/tắt lật cam | T=Teacher | ESC=Thoát"
        )
        self.hint_label.setFont(QFont("Arial", 10))
        self.hint_label.setStyleSheet("color: #aaaaaa;")

        right_panel.addWidget(mode_box)
        right_panel.addWidget(big_target_box)
        right_panel.addWidget(info_box)
        right_panel.addWidget(prog_box)
        right_panel.addWidget(ref_box)
        right_panel.addWidget(self.btn_teacher)
        right_panel.addStretch()
        right_panel.addWidget(self.hint_label)

        main_layout.addWidget(camera_box, stretch=3)
        main_layout.addLayout(right_panel, stretch=2)

        self.update_ui_text()
        self.update_progress_quick_view()

    # ---------- Mode switching ----------
    def on_menu_clicked(self):
        self.mode = "menu"
        self.correct_streak = 0
        self.stable_label = "..."
        self.update_ui_text()

    def on_learn_clicked(self):
        self.mode = "learn"
        self.learn_idx = 0
        self.correct_streak = 0
        self.stable_label = "..."
        self.update_ui_text()

    def on_quiz_clicked(self):
        self.mode = "quiz"
        self.score = 0
        self.correct_streak = 0
        self.stable_label = "..."
        self.quiz_target = random.choice(LEARN_LETTERS)
        self.update_ui_text()

    def on_snake_clicked(self):
        self.mode = "snake"
        self.snake_game.reset()
        self.correct_streak = 0
        self.stable_label = "..."
        self.update_ui_text()

    def on_spelling_clicked(self):
        self.mode = "spelling"
        self.correct_streak = 0
        self.stable_label = "..."
        self.spell_score = 0
        self.spell_index = 0
        self.next_spelling_word()
        self.update_ui_text()

    def next_spelling_word(self):
        if not SPELLING_WORDS:
            self.spell_target = None
            self.spell_current = ""
            return
        self.spell_target = SPELLING_WORDS[self.spell_index % len(SPELLING_WORDS)]
        self.spell_index += 1
        self.spell_current = ""

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        # Lưu lại trạng thái flip cam
        self.settings.setValue("flip_camera", "true" if self.flip_camera else "false")
        event.accept()

    # ---------- UI text update ----------
    def update_ui_text(self):
        self.lbl_mode.setText(f"Mode: {self.mode.upper()}")

        if self.mode == "learn":
            target = LEARN_LETTERS[self.learn_idx]
            self.lbl_target.setText(f"Target: {target}")
            self.lbl_score.setText("Score: -")
            self.lbl_big_target.setText(target)
            self.update_reference_image(target)
        elif self.mode == "quiz":
            self.lbl_target.setText(f"Target: {self.quiz_target}")
            self.lbl_score.setText(f"Score: {self.score}")
            self.lbl_big_target.setText(self.quiz_target if self.quiz_target else "-")
            self.update_reference_image(self.quiz_target)
        elif self.mode == "snake":
            self.lbl_target.setText("Snake Game")
            self.lbl_score.setText(f"Score: {self.snake_game.score}")
            self.lbl_big_target.setText("🐍")
            self.update_reference_image(None)
        elif self.mode == "spelling":
            self.lbl_target.setText(f"Word: {self.spell_target}")
            self.lbl_score.setText(f"Score: {self.spell_score}")
            self.lbl_big_target.setText(self.spell_target if self.spell_target else "-")
            self.update_reference_image(None)
        else:
            self.lbl_target.setText("Target: -")
            self.lbl_score.setText("Score: -")
            self.lbl_big_target.setText("-")
            self.update_reference_image(None)

        self.lbl_prediction.setText(f"Prediction: {self.stable_label}")
        self.lbl_streak.setText(f"Progress: {self.correct_streak} / {STREAK_REQUIRED}")
        self.update_progress_quick_view()

    def update_progress_quick_view(self):
        weak = self.progress.get_weak_letters(top_k=3)
        if not weak:
            text = "Chưa có dữ liệu. Hãy luyện tập để xem chữ nào còn yếu nhé."
        else:
            parts = []
            for letter, acc, total in weak:
                parts.append(f"{letter}: {acc*100:.0f}% (n={total})")
            text = "Cần luyện thêm: " + ", ".join(parts)
        self.lbl_weak_letters.setText(text)

    def update_reference_image(self, letter):
        if letter is None:
            self.ref_label.setText("No image")
            self.ref_label.setPixmap(QPixmap())
            return

        path = f"{REFERENCE_DIR}/Sample_{letter}.jpeg"
        img = cv2.imread(path)
        if img is None:
            self.ref_label.setText(f"No image for {letter}")
            self.ref_label.setPixmap(QPixmap())
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.ref_label.width(), self.ref_label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ref_label.setPixmap(pixmap)
        self.ref_label.setText("")

    # ---------- Core: lấy ROI bàn tay ----------
    def get_hand_roi(self, frame):
        hands, _ = self.detector.findHands(frame, draw=False)
        if not hands:
            return None, None

        h_frame, w_frame = frame.shape[:2]
        hand = hands[0]
        x, y, w, h = hand['bbox']

        cx, cy = x + w // 2, y + h // 2
        side = int(max(w, h) * OFFSET_SCALE)

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_frame, cx + side // 2)
        y2 = min(h_frame, cy + side // 2)

        img_crop = frame[y1:y2, x1:x2]
        if img_crop.size == 0:
            return None, None

        img_roi = cv2.resize(img_crop, IMG_SIZE)
        return img_roi, (x1, y1, x2, y2)

    # ---------- Teacher mode toggle ----------
    def on_teacher_toggled(self, checked):
        self.teacher_mode = checked
        self.btn_teacher.setText("Teacher Mode: ON" if checked else "Teacher Mode: OFF")

    # ---------- Frame loop ----------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # FPS
        now = time.time()
        dt = now - getattr(self, "prev_frame_time", now)
        self.prev_frame_time = now
        if dt > 0:
            self.current_fps = 1.0 / dt
        else:
            self.current_fps = 0.0

        if self.flip_camera:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # Header
        cv2.rectangle(display, (0, 0), (w_frame, 35), (0, 0, 0), -1)
        cv2.putText(display, "ASL Learning App + Snake Game",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ========== MODE SNAKE ==========
        if self.mode == "snake":
            hands, _ = self.detector.findHands(frame, draw=False)
            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]
                display = self.snake_game.update(display, pointIndex)
            else:
                cv2.putText(display, "No hand detected...",
                            (10, h_frame - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.lbl_prediction.setText("Prediction: -")
            self.lbl_streak.setText("-")
            self.lbl_target.setText("Snake Game")
            self.lbl_big_target.setText("🐍")
            self.lbl_score.setText(f"Score: {self.snake_game.score}")

        # ========== MODE LEARN / QUIZ / SPELLING ==========
        elif self.mode in ["learn", "quiz", "spelling"]:
            img_roi, bbox = self.get_hand_roi(frame)
            stable_pred = None

            if img_roi is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)

                if now - self.last_pred_time >= PRED_INTERVAL:
                    self.last_pred_time = now
                    pred_label, conf = self.asl_model.predict_letter(img_roi)

                    if pred_label is not None and conf >= CONF_THRESH:
                        self.recent_preds.append(pred_label)
                        self.last_conf = conf
                        values, counts = np.unique(list(self.recent_preds), return_counts=True)
                        maj_label = values[np.argmax(counts)]
                        maj_count = counts[np.argmax(counts)]
                        if maj_count >= 3 or len(self.recent_preds) < 3:
                            stable_pred = maj_label
                    # Nếu conf thấp: không cập nhật stable_pred
            else:
                cv2.putText(display, "No hand detected...",
                            (10, h_frame - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Xác định target hiện tại
            target = None
            if self.mode == "learn":
                target = LEARN_LETTERS[self.learn_idx]
            elif self.mode == "quiz":
                target = self.quiz_target
            elif self.mode == "spelling":
                if self.spell_target and len(self.spell_current) < len(self.spell_target):
                    target = self.spell_target[len(self.spell_current)]
                else:
                    target = None

            # Cập nhật dựa trên stable_pred
            if stable_pred is not None and target is not None:
                self.stable_label = f"{stable_pred} ({self.last_conf:.2f})"

                # Ghi tiến độ (có thể đúng hoặc sai)
                self.progress.record_letter(target, stable_pred == target)

                if stable_pred == target:
                    self.correct_streak += 1

            # Khi đủ streak
            if self.correct_streak >= STREAK_REQUIRED and target is not None:
                if self.mode == "learn":
                    completed_letter = LEARN_LETTERS[self.learn_idx]
                    self.learn_idx = (self.learn_idx + 1) % len(LEARN_LETTERS)
                    self.correct_streak = 0
                    self.stable_label = "..."
                    self.update_reference_image(LEARN_LETTERS[self.learn_idx])
                elif self.mode == "quiz":
                    completed_letter = self.quiz_target
                    self.score += 1
                    self.quiz_target = random.choice(LEARN_LETTERS)
                    self.correct_streak = 0
                    self.stable_label = "..."
                elif self.mode == "spelling":
                    completed_letter = target
                    if self.spell_target and target:
                        self.spell_current += target
                        if self.spell_current == self.spell_target:
                            self.spell_score += 1
                            self.progress.record_word(self.spell_target, True)
                            self.next_spelling_word()
                    self.correct_streak = 0
                    self.stable_label = "..."

                # Hiệu ứng correct
                self.correct_effect_active = True
                self.correct_effect_end_time = now + 1.2
                self.last_correct_target = completed_letter
                self.update_progress_quick_view()

            # Cập nhật info panel
            self.lbl_prediction.setText(f"Prediction: {self.stable_label}")
            if self.mode == "learn":
                self.lbl_streak.setText(f"Progress: {self.correct_streak} / {STREAK_REQUIRED}")
                self.lbl_target.setText(f"Target: {LEARN_LETTERS[self.learn_idx]}")
                self.lbl_big_target.setText(LEARN_LETTERS[self.learn_idx])
                self.lbl_score.setText("Score: -")
            elif self.mode == "quiz":
                self.lbl_streak.setText(f"Progress: {self.correct_streak} / {STREAK_REQUIRED}")
                self.lbl_target.setText(f"Target: {self.quiz_target}")
                self.lbl_big_target.setText(self.quiz_target if self.quiz_target else "-")
                self.lbl_score.setText(f"Score: {self.score}")
            elif self.mode == "spelling":
                self.lbl_target.setText(f"Word: {self.spell_target}")
                self.lbl_big_target.setText(self.spell_target if self.spell_target else "-")
                self.lbl_score.setText(f"Score: {self.spell_score}")
                self.lbl_streak.setText(
                    f"Typed: {self.spell_current} | Streak: {self.correct_streak}/{STREAK_REQUIRED}"
                )

        # MODE MENU: chỉ hiển thị camera bình thường
        else:
            cv2.putText(display, "MENU - Select mode on the right",
                        (10, h_frame - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ========== HIỆU ỨNG CORRECT OVERLAY ==========
        if self.correct_effect_active:
            if time.time() <= self.correct_effect_end_time:
                overlay = display.copy()

                box_w = int(w_frame * 0.7)
                box_h = int(h_frame * 0.3)
                x1 = (w_frame - box_w) // 2
                y1 = (h_frame - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)

                text_main = "CORRECT!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 2
                thickness = 4
                (tw, th), _ = cv2.getTextSize(text_main, font, scale, thickness)
                tx = x1 + (box_w - tw) // 2
                ty = y1 + box_h // 2
                cv2.putText(display, text_main, (tx, ty),
                            font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

                if self.last_correct_target is not None:
                    small = f"Letter: {self.last_correct_target}"
                    (tw2, th2), _ = cv2.getTextSize(small, font, 0.8, 2)
                    sx = x1 + (box_w - tw2) // 2
                    sy = ty + th + 20
                    cv2.putText(display, small, (sx, sy),
                                font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                self.correct_effect_active = False
                self.last_correct_target = None

        # ========== TEACHER MODE OVERLAY ==========
        if self.teacher_mode:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, f"FPS: {self.current_fps:.1f}",
                        (10, 60), font, 0.7, (255, 255, 255), 2)

            bar_w = 200
            bar_h = 20
            x1 = 10
            y1 = h_frame - 60
            cv2.rectangle(display, (x1, y1), (x1 + bar_w, y1 + bar_h),
                          (255, 255, 255), 1)
            conf_norm = max(0.0, min(1.0, self.last_conf))
            cv2.rectangle(display, (x1, y1),
                          (x1 + int(bar_w * conf_norm), y1 + bar_h),
                          (0, 255, 0), -1)
            cv2.putText(display, f"Conf: {self.last_conf:.2f}",
                        (x1, y1 - 5), font, 0.5, (255, 255, 255), 1)

        # Render ra QLabel camera
        rgb_image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(pixmap)

    # ---------- Keyboard shortcuts ----------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_1:
            self.on_learn_clicked()
        elif event.key() == Qt.Key_2:
            self.on_quiz_clicked()
        elif event.key() == Qt.Key_3:
            self.on_snake_clicked()
        elif event.key() == Qt.Key_4:
            self.on_spelling_clicked()
        elif event.key() == Qt.Key_0:
            self.on_menu_clicked()
        elif event.key() == Qt.Key_R:
            if self.mode == "snake":
                self.snake_game.reset()
                self.lbl_score.setText(f"Score: {self.snake_game.score}")
        elif event.key() == Qt.Key_F:
            self.flip_camera = not self.flip_camera
            self.settings.setValue("flip_camera", "true" if self.flip_camera else "false")
            print("Flip camera:", self.flip_camera)
        elif event.key() == Qt.Key_T:
            self.teacher_mode = not self.teacher_mode
            self.btn_teacher.setChecked(self.teacher_mode)
            self.btn_teacher.setText("Teacher Mode: ON" if self.teacher_mode else "Teacher Mode: OFF")


# ========== 4. LANDING WINDOW – MÀN HÌNH NGOÀI ==========

class LandingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("ASL Learning Suite")
        self.setMinimumSize(800, 500)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QLabel {
                color: #e5e7eb;
            }
            QPushButton {
                background-color: #22c55e;
                color: #0f172a;
                border-radius: 10px;
                padding: 10px 18px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
            QPushButton:pressed {
                background-color: #15803d;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("ASL Learning App")
        title.setFont(QFont("Arial", 32, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Interactive ASL Learning • Real-time Hand Tracking • Gamified Practice")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #9ca3af;")

        features_box = QGroupBox("Features")
        f_layout = QVBoxLayout()
        f_layout.setSpacing(6)

        l1 = QLabel("• Learn A–Z with real-time ASL recognition")
        l2 = QLabel("• Quiz & Spelling modes to test your skills and track progress")
        l3 = QLabel("• Snake game controlled entirely by your hand")
        for lb in [l1, l2, l3]:
            lb.setFont(QFont("Arial", 11))
            f_layout.addWidget(lb)

        features_box.setLayout(f_layout)
        features_box.setStyleSheet("""
            QGroupBox {
                border: 1px solid #1f2937;
                border-radius: 12px;
                margin-top: 10px;
                color: #e5e7eb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)
        btn_row.setAlignment(Qt.AlignCenter)

        self.btn_start = QPushButton("Start Learning")
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: #f9fafb;
                border-radius: 10px;
                padding: 10px 18px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        """)

        self.btn_start.clicked.connect(self.open_asl_app)
        self.btn_exit.clicked.connect(self.close)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_exit)

        footer = QLabel("Tip: Use keys 1–4 to switch modes, F to mirror camera, T để bật Teacher Mode.")
        footer.setFont(QFont("Arial", 10))
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #9ca3af;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(features_box)
        layout.addLayout(btn_row)
        layout.addWidget(footer)

    def open_asl_app(self):
        self.asl_window = ASLLearningApp()
        self.asl_window.show()
        self.close()


# ========== MAIN ==========

def main():
    app = QApplication(sys.argv)
    landing = LandingWindow()
    landing.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
