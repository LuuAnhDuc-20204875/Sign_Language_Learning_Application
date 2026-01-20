from __future__ import annotations

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QGroupBox, QGridLayout, QMessageBox
)

from core.config import AppConfig
from core.user_store import UserStore
from ui.theme import APP_QSS
from ui.dialogs import LoginDialog, RegisterDialog
from app.main_window import ASLLearningApp


class LandingWindow(QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.store = UserStore("users.json")
        self.settings = QSettings("ASLExample", "ASLLearningSuiteV2")
        self.username = None  # None = Guest
        # Remember last logged-in user
        try:
            last_u = (self.settings.value("last_username", "") or "").strip()
            if last_u and last_u in self.store.data.get("users", {}):
                self.username = last_u
        except Exception:
            pass
        self.init_ui()
        self.apply_theme()
        self.refresh_user_view()

    def apply_theme(self):
        self.setStyleSheet(APP_QSS)

    def init_ui(self):
        self.setWindowTitle(self.cfg.app_title)
        self.setMinimumSize(980, 620)

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(26, 26, 26, 26)
        lay.setSpacing(18)

        title = QLabel(self.cfg.app_title)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 22, QFont.Bold))

        subtitle = QLabel("Interactive ASL Learning • Real-time Hand Tracking • Gamified Practice")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("Muted")

        # user row
        user_row = QHBoxLayout()
        user_row.addStretch()
        self.lbl_user = QLabel("User: Guest")
        self.lbl_user.setObjectName("Pill")
        user_row.addWidget(self.lbl_user)
        user_row.addStretch()

        # highlights
        features = QGroupBox("Highlights")
        f_lay = QVBoxLayout(features)
        f_lay.setContentsMargins(16, 14, 16, 14)
        f_lay.setSpacing(8)
        for t in [
            "Learn A–Z with real-time ASL recognition (streak-based)",
            "Quiz & Spelling modes with progress tracking",
            "Snake game controlled by your hand (index finger)",
            "Teacher Mode: FPS + confidence visualization",
        ]:
            lb = QLabel("• " + t)
            lb.setWordWrap(True)
            f_lay.addWidget(lb)

        # highest score box
        scores = QGroupBox("Highest Scores")
        s_lay = QGridLayout(scores)
        s_lay.setContentsMargins(16, 14, 16, 14)
        s_lay.setHorizontalSpacing(18)
        s_lay.setVerticalSpacing(10)

        s_lay.addWidget(QLabel("Quiz:"), 0, 0, alignment=Qt.AlignRight)
        self.lbl_best_quiz = QLabel("0")
        self.lbl_best_quiz.setFont(QFont("Arial", 12, QFont.Bold))
        s_lay.addWidget(self.lbl_best_quiz, 0, 1)

        s_lay.addWidget(QLabel("Snake:"), 0, 2, alignment=Qt.AlignRight)
        self.lbl_best_snake = QLabel("0")
        self.lbl_best_snake.setFont(QFont("Arial", 12, QFont.Bold))
        s_lay.addWidget(self.lbl_best_snake, 0, 3)

        s_lay.addWidget(QLabel("Spelling:"), 1, 0, alignment=Qt.AlignRight)
        self.lbl_best_spelling = QLabel("0")
        self.lbl_best_spelling.setFont(QFont("Arial", 12, QFont.Bold))
        s_lay.addWidget(self.lbl_best_spelling, 1, 1)

        self.lbl_scores_hint = QLabel("(* Nếu chưa đăng nhập thì điểm sẽ lưu theo Guest)")
        self.lbl_scores_hint.setObjectName("Muted")
        s_lay.addWidget(self.lbl_scores_hint, 2, 0, 1, 4)

        # buttons
        btns = QHBoxLayout()
        btns.setSpacing(12)

        self.btn_login = QPushButton("Đăng nhập")
        self.btn_register = QPushButton("Đăng ký")
        self.btn_start = QPushButton("Start Learning")
        self.btn_exit = QPushButton("Exit")

        self.btn_start.setDefault(True)

        btns.addStretch()
        btns.addWidget(self.btn_login)
        btns.addWidget(self.btn_register)
        btns.addSpacing(12)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_exit)
        btns.addStretch()

        foot = QLabel("Tip: Use 1–4 to switch modes, F to flip camera, T to toggle Teacher mode.")
        foot.setAlignment(Qt.AlignCenter)
        foot.setObjectName("Muted")

        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addLayout(user_row)
        lay.addWidget(features)
        lay.addWidget(scores)
        lay.addLayout(btns)
        lay.addWidget(foot)

        self.btn_login.clicked.connect(self.on_login)
        self.btn_register.clicked.connect(self.on_register)
        self.btn_start.clicked.connect(self.open_asl_app)
        self.btn_exit.clicked.connect(self.close)

    def refresh_user_view(self):
        who = self.username or "Guest"
        self.lbl_user.setText(f"User: {who}")

        q = self.store.get_high_score(self.username, "quiz")
        s = self.store.get_high_score(self.username, "snake")
        sp = self.store.get_high_score(self.username, "spelling")

        self.lbl_best_quiz.setText(str(q))
        self.lbl_best_snake.setText(str(s))
        self.lbl_best_spelling.setText(str(sp))

    def on_login(self):
        dlg = LoginDialog(self.store, self)
        if dlg.exec_() == dlg.Accepted and dlg.username:
            self.username = dlg.username
            try:
                self.settings.setValue("last_username", self.username)
            except Exception:
                pass
            self.refresh_user_view()

    def on_register(self):
        dlg = RegisterDialog(self.store, self)
        if dlg.exec_() == dlg.Accepted and dlg.username:
            # auto-fill after register: treat as logged in
            self.username = dlg.username
            try:
                self.settings.setValue("last_username", self.username)
            except Exception:
                pass
            self.refresh_user_view()

    def open_asl_app(self):
        try:
            self.asl_window = ASLLearningApp(self.cfg, username=self.username, user_store=self.store)
            self.asl_window.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Start failed", f"Không thể khởi động ứng dụng:\n{e}")
