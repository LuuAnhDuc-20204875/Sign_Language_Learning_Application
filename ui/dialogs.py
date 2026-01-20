from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QCheckBox, QComboBox
)

from core.config import AppConfig
from core.user_store import UserStore


class SettingsDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.cfg = cfg
        self._ok = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        title = QLabel("Application Settings")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        lay.addWidget(title)

        # model path
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model path:"))
        self.ed_model = QLineEdit(cfg.model_path)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.clicked.connect(self.browse_model)
        row1.addWidget(self.ed_model, 1)
        row1.addWidget(btn_browse_model)
        lay.addLayout(row1)

        # reference dir
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Reference dir:"))
        self.ed_ref = QLineEdit(cfg.reference_dir)
        btn_browse_ref = QPushButton("Browse")
        btn_browse_ref.clicked.connect(self.browse_ref)
        row2.addWidget(self.ed_ref, 1)
        row2.addWidget(btn_browse_ref)
        lay.addLayout(row2)

        # MCQ dataset dir
        row2b = QHBoxLayout()
        row2b.addWidget(QLabel("MCQ dataset dir:"))
        self.ed_mcq = QLineEdit(getattr(cfg, "multiplechoice_dir", "MCQs"))
        btn_browse_mcq = QPushButton("Browse")
        btn_browse_mcq.clicked.connect(self.browse_mcq)
        row2b.addWidget(self.ed_mcq, 1)
        row2b.addWidget(btn_browse_mcq)
        lay.addLayout(row2b)

        # camera index
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Camera index:"))
        self.sp_cam = QSpinBox()
        self.sp_cam.setRange(0, 10)
        self.sp_cam.setValue(cfg.camera_index)
        row3.addWidget(self.sp_cam)
        row3.addStretch()
        lay.addLayout(row3)

        # UI fonts
        row_ui = QHBoxLayout()
        row_ui.addWidget(QLabel("UI font (pt):"))
        self.sp_ui_font = QSpinBox()
        self.sp_ui_font.setRange(8, 20)
        self.sp_ui_font.setValue(int(getattr(cfg, "base_font_pt", 11)))
        row_ui.addWidget(self.sp_ui_font)
        row_ui.addSpacing(10)
        row_ui.addWidget(QLabel("Target font (pt):"))
        self.sp_target_font = QSpinBox()
        self.sp_target_font.setRange(24, 140)
        self.sp_target_font.setValue(int(getattr(cfg, "target_font_pt", 56)))
        row_ui.addWidget(self.sp_target_font)
        row_ui.addStretch()
        lay.addLayout(row_ui)

        # thresholds
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Conf threshold:"))
        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.0, 1.0)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(cfg.conf_thresh)
        row4.addWidget(self.sp_conf)
        row4.addSpacing(10)
        row4.addWidget(QLabel("Pred interval (s):"))
        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setRange(0.05, 2.0)
        self.sp_interval.setSingleStep(0.05)
        self.sp_interval.setValue(cfg.pred_interval)
        row4.addWidget(self.sp_interval)
        lay.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Streak required:"))
        self.sp_streak = QSpinBox()
        self.sp_streak.setRange(1, 20)
        self.sp_streak.setValue(cfg.streak_required)
        row5.addWidget(self.sp_streak)
        row5.addStretch()
        lay.addLayout(row5)

        # Study suggestions
        row6 = QHBoxLayout()
        row6.addWidget(QLabel("Study suggestions:"))
        self.cb_suggest = QCheckBox("Enable popup")
        self.cb_suggest.setChecked(bool(getattr(cfg, "suggestion_enabled", True)))
        row6.addWidget(self.cb_suggest)
        row6.addSpacing(10)
        row6.addWidget(QLabel("Snooze (min):"))
        self.cmb_snooze = QComboBox()
        self.cmb_snooze.addItems(["5", "10", "15"])
        current = str(int(getattr(cfg, "suggestion_snooze_minutes", 10)))
        if current not in ("5", "10", "15"):
            current = "10"
        self.cmb_snooze.setCurrentText(current)
        row6.addWidget(self.cmb_snooze)
        row6.addStretch()
        lay.addLayout(row6)

        # buttons
        btns = QHBoxLayout()
        btns.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Save")
        btn_ok.clicked.connect(self.on_ok)
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        lay.addLayout(btns)

        self.setStyleSheet("""
        QDialog { background: #0b1220; color: #e5e7eb; }
        QLabel { color: #e5e7eb; }
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            padding: 6px 8px;
            color: #e5e7eb;
        }
        QPushButton {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            padding: 8px 10px;
            color: #e5e7eb;
            font-weight: 600;
        }
        QPushButton:hover { background: rgba(255,255,255,0.12); }
        """)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model (.h5) or choose any file", "", "Model (*.h5);;All files (*)")
        if path:
            self.ed_model.setText(path)

    def browse_ref(self):
        path = QFileDialog.getExistingDirectory(self, "Select reference images folder", "")
        if path:
            self.ed_ref.setText(path)

    def browse_mcq(self):
        path = QFileDialog.getExistingDirectory(self, "Select MCQ dataset folder (e.g., MCQs)", "")
        if path:
            self.ed_mcq.setText(path)

    def on_ok(self):
        self.cfg.model_path = self.ed_model.text().strip()
        self.cfg.reference_dir = self.ed_ref.text().strip()
        self.cfg.multiplechoice_dir = self.ed_mcq.text().strip() or getattr(self.cfg, "multiplechoice_dir", "MCQs")
        self.cfg.camera_index = int(self.sp_cam.value())
        self.cfg.conf_thresh = float(self.sp_conf.value())
        self.cfg.pred_interval = float(self.sp_interval.value())
        self.cfg.streak_required = int(self.sp_streak.value())

        # UI
        self.cfg.base_font_pt = int(self.sp_ui_font.value())
        self.cfg.target_font_pt = int(self.sp_target_font.value())

        # Study suggestions
        self.cfg.suggestion_enabled = bool(self.cb_suggest.isChecked())
        try:
            self.cfg.suggestion_snooze_minutes = int(self.cmb_snooze.currentText())
        except Exception:
            self.cfg.suggestion_snooze_minutes = int(getattr(self.cfg, "suggestion_snooze_minutes", 10))
        self._ok = True
        self.accept()

    @property
    def ok(self) -> bool:
        return self._ok


class LoginDialog(QDialog):
    def __init__(self, store: UserStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Đăng nhập")
        self.setModal(True)
        self.store = store
        self.username: Optional[str] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        title = QLabel("Đăng nhập")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        lay.addWidget(title)

        # Username
        row_u = QHBoxLayout()
        row_u.addWidget(QLabel("Tên đăng nhập:"))
        self.ed_user = QLineEdit()
        self.ed_user.setPlaceholderText("vd: lan_anh")
        row_u.addWidget(self.ed_user)
        lay.addLayout(row_u)

        # Password
        row_p = QHBoxLayout()
        row_p.addWidget(QLabel("Mật khẩu:"))
        self.ed_pass = QLineEdit()
        self.ed_pass.setEchoMode(QLineEdit.Password)
        self.ed_pass.setPlaceholderText("••••")
        row_p.addWidget(self.ed_pass)
        lay.addLayout(row_p)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QPushButton("Hủy")
        self.btn_ok = QPushButton("Đăng nhập")
        self.btn_ok.setDefault(True)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        lay.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.on_login)

    def on_login(self):
        u = self.ed_user.text().strip()
        p = self.ed_pass.text()
        res = self.store.login(u, p)
        if not res.ok:
            QMessageBox.warning(self, "Đăng nhập thất bại", res.message)
            return
        self.username = res.username
        self.accept()


class RegisterDialog(QDialog):
    def __init__(self, store: UserStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Đăng ký")
        self.setModal(True)
        self.store = store
        self.username: Optional[str] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        title = QLabel("Tạo tài khoản")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        lay.addWidget(title)

        hint = QLabel("Tên đăng nhập: 3–20 ký tự (chữ/số/_). Mật khẩu: tối thiểu 4 ký tự.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        row_u = QHBoxLayout()
        row_u.addWidget(QLabel("Tên đăng nhập:"))
        self.ed_user = QLineEdit()
        self.ed_user.setPlaceholderText("vd: lan_anh")
        row_u.addWidget(self.ed_user)
        lay.addLayout(row_u)

        row_p = QHBoxLayout()
        row_p.addWidget(QLabel("Mật khẩu:"))
        self.ed_pass = QLineEdit()
        self.ed_pass.setEchoMode(QLineEdit.Password)
        self.ed_pass.setPlaceholderText("••••")
        row_p.addWidget(self.ed_pass)
        lay.addLayout(row_p)

        row_p2 = QHBoxLayout()
        row_p2.addWidget(QLabel("Nhập lại:"))
        self.ed_pass2 = QLineEdit()
        self.ed_pass2.setEchoMode(QLineEdit.Password)
        self.ed_pass2.setPlaceholderText("••••")
        row_p2.addWidget(self.ed_pass2)
        lay.addLayout(row_p2)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QPushButton("Hủy")
        self.btn_ok = QPushButton("Đăng ký")
        self.btn_ok.setDefault(True)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        lay.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.on_register)

    def on_register(self):
        u = self.ed_user.text().strip()
        p1 = self.ed_pass.text()
        p2 = self.ed_pass2.text()
        if p1 != p2:
            QMessageBox.warning(self, "Lỗi", "Mật khẩu nhập lại không khớp.")
            return
        res = self.store.register(u, p1)
        if not res.ok:
            QMessageBox.warning(self, "Đăng ký thất bại", res.message)
            return
        QMessageBox.information(self, "OK", "Đăng ký thành công! Bạn có thể đăng nhập.")
        self.username = res.username
        self.accept()
