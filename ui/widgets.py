from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QFrame, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QProgressBar


class Card(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet("""
        QFrame#Card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
        }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 14, 14, 14)
        self.layout.setSpacing(10)

        if title:
            ttl = QLabel(title)
            ttl.setFont(QFont("Arial", 11, QFont.Bold))
            ttl.setStyleSheet("color:#cbd5e1;")
            self.layout.addWidget(ttl)


class Pill(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                color: #e5e7eb;
                font-size: 10.5pt;
            }
        """)


class BigTarget(QLabel):
    def __init__(self, parent=None):
        super().__init__("-", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Arial", 56, QFont.Bold))
        self.setStyleSheet("color:#60a5fa; padding: 6px;")


class ModeButton(QPushButton):
    def __init__(self, text: str, emoji: str = "", parent=None):
        super().__init__(f"{emoji}  {text}" if emoji else text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setStyleSheet("""
            QPushButton{
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 10px 12px;
                text-align: left;
                color: #e5e7eb;
                font-weight: 650;
            }
            QPushButton:hover{ background: rgba(255,255,255,0.10); }
            QPushButton:checked{
                background: rgba(37,99,235,0.35);
                border: 1px solid rgba(37,99,235,0.85);
            }
        """)
        self.setCheckable(True)


def make_kv_row(label: str, value: str) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)

    l = QLabel(label)
    l.setStyleSheet("color:#94a3b8;")
    v = QLabel(value)
    v.setStyleSheet("color:#e5e7eb; font-weight:600;")
    lay.addWidget(l)
    lay.addStretch()
    lay.addWidget(v)
    return w


class StreakBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.title = QLabel("Streak")
        self.title.setStyleSheet("color:#94a3b8;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFormat("%p%")
        lay.addWidget(self.title)
        lay.addWidget(self.bar)

    def set_progress(self, current: int, required: int):
        if required <= 0:
            self.bar.setValue(0)
            self.bar.setFormat("—")
            return
        pct = int(max(0, min(100, (current / required) * 100)))
        self.bar.setValue(pct)
        self.bar.setFormat(f"{current}/{required}")
