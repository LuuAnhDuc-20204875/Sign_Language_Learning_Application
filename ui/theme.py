# ui/theme.py
# Centralized theme stylesheet for the whole app.

APP_QSS = r"""
/* =====================
   Base
===================== */
QMainWindow {
  background: #0b1220;
}
QWidget {
  font-family: "Segoe UI";
}
QLabel {
  color: #e5e7eb;
}

/* Typography helpers */
QLabel#H1 {
  font-size: 18px;
  font-weight: 800;
  color: #f9fafb;
}
QLabel#Muted {
  color: #9ca3af;
}

/* =====================
   Cards & containers
===================== */
QFrame#Card {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #0f1b2d,
    stop:1 #0b1526);
  border: 1px solid #1f2a3d;
  border-radius: 14px;
}
QFrame#TopBar {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #0f1b2d,
    stop:1 #0b1526);
  border: 1px solid #1f2a3d;
  border-radius: 16px;
}
QLabel#VideoFrame {
  background: #000000;
  border-radius: 12px;
  border: 1px solid #22314a;
}

/* Reference image frame */
QLabel#RefFrame {
  background: #0b1526;
  border-radius: 12px;
  border: 1px solid #22314a;
}

/* =====================
   Buttons
===================== */
QPushButton {
  background: #2563eb;
  color: white;
  border-radius: 10px;
  padding: 10px 12px;
  font-weight: 700;
}
QPushButton:hover { background: #1d4ed8; }
QPushButton:pressed { background: #1e40af; }

QPushButton#ModeBtn {
  background: #0b1526;
  color: #e5e7eb;
  border: 1px solid #22314a;
  border-radius: 12px;
  padding: 10px 12px;
  text-align: left;
}
QPushButton#ModeBtn:hover {
  background: #0f1b2d;
  border-color: #2e4370;
}
QPushButton#ModeBtn:checked {
  background: #13264a;
  border-color: #3b82f6;
}

/* Icon tool buttons */
QToolButton#IconBtn {
  background: #0b1526;
  border: 1px solid #22314a;
  border-radius: 10px;
  padding: 6px;
}
QToolButton#IconBtn:hover { border-color: #3b82f6; }

/* Pills on top bar */
QLabel#Pill {
  background: #0b1526;
  border: 1px solid #22314a;
  border-radius: 999px;
  padding: 6px 10px;
  color: #e5e7eb;
  font-weight: 700;
}

/* Big target */
QLabel#BigTarget {
  color: #60a5fa;
}

/* Prediction label */
QLabel#PredLabel {
  font-size: 13px;
  font-weight: 800;
  color: #f9fafb;
}

/* =====================
   Progress bars
===================== */
QProgressBar {
  background: #0b1526;
  border: 1px solid #22314a;
  border-radius: 10px;
  height: 18px;
  text-align: center;
  color: #e5e7eb;
  font-weight: 700;
}
QProgressBar::chunk {
  border-radius: 10px;
  background: #22c55e;
}

/* =====================
   Splitter + Scroll
===================== */
QSplitter::handle {
  background: #1f2a3d;
}
QScrollArea#RightScroll {
  border: 0;
  background: transparent;
}
QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 6px 2px 6px 2px;
}
QScrollBar::handle:vertical {
  background: #22314a;
  border-radius: 5px;
  min-height: 20px;
}
QScrollBar::handle:vertical:hover {
  background: #2e4370;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
  height: 0px;
}
"""
