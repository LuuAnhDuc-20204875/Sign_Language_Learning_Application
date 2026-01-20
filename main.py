from __future__ import annotations

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from core.config import AppConfig
from app.landing import LandingWindow


def main():
    cfg = AppConfig.load("config.json")
    app = QApplication(sys.argv)
    # Default app font (DPI-friendly)
    app.setFont(QFont("Segoe UI", int(getattr(cfg, "base_font_pt", 11))))
    landing = LandingWindow(cfg)
    landing.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
