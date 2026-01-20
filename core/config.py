from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class AppConfig:
    app_title: str = "ASL Learning Suite"
    model_path: str = "models/asl_model.h5"
    reference_dir: str = "EXAMPLE_IMG"
    food_image: str = "Donut.png"

    camera_index: int = 1
    frame_width: int = 1280
    frame_height: int = 720

    img_size: Tuple[int, int] = (224, 224)

    pred_interval: float = 0.30
    conf_thresh: float = 0.70
    streak_required: int = 5
    offset_scale: float = 1.2

    # UI
    # NOTE: Using point sizes (pt) helps DPI scaling on Windows.
    base_font_pt: int = 11          # default app font size
    target_font_pt: int = 56        # "Current Target" big letter size

    # Study suggestions (popup)
    suggestion_enabled: bool = True
    suggestion_snooze_minutes: int = 10  # default 10 minutes (user can set 5/10/15)

    # Multiple choice assets (dataset folder, e.g., MCQs/A/*.jpg)
    multiplechoice_dir: str = "MCQs"

    spelling_words: List[str] = None

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))

        cfg = AppConfig()
        cfg.app_title = data.get("app_title", cfg.app_title)
        cfg.model_path = data.get("model_path", cfg.model_path)
        cfg.reference_dir = data.get("reference_dir", cfg.reference_dir)
        cfg.food_image = data.get("food_image", cfg.food_image)

        cfg.camera_index = int(data.get("camera_index", cfg.camera_index))
        cfg.frame_width = int(data.get("frame_width", cfg.frame_width))
        cfg.frame_height = int(data.get("frame_height", cfg.frame_height))

        img_size = data.get("img_size", list(cfg.img_size))
        cfg.img_size = (int(img_size[0]), int(img_size[1]))

        cfg.pred_interval = float(data.get("pred_interval", cfg.pred_interval))
        cfg.conf_thresh = float(data.get("conf_thresh", cfg.conf_thresh))
        cfg.streak_required = int(data.get("streak_required", cfg.streak_required))
        cfg.offset_scale = float(data.get("offset_scale", cfg.offset_scale))

        # UI
        cfg.base_font_pt = int(data.get("base_font_pt", cfg.base_font_pt))
        cfg.target_font_pt = int(data.get("target_font_pt", cfg.target_font_pt))

        cfg.suggestion_enabled = bool(data.get("suggestion_enabled", cfg.suggestion_enabled))
        cfg.suggestion_snooze_minutes = int(data.get("suggestion_snooze_minutes", cfg.suggestion_snooze_minutes))
        cfg.multiplechoice_dir = str(data.get("multiplechoice_dir", cfg.multiplechoice_dir))

        cfg.spelling_words = data.get("spelling_words") or ["HELLO", "YES", "NO", "LOVE", "NAME"]
        return cfg

    def save(self, path: str | Path) -> None:
        """Persist config to json (so UI settings are easy to change & keep)."""
        p = Path(path)
        data = {
            "app_title": self.app_title,
            "model_path": self.model_path,
            "reference_dir": self.reference_dir,
            "food_image": self.food_image,
            "camera_index": int(self.camera_index),
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "img_size": [int(self.img_size[0]), int(self.img_size[1])],
            "pred_interval": float(self.pred_interval),
            "conf_thresh": float(self.conf_thresh),
            "streak_required": int(self.streak_required),
            "offset_scale": float(self.offset_scale),
            "spelling_words": list(self.spelling_words or []),
            # UI
            "base_font_pt": int(self.base_font_pt),
            "target_font_pt": int(self.target_font_pt),
            # Study suggestions
            "suggestion_enabled": bool(self.suggestion_enabled),
            "suggestion_snooze_minutes": int(self.suggestion_snooze_minutes),
            # MCQ dataset
            "multiplechoice_dir": str(self.multiplechoice_dir),
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
