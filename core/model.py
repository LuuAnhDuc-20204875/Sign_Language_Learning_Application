# core/model.py
from __future__ import annotations

import os
from typing import Tuple, Optional

import cv2
import numpy as np

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model
except Exception as e:
    raise ImportError(
        "Không import được TensorFlow. Nếu gặp lỗi DLL trên Windows, "
        "thường là do thiếu Microsoft Visual C++ Runtime hoặc bị lệch version "
        "protobuf/numpy/tensorflow.\n"
        f"Chi tiết lỗi: {repr(e)}"
    ) from e

# Import config trong project
try:
    from core.config import LABELS, IMG_SIZE
except Exception:
    # fallback để file vẫn chạy nếu config bị đổi tên
    IMG_SIZE = (224, 224)
    LABELS = [
        "A","B","C","D","E","F","G","H","I","J",
        "K","L","M","N","O","P","Q","R","S","T",
        "U","V","W","X","Y","Z",
        "del","nothing","space"
    ]


class ASLModel:
    def __init__(self, model_path: str, labels=None):
        if labels is None:
            labels = LABELS
        self.labels = list(labels)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = load_model(model_path, compile=False)
        print("✅ Loaded ASL model from:", model_path)

        out_units = self.model.output_shape[-1]
        if out_units != len(self.labels):
            raise ValueError(
                f"Model output has {out_units} units, "
                f"but labels has {len(self.labels)} classes!"
            )

    def predict_letter(self, img_roi_bgr: np.ndarray):
        if img_roi_bgr.shape[:2] != IMG_SIZE:
            img_roi_bgr = cv2.resize(img_roi_bgr, IMG_SIZE)

        x = cv2.cvtColor(img_roi_bgr, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x, verbose=0)[0]
        index = int(np.argmax(preds))
        conf = float(np.max(preds))

        if 0 <= index < len(self.labels):
            return self.labels[index], conf
        return None, conf
