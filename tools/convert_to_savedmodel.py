from __future__ import annotations

from pathlib import Path
from tensorflow.keras.models import load_model

# Edit paths:
IN_H5 = Path(r"models/asl_model.h5")     # .h5 file
OUT_DIR = Path(r"models/asl_savedmodel") # folder output

m = load_model(str(IN_H5), compile=False)
m.save(str(OUT_DIR))
print("SavedModel exported to:", OUT_DIR.resolve())
