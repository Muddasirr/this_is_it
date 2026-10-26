from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PADEL_DATA_DIR", ROOT / "data" / "app"))
MATCH_DIR = DATA_DIR / "matches"
DB_URL = os.environ.get("PADEL_DB_URL", f"sqlite:///{DATA_DIR / 'padel.db'}")
DEVICE = os.environ.get("PADEL_DEVICE") or None  # None = cuda if available else cpu
POSE_STRIDE = int(os.environ.get("PADEL_POSE_STRIDE", "2"))
MAX_FRAMES = int(os.environ.get("PADEL_MAX_FRAMES", "0"))  # 0 = whole video
# kaggle = run detection + analysis on the Kaggle GPU (account in ../.env), local = this machine
BACKEND = os.environ.get("PADEL_BACKEND", "kaggle")
CORS = os.environ.get("PADEL_CORS", "http://localhost:5175,http://127.0.0.1:5175").split(",")
WEB_DIST = ROOT / "web" / "dist"

MATCH_DIR.mkdir(parents=True, exist_ok=True)
