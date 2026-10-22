"""Minimal settings for Kaggle (mirrors local CourtLens vision defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def resolve_yolo_weights(name: str | None = None) -> str:
    """Resolve a YOLO weights name to an existing file when possible."""
    raw = (name or "yolov5x_football.pt").strip() or "yolov5x_football.pt"
    given = Path(raw)
    if given.is_file() and given.stat().st_size > 1000:
        return str(given.resolve())

    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / raw,
        here.parent / raw,
        given,
    ]
    stem = given.name.lower()
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), here.parents[2] if len(here.parents) >= 3 else here.parent]
    names = [stem]
    if "yolov5x" in stem or stem in {"best.pt", "yolov5x_football.pt"}:
        names = ["yolov5x_football.pt"]
    for root in search_roots:
        if not root.is_dir():
            continue
        for n in names:
            try:
                hits = list(root.rglob(n))
            except OSError:
                hits = []
            candidates.extend(hits[:8])
        candidates.append(root / "data/eval/football_analysis_model/best.pt")
        candidates.append(root / "football_analysis_model/best.pt")
    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size > 1000:
                return str(cand.resolve())
        except OSError:
            continue
    return raw


@dataclass
class Settings:
    yolo_model: str = "yolov5x_football.pt"
    vision_device: str = "0"  # GPU
    vision_frame_stride: int = 5  # same as local backend default
    vision_conf: float = 0.25
    vision_imgsz: int = 640
    vision_max_frames: int = 0
    # OCR on by default but heavily throttled in yolo_backend (avatars + numbers)
    vision_jersey_ocr: bool = True
    vision_tracker: str = "botsort"
    vision_reid: bool = True
    vision_reid_model: str = "osnet_x0_25"
    vision_reid_every: int = 3


def get_settings() -> Settings:
    ocr_raw = os.environ.get("VISION_JERSEY_OCR", "1").strip().lower()
    reid_raw = os.environ.get("VISION_REID", "1").strip().lower()
    return Settings(
        yolo_model=os.environ.get("YOLO_MODEL", "yolov5x_football.pt"),
        vision_device=os.environ.get("VISION_DEVICE", "0"),
        vision_frame_stride=int(os.environ.get("VISION_FRAME_STRIDE", "5")),
        vision_conf=float(os.environ.get("VISION_CONF", "0.25")),
        vision_imgsz=int(os.environ.get("VISION_IMGSZ", "640")),
        vision_max_frames=int(os.environ.get("VISION_MAX_FRAMES", "0")),
        vision_jersey_ocr=ocr_raw not in {"0", "false", "off", "no"},
        vision_tracker=os.environ.get("VISION_TRACKER", "botsort").strip().lower(),
        vision_reid=reid_raw not in {"0", "false", "off", "no"},
        vision_reid_model=os.environ.get("VISION_REID_MODEL", "osnet_x0_25"),
        vision_reid_every=int(os.environ.get("VISION_REID_EVERY", "3")),
    )
