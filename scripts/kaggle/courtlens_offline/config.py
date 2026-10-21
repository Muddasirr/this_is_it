"""Minimal settings for Kaggle (mirrors local CourtLens vision defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    yolo_model: str = "yolov8n.pt"
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
        yolo_model=os.environ.get("YOLO_MODEL", "yolov8n.pt"),
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
