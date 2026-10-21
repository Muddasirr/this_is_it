"""Stable vision types (Kaggle offline copy — no app settings)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Detection:
    track_key: str
    camera_label: str
    start_ms: int
    end_ms: int
    confidence: float
    bbox_series: list[dict[str, Any]] | None = None
    class_name: str = "person"
    features: dict[str, Any] | None = None


@dataclass
class EventCandidate:
    event_type: str  # goal | shot | pass
    camera_label: str
    camera_ts_ms: int
    confidence: float
    track_keys: list[str]
    evidence: dict[str, Any] | None = None


class VisionBackend(ABC):
    """Swap YOLO/ByteTrack/etc. without changing product APIs."""

    name: str
    version: str
    last_run_stats: dict[str, Any]

    @abstractmethod
    def detect_and_track(self, video_path: str, camera_label: str) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        raise NotImplementedError
