"""Stable vision interface — pretrained models behind a swappable backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings


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


class StubVisionBackend(VisionBackend):
    """Deterministic stub for pipeline wiring without GPU/weights."""

    name = "stub-pretrained"
    version = "0.1.0"

    def __init__(self) -> None:
        self.last_run_stats = {}

    def detect_and_track(self, video_path: str, camera_label: str) -> list[Detection]:
        self.last_run_stats = {"backend": "stub", "person_tracks": 1, "ball_detections": 0}
        return [
            Detection(
                track_key=f"{camera_label}-t1",
                camera_label=camera_label,
                start_ms=0,
                end_ms=5000,
                confidence=0.4,
                class_name="person",
            )
        ]

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import infer_event_candidates

        return infer_event_candidates(detections, camera_label)


def get_vision_backend() -> VisionBackend:
    settings = get_settings()
    if settings.vision_backend == "stub":
        return StubVisionBackend()
    if settings.vision_backend == "kaggle":
        from app.services.kaggle_backend import KaggleTrackBackend

        return KaggleTrackBackend()
    if settings.vision_backend == "modal":
        from app.services.modal_backend import ModalTrackBackend

        return ModalTrackBackend()
    from app.services.yolo_backend import YoloTrackBackend

    return YoloTrackBackend()
