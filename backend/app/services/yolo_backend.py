"""Pretrained YOLO detection + ByteTrack tracking (Ultralytics)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from app.core.config import get_settings
from app.services.vision import Detection, EventCandidate, VisionBackend

logger = logging.getLogger(__name__)

# COCO class ids
PERSON = 0
SPORTS_BALL = 32


class YoloTrackBackend(VisionBackend):
    name = "yolov8n-bytetrack"
    version = "0.1.0"

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.yolo_model
        self.device = settings.vision_device
        self.frame_stride = max(1, settings.vision_frame_stride)
        self.conf = settings.vision_conf
        self.imgsz = settings.vision_imgsz
        self.max_frames = settings.vision_max_frames  # 0 = full video
        self._model = None
        self.last_run_stats: dict[str, Any] = {}

    def _load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        logger.info("Loading YOLO model %s on %s", self.model_name, self.device)
        self._model = YOLO(self.model_name)
        return self._model

    def detect_and_track(self, video_path: str, camera_label: str) -> list[Detection]:
        import cv2

        model = self._load()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # track_id -> accumulated observations
        person_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        ball_hits = 0
        frames_processed = 0
        t0 = time.perf_counter()
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.frame_stride != 0:
                    frame_idx += 1
                    continue
                if self.max_frames and frames_processed >= self.max_frames:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    classes=[PERSON, SPORTS_BALL],
                    verbose=False,
                )
                frames_processed += 1
                ts_ms = int((frame_idx / fps) * 1000)
                result = results[0]

                if result.boxes is None or len(result.boxes) == 0:
                    frame_idx += 1
                    continue

                boxes = result.boxes
                ids = boxes.id
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].tolist()
                    if cls_id == SPORTS_BALL:
                        ball_hits += 1
                    if ids is None:
                        continue
                    tid = int(ids[i].item())
                    obs = {
                        "ts_ms": ts_ms,
                        "confidence": conf,
                        "bbox": {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        },
                        "frame_h": float(frame.shape[0]),
                        "frame_w": float(frame.shape[1]),
                        "class_name": "sports_ball" if cls_id == SPORTS_BALL else "person",
                    }
                    if cls_id == SPORTS_BALL:
                        person_tracks[tid + 100000].append(obs)
                    elif cls_id == PERSON:
                        person_tracks[tid].append(obs)
                frame_idx += 1
        finally:
            cap.release()

        elapsed = time.perf_counter() - t0
        detections: list[Detection] = []
        for tid, obs in person_tracks.items():
            if not obs:
                continue
            class_name = obs[0].get("class_name", "person")
            prefix = "b" if class_name == "sports_ball" else "p"
            key_id = tid - 100000 if class_name == "sports_ball" else tid
            # keep a sparse bbox series (first, mid, last) to avoid huge rows
            series = [obs[0], obs[len(obs) // 2], obs[-1]] if len(obs) >= 3 else obs
            detections.append(
                Detection(
                    track_key=f"{camera_label}-{prefix}{key_id}",
                    camera_label=camera_label,
                    start_ms=obs[0]["ts_ms"],
                    end_ms=obs[-1]["ts_ms"],
                    confidence=sum(o["confidence"] for o in obs) / len(obs),
                    bbox_series=[
                        {
                            "ts_ms": o["ts_ms"],
                            **o["bbox"],
                            "confidence": o["confidence"],
                            "frame_h": o.get("frame_h"),
                            "frame_w": o.get("frame_w"),
                        }
                        for o in series
                    ],
                    class_name=class_name,
                )
            )

        video_seconds = (total_frames / fps) if total_frames and fps else 0
        self.last_run_stats = {
            "camera_label": camera_label,
            "model": self.model_name,
            "device": self.device,
            "fps_source": round(fps, 3),
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "frame_stride": self.frame_stride,
            "elapsed_sec": round(elapsed, 3),
            "processing_fps": round(frames_processed / elapsed, 3) if elapsed else 0,
            "realtime_factor": round(video_seconds / elapsed, 3) if elapsed and video_seconds else None,
            "person_tracks": len(detections),
            "ball_detections": ball_hits,
            "ball_visible": ball_hits > 0,
            "player_visible": len(detections) > 0,
            "video_seconds_probed": round(video_seconds, 2),
        }
        logger.info("YOLO track done: %s", self.last_run_stats)
        return detections

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import infer_event_candidates

        return infer_event_candidates(detections, camera_label)


def run_feasibility_on_video(video_path: str, camera_label: str = "A") -> dict[str, Any]:
    """One-shot Phase 0 report for a public/online clip."""
    from app.services.media import probe_video, validate_probe

    probe = probe_video(video_path)
    validation_errors = validate_probe(probe, min_duration_ms=30_000)

    backend = YoloTrackBackend()
    detections = backend.detect_and_track(video_path, camera_label)
    stats = backend.last_run_stats

    go = bool(stats.get("player_visible")) and not any(
        "Unsupported codec" in e or "Frame rate too low" in e for e in validation_errors
    )
    notes: list[str] = []
    if not stats.get("player_visible"):
        notes.append("No person tracks — footage may be too far/blurry or wrong content.")
    if not stats.get("ball_visible"):
        notes.append("Ball never detected (common on wide CCTV / small futsal ball).")
    if stats.get("realtime_factor") is not None and stats["realtime_factor"] < 0.5:
        notes.append("Slower than 0.5× realtime at current settings — raise stride or use GPU.")

    return {
        "probe": probe,
        "validation_errors": validation_errors,
        "detection": stats,
        "sample_tracks": [
            {
                "track_key": d.track_key,
                "start_ms": d.start_ms,
                "end_ms": d.end_ms,
                "confidence": round(d.confidence, 3),
            }
            for d in detections[:20]
        ],
        "go_no_go": {
            "players_detectable": bool(stats.get("player_visible")),
            "ball_detectable": bool(stats.get("ball_visible")),
            "decode_ok": len(validation_errors) == 0 or all("Duration" in e for e in validation_errors),
            "suggested_continue": go,
            "notes": notes,
        },
    }
