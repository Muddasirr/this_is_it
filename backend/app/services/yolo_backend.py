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

# Keep dense enough series for possession events without huge JSON rows
MAX_PERSON_SERIES = 400
MAX_BALL_SERIES = 2000


class YoloTrackBackend(VisionBackend):
    name = "yolov8n-bytetrack"
    version = "0.3.0"

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
        import numpy as np

        model = self._load()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        person_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        color_sums: dict[int, Any] = {}
        color_counts: dict[int, int] = defaultdict(int)
        jersey_votes: dict[int, list[tuple[int, float]]] = defaultdict(list)
        ball_hits = 0
        frames_processed = 0
        t0 = time.perf_counter()
        frame_idx = 0
        ocr_every = max(4, 12 // max(1, self.frame_stride))  # ~sparse OCR for CPU

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
                        hist = _upper_body_hsv_hist(frame, xyxy)
                        if hist is not None:
                            if tid not in color_sums:
                                color_sums[tid] = hist
                            else:
                                color_sums[tid] = color_sums[tid] + hist
                            color_counts[tid] += 1
                        # Occasional jersey OCR when box is large enough
                        if frames_processed % ocr_every == 0 and (xyxy[2] - xyxy[0]) >= 36:
                            from app.services.jersey_ocr import read_jersey_number

                            hit = read_jersey_number(frame, xyxy)
                            if hit:
                                jersey_votes[tid].append((int(hit["number"]), float(hit["confidence"])))
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
            series_obs = _downsample_series(
                obs, MAX_BALL_SERIES if class_name == "sports_ball" else MAX_PERSON_SERIES
            )
            feats: dict[str, Any] = {"class": class_name}
            if class_name == "person" and tid in color_sums and color_counts[tid]:
                mean_hist = (color_sums[tid] / max(1, color_counts[tid])).astype(float)
                # L2 normalize for cosine similarity later
                norm = float(np.linalg.norm(mean_hist)) or 1.0
                feats["color_hist"] = (mean_hist / norm).tolist()
                feats["color_samples"] = color_counts[tid]
            if class_name == "person" and jersey_votes.get(tid):
                number, jconf = _majority_jersey(jersey_votes[tid])
                if number is not None:
                    feats["jersey_number"] = number
                    feats["jersey_confidence"] = jconf
                    feats["jersey_votes"] = len(jersey_votes[tid])

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
                        for o in series_obs
                    ],
                    class_name=class_name,
                    features=feats,
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
            "person_tracks": sum(1 for d in detections if d.class_name == "person"),
            "ball_tracks": sum(1 for d in detections if d.class_name == "sports_ball"),
            "ball_detections": ball_hits,
            "ball_visible": ball_hits > 0,
            "player_visible": any(d.class_name == "person" for d in detections),
            "video_seconds_probed": round(video_seconds, 2),
            "event_engine": "possession_v1",
            "jersey_ocr_tracks": sum(1 for d in detections if (d.features or {}).get("jersey_number")),
        }
        logger.info("YOLO track done: %s", self.last_run_stats)
        return detections

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import infer_event_candidates

        return infer_event_candidates(detections, camera_label)


def _majority_jersey(votes: list[tuple[int, float]]) -> tuple[int | None, float]:
    if not votes:
        return None, 0.0
    scores: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for num, conf in votes:
        scores[num] += conf
        counts[num] += 1
    best = max(scores.keys(), key=lambda n: (scores[n], counts[n]))
    return best, round(scores[best] / max(1, counts[best]), 3)


def _downsample_series(obs: list[dict[str, Any]], max_n: int) -> list[dict[str, Any]]:
    if len(obs) <= max_n:
        return obs
    step = max(1, len(obs) // max_n)
    sampled = obs[::step]
    if sampled[-1] is not obs[-1]:
        sampled = sampled + [obs[-1]]
    return sampled[:max_n]


def _upper_body_hsv_hist(frame, xyxy: list[float]):
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    # jersey-ish upper half of the person box
    y_mid = y1 + max(1, (y2 - y1) // 2)
    crop = frame[y1:y_mid, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    return hist.flatten().astype(np.float32)


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
