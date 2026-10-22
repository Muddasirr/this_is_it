"""Pretrained YOLO detection + BoT-SORT (default) / ByteTrack + OSNet ReID."""

from __future__ import annotations

import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.config import get_settings, resolve_yolo_weights
from app.services.vision import Detection, EventCandidate, VisionBackend

logger = logging.getLogger(__name__)

# COCO class ids
PERSON = 0
SPORTS_BALL = 32

# Keep dense enough series for possession events without huge JSON rows
MAX_PERSON_SERIES = 400
MAX_BALL_SERIES = 2000

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


class YoloTrackBackend(VisionBackend):
    name = "yolov5x-football-botsort-osnet"
    version = "0.5.0"

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = resolve_yolo_weights(settings.yolo_model)
        self.device = settings.vision_device
        self.frame_stride = max(1, settings.vision_frame_stride)
        self.conf = settings.vision_conf
        self.imgsz = settings.vision_imgsz
        self.max_frames = settings.vision_max_frames  # 0 = full video
        self.tracker_name = (settings.vision_tracker or "botsort").strip().lower()
        self.reid_on = bool(getattr(settings, "vision_reid", True))
        self.reid_every = max(1, int(getattr(settings, "vision_reid_every", 3) or 3))
        self._model = None
        self._coco_ball_model = None
        self._class_map: dict[int, str] | None = None
        self.last_run_stats: dict[str, Any] = {}

    def _tracker_cfg(self) -> str:
        name = self.tracker_name if self.tracker_name in {"botsort", "bytetrack"} else "botsort"
        path = _CFG_DIR / f"{name}.yaml"
        if path.is_file():
            return str(path)
        # Fall back to Ultralytics built-in name
        return f"{name}.yaml"

    def _load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        weights = resolve_yolo_weights(self.model_name)
        self.model_name = weights
        logger.info("Loading YOLO model %s on %s", weights, self.device)
        self._model = YOLO(weights)
        self._class_map = _build_class_map(self._model)
        return self._model

    def _load_coco_ball(self):
        """COCO yolov8n for tiny-ball retry when primary is a futsal fine-tune."""
        if self._coco_ball_model is not None:
            return self._coco_ball_model
        from ultralytics import YOLO

        coco = resolve_yolo_weights("yolov8n.pt")
        logger.info("Loading COCO ball fallback %s on %s", coco, self.device)
        self._coco_ball_model = YOLO(coco)
        return self._coco_ball_model

    def detect_and_track(self, video_path: str, camera_label: str) -> list[Detection]:
        import cv2
        import numpy as np

        model = self._load()
        class_map: dict[int, str] = self._class_map or {
            PERSON: "person",
            SPORTS_BALL: "sports_ball",
        }
        track_classes = sorted(class_map.keys())
        is_coco = PERSON in class_map and SPORTS_BALL in class_map and class_map.get(PERSON) == "person"
        # Futsal fine-tunes often can't see the ball — keep a COCO sports_ball retry.
        need_coco_ball = not is_coco
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        person_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        color_sums: dict[int, Any] = {}
        color_counts: dict[int, int] = defaultdict(int)
        reid_embs: dict[int, Any] = {}
        jersey_hsv_sums: dict[int, list[dict]] = defaultdict(list)
        jersey_votes: dict[int, list[tuple[int, float]]] = defaultdict(list)
        best_thumbs: dict[int, tuple[float, str]] = {}
        ball_hits = 0
        frames_processed = 0
        person_box_total = 0
        person_box_with_id = 0
        t0 = time.perf_counter()
        frame_idx = 0
        from app.services.jersey_ocr import jersey_ocr_enabled, read_jersey_number

        ocr_on = jersey_ocr_enabled()
        ocr_every = max(90, 450 // max(1, self.frame_stride))
        ball_every = 1
        tracker_cfg = self._tracker_cfg()
        # Per-job dir when configured: a shared temp dir lets later videos on the
        # same machine overwrite earlier crops that kit assignment still reads.
        thumb_root = Path(get_settings().vision_thumb_dir or Path(tempfile.gettempdir()) / "courtlens_thumbs")
        thumb_dir = thumb_root / camera_label
        thumb_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                if frame_idx % self.frame_stride != 0:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                if self.max_frames and frames_processed >= self.max_frames:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    classes=track_classes,
                    tracker=tracker_cfg,
                    verbose=False,
                )
                frames_processed += 1
                ts_ms = int((frame_idx / fps) * 1000)
                result = results[0]
                frame_h = float(frame.shape[0])
                frame_w = float(frame.shape[1])
                ball_seen_this_frame = False
                ocr_budget = 1 if (ocr_on and frames_processed % ocr_every == 0) else 0
                do_reid = self.reid_on and (frames_processed % self.reid_every == 0)
                reid_boxes: list[list[float]] = []
                reid_tids: list[int] = []

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    ids = boxes.id
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].tolist()
                        tid = _safe_track_id(ids, i)

                        mapped = class_map.get(cls_id)
                        if mapped == "sports_ball":
                            # Balls often lack tracker IDs — still keep the observation.
                            ball_hits += 1
                            ball_seen_this_frame = True
                            person_tracks[100000].append(
                                _obs(ts_ms, conf, xyxy, frame_h, frame_w, "sports_ball")
                            )
                            continue

                        if mapped == "referee":
                            if tid is None:
                                continue
                            person_tracks[tid].append(
                                _obs(ts_ms, conf, xyxy, frame_h, frame_w, "referee")
                            )
                            # thumb for overlay / Kit X — no OCR/kit clustering
                            bw = max(1.0, xyxy[2] - xyxy[0])
                            bh = max(1.0, xyxy[3] - xyxy[1])
                            thumb_score = bw * bh * conf
                            if thumb_score > best_thumbs.get(tid, (0.0, ""))[0] and bw >= 28 and bh >= 40:
                                x1 = max(0, int(xyxy[0]))
                                y1 = max(0, int(xyxy[1]))
                                x2 = min(frame.shape[1], int(xyxy[2]))
                                y2 = min(frame.shape[0], int(xyxy[3]))
                                crop = frame[y1:y2, x1:x2]
                                if crop.size:
                                    path = str(thumb_dir / f"r{tid}.jpg")
                                    cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                                    best_thumbs[tid] = (thumb_score, path)
                            continue

                        if mapped != "person":
                            continue
                        person_box_total += 1
                        if tid is None:
                            continue
                        person_box_with_id += 1

                        obs = _obs(ts_ms, conf, xyxy, frame_h, frame_w, "person")
                        person_tracks[tid].append(obs)
                        hist = _upper_body_hsv_hist(frame, xyxy)
                        if hist is not None:
                            if tid not in color_sums:
                                color_sums[tid] = hist
                            else:
                                color_sums[tid] = color_sums[tid] + hist
                            color_counts[tid] += 1
                        jhsv = _jersey_hsv_for_box(frame, xyxy)
                        if jhsv is not None:
                            jersey_hsv_sums[tid].append(jhsv)
                        if do_reid:
                            reid_boxes.append(xyxy)
                            reid_tids.append(tid)
                        # Best in-match crop for avatar (prefer large, confident box)
                        bw = max(1.0, xyxy[2] - xyxy[0])
                        bh = max(1.0, xyxy[3] - xyxy[1])
                        thumb_score = bw * bh * conf
                        if thumb_score > best_thumbs.get(tid, (0.0, ""))[0] and bw >= 28 and bh >= 40:
                            x1 = max(0, int(xyxy[0]))
                            y1 = max(0, int(xyxy[1]))
                            x2 = min(frame.shape[1], int(xyxy[2]))
                            y2 = min(frame.shape[0], int(xyxy[3]))
                            crop = frame[y1:y2, x1:x2]
                            if crop.size:
                                path = str(thumb_dir / f"p{tid}.jpg")
                                cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                                best_thumbs[tid] = (thumb_score, path)
                        if ocr_budget > 0 and bw >= 36 and len(jersey_votes.get(tid, [])) < 2:
                            hit = read_jersey_number(frame, xyxy)
                            ocr_budget -= 1
                            if hit:
                                jersey_votes[tid].append((int(hit["number"]), float(hit["confidence"])))

                    if do_reid and reid_boxes:
                        try:
                            from app.services.reid import embed_frame_boxes, ema_update

                            embs = embed_frame_boxes(frame, reid_boxes, device=self.device)
                            for tid, emb in zip(reid_tids, embs):
                                if emb is None:
                                    continue
                                reid_embs[tid] = ema_update(reid_embs.get(tid), emb)
                        except Exception as e:
                            logger.warning("ReID embed failed: %s", e)

                # Futsal balls are tiny on wide CCTV — sparse second pass (not every frame)
                if not ball_seen_this_frame and frames_processed % ball_every == 0:
                    ball_imgsz = max(self.imgsz, 960)
                    ball_conf = min(self.conf, 0.12)
                    ball_cls_ids = [cid for cid, name in class_map.items() if name == "sports_ball"]
                    ball_model = model
                    if need_coco_ball:
                        ball_model = self._load_coco_ball()
                        ball_cls_ids = [SPORTS_BALL]
                    if ball_cls_ids:
                        ball_results = ball_model.predict(
                            frame,
                            conf=ball_conf,
                            imgsz=ball_imgsz,
                            device=self.device,
                            classes=ball_cls_ids,
                            verbose=False,
                        )
                        bboxes = ball_results[0].boxes
                        if bboxes is not None and len(bboxes) > 0:
                            best_i = int(bboxes.conf.argmax().item())
                            conf = float(bboxes.conf[best_i].item())
                            xyxy = bboxes.xyxy[best_i].tolist()
                            ball_hits += 1
                            person_tracks[100000].append(
                                _obs(ts_ms, conf, xyxy, frame_h, frame_w, "sports_ball")
                            )

                if frames_processed % 50 == 0:
                    elapsed_so_far = time.perf_counter() - t0
                    pps = frames_processed / elapsed_so_far if elapsed_so_far else 0
                    logger.info(
                        "YOLO progress frames=%s ball_hits=%s %.1f fps elapsed=%.0fs",
                        frames_processed,
                        ball_hits,
                        pps,
                        elapsed_so_far,
                    )

                frame_idx += 1
        finally:
            cap.release()

        elapsed = time.perf_counter() - t0
        detections: list[Detection] = []
        for tid, obs in person_tracks.items():
            if not obs:
                continue
            class_name = obs[0].get("class_name", "person")
            if class_name == "sports_ball":
                prefix, key_id = "b", tid - 100000
            elif class_name == "referee":
                prefix, key_id = "r", tid
            else:
                prefix, key_id = "p", tid
            series_obs = _downsample_series(
                obs, MAX_BALL_SERIES if class_name == "sports_ball" else MAX_PERSON_SERIES
            )
            # Fill short gaps in ball track (football_analysis interpolate)
            if class_name == "sports_ball":
                from app.services.possession import interpolate_bbox_series

                series_obs = interpolate_bbox_series(series_obs, max_gap_ms=500)
            feats: dict[str, Any] = {"class": class_name}
            if class_name == "referee" and tid in best_thumbs:
                feats["thumb_path"] = best_thumbs[tid][1]
                feats["is_referee"] = True
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
            if class_name == "person" and tid in best_thumbs:
                feats["thumb_path"] = best_thumbs[tid][1]
                feats["thumb_score"] = float(best_thumbs[tid][0])
            if class_name == "person" and tid in reid_embs and reid_embs[tid] is not None:
                feats["reid_emb"] = np.asarray(reid_embs[tid], dtype=float).tolist()
            if class_name == "person" and jersey_hsv_sums.get(tid):
                from app.services.kit_color import mean_descriptor

                jd = mean_descriptor(jersey_hsv_sums[tid])
                if jd:
                    feats["jersey_hsv"] = jd

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
        from app.services.court_format import court_settings

        person_dets = [d for d in detections if d.class_name == "person"]
        person_pts = sum(len(d.bbox_series or []) for d in person_dets)
        ball_coverage_pct = round(100.0 * ball_hits / max(1, frames_processed), 2)
        id_rate = (
            round(100.0 * person_box_with_id / max(1, person_box_total), 2) if person_box_total else 0.0
        )
        self.last_run_stats = {
            "camera_label": camera_label,
            "model": self.model_name,
            "device": self.device,
            "tracker": self.tracker_name,
            "tracker_cfg": tracker_cfg,
            "reid": self.reid_on,
            "reid_tracks": sum(1 for d in person_dets if (d.features or {}).get("reid_emb")),
            "person_boxes": person_box_total,
            "person_boxes_with_id": person_box_with_id,
            "id_assignment_pct": id_rate,
            "fps_source": round(fps, 3),
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "frame_stride": self.frame_stride,
            "elapsed_sec": round(elapsed, 3),
            "processing_fps": round(frames_processed / elapsed, 3) if elapsed else 0,
            "realtime_factor": round(video_seconds / elapsed, 3) if elapsed and video_seconds else None,
            "person_tracks": len(person_dets),
            "referee_tracks": sum(1 for d in detections if d.class_name == "referee"),
            "person_bbox_samples": person_pts,
            "ball_tracks": sum(1 for d in detections if d.class_name == "sports_ball"),
            "ball_detections": ball_hits,
            "ball_coverage_pct": ball_coverage_pct,
            "ball_visible": ball_hits > 0,
            "player_visible": bool(person_dets),
            "thumbs_written": sum(1 for d in person_dets if (d.features or {}).get("thumb_path")),
            "court_format": court_settings()["format"],
            "video_seconds_probed": round(video_seconds, 2),
            "event_engine": "possession_v1",
            "jersey_ocr_tracks": sum(1 for d in detections if (d.features or {}).get("jersey_number")),
        }
        logger.info("YOLO track done: %s", self.last_run_stats)

        # Futsal fine-tunes often miss facility CCTV (domain shift). Fall back to COCO.
        if (
            not person_dets
            and not getattr(self, "_coco_fallback_attempted", False)
            and Path(str(self.model_name)).name not in {"yolov8n.pt", "yolo11n.pt", "yolov8s.pt"}
        ):
            logger.warning(
                "No players with %s — falling back to yolov8n.pt for this clip",
                self.model_name,
            )
            self._coco_fallback_attempted = True
            primary = self.model_name
            self.model_name = resolve_yolo_weights("yolov8n.pt")
            self._model = None
            self._class_map = None
            try:
                dets = self.detect_and_track(video_path, camera_label)
                self.last_run_stats["primary_model"] = primary
                self.last_run_stats["fallback_model"] = self.model_name
                return dets
            finally:
                self._coco_fallback_attempted = False

        return detections

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import infer_event_candidates

        return infer_event_candidates(detections, camera_label)


def _build_class_map(model) -> dict[int, str]:
    """Map detector class ids → person | sports_ball | referee.

    Football/futsal fine-tunes (player/goalkeeper/ball/referee) keep refs as
    their own class so they never enter Kit A/B. COCO has no referee.
    """
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    out: dict[int, str] = {}
    for cid, name in names.items():
        n = str(name).strip().lower().replace(" ", "_")
        if n in {"person", "player", "goalkeeper", "gk"}:
            out[int(cid)] = "person"
        elif n in {"sports_ball", "sportsball", "ball", "football", "soccer_ball"}:
            out[int(cid)] = "sports_ball"
        elif n in {"referee", "ref", "official"}:
            out[int(cid)] = "referee"
    if not out:
        out = {PERSON: "person", SPORTS_BALL: "sports_ball"}
    logger.info("YOLO class map: %s", {names.get(k, k): v for k, v in out.items()})
    return out


def _majority_jersey(votes: list[tuple[int, float]]) -> tuple[int | None, float]:
    from app.services.jersey_ocr import prefer_jersey_number

    return prefer_jersey_number(votes)


def _safe_track_id(ids, index: int) -> int | None:
    """ByteTrack often leaves sports_ball (and some persons) without an id."""
    if ids is None:
        return None
    try:
        val = ids[index].item()
    except Exception:
        return None
    if val is None:
        return None
    try:
        import math

        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def _obs(
    ts_ms: int,
    conf: float,
    xyxy: list[float],
    frame_h: float,
    frame_w: float,
    class_name: str,
) -> dict[str, Any]:
    return {
        "ts_ms": ts_ms,
        "confidence": conf,
        "bbox": {
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
        },
        "frame_h": frame_h,
        "frame_w": frame_w,
        "class_name": class_name,
    }


def _downsample_series(obs: list[dict[str, Any]], max_n: int) -> list[dict[str, Any]]:
    if len(obs) <= max_n:
        return obs
    step = max(1, len(obs) // max_n)
    sampled = obs[::step]
    if sampled[-1] is not obs[-1]:
        sampled = sampled + [obs[-1]]
    return sampled[:max_n]


def _upper_body_hsv_hist(frame, xyxy: list[float]):
    """Legacy 8×8 H×S hist (compat). Prefer jersey_hsv via kit_color."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    y_mid = y1 + max(1, (y2 - y1) // 2)
    crop = frame[y1:y_mid, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Mask green pitch so grass doesn't own the histogram
    green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    mask = cv2.bitwise_not(green)
    hist = cv2.calcHist([hsv], [0, 1], mask, [8, 8], [0, 180, 0, 256])
    return hist.flatten().astype(np.float32)


def _jersey_hsv_for_box(frame, xyxy: list[float]) -> dict | None:
    from app.services.kit_color import descriptor_from_frame_box

    return descriptor_from_frame_box(frame, xyxy)


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
