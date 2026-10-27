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
        # FP16 measured on Modal L4 (SoccerTrack MOT): +8–18 % fps but stitched IDF1
        # 0.579 → 0.562 / 0.569 → 0.551 and more ID switches — accuracy wins, stay FP32.
        self.half = False
        self.frame_stride = max(1, settings.vision_frame_stride)
        self.target_fps = float(getattr(settings, "vision_target_fps", 0) or 0)
        self.team_embed_on = bool(getattr(settings, "vision_team_embed", True))
        self.conf = settings.vision_conf
        self.imgsz = settings.vision_imgsz
        self.imgsz_max = int(getattr(settings, "vision_imgsz_max", 1920) or 1920)
        self.max_frames = settings.vision_max_frames  # 0 = full video
        self.tracker_name = (settings.vision_tracker or "botsort").strip().lower()
        self.reid_on = bool(getattr(settings, "vision_reid", True))
        self.reid_every = max(1, int(getattr(settings, "vision_reid_every", 3) or 3))
        self._model = None
        self._coco_ball_model = None
        self._ball_model = None
        self.ball_model_path = (getattr(settings, "vision_ball_model", "") or "").strip()
        self._ball_class_ids: list[int] = []
        # camera → learned spotter output (action_spotting.py); empty = heuristic events
        self.spotted: dict[str, list[dict[str, Any]]] = {}
        # Benchmarks set a list here to receive every processed frame's person boxes
        # (frame_idx, track_id, x1, y1, x2, y2, conf) — bbox_series is downsampled.
        self.frame_log: list[tuple] | None = None
        self._class_map: dict[int, str] | None = None
        self.last_run_stats: dict[str, Any] = {}

    def _tracker_cfg(self, static_camera: bool = False) -> str:
        name = self.tracker_name if self.tracker_name in {"botsort", "bytetrack"} else "botsort"
        gmc = (getattr(get_settings(), "vision_gmc", "auto") or "auto").strip().lower()
        if name == "botsort" and (gmc == "off" or (gmc == "auto" and static_camera)):
            name = "botsort_static"
        path = _CFG_DIR / f"{name}.yaml"
        if path.is_file():
            return str(path)
        # Fall back to Ultralytics built-in name
        return f"{name}.yaml"

    def _football_profile(self, video_path: str) -> int:
        """Football on broadcast-size frames: people from the football fine-tune (it has a referee
        class) at its own input size; the ball pass keeps the general model. Returns that input
        size, or 0 when the profile does not apply (futsal, panoramas, weights missing)."""
        from app.services.action_spotting import is_football

        settings = get_settings()
        if not is_football() or self._model is not None:
            return 0
        import cv2

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        cap.release()
        weights = resolve_yolo_weights(settings.yolo_model_football)
        if not width or width > settings.vision_football_max_w or not Path(weights).is_file():
            return 0
        if not self.ball_model_path:
            self.ball_model_path = self.model_name  # the general model still finds the ball best
        self.model_name = weights
        return settings.vision_imgsz_football

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

    def _load_ball_model(self):
        """Separate instance of the primary weights for the hi-res ball pass.

        Calling .predict on the tracking model shares its predictor: the ball-only
        results then feed the person tracker's update callback and every person
        track is marked lost on those frames.
        """
        if self._ball_model is None:
            from ultralytics import YOLO

            if self.ball_model_path:
                weights = resolve_yolo_weights(self.ball_model_path)
                logger.info("Loading dedicated ball model %s on %s", weights, self.device)
                self._ball_model = YOLO(weights)
                self._ball_class_ids = [
                    cid for cid, name in _build_class_map(self._ball_model).items() if name == "sports_ball"
                ]
            else:
                self._load()  # resolves self.model_name to a weights path
                self._ball_model = YOLO(self.model_name)
        return self._ball_model

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

        football_imgsz = self._football_profile(video_path)
        model = self._load()
        reset_tracker(model)
        class_map: dict[int, str] = self._class_map or {
            PERSON: "person",
            SPORTS_BALL: "sports_ball",
        }
        track_classes = sorted(class_map.keys())
        is_coco = PERSON in class_map and SPORTS_BALL in class_map and class_map.get(PERSON) == "person"
        # Futsal fine-tunes often can't see the ball — keep a COCO sports_ball retry.
        need_coco_ball = not is_coco
        dedicated_ball = bool(self.ball_model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        full_imgsz = auto_imgsz(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0), self.imgsz_max
        )
        imgsz = football_imgsz or self.imgsz or full_imgsz
        if self.target_fps > 0:
            # Same temporal density on 25 and 60 fps sources (fast players need ~10 Hz to keep IDs)
            self.frame_stride = max(1, int(round(fps / self.target_fps)))
        team_crops: dict[int, list[Any]] = defaultdict(list)
        team_crop_frame: dict[int, int] = {}

        person_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        color_sums: dict[int, Any] = {}
        color_counts: dict[int, int] = defaultdict(int)
        reid_embs: dict[int, Any] = {}
        jersey_hsv_sums: dict[int, list[dict]] = defaultdict(list)
        jersey_votes: dict[int, list[tuple[int, float]]] = defaultdict(list)
        best_thumbs: dict[int, tuple[float, str]] = {}
        ball_hits = 0
        # (ts_ms, x1, y1, x2, y2, conf, frame_h, frame_w) — every ball candidate; the match
        # ball is chosen after the loop (ball_select: static spares out, trajectory first)
        ball_cands: list[tuple] = []
        frames_processed = 0
        person_box_total = 0
        person_box_with_id = 0
        t0 = time.perf_counter()
        frame_idx = 0
        from app.services.jersey_ocr import jersey_reader

        read_jersey_number = jersey_reader()
        ocr_on = read_jersey_number is not None
        # Trained classifier is cheap: more reads per frame and per track for a real vote;
        # Tesseract (opt-in only) keeps the old throttle.
        fast_reader = getattr(read_jersey_number, "__name__", "") == "read_jersey_number_cls"
        ocr_every = 5 if fast_reader else max(90, 450 // max(1, self.frame_stride))
        ocr_per_frame = 6 if fast_reader else 1
        ocr_per_track = 8 if fast_reader else 2
        ball_every = 1
        static_camera = camera_is_static(video_path)
        tracker_cfg = self._tracker_cfg(static_camera)
        # Per-job dir when configured: a shared temp dir lets later videos on the
        # same machine overwrite earlier crops that kit assignment still reads.
        thumb_root = Path(get_settings().vision_thumb_dir or Path(tempfile.gettempdir()) / "courtlens_thumbs")
        thumb_dir = thumb_root / camera_label
        thumb_dir.mkdir(parents=True, exist_ok=True)

        # Seconds per stage (decode / tracker / ball pass / ReID / crop models) → last_run_stats
        timing: dict[str, float] = defaultdict(float)
        try:
            while True:
                t_read = time.perf_counter()
                if frame_idx % self.frame_stride != 0:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    timing["decode"] += time.perf_counter() - t_read
                    continue
                ok, frame = cap.read()
                timing["decode"] += time.perf_counter() - t_read
                if not ok:
                    break
                if self.max_frames and frames_processed >= self.max_frames:
                    break

                t_track = time.perf_counter()
                results = model.track(
                    frame,
                    persist=True,
                    conf=self.conf,
                    imgsz=imgsz,
                    device=self.device,
                    classes=track_classes,
                    tracker=tracker_cfg,
                    half=self.half,
                    verbose=False,
                )
                timing["track"] += time.perf_counter() - t_track
                frames_processed += 1
                ts_ms = int((frame_idx / fps) * 1000)
                result = results[0]
                frame_h = float(frame.shape[0])
                frame_w = float(frame.shape[1])
                ball_seen_this_frame = False
                ocr_budget = ocr_per_frame if (ocr_on and frames_processed % ocr_every == 0) else 0
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
                            if dedicated_ball:
                                continue  # the dedicated ball pass below replaces these
                            # Balls often lack tracker IDs — still keep the observation.
                            ball_hits += 1
                            ball_seen_this_frame = True
                            ball_cands.append((ts_ms, *xyxy, conf, frame_h, frame_w))
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
                            if thumb_score > best_thumbs.get(tid, (0.0, ""))[0] and bw >= THUMB_MIN_W and bh >= THUMB_MIN_H:
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
                        if self.frame_log is not None:
                            self.frame_log.append((frame_idx, tid, *xyxy, conf))
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
                        if self.team_embed_on:
                            _sample_team_crop(frame, xyxy, tid, frames_processed, team_crops, team_crop_frame)
                        # Best in-match crop for avatar (prefer large, confident box)
                        bw = max(1.0, xyxy[2] - xyxy[0])
                        bh = max(1.0, xyxy[3] - xyxy[1])
                        thumb_score = bw * bh * conf
                        if thumb_score > best_thumbs.get(tid, (0.0, ""))[0] and bw >= THUMB_MIN_W and bh >= THUMB_MIN_H:
                            x1 = max(0, int(xyxy[0]))
                            y1 = max(0, int(xyxy[1]))
                            x2 = min(frame.shape[1], int(xyxy[2]))
                            y2 = min(frame.shape[0], int(xyxy[3]))
                            crop = frame[y1:y2, x1:x2]
                            if crop.size:
                                path = str(thumb_dir / f"p{tid}.jpg")
                                cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                                best_thumbs[tid] = (thumb_score, path)
                        if ocr_budget > 0 and bw >= 36 and len(jersey_votes.get(tid, [])) < ocr_per_track:
                            hit = read_jersey_number(frame, xyxy)
                            ocr_budget -= 1
                            if hit:
                                jersey_votes[tid].append((int(hit["number"]), float(hit["confidence"])))

                    if do_reid and reid_boxes:
                        try:
                            from app.services.reid import embed_frame_boxes, ema_update

                            t_reid = time.perf_counter()
                            embs = embed_frame_boxes(frame, reid_boxes, device=self.device)
                            timing["reid"] += time.perf_counter() - t_reid
                            for tid, emb in zip(reid_tids, embs):
                                if emb is None:
                                    continue
                                reid_embs[tid] = ema_update(reid_embs.get(tid), emb)
                        except Exception as e:
                            logger.warning("ReID embed failed: %s — ReID off for the rest of this video", e)
                            self.reid_on = False

                # Futsal balls are tiny on wide CCTV — sparse second pass (not every frame)
                # Ball pass: always when no ball was seen; on frames too wide for the input
                # (tiled, native resolution) also every 2nd frame — the primary may only
                # see a spare ball there. On frames that fit, a second pass at the same
                # resolution adds nothing but a lower threshold, so skip it (CPU cost).
                ball_imgsz = max(full_imgsz if football_imgsz else imgsz, 960)
                tiled = len(ball_tiles(frame.shape[1], frame.shape[0], ball_imgsz)) > 1
                if (not ball_seen_this_frame or (tiled and frames_processed % 2 == 0)) and frames_processed % ball_every == 0:
                    ball_conf = min(self.conf, 0.12)
                    ball_cls_ids = [cid for cid, name in class_map.items() if name == "sports_ball"]
                    ball_model = self._load_ball_model()
                    if dedicated_ball:
                        ball_cls_ids = self._ball_class_ids
                    elif need_coco_ball:
                        ball_model = self._load_coco_ball()
                        ball_cls_ids = [SPORTS_BALL]
                    if ball_cls_ids:
                        # Frames wider than the detector input are shrunk and a ~10 px ball vanishes:
                        # tile at native resolution instead (SAHI-style), offsets mapped back.
                        tiles = ball_tiles(frame.shape[1], frame.shape[0], ball_imgsz)
                        t_ball = time.perf_counter()
                        timing["ball_passes"] += 1
                        ball_results = ball_model.predict(
                            [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles] if len(tiles) > 1 else frame,
                            conf=ball_conf,
                            imgsz=ball_imgsz,
                            device=self.device,
                            classes=ball_cls_ids,
                            half=self.half,
                            verbose=False,
                        )
                        timing["ball"] += time.perf_counter() - t_ball
                        found = []
                        for (x0, y0, _x1, _y1), res in zip(tiles, ball_results):
                            bb = res.boxes
                            if bb is None or not len(bb):
                                continue
                            for j in bb.conf.argsort(descending=True)[:BALL_CANDIDATES_PER_FRAME].tolist():
                                bx1, by1, bx2, by2 = bb.xyxy[j].tolist()
                                found.append((float(bb.conf[j].item()), [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]))
                        for conf_b, box in sorted(found, key=lambda t: -t[0])[:BALL_CANDIDATES_PER_FRAME]:
                            ball_cands.append((ts_ms, *box, conf_b, frame_h, frame_w))
                        if found:
                            ball_hits += 1

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
        from app.services.ball_select import select_ball
        from app.services.ball_select import summary as ball_summary

        chosen_ball = select_ball([c[:6] for c in ball_cands], frames_processed)
        dims = {c[0]: (c[6], c[7]) for c in ball_cands}
        if chosen_ball:
            person_tracks[100000] = [
                _obs(c[0], c[5], list(c[1:5]), dims[c[0]][0], dims[c[0]][1], "sports_ball") for c in chosen_ball
            ]
        t_crops = time.perf_counter()
        team_embs = _embed_team_crops(team_crops, self.device) if self.team_embed_on else {}
        crop_sem = _crop_semantics(best_thumbs, person_tracks, self.device)
        timing["crop_models"] += time.perf_counter() - t_crops
        detections: list[Detection] = []
        for tid, obs in person_tracks.items():
            if not obs:
                continue
            class_name = track_class(obs)
            if class_name == "sports_ball":
                prefix, key_id = "b", tid - 100000
            elif class_name == "referee":
                prefix, key_id = "r", tid
            else:
                prefix, key_id = "p", tid
            series_obs = _downsample_series(
                obs, MAX_BALL_SERIES if class_name == "sports_ball" else MAX_PERSON_SERIES
            )
            feats: dict[str, Any] = {"class": class_name}
            if class_name == "sports_ball":
                # Raw candidates (ts, x1, y1, x2, y2, conf) so ball selection can be re-tuned offline
                feats["ball_candidates"] = [[int(c[0]), *(round(float(v), 1) for v in c[1:5]), round(float(c[5]), 3)]
                                            for c in ball_cands]
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
            if class_name == "person" and tid in team_embs:
                feats.update(team_embs[tid])
            if class_name == "person" and tid in crop_sem:
                feats.update(crop_sem[tid])  # role_score + colour_profile (identity: bystanders, kits)
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
                    bbox_series=_flat_series(series_obs, interpolate=class_name == "sports_ball"),
                    class_name=class_name,
                    features=feats,
                )
            )

        video_seconds = (total_frames / fps) if total_frames and fps else 0
        from app.services.court_format import court_settings

        person_dets = [d for d in detections if d.class_name == "person"]
        person_pts = sum(len(d.bbox_series or []) for d in person_dets)
        # Share of processed frames with a selected match ball (was candidates / frames, >100 %)
        ball_coverage_pct = round(100.0 * len(chosen_ball) / max(1, frames_processed), 2)
        id_rate = (
            round(100.0 * person_box_with_id / max(1, person_box_total), 2) if person_box_total else 0.0
        )
        self.last_run_stats = {
            "camera_label": camera_label,
            "model": self.model_name,
            "device": self.device,
            "tracker": self.tracker_name,
            "imgsz": imgsz,
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
            "static_camera": static_camera,
            "tracker_cfg_name": Path(tracker_cfg).stem,
            "stage_seconds": {k: round(v, 1) for k, v in timing.items()},
            "realtime_factor": round(video_seconds / elapsed, 3) if elapsed and video_seconds else None,
            "person_tracks": len(person_dets),
            "referee_tracks": sum(1 for d in detections if d.class_name == "referee"),
            "person_bbox_samples": person_pts,
            "ball_tracks": sum(1 for d in detections if d.class_name == "sports_ball"),
            "ball_detections": ball_hits,
            **ball_summary([c[:6] for c in ball_cands], chosen_ball, frames_processed),
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

        from app.services import action_spotting

        if action_spotting.available(self.device):
            t_spot = time.perf_counter()
            self.spotted[camera_label] = action_spotting.spot(video_path, detections, _cuda(self.device))
            self.last_run_stats["spotted_events"] = len(self.spotted[camera_label])
            self.last_run_stats["spotted"] = self.spotted[camera_label]  # raw (≥0.05) → summary.json for tuning
            self.last_run_stats.setdefault("stage_seconds", {})["spotting"] = round(time.perf_counter() - t_spot, 1)
        return detections

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import events_from_spotting, infer_event_candidates

        if self.spotted.get(camera_label):
            return events_from_spotting(self.spotted[camera_label], detections, camera_label)
        return infer_event_candidates(detections, camera_label)


def _cuda(device: str) -> str:
    from app.services.reid import torch_device

    return torch_device(device)


REFEREE_TRACK_SHARE = 0.2  # of a track's frames called referee ⇒ the track is a referee


def track_class(obs: list[dict]) -> str:
    """A track's class from all its frames, not just the first. The football detector misses
    refs often (47.5 % of referee boxes called referee on SoccerNet GSR) but almost never calls a
    player one (0.1 %), so a minority of referee frames is enough."""
    first = obs[0].get("class_name", "person")
    if first == "sports_ball":
        return first
    refs = sum(1 for o in obs if o.get("class_name") == "referee")
    return "referee" if refs >= 2 and refs >= REFEREE_TRACK_SHARE * len(obs) else "person"


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


BALL_CANDIDATES_PER_FRAME = 5
TEAM_CROPS_PER_TRACK = 6
TEAM_CROP_GAP = 8  # processed frames between samples of one track
TEAM_CROP_MIN_H = 32  # px — smaller crops carry no kit pattern


def _sample_team_crop(frame, xyxy, tid: int, frame_no: int, crops: dict, last: dict) -> None:
    """Keep up to TEAM_CROPS_PER_TRACK crops per track, spread over its lifetime."""
    import cv2

    if len(crops[tid]) >= TEAM_CROPS_PER_TRACK or frame_no - last.get(tid, -TEAM_CROP_GAP) < TEAM_CROP_GAP:
        return
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if y2 - y1 < TEAM_CROP_MIN_H or x2 - x1 < 12:
        return
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] > 160:  # bound memory; SigLIP resizes to 224 anyway
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * 160 / crop.shape[0])), 160))
    crops[tid].append(crop.copy())
    last[tid] = frame_no


def _torso(crop):
    h, w = crop.shape[:2]
    return crop[int(0.15 * h) : max(int(0.15 * h) + 1, int(0.55 * h)), int(0.2 * w) : max(int(0.2 * w) + 1, int(0.8 * w))]


def _embed_team_crops(crops: dict[int, list[Any]], device: str) -> dict[int, dict[str, list[float]]]:
    """Per track: median torso jersey colour (Lab) and mean SigLIP torso embedding.

    Colour is the primary team cue — on blurry CCTV crops whole-body SigLIP
    clustered on background/blur (7/12 correct) vs torso colour 11/12.
    """
    import numpy as np

    from app.services.team_assigner import jersey_colour_from_crop
    from app.services.team_embed import embed_crops

    out: dict[int, dict[str, list[float]]] = {}
    for tid, cs in crops.items():
        cols = [o[0] for c in cs if (o := jersey_colour_from_crop(c)) is not None]
        if cols:
            out.setdefault(tid, {})["team_colour"] = np.round(np.median(np.stack(cols), axis=0), 2).tolist()
    order = [(tid, c) for tid, cs in crops.items() for c in cs]
    embs = embed_crops([_torso(c) for _, c in order], device=device) if order else None
    if embs is not None:
        idx: dict[int, list[int]] = {}
        for i, (tid, _) in enumerate(order):
            idx.setdefault(tid, []).append(i)
        for tid, rows in idx.items():
            m = embs[rows].mean(axis=0)
            out.setdefault(tid, {})["team_emb"] = np.round(m / (np.linalg.norm(m) + 1e-12), 4).tolist()
    return out


def _crop_semantics(best_thumbs: dict, person_tracks: dict, device: str) -> dict[int, dict]:
    """Zero-shot role score + kit colour profile per person track from its best thumbnail (role_filter.py)."""
    import cv2

    from app.core.config import get_settings

    if not getattr(get_settings(), "vision_role_filter", True):
        return {}
    tids = [tid for tid in best_thumbs if person_tracks.get(tid) and track_class(person_tracks[tid]) == "person"]
    imgs = [cv2.imread(str(best_thumbs[tid][1])) for tid in tids]
    from app.services.role_filter import crop_semantics

    out = {}
    for tid, img, sem in zip(tids, imgs, crop_semantics(imgs, device)):
        if not sem:
            continue
        if img is None or img.shape[0] < ROLE_MIN_H:
            sem = {k: v for k, v in sem.items() if k != "role_score"}
        out[tid] = sem
    return out


# Smallest box worth a crop (kit colour, avatar). 28 x 40 suited futsal; football panorama players
# are ~16-28 px wide (SoccerTrack GT 21x54, 16x44), and at 28 px 61 % of their tracks got no crop,
# hence no kit — identity linked 14 of 756 tracks.
THUMB_MIN_W, THUMB_MIN_H = 12, 30
# Below this crop height the zero-shot player-vs-bystander score is noise (mean 0.01 on the
# panorama): keep the colour profile, drop the role score so the filter never acts on it.
ROLE_MIN_H = 60


STATIC_SAMPLES = 30
STATIC_MAX_SHIFT = 1.0  # px at 320 px width (90th percentile frame-to-frame shift)


def camera_is_static(video_path: str) -> bool:
    """True when the view does not pan / zoom: phase-correlation shift between frame pairs
    sampled across the video stays under STATIC_MAX_SHIFT (players moving barely shift it)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if n < 10:
        cap.release()
        return False
    gap = max(1, int(round(fps / 5)))  # 0.2 s apart
    shifts = []
    for start in np.linspace(0, max(0, n - gap - 1), STATIC_SAMPLES).astype(int):
        pair = []
        for f in (start, start + gap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (320, max(1, int(320 * g.shape[0] / g.shape[1]))))
            pair.append(np.float32(g))
        if len(pair) == 2:
            (dx, dy), _ = cv2.phaseCorrelate(pair[0], pair[1])
            shifts.append(float(np.hypot(dx, dy)))
    cap.release()
    return bool(shifts) and float(np.percentile(shifts, 90)) < STATIC_MAX_SHIFT


def ball_tiles(width: int, height: int, tile: int, overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    """Native-resolution tiles (x0, y0, x1, y1) covering the frame; one tile if it already fits."""
    if width <= tile * 1.1 and height <= tile * 1.1:
        return [(0, 0, width, height)]

    def spans(n: int) -> list[tuple[int, int]]:
        if n <= tile:
            return [(0, n)]
        step = int(tile * (1 - overlap))
        starts = list(range(0, n - tile, step)) + [n - tile]
        return [(s, s + tile) for s in sorted(set(starts))]

    return [(x0, y0, x1, y1) for y0, y1 in spans(height) for x0, x1 in spans(width)]


def auto_imgsz(width: int, height: int, cap: int = 1920) -> int:
    """Detector input size: never shrink below the frame's long side, up to `cap`.

    Wide fixed cameras make players tiny (16–28 px on a 4096 px panorama); the
    old fixed 640 shrank them to ~4 px and missed 84 % of them.
    """
    long_side = max(width, height)
    if long_side <= 0:
        return 640
    return int(max(640, min(cap, (long_side + 31) // 32 * 32)))


def reset_tracker(model) -> None:
    """Drop Ultralytics tracker state so a new video starts with fresh track IDs.

    `model.track(persist=True)` keeps trackers on the cached predictor, so camera B
    (or the next clip) was being associated against camera A's last frame.
    """
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        del predictor.trackers


def _flat_series(series_obs: list[dict[str, Any]], *, interpolate: bool) -> list[dict[str, Any]]:
    """Public bbox_series shape (flat x1..y2), with ball gaps filled when asked."""
    flat = [
        {
            "ts_ms": o["ts_ms"],
            **o["bbox"],
            "confidence": o["confidence"],
            "frame_h": o.get("frame_h"),
            "frame_w": o.get("frame_w"),
        }
        for o in series_obs
    ]
    if interpolate:
        # Fill short gaps in ball track (football_analysis interpolate) — needs flat boxes
        from app.services.possession import interpolate_bbox_series

        flat = interpolate_bbox_series(flat, max_gap_ms=500)
    return flat


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
