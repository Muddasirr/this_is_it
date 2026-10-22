#!/usr/bin/env python3
"""Render track overlay (football_analysis-style) on a short clip.

Uses fine-tuned player/ref/ball detector when present, KMeans jersey teams
(sticky per track id), ellipse feet markers, ball triangle.

  OVERLAY_VIDEO=... OVERLAY_OUT=... python scripts/eval/render_track_overlay.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_BACKEND = Path("/app") if Path("/app/app/services").is_dir() else Path(__file__).resolve().parents[2] / "backend"
_ROOT = _BACKEND.parent if (_BACKEND / "app").is_dir() else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BACKEND))

_BALL = (0, 255, 0)
_REF = (0, 255, 255)
_UNK = (160, 160, 160)
_TEAM_A = (40, 40, 220)  # red-ish BGR default until KMeans fits
_TEAM_B = (220, 180, 40)


def _resolve_weights() -> str:
    env = os.environ.get("OVERLAY_WEIGHTS") or os.environ.get("YOLO_MODEL")
    cands = []
    if env:
        cands.append(Path(env))
        cands.append(_BACKEND / env)
        cands.append(_ROOT / env)
    cands += [
        _BACKEND / "yolov5x_football.pt",
        _ROOT / "yolov5x_football.pt",
        _ROOT / "data/eval/football_analysis_model/best.pt",
        _BACKEND / "yolov8n_football.pt",
        _ROOT / "data/eval/roboflow_football_players_v12/yolov8n_football_v12_best.pt",
        _BACKEND / "yolov8n_futsal.pt",
        _ROOT / "yolov8n_futsal.pt",
        _BACKEND / "yolov8n.pt",
        _ROOT / "yolov8n.pt",
    ]
    for p in cands:
        if p.is_file() and p.stat().st_size > 1000:
            return str(p.resolve())
    return "yolov5x_football.pt"


def _tracker_cfg() -> str:
    name = (os.environ.get("VISION_TRACKER") or "botsort").strip().lower()
    if name not in {"botsort", "bytetrack"}:
        name = "botsort"
    path = _BACKEND / "cfg" / f"{name}.yaml"
    return str(path) if path.is_file() else f"{name}.yaml"


def _class_ids(model) -> dict[str, list[int]]:
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    buckets: dict[str, list[int]] = {"person": [], "ball": [], "referee": []}
    for cid, name in names.items():
        n = str(name).strip().lower().replace(" ", "_")
        if n in {"person", "player", "goalkeeper", "gk"}:
            buckets["person"].append(int(cid))
        elif n in {"sports_ball", "ball", "football", "soccer_ball"}:
            buckets["ball"].append(int(cid))
        elif n in {"referee", "ref", "official"}:
            buckets["referee"].append(int(cid))
    if not buckets["person"] and not buckets["ball"]:
        buckets = {"person": [0], "ball": [32], "referee": []}
    return buckets


def _ellipse(frame, bbox, color, track_id=None):
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    y2i = int(y2)
    x_c = int((x1 + x2) / 2)
    width = max(8, int(x2 - x1))
    cv2.ellipse(
        frame,
        center=(x_c, y2i),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )
    if track_id is not None:
        label = f"{track_id}"
        tw = 40 if track_id < 100 else 48
        cv2.rectangle(
            frame,
            (x_c - tw // 2, y2i + 5),
            (x_c + tw // 2, y2i + 25),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x_c - tw // 2 + 6, y2i + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
    return frame


def _triangle(frame, bbox, color):
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    x = int((x1 + x2) / 2)
    y = int(y1)
    pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
    return frame


def main() -> None:
    import cv2
    from ultralytics import YOLO
    from app.services.team_assigner import TeamAssigner

    video = Path(os.environ.get("OVERLAY_VIDEO", str(_ROOT / "testing_videos/08fd33_4.mp4")))
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")

    out = Path(
        os.environ.get(
            "OVERLAY_OUT",
            str(_ROOT / "data/eval/overlays" / f"{video.stem}_tracks.mp4"),
        )
    )
    if str(out).startswith("/host_data") and not Path("/host_data").is_dir():
        out = _ROOT / "data/eval/overlays" / Path(out).name
    out.parent.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(os.environ.get("OVERLAY_FRAME_STRIDE", "1")))
    conf = float(os.environ.get("OVERLAY_CONF", "0.15"))
    imgsz = int(os.environ.get("OVERLAY_IMGSZ", "960"))
    device = os.environ.get("VISION_DEVICE", "cpu")
    model_path = _resolve_weights()

    print(f"video={video}", flush=True)
    print(f"weights={model_path} stride={stride} conf={conf} imgsz={imgsz}", flush=True)
    print(f"out={out}", flush=True)

    model = YOLO(model_path)
    buckets = _class_ids(model)
    track_classes = sorted(buckets["person"] + buckets["ball"] + buckets["referee"])
    tracker = _tracker_cfg()
    person_like = set(buckets["person"])
    ball_like = set(buckets["ball"])
    ref_like = set(buckets["referee"])
    is_custom = bool(buckets["referee"] or (0 in person_like and "player" in str(model.names).lower()) or len(person_like) > 0 and 32 not in ball_like)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer {out}")

    assigner = TeamAssigner()
    team_fit_done = False
    sticky_kit: dict[int, str] = {}
    team_color: dict[str, tuple] = {"A": _TEAM_A, "B": _TEAM_B}

    person_ids: set[int] = set()
    ref_ids: set[int] = set()
    ball_hits = 0
    frames_in = 0
    frames_tracked = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_in += 1
        if idx % stride != 0:
            writer.write(frame)
            idx += 1
            continue

        results = model.track(
            frame,
            persist=True,
            conf=conf,
            imgsz=imgsz,
            device=device,
            classes=track_classes,
            tracker=tracker,
            verbose=False,
        )
        frames_tracked += 1
        annot = frame.copy()
        result = results[0]

        # Collect player crops on early frames to fit teams
        frame_players: list[tuple] = []  # (tid, cls, conf, xyxy, crop)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            ids = boxes.id
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                c = float(boxes.conf[i].item())
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                tid = None
                if ids is not None:
                    try:
                        tid = int(ids[i].item())
                    except Exception:
                        tid = None
                crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                frame_players.append((tid, cls_id, c, (x1, y1, x2, y2), crop))

        if not team_fit_done and frames_tracked <= 60:
            crops = [
                crop
                for tid, cls_id, c, xyxy, crop in frame_players
                if cls_id in person_like
                and crop is not None
                and crop.size > 0
                and (xyxy[3] - xyxy[1]) >= 48
            ]
            if len(crops) >= 4 and assigner.fit(crops):
                team_fit_done = True
                ca = assigner.team_colors["A"]
                cb = assigner.team_colors["B"]
                team_color["A"] = (int(ca[0]), int(ca[1]), int(ca[2]))
                team_color["B"] = (int(cb[0]), int(cb[1]), int(cb[2]))
                print(f"  team colours fitted at f={idx}: A={team_color['A']} B={team_color['B']}", flush=True)

        for tid, cls_id, c, xyxy, crop in frame_players:
            x1, y1, x2, y2 = xyxy
            if cls_id in ball_like:
                ball_hits += 1
                _triangle(annot, xyxy, _BALL)
                continue
            if cls_id in ref_like:
                if tid is not None:
                    ref_ids.add(tid)
                _ellipse(annot, xyxy, _REF, tid)
                continue

            # person / player / gk
            kit = None
            if tid is not None:
                person_ids.add(tid)
                if tid in sticky_kit:
                    kit = sticky_kit[tid]
                elif team_fit_done and crop is not None and crop.size:
                    kit = assigner.predict(crop, sticky_id=tid)
                    if kit:
                        sticky_kit[tid] = kit
            color = team_color.get(kit or "", _UNK) if kit else _UNK
            if not team_fit_done and tid is not None:
                # pre-fit: colour by id
                palette = [(66, 133, 244), (219, 68, 55), (15, 157, 88), (244, 180, 0)]
                color = palette[tid % len(palette)]
            _ellipse(annot, xyxy, color, tid)

        # HUD
        a_n = sum(1 for k in sticky_kit.values() if k == "A")
        b_n = sum(1 for k in sticky_kit.values() if k == "B")
        hud = (
            f"f={idx} ids={len(person_ids)} ref={len(ref_ids)} ball={ball_hits} "
            f"A={a_n} B={b_n} {'KMeans' if team_fit_done else 'fitting…'}"
        )
        cv2.putText(annot, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(annot)
        idx += 1
        if frames_tracked % 50 == 0:
            print(f"  … tracked={frames_tracked} ids={len(person_ids)} ball={ball_hits}", flush=True)

    cap.release()
    writer.release()
    summary = {
        "video": str(video),
        "out": str(out),
        "weights": model_path,
        "custom_detector": is_custom,
        "frames_in": frames_in,
        "frames_tracked": frames_tracked,
        "unique_person_ids": len(person_ids),
        "unique_ref_ids": len(ref_ids),
        "ball_hits": ball_hits,
        "team_fit": team_fit_done,
        "sticky_teams": len(sticky_kit),
        "stride": stride,
        "conf": conf,
        "imgsz": imgsz,
    }
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}", flush=True)
    print(
        f"frames={frames_in} tracked={frames_tracked} "
        f"players={len(person_ids)} refs={len(ref_ids)} ball={ball_hits} "
        f"team_fit={team_fit_done}",
        flush=True,
    )


if __name__ == "__main__":
    main()
