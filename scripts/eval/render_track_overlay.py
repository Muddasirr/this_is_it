#!/usr/bin/env python3
"""Render YOLO + BoT-SORT track overlay on a short clip.

  docker run --rm \
    -v $PWD/backend:/app -v $PWD/data:/host_data \
    -v $PWD/testing_videos:/videos -v $PWD/scripts:/scripts \
    -e OVERLAY_VIDEO=/videos/08fd33_4.mp4 \
    -e OVERLAY_OUT=/host_data/eval/overlays/08fd33_4_tracks.mp4 \
    -w /app <paas-api-image> \
    python -u /scripts/eval/render_track_overlay.py

Draws person boxes (coloured by track id), ball boxes, and a thin torso ROI
so you can see what kit/OCR would sample. OCR/kit assign are OFF (preview only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND = Path("/app") if Path("/app/app/services").is_dir() else Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(_BACKEND))

PERSON = 0
SPORTS_BALL = 32

# Palette for track ids (BGR)
_COLORS = [
    (66, 133, 244),
    (219, 68, 55),
    (244, 180, 0),
    (15, 157, 88),
    (171, 71, 188),
    (0, 172, 193),
    (255, 112, 67),
    (124, 179, 66),
    (92, 107, 192),
    (236, 64, 122),
]
_BALL = (0, 255, 255)
_TORSO = (180, 180, 180)


def _color(tid: int) -> tuple[int, int, int]:
    return _COLORS[int(tid) % len(_COLORS)]


def _torso_roi(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Same relative band as kit_color / jersey OCR (approx)."""
    bw, bh = x2 - x1, y2 - y1
    return (
        x1 + int(bw * 0.18),
        y1 + int(bh * 0.28),
        x1 + int(bw * 0.82),
        y1 + int(bh * 0.72),
    )


def _tracker_cfg() -> str:
    name = (os.environ.get("VISION_TRACKER") or "botsort").strip().lower()
    if name not in {"botsort", "bytetrack"}:
        name = "botsort"
    path = _BACKEND / "cfg" / f"{name}.yaml"
    return str(path) if path.is_file() else f"{name}.yaml"


def main() -> None:
    import cv2
    from ultralytics import YOLO

    video = Path(os.environ.get("OVERLAY_VIDEO", "/videos/08fd33_4.mp4"))
    if not video.is_file():
        alt = Path(__file__).resolve().parents[2] / "testing_videos" / "08fd33_4.mp4"
        video = alt if alt.is_file() else video
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")

    out = Path(
        os.environ.get(
            "OVERLAY_OUT",
            "/host_data/eval/overlays/08fd33_4_tracks.mp4",
        )
    )
    if str(out).startswith("/host_data") and not Path("/host_data").is_dir():
        out = Path(__file__).resolve().parents[2] / "data" / "eval" / "overlays" / "08fd33_4_tracks.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # stride=1: detect every frame so boxes don't freeze/lag behind players
    stride = max(1, int(os.environ.get("OVERLAY_FRAME_STRIDE", "1")))
    conf = float(os.environ.get("OVERLAY_CONF", "0.20"))
    imgsz = int(os.environ.get("OVERLAY_IMGSZ", "960"))
    device = os.environ.get("VISION_DEVICE", "cpu")
    show_torso = os.environ.get("OVERLAY_TORSO", "1") not in {"0", "false", "False"}

    weights = _BACKEND / "yolov8n.pt"
    if not weights.is_file():
        weights = Path("/app/yolov8n.pt")
    model_path = str(weights) if weights.is_file() else "yolov8n.pt"

    print(f"video={video}", flush=True)
    print(f"out={out} stride={stride} conf={conf} imgsz={imgsz} device={device}", flush=True)

    model = YOLO(model_path)
    tracker = _tracker_cfg()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # Output plays at source fps; we hold last annotated frame across skipped reads
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer {out}")

    person_ids: set[int] = set()
    ball_hits = 0
    frames_in = 0
    frames_tracked = 0
    idx = 0
    last_annot = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_in += 1

        do_track = idx % stride == 0
        if do_track:
            results = model.track(
                frame,
                persist=True,
                conf=conf,
                imgsz=imgsz,
                device=device,
                classes=[PERSON, SPORTS_BALL],
                tracker=tracker,
                verbose=False,
            )
            frames_tracked += 1
            result = results[0]
            # Fresh boxes every tracked frame — never reuse stale coords on new pixels
            last_boxes: list[tuple] = []
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
                    last_boxes.append((cls_id, c, x1, y1, x2, y2, tid))
                    if cls_id == SPORTS_BALL:
                        ball_hits += 1
                    elif tid is not None:
                        person_ids.add(tid)

        # Always draw on the *current* frame (no frozen overlay bitmap)
        annot = frame.copy()
        for cls_id, c, x1, y1, x2, y2, tid in last_boxes if do_track else getattr(main, "_hold_boxes", []):
            if not do_track:
                break
        if do_track:
            main._hold_boxes = last_boxes  # type: ignore[attr-defined]
        hold = getattr(main, "_hold_boxes", [])
        # When stride>1, skip writing stale boxes on non-tracked frames (avoids lag)
        draw_list = last_boxes if do_track else ([] if stride > 1 else hold)
        if do_track:
            draw_list = last_boxes
        for cls_id, c, x1, y1, x2, y2, tid in draw_list:
            if cls_id == SPORTS_BALL:
                cv2.rectangle(annot, (x1, y1), (x2, y2), _BALL, 2)
                cv2.putText(
                    annot,
                    f"ball {c:.2f}",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _BALL,
                    1,
                    cv2.LINE_AA,
                )
                continue
            if tid is None:
                color = (160, 160, 160)
                label = f"? {c:.2f}"
            else:
                color = _color(tid)
                label = f"id={tid} {c:.2f}"
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
            if show_torso:
                tx1, ty1, tx2, ty2 = _torso_roi(x1, y1, x2, y2)
                cv2.rectangle(annot, (tx1, ty1), (tx2, ty2), _TORSO, 1)
            cv2.putText(
                annot,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        hud = f"f={idx} tracked={frames_tracked} ids={len(person_ids)} ball={ball_hits}"
        cv2.putText(
            annot,
            hud,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(annot)
        idx += 1
        if frames_tracked and frames_tracked % 50 == 0 and do_track:
            print(f"  … tracked_frames={frames_tracked} ids={len(person_ids)} ball={ball_hits}", flush=True)

    cap.release()
    writer.release()

    summary = {
        "video": str(video),
        "out": str(out),
        "frames_in": frames_in,
        "frames_tracked": frames_tracked,
        "unique_person_ids": len(person_ids),
        "person_ids": sorted(person_ids),
        "ball_hits": ball_hits,
        "stride": stride,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "tracker": tracker,
    }
    meta = out.with_suffix(".json")
    meta.write_text(__import__("json").dumps(summary, indent=2))
    print(f"\nWrote {out}", flush=True)
    print(
        f"frames_in={frames_in} tracked={frames_tracked} "
        f"unique_ids={len(person_ids)} ball_hits={ball_hits}",
        flush=True,
    )
    print(f"meta={meta}", flush=True)


if __name__ == "__main__":
    main()
