"""Detection + tracking accuracy vs MOT ground truth (SoccerTrack v2 `mot/`).

Runs the product's YoloTrackBackend on a clip at one or more input sizes and
scores every processed frame against the GT boxes with py-motmetrics:
detection recall / precision (IoU ≥ 0.5) and tracking MOTA / IDF1 / ID switches.

  python -m app.scripts.bench_mot --clip mot/clips/117093.mp4 --gt mot/117093.txt \\
      --imgsz 640,1280,1920 --out bench_mot.json [--max-frames 600]

GT covers the 22 players (incl. goalkeepers) only — referees / staff we detect
count as false positives, so precision is a lower bound.

--teams <match>_teams.json (white/blue/goalkeeper GT ids) also scores the SigLIP
team split: each matched detection's predicted kit vs its true team, best A/B
mapping, goalkeepers excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_gt(path: Path) -> dict[int, list[tuple[int, list[float]]]]:
    gt: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        f, tid, x, y, w, h = line.split(",")[:6]
        gt[int(f)].append((int(tid), [float(x), float(y), float(w), float(h)]))
    return gt


def _iou_distance(a: list[list[float]], b: list[list[float]], min_iou: float = 0.5) -> np.ndarray:
    """1 - IoU for xywh boxes; NaN (= cannot match) below min_iou.

    Own implementation: motmetrics' iou_matrix uses np.asfarray (removed in NumPy 2).
    """
    if not a or not b:
        return np.empty((len(a), len(b)))
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ax2, ay2, bx2, by2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3], B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(A[:, None, 0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(A[:, None, 1], B[:, 1]), 0, None)
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + B[:, 2] * B[:, 3] - inter
    iou = inter / np.maximum(union, 1e-9)
    return np.where(iou >= min_iou, 1.0 - iou, np.nan)


def score(gt: dict, log: list[tuple], frames: list[int]) -> dict:
    import motmetrics as mm

    pred: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for f, tid, x1, y1, x2, y2, _conf in log:
        pred[int(f)].append((int(tid if tid is not None else -1), [x1, y1, x2 - x1, y2 - y1]))
    acc = mm.MOTAccumulator(auto_id=True)
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        dist = _iou_distance([b for _, b in g], [b for _, b in p])
        acc.update([i for i, _ in g], [i for i, _ in p], dist)
    m = mm.metrics.create().compute(
        acc,
        metrics=["num_frames", "recall", "precision", "mota", "idf1", "num_switches",
                 "mostly_tracked", "num_unique_objects"],
        name="run",
    ).iloc[0]
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in m.items()}


def team_accuracy(gt: dict, log: list[tuple], frames: list[int], kit_of: dict[int, str], teams: dict) -> dict:
    """Per matched (IoU ≥ 0.5) detection: predicted kit vs true team (field players only)."""
    from app.services.reid import linear_assignment_max

    team_of = {i: "white" for i in teams["white"]} | {i: "blue" for i in teams["blue"]}
    pred: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for f, tid, x1, y1, x2, y2, _c in log:
        pred[int(f)].append((tid, [x1, y1, x2 - x1, y2 - y1]))
    pairs = []
    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        if not g or not p:
            continue
        d = _iou_distance([b for _, b in g], [b for _, b in p])
        for i, j in linear_assignment_max(np.where(np.isnan(d), -np.inf, -d)):
            gid, tid = g[i][0], p[j][0]
            if gid in team_of and kit_of.get(tid) in {"A", "B"}:
                pairs.append((team_of[gid], kit_of[tid]))
    if not pairs:
        return {"team_pairs": 0, "team_accuracy": None}
    same = sum(1 for t, k in pairs if (t, k) in {("white", "A"), ("blue", "B")})
    return {"team_pairs": len(pairs), "team_accuracy": round(max(same, len(pairs) - same) / len(pairs), 4),
            "kit_x_tracks": sum(1 for k in kit_of.values() if k == "X")}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--imgsz", default="640", help="comma-separated input sizes to compare")
    ap.add_argument("--max-frames", type=int, default=0, help="processed-frame cap per run (0 = whole clip)")
    ap.add_argument("--teams", type=Path, default=None, help="<match>_teams.json → score team split")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    gt = load_gt(args.gt)
    teams = json.loads(args.teams.read_text()) if args.teams else None
    results = []
    for size in [int(s) for s in args.imgsz.split(",") if s != ""]:
        os.environ.update(VISION_IMGSZ=str(size), VISION_REID="0", VISION_JERSEY_OCR="0",
                          VISION_TEAM_EMBED="1" if teams else "0", VISION_MAX_FRAMES=str(args.max_frames))
        from app.core.config import get_settings

        get_settings.cache_clear()
        from app.services.yolo_backend import YoloTrackBackend

        backend = YoloTrackBackend()
        backend.frame_log = []
        t0 = time.time()
        dets = backend.detect_and_track(str(args.clip), "A")
        stats = backend.last_run_stats
        # Exactly the frames the detector looked at (every frame_stride-th, from 0)
        frames = [f for f in range(0, backend.frame_stride * int(stats.get("frames_processed") or 0),
                                   backend.frame_stride) if f in gt]
        row = {
            "imgsz": size,
            "model": str(backend.model_name).rsplit("/", 1)[-1],
            "frame_stride": backend.frame_stride,
            "imgsz_used": stats.get("imgsz"),
            "frames_processed": stats.get("frames_processed"),
            "processing_fps": stats.get("processing_fps"),
            "seconds": round(time.time() - t0, 1),
            **score(gt, backend.frame_log, frames),
        }
        if teams:
            import numpy as np

            from app.services.team_embed import _team_features, split_teams

            # Same features the product clusters on: torso colour + 0.25 × SigLIP torso
            tracks = [d for d in dets if d.class_name == "person"
                      and ((d.features or {}).get("team_colour") or (d.features or {}).get("team_emb"))]
            rows = [(i, np.asarray(d.features["team_colour"]) if d.features.get("team_colour") else None,
                     np.asarray(d.features["team_emb"]) if d.features.get("team_emb") else None,
                     float(max(1, d.end_ms - d.start_ms))) for i, d in enumerate(tracks)]
            kits = split_teams(_team_features(rows), [r[3] for r in rows], reduce=False) if rows else []
            kit_of = {int(d.track_key.rsplit("p", 1)[1]): k for d, k in zip(tracks, kits)}
            row.update(team_accuracy(gt, backend.frame_log, frames, kit_of, teams))
        results.append(row)
        print(json.dumps(row), flush=True)
    out = {"clip": str(args.clip), "gt": str(args.gt), "runs": results}
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
