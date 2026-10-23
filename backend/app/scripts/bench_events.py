"""Pass-detection precision/recall on labelled DFL event clips (real detector + events engine).

Ground truth: Kaggle `alejopaullier/dfl-clips` — ~2.5 s clips cut around the
DFL Bundesliga Data Shootout competition labels, one class per file name:
`play-*` (pass / cross) = positive; `background-*`, `challenge-*`, `throwin-*` = negative.
A clip counts as "pass predicted" when the events engine emits a pass within
±`--window-ms` of the clip centre (where the labelled event sits).

  python -m app.scripts.bench_events --clips /kaggle/input/dfl-clips --out bench.json \\
      --per-class 120 [--stride 1] [--save-dets dir]

Default sampling = the product's (VISION_TARGET_FPS). --save-dets writes each
clip's detections so the events engine can be re-tuned on CPU without the GPU:
`python -m app.scripts.bench_events --replay dir --out bench.json`.

Uses the same YoloTrackBackend.detect_and_track + infer_events as the product
(one loaded model; tracker reset per clip).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

CLASSES = ("play", "background", "challenge", "throwin")


def _label(path: Path) -> str | None:
    m = re.match(r"([a-z]+)-", path.name)
    return m.group(1) if m and m.group(1) in CLASSES else None


def sample_clips(root: Path, per_class: int, seed: int = 0) -> list[tuple[Path, str]]:
    by_class: dict[str, list[Path]] = {c: [] for c in CLASSES}
    for p in root.rglob("*.mp4"):
        lab = _label(p)
        if lab:
            by_class[lab].append(p)
    rng = random.Random(seed)
    out: list[tuple[Path, str]] = []
    for c, paths in by_class.items():
        paths.sort()
        rng.shuffle(paths)
        # Negatives that look like play (challenge/throw-in) are rarer — take what exists
        out.extend((p, c) for p in paths[: per_class if c in {"play", "background"} else per_class // 3])
    return out


def score(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["label"] == "play" and r["pred_pass"])
    fn = sum(1 for r in rows if r["label"] == "play" and not r["pred_pass"])
    fp = sum(1 for r in rows if r["label"] != "play" and r["pred_pass"])
    tn = sum(1 for r in rows if r["label"] != "play" and not r["pred_pass"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    fp_by = Counter(r["label"] for r in rows if r["label"] != "play" and r["pred_pass"])
    n_by = Counter(r["label"] for r in rows)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "false_positive_rate_by_class": {c: round(fp_by[c] / n_by[c], 3) for c in n_by if c != "play"},
        "clips_by_class": dict(n_by),
        "ball_seen_pct": round(100 * sum(r["ball_samples"] > 0 for r in rows) / max(1, len(rows)), 1),
    }


def _score_clip(name: str, label: str, dets, dur_ms: int, window_ms: int, balls: int, persons: int) -> dict:
    from app.services.events import infer_event_candidates

    centre = dur_ms // 2
    events = infer_event_candidates(dets, "A")
    passes = [e.camera_ts_ms for e in events if e.event_type == "pass"]
    return {
        "clip": name,
        "label": label,
        "pred_pass": any(abs(ts - centre) <= window_ms for ts in passes),
        "passes": passes,
        "events": dict(Counter(e.event_type for e in events)),
        "ball_samples": balls,
        "person_tracks": persons,
    }


def _replay(args) -> dict:
    """Score saved per-clip detections with the current events engine (CPU, seconds)."""
    from app.services.vision import Detection

    rows = []
    for f in sorted(args.replay.glob("*.json")):
        data = json.loads(f.read_text())
        dets = [Detection(camera_label="A", features={"class": d["class_name"]}, **d) for d in data["detections"]]
        balls = sum(len(d.bbox_series or []) for d in dets if d.class_name == "sports_ball")
        persons = sum(1 for d in dets if d.class_name == "person")
        rows.append(_score_clip(f.stem, data["label"], dets, data["duration_ms"], args.window_ms, balls, persons))
    result = {"replay": str(args.replay), "window_ms": args.window_ms, "pass_vs_not": score(rows), "rows": rows}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["pass_vs_not"], indent=2))
    return result


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--stride", type=int, default=0, help="0 = product sampling (VISION_TARGET_FPS)")
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--save-dets", type=Path, default=None)
    ap.add_argument("--replay", type=Path, default=None, help="score saved detections (no detector)")
    args = ap.parse_args(argv)
    if args.replay:
        return _replay(args)

    if args.stride:
        os.environ["VISION_FRAME_STRIDE"] = str(args.stride)
        os.environ["VISION_TARGET_FPS"] = "0"
    os.environ.setdefault("VISION_JERSEY_OCR", "0")  # identity is not scored here
    os.environ.setdefault("VISION_TEAM_EMBED", "0")
    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.services.yolo_backend import YoloTrackBackend

    clips = sample_clips(args.clips, args.per_class)
    backend = YoloTrackBackend()
    rows = []
    t0 = time.time()
    for i, (path, label) in enumerate(clips):
        try:
            dets = backend.detect_and_track(str(path), "A")
        except Exception as exc:  # noqa: BLE001 — a bad clip must not stop the benchmark
            rows.append({"clip": path.name, "label": label, "error": str(exc), "pred_pass": False, "ball_samples": 0})
            continue
        stats = backend.last_run_stats
        dur_ms = int(1000 * (stats.get("video_seconds_probed") or 2.5))
        if args.save_dets:
            args.save_dets.mkdir(parents=True, exist_ok=True)
            (args.save_dets / f"{path.stem}.json").write_text(json.dumps({
                "label": label, "duration_ms": dur_ms,
                "detections": [
                    {"track_key": d.track_key, "class_name": d.class_name, "start_ms": d.start_ms,
                     "end_ms": d.end_ms, "confidence": d.confidence, "bbox_series": d.bbox_series}
                    for d in dets
                ],
            }))
        rows.append(_score_clip(path.name, label, dets, dur_ms, args.window_ms,
                                int(stats.get("ball_detections") or 0), int(stats.get("person_tracks") or 0)))
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(clips)} clips · {time.time() - t0:.0f}s · running {score(rows)}", flush=True)

    result = {
        "dataset": "alejopaullier/dfl-clips",
        "model": str(backend.model_name).rsplit("/", 1)[-1],
        "stride": args.stride,
        "window_ms": args.window_ms,
        "seconds": round(time.time() - t0),
        "pass_vs_not": score(rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return result


if __name__ == "__main__":
    main()
