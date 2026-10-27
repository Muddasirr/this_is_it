#!/usr/bin/env python3
"""Score a live-app football match against the DFL Bundesliga Data Shootout labels.

  python scripts/eval/score_dfl.py --labels data/eval/dfl_broadcast/dfl_train_labels.csv \
      --video-id 1606b0e6_0 --clip-start-s 600 --clip-end-s 1320 --match <match uuid>

The clip is a stitch of the 30 s chunks of one half (chunk i = [(i-1)·30, i·30) s). DFL only
labels the in-play windows (start…end rows), so predictions are scored inside those windows only,
one-to-one within ±tol. `play` = pass; `challenge` ~ tackle/foul/dribble/interception;
`throwin` ~ kick_in. Also prints the best global offset as an alignment sanity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PASSY = {"pass", "interception"}
CHALLENGE = {"tackle", "foul", "dribble", "interception"}


def load_truth(path: Path, video_id: str, t0: float, t1: float) -> tuple[list[dict], list[tuple[int, int]]]:
    events, windows, start = [], [], None
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if row["video_id"] != video_id:
            continue
        t = float(row["time"])
        if row["event"] == "start":
            start = t
        elif row["event"] == "end":
            if start is not None and t0 <= start < t1:
                windows.append((int((start - t0) * 1000), int((min(t, t1) - t0) * 1000)))
        elif t0 <= t < t1:
            events.append({"ts_ms": int((t - t0) * 1000), "type": row["event"], "attrs": row["event_attributes"]})
    return events, windows


def match(truth: list[int], pred: list[int], tol: int) -> int:
    used, tp = set(), 0
    for t in sorted(truth):
        cands = [j for j, p in enumerate(pred) if j not in used and abs(p - t) <= tol]
        if cands:
            used.add(min(cands, key=lambda j: abs(pred[j] - t)))
            tp += 1
    return tp


def prf(truth: list[int], pred: list[int], tol: int) -> dict:
    tp = match(truth, pred, tol)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return {"truth": len(truth), "pred": len(pred), "tp": tp, "precision": round(p, 2), "recall": round(r, 2),
            "f1": round(2 * p * r / (p + r), 2) if p + r else 0.0}


def match_events(match_id: str) -> list[tuple[str, int]]:
    out = subprocess.run(
        ["docker", "compose", "exec", "-T", "db", "psql", "-U", "futsal", "-d", "futsal", "-At", "-F", "\t", "-c",
         f"select event_type, canonical_ts_ms from match_events where match_id='{match_id}' and is_canonical"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [(t, int(ms)) for t, ms in (line.split("\t") for line in out.splitlines() if line.strip())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--clip-start-s", type=float, required=True)
    ap.add_argument("--clip-end-s", type=float, required=True)
    ap.add_argument("--match", help="match uuid (reads the app DB via docker compose)")
    ap.add_argument("--events-tsv", type=Path, help="event_type<TAB>ts_ms instead of --match")
    ap.add_argument("--tol-ms", type=int, default=1000)
    args = ap.parse_args()

    truth, windows = load_truth(args.labels, args.video_id, args.clip_start_s, args.clip_end_s)
    if args.events_tsv:
        rows = [ln.split("\t") for ln in args.events_tsv.read_text().splitlines() if ln.strip()]
        pred = [(t, int(ms)) for t, ms in rows]
    else:
        pred = match_events(args.match)
    tol = args.tol_ms

    def inwin(ms: int) -> bool:
        return any(a - tol <= ms <= b + tol for a, b in windows)

    def preds(types: set[str], off: int = 0) -> list[int]:
        return [ms + off for t, ms in pred if t in types and inwin(ms + off)]

    def gt(kind: str) -> list[int]:
        return [e["ts_ms"] for e in truth if e["type"] == kind]

    best = max(range(-5000, 5001, 250), key=lambda o: match(gt("play"), preds(PASSY, o), tol))
    report = {
        "labelled_seconds": round(sum(b - a for a, b in windows) / 1000, 1),
        "pass": prf(gt("play"), preds(PASSY), tol),
        "pass_strict_type": prf(gt("play"), preds({"pass"}), tol),
        "challenge": prf(gt("challenge"), preds(CHALLENGE), tol),
        "throwin": prf(gt("throwin"), preds({"kick_in"}), tol),
        "pass_tol_2s": prf(gt("play"), preds(PASSY), 2000),
        "best_pass_offset_ms": best,
        "predicted_types": dict(Counter(t for t, _ in pred)),
        "predicted_in_windows": dict(Counter(t for t, ms in pred if inwin(ms))),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
