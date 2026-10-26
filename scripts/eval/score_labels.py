#!/usr/bin/env python3
"""Score a run's events against labels exported from the app's Label page (/matches/<id>/label).

  python scripts/eval/score_labels.py --labels labels-<match>.json --run <run_offline out dir> [--tol-ms 1000]
         [--offset-ms 0]   # label video time − run video time (e.g. the run used a trimmed clip)

Per class (pass / dribble / shot / goal): precision, recall, F1 with one-to-one matching
within ±tol (same matcher as bench_soccertrack), scored only over the labelled time span.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.scripts.bench_soccertrack import match  # noqa: E402


def score(labels: list[dict], events: list[dict], tol_ms: int) -> dict:
    lo = min((l["ts_ms"] for l in labels), default=0) - tol_ms
    hi = max((l["ts_ms"] for l in labels), default=0) + tol_ms
    out = {}
    for cls in ("pass", "dribble", "shot", "goal"):
        t = [{"ts_ms": l["ts_ms"]} for l in labels if l["type"] == cls]
        p = [{"ts_ms": e["ts_ms"]} for e in events if e["event_type"] == cls and lo <= e["ts_ms"] <= hi]
        tp = len(match(t, p, tol_ms))
        prec = tp / len(p) if p else None
        rec = tp / len(t) if t else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        out[cls] = {"truth": len(t), "pred": len(p), "tp": tp,
                    "precision": round(prec, 3) if prec is not None else None,
                    "recall": round(rec, 3) if rec is not None else None,
                    "f1": round(f1, 3) if f1 is not None else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tol-ms", type=int, default=1000)
    ap.add_argument("--offset-ms", type=int, default=0)
    args = ap.parse_args()
    labels = [{**l, "ts_ms": int(l["ts_ms"]) - args.offset_ms} for l in json.loads(args.labels.read_text())["events"]]
    events = json.loads((args.run / "canonical_events.json").read_text())
    print(json.dumps(score(labels, events, args.tol_ms), indent=1))


if __name__ == "__main__":
    main()
