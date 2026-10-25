"""Score predicted goals/shots against scoreboard-derived goal ground truth.

A goal at ground-truth time g (when the scoreboard changed) is matched by a
prediction in [g - lead, g + lag]: the shot happens seconds before the overlay
updates. A goal counts as "found" by a predicted goal, and as "shot-covered"
by a predicted shot or goal. Precision is per predicted goal.

  python -m app.scripts.bench_goals --gt data/futsal_gt/spain_france_goals.json \\
      --pred data/kaggle_out/gpu_eval_v9/eval/spain_france_u19_2023__yolov8m/canonical_events.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _events(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("events", [])


def _type(e: dict) -> str:
    return e.get("event_type") or e.get("type") or ""


def _match(gt_ts: list[int], pred_ts: list[int], lead_ms: int, lag_ms: int) -> int:
    """Greedy one-to-one matches in time order."""
    used, hits = set(), 0
    for g in sorted(gt_ts):
        for i, p in enumerate(pred_ts):
            if i not in used and g - lead_ms <= p <= g + lag_ms:
                used.add(i)
                hits += 1
                break
    return hits


def score(gt: list[dict], pred: list[dict], lead_ms: int = 15000, lag_ms: int = 2000) -> dict:
    g = sorted(e["ts_ms"] for e in gt if _type(e) == "goal")
    pg = sorted(e["ts_ms"] for e in pred if _type(e) == "goal")
    ps = sorted(e["ts_ms"] for e in pred if _type(e) in ("shot", "goal"))
    found = _match(g, pg, lead_ms, lag_ms)
    covered = _match(g, ps, lead_ms, lag_ms)
    return {
        "gt_goals": len(g),
        "pred_goals": len(pg),
        "pred_shots": sum(1 for e in pred if _type(e) == "shot"),
        "goal_recall": round(found / len(g), 3) if g else None,
        "goal_precision": round(found / len(pg), 3) if pg else None,
        "goal_shot_recall": round(covered / len(g), 3) if g else None,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--lead-ms", type=int, default=15000)
    ap.add_argument("--lag-ms", type=int, default=2000)
    args = ap.parse_args(argv)
    result = score(_events(args.gt), _events(args.pred), args.lead_ms, args.lag_ms)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
