#!/usr/bin/env python3
"""Score a live-app football match against SoccerTrack v2 ball-action labels (HF atomscott/soccertrack-v2, CC BY 4.0).

  python scripts/eval/score_soccertrack.py \
      --labels data/eval/soccertrack2/bas/128057/128057_12_class_events.json \
      --half 1 --clip-start-s 0 --clip-end-s 720 --match <match uuid>

SoccerTrack v2 is ten full matches from a fixed panoramic camera with every on-ball action labelled
frame-aligned (`position` = ms from the half's first frame), so — unlike the DFL chunks — the whole
clip is labelled and every prediction counts. One-to-one matching within ±tol.
  pass        PASS / HIGH PASS / CROSS        vs app pass + interception
  shot        SHOT (+ GOAL)                   vs app shot + goal
  goal        GOAL                            vs app goal
  throw_in    THROW IN                        vs app kick_in
  free_kick   FREE KICK                       vs app free_kick
  dribble     DRIVE (carry)                   vs app dribble
  touch       any labelled action             vs any app event
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_dfl import match, match_events, prf  # noqa: E402

KINDS = {
    "pass": ({"PASS", "HIGH PASS", "CROSS"}, {"pass", "interception"}),
    "pass_strict_type": ({"PASS", "HIGH PASS", "CROSS"}, {"pass"}),
    "shot": ({"SHOT", "GOAL"}, {"shot", "goal"}),
    "goal": ({"GOAL"}, {"goal"}),
    "throw_in": ({"THROW IN"}, {"kick_in"}),
    "free_kick": ({"FREE KICK"}, {"free_kick"}),
    "dribble": ({"DRIVE"}, {"dribble"}),
}


def load_truth(path: Path, half: int, t0: float, t1: float) -> list[dict]:
    out = []
    for a in json.loads(path.read_text())["actions"]:
        ms = int(a["position"])
        if a["gameTime"].startswith(f"{half} ") and t0 * 1000 <= ms < t1 * 1000:
            out.append({"ts_ms": ms - int(t0 * 1000), "type": a["label"], "team": a.get("team")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--clip-start-s", type=float, default=0)
    ap.add_argument("--clip-end-s", type=float, required=True)
    ap.add_argument("--match", help="match uuid (reads the app DB via docker compose)")
    ap.add_argument("--events-tsv", type=Path, help="event_type<TAB>ts_ms instead of --match")
    ap.add_argument("--tol-ms", type=int, default=1000)
    args = ap.parse_args()

    truth = load_truth(args.labels, args.half, args.clip_start_s, args.clip_end_s)
    if args.events_tsv:
        rows = [ln.split("\t") for ln in args.events_tsv.read_text().splitlines() if ln.strip()]
        pred = [(t, int(ms)) for t, ms in rows]
    else:
        pred = match_events(args.match)
    tol = args.tol_ms

    def gt(labels: set[str]) -> list[int]:
        return [e["ts_ms"] for e in truth if e["type"] in labels]

    def preds(types: set[str] | None, off: int = 0) -> list[int]:
        return [ms + off for t, ms in pred if types is None or t in types]

    passes = KINDS["pass"]
    best = max(range(-5000, 5001, 250), key=lambda o: match(gt(passes[0]), preds(passes[1], o), tol))
    report = {k: prf(gt(lab), preds(app), tol) for k, (lab, app) in KINDS.items()}
    report["touch"] = prf(gt({e["type"] for e in truth}), preds(None), tol)
    report["pass_tol_2s"] = prf(gt(passes[0]), preds(passes[1]), 2000)
    report["best_pass_offset_ms"] = best
    report["truth_types"] = dict(Counter(e["type"] for e in truth))
    report["predicted_types"] = dict(Counter(t for t, _ in pred))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
