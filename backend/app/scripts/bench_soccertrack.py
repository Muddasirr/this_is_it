"""Score a run against SoccerTrack v2 full-match labels (fixed whole-pitch camera).

Ground truth: `bas/<match>/<match>_12_class_events.json` (Hugging Face
atomscott/soccertrack-v2, CC BY 4.0) — every ball action with its video frame,
team side and player. Our output: `canonical_events.json` from run_offline.

  python -m app.scripts.bench_soccertrack --run out/117093_1st \\
      --events bas/117093/117093_12_class_events.json --half 1 --out score.json

Per class (greedy one-to-one match within ±tolerance): precision / recall / F1.
Team split: on matched passes, our passer's kit (A/B) vs the true team side,
under the better of the two letter mappings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# SoccerTrack label → our event type
CLASS_MAP = {
    "PASS": "pass",
    "HIGH PASS": "pass",
    "CROSS": "pass",
    "SHOT": "shot",
    "GOAL": "goal",
    "DRIVE": "drive",  # a controlling touch; compared with our dribbles, reported separately
}
OURS_FOR = {"pass": "pass", "shot": "shot", "goal": "goal", "drive": "dribble"}


def load_truth(path: Path, half: int) -> list[dict]:
    data = json.loads(path.read_text())
    fps = float(data.get("fps") or 25.0)
    out = []
    for a in data["actions"]:
        if not str(a.get("gameTime", "")).startswith(f"{half} "):
            continue
        cls = CLASS_MAP.get(a.get("label"))
        if cls is None:
            continue
        ts = int(a["position"]) if a.get("position") not in (None, "") else int(round(a["frame"] * 1000 / fps))
        out.append({"type": cls, "ts_ms": ts, "team": a.get("team"), "label": a.get("label")})
    return out


def load_pred(run: Path) -> list[dict]:
    rows = json.loads((run / "canonical_events.json").read_text())
    out = []
    for e in rows:
        kit = None
        for p in e.get("participants") or []:
            if p.get("role") in {"passer", "shooter", "scorer", "carrier"} and p.get("player"):
                name = str(p["player"])
                kit = "A" if name.startswith("Kit A") else "B" if name.startswith("Kit B") else None
                break
        out.append({"type": e["event_type"], "ts_ms": int(e["ts_ms"]), "kit": kit})
    return out


def match(truth: list[dict], pred: list[dict], tol_ms: int) -> list[tuple[dict, dict]]:
    """Greedy nearest-first one-to-one matching within tolerance."""
    pairs = sorted(
        ((abs(t["ts_ms"] - p["ts_ms"]), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred)
         if abs(t["ts_ms"] - p["ts_ms"]) <= tol_ms),
    )
    used_t, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        out.append((truth[i], pred[j]))
    return out


def score(truth: list[dict], pred: list[dict], tol_ms: int = 1000, duration_ms: int | None = None) -> dict:
    if duration_ms:  # only score the part of the half that was processed
        truth = [t for t in truth if t["ts_ms"] <= duration_ms]
    by_class = {}
    team_pairs: list[tuple[str, str]] = []
    for cls, ours in OURS_FOR.items():
        t = [x for x in truth if x["type"] == cls]
        p = [x for x in pred if x["type"] == ours]
        m = match(t, p, tol_ms)
        tp = len(m)
        prec = tp / len(p) if p else None
        rec = tp / len(t) if t else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        by_class[cls] = {
            "truth": len(t), "pred": len(p), "tp": tp,
            "precision": round(prec, 3) if prec is not None else None,
            "recall": round(rec, 3) if rec is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
        }
        if cls == "pass":
            team_pairs = [(tt["team"], pp["kit"]) for tt, pp in m if tt.get("team") and pp.get("kit")]
    same = sum(1 for team, kit in team_pairs if (team, kit) in {("left", "A"), ("right", "B")})
    flip = sum(1 for team, kit in team_pairs if (team, kit) in {("left", "B"), ("right", "A")})
    team_acc = max(same, flip) / len(team_pairs) if team_pairs else None
    return {
        "tolerance_ms": tol_ms,
        "by_class": by_class,
        "team_split": {
            "matched_passes_with_team": len(team_pairs),
            "accuracy": round(team_acc, 3) if team_acc is not None else None,
            "note": "passer kit vs true team side, best letter mapping",
        },
        "truth_counts": dict(Counter(t["label"] for t in truth)),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--tolerance-ms", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    summary = json.loads((args.run / "summary.json").read_text()) if (args.run / "summary.json").exists() else {}
    vision = ((summary.get("stages") or {}).get("detect_track") or {}).get("vision_stats") or [{}]
    duration_ms = int(1000 * float(vision[0].get("video_seconds_probed") or 0)) or None
    if vision[0].get("frames_processed") and vision[0].get("total_frames"):
        processed = vision[0]["frames_processed"] * vision[0].get("frame_stride", 1)
        duration_ms = min(duration_ms or 10**9, int(1000 * processed / float(vision[0].get("fps_source") or 25)))
    result = score(load_truth(args.events, args.half), load_pred(args.run), args.tolerance_ms, duration_ms)
    result["run"] = str(args.run)
    result["scored_ms"] = duration_ms
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return result


if __name__ == "__main__":
    main()
