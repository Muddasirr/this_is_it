"""Team split + bystander filter vs hand-labelled tracks (bench_labels/<video>_roles.json).

Labels are per track: two team letters, G (goalkeeper), O (not a player). Scored
from a run's players.json (track_key → Kit A / Kit B):
  team_accuracy     labelled team tracks on the right kit, best A/B mapping, among assigned tracks
  bystanders_kept   O tracks the run still counts as players (lower is better)
  team_assigned     labelled team tracks that ended up in any player

  python -m app.scripts.bench_teams --run <run dir> --labels app/scripts/bench_labels/test_video_2_roles.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(players: list[dict], labels: dict[str, str]) -> dict:
    kit = {k: p["team"] for p in players for k in p.get("track_keys") or []}
    teams = sorted({v for v in labels.values() if v not in {"G", "O"}})
    pairs = [(labels[k], kit[k]) for k in labels if labels[k] in teams and kit.get(k) in {"Kit A", "Kit B"}]
    if len(teams) == 2 and pairs:
        same = sum(1 for t, k in pairs if (t == teams[0]) == (k == "Kit A"))
        acc = max(same, len(pairs) - same) / len(pairs)
    else:
        acc = None
    n_team = sum(1 for v in labels.values() if v in teams)
    n_other = sum(1 for v in labels.values() if v == "O")
    kept = sum(1 for k, v in labels.items() if v == "O" and k in kit)
    return {
        "team_accuracy": round(acc, 3) if acc is not None else None,
        "team_tracks_scored": len(pairs),
        "team_assigned": f"{len(pairs)}/{n_team}",
        "bystanders_kept": f"{kept}/{n_other}",
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run dir with players.json")
    ap.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args(argv)
    result = score(json.loads((args.run / "players.json").read_text()), json.loads(args.labels.read_text())["labels"])
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
