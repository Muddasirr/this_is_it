#!/usr/bin/env python3
"""Re-run possession events + player table from an existing detections.json (no YOLO)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OFFLINE = Path(__file__).resolve().parent / "courtlens_offline"
sys.path.insert(0, str(OFFLINE))

from events import infer_event_candidates  # noqa: E402
from identity_offline import PlayerCluster, detections_to_tracks  # noqa: E402
from stats_offline import build_player_stats  # noqa: E402
from vision import Detection  # noqa: E402


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_out/v16/courtlens_out")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/kaggle_out/v16_reinfer")
    out.mkdir(parents=True, exist_ok=True)

    raw = json.loads((src / "detections.json").read_text())
    dets = [
        Detection(
            track_key=d["track_key"],
            camera_label=d.get("camera_label", "A"),
            start_ms=d["start_ms"],
            end_ms=d["end_ms"],
            confidence=d["confidence"],
            class_name=d["class_name"],
            features=d.get("features") or {},
            bbox_series=d.get("bbox_series") or [],
        )
        for d in raw
    ]
    events = infer_event_candidates(dets, "A")
    print("events", dict(Counter(e.event_type for e in events)))

    tracks = detections_to_tracks(dets)
    pj = json.loads((src / "players.json").read_text())
    ttp = json.loads((src / "track_to_player.json").read_text())
    thumbs = src / "thumbs"
    players = []
    for p in pj:
        local = thumbs / f"{p['player_key']}.jpg"
        players.append(
            PlayerCluster(
                player_key=p["player_key"],
                display_name=p["display_name"],
                kit=p.get("kit") or "A",
                shirt_number=p.get("shirt_number"),
                jersey_confidence=float(p.get("jersey_confidence") or 0),
                track_keys=p.get("track_keys") or [],
                color_hist=None,
                thumb_path=str(local) if local.exists() else None,
            )
        )

    stats = build_player_stats(tracks, players, ttp, events)
    table = [
        {
            "name": r["display_name"],
            "kit": r.get("kit"),
            "shirt": r.get("shirt_number"),
            "touches": r["touches"],
            "possession_pct": r["possession_pct"],
            "distance": r["distance"],
            "pass": r["pass"],
            "shot": r["shot"],
            "goal": r["goal"],
            "avatar": r.get("avatar_path"),
        }
        for r in stats["players"]
    ]
    (out / "player_table.json").write_text(json.dumps(table, indent=2))
    (out / "match_totals.json").write_text(json.dumps(stats["totals"], indent=2))
    (out / "events.json").write_text(
        json.dumps(
            [
                {
                    "event_type": e.event_type,
                    "camera_ts_ms": e.camera_ts_ms,
                    "confidence": round(e.confidence, 4),
                    "track_keys": e.track_keys,
                    "evidence": e.evidence,
                }
                for e in events
            ],
            indent=2,
        )
    )
    (src / "player_table_fixed.json").write_text(json.dumps(table, indent=2))
    print("totals", stats["totals"])
    print("wrote", out)


if __name__ == "__main__":
    main()
