#!/usr/bin/env python3
"""CourtLens full pipeline on Kaggle GPU — same logic as local YOLO backend."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> None:
    import torch

    video = os.environ.get("COURTLENS_VIDEO")
    weights = os.environ.get("COURTLENS_WEIGHTS")
    out_dir = Path(os.environ.get("COURTLENS_OUT", "/kaggle/working/courtlens_out"))
    camera = os.environ.get("COURTLENS_CAMERA", "A")

    if not video:
        # auto-discover: COURTLENS_VIDEO_MATCH substring, else facility CCTV, else any mp4
        import glob

        all_mp4 = sorted(glob.glob("/kaggle/input/**/*.mp4", recursive=True))
        match = os.environ.get("COURTLENS_VIDEO_MATCH", "").strip().lower()
        cands: list[str] = []
        if match:
            cands = [p for p in all_mp4 if match in Path(p).name.lower()]
        if not cands:
            cands = [p for p in all_mp4 if Path(p).name == "test_video_2.mp4"]
        if not cands:
            cands = all_mp4
        assert cands, "Set COURTLENS_VIDEO or put an .mp4 under /kaggle/input"
        video = cands[0]
        if len(cands) > 1:
            print("video candidates:", [Path(p).name for p in cands])
            print("selected:", Path(video).name, "(set COURTLENS_VIDEO_MATCH to pick)")
    if not weights:
        import glob

        cands = glob.glob("/kaggle/input/**/yolov8n.pt", recursive=True)
        weights = cands[0] if cands else "yolov8n.pt"

    cuda = torch.cuda.is_available()
    print("cuda:", cuda, torch.cuda.get_device_name(0) if cuda else "cpu")
    print("torch:", torch.__version__)
    if not cuda:
        print(
            "WARNING: No GPU — CourtLens will use CPU. "
            "In Kaggle notebook settings set Accelerator = GPU T4."
        )

    os.environ.setdefault("VISION_DEVICE", "0" if cuda else "cpu")
    os.environ.setdefault("VISION_FRAME_STRIDE", "5")  # local default
    os.environ.setdefault("YOLO_MODEL", weights)

    from config import get_settings
    from identity_offline import detections_to_tracks, propose_players
    from stats_offline import build_player_stats
    from yolo_backend import YoloTrackBackend

    settings = get_settings()
    print("settings:", settings)

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_out = out_dir / "thumbs"
    thumbs_out.mkdir(exist_ok=True)

    backend = YoloTrackBackend()
    backend.device = "0" if cuda else "cpu"
    backend.model_name = weights
    backend.frame_stride = settings.vision_frame_stride

    print("detect_and_track…", video)
    detections = backend.detect_and_track(video, camera)
    detect_stats = dict(backend.last_run_stats)
    print("detect stats:", json.dumps(detect_stats, indent=2))

    print("infer_events…")
    events = backend.infer_events(detections, camera)
    print("events:", len(events), "by type", _count_events(events))

    tracks = detections_to_tracks(detections)
    players, track_to_player = propose_players(tracks)
    print("players:", len(players), [p.display_name for p in players])

    # copy thumbs into output
    for p in players:
        if p.thumb_path and Path(p.thumb_path).exists():
            dest = thumbs_out / f"{p.player_key}.jpg"
            shutil.copy2(p.thumb_path, dest)
            p.thumb_path = str(dest)

    stats = build_player_stats(tracks, players, track_to_player, events)
    stats["detection"] = detect_stats
    stats["cuda"] = cuda
    stats["video"] = video
    stats["weights"] = weights

    # serializable detections (drop huge raw if needed — keep features + series)
    det_dump = [
        {
            "track_key": d.track_key,
            "camera_label": d.camera_label,
            "start_ms": d.start_ms,
            "end_ms": d.end_ms,
            "confidence": round(d.confidence, 4),
            "class_name": d.class_name,
            "features": {
                k: v
                for k, v in (d.features or {}).items()
                if k != "bbox_series"
            },
            "bbox_series": d.bbox_series,
        }
        for d in detections
    ]
    evt_dump = [
        {
            "event_type": e.event_type,
            "camera_label": e.camera_label,
            "camera_ts_ms": e.camera_ts_ms,
            "confidence": round(e.confidence, 4),
            "track_keys": e.track_keys,
            "evidence": e.evidence,
        }
        for e in events
    ]
    players_dump = [
        {
            "player_key": p.player_key,
            "display_name": p.display_name,
            "kit": p.kit,
            "shirt_number": p.shirt_number,
            "jersey_confidence": p.jersey_confidence,
            "track_keys": p.track_keys,
            "thumb_path": p.thumb_path,
        }
        for p in players
    ]

    (out_dir / "summary.json").write_text(json.dumps(detect_stats, indent=2))
    (out_dir / "detections.json").write_text(json.dumps(det_dump))
    (out_dir / "events.json").write_text(json.dumps(evt_dump, indent=2))
    (out_dir / "players.json").write_text(json.dumps(players_dump, indent=2))
    (out_dir / "match_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "track_to_player.json").write_text(json.dumps(track_to_player, indent=2))

    # compact player table for quick view
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
        }
        for r in stats["players"]
    ]
    (out_dir / "player_table.json").write_text(json.dumps(table, indent=2))
    print("player_table:")
    print(json.dumps(table, indent=2))
    print("wrote", out_dir)


def _count_events(events) -> dict:
    from collections import Counter

    return dict(Counter(e.event_type for e in events))


if __name__ == "__main__":
    main()
