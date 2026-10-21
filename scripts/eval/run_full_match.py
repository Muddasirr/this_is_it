#!/usr/bin/env python3
"""Full single-camera match run: detect → possession → events → offline stats."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core import config  # noqa: E402


def main() -> None:
    video = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "testing_videos/test_video_2.mp4")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/eval/test_video_2_full")
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(config.get_settings, "cache_clear"):
        config.get_settings.cache_clear()

    from app.services.events import infer_event_candidates
    from app.services.possession import build_possession_timeline
    from app.services.yolo_backend import YoloTrackBackend

    def dump(name: str, obj) -> None:
        (out / name).write_text(json.dumps(obj, indent=2, default=str))

    t0 = time.perf_counter()
    print(f"START {time.ctime()} video={video}", flush=True)
    backend = YoloTrackBackend()
    print(
        f"model={backend.model_name} stride={backend.frame_stride} "
        f"max_frames={backend.max_frames} device={backend.device}",
        flush=True,
    )

    dets = backend.detect_and_track(str(video), "A")
    summary = dict(backend.last_run_stats)
    dump("summary.json", summary)
    print(
        f"DETECT persons={summary.get('person_tracks')} ball_hits={summary.get('ball_detections')} "
        f"ball_cov={summary.get('ball_coverage_pct')} ocr={summary.get('jersey_ocr_tracks')} "
        f"thumbs={summary.get('thumbs_written')} wall={time.perf_counter()-t0:.0f}s",
        flush=True,
    )

    det_payload = [
        {
            "track_key": d.track_key,
            "camera_label": d.camera_label,
            "start_ms": d.start_ms,
            "end_ms": d.end_ms,
            "confidence": d.confidence,
            "class_name": d.class_name,
            "features": d.features,
            "bbox_series": d.bbox_series,
        }
        for d in dets
    ]
    dump("detections.json", det_payload)

    pos = build_possession_timeline(dets)
    dump(
        "possession.json",
        {
            "ball_samples": pos.get("ball_samples"),
            "diag": pos.get("diag"),
            "radius": pos.get("radius"),
            "n_segments": len(pos.get("segments") or []),
            "segments": pos.get("segments"),
        },
    )
    print(
        f"POSSESSION segments={len(pos.get('segments') or [])} "
        f"ball_samples={pos.get('ball_samples')}",
        flush=True,
    )

    events = infer_event_candidates(dets, "A")
    ev = [
        {
            "type": e.event_type,
            "ts_ms": e.camera_ts_ms,
            "confidence": e.confidence,
            "track_keys": e.track_keys,
            "evidence": e.evidence,
        }
        for e in events
    ]
    dump("events.json", ev)
    print(f"EVENTS {dict(Counter(e['type'] for e in ev))} total={len(ev)}", flush=True)

    # Offline per-player rollup (no DB): possession time + event credits
    person_dets = [d for d in dets if d.class_name == "person"]
    by_player: dict[str, dict] = {}
    for d in person_dets:
        feats = d.features or {}
        by_player[d.track_key] = {
            "track_key": d.track_key,
            "jersey_number": feats.get("jersey_number"),
            "jersey_confidence": feats.get("jersey_confidence"),
            "thumb_path": feats.get("thumb_path"),
            "duration_ms": max(0, d.end_ms - d.start_ms),
            "bbox_samples": len(d.bbox_series or []),
            "passes": 0,
            "shots": 0,
            "goals": 0,
            "possession_ms": 0,
        }

    for seg in pos.get("segments") or []:
        holder = seg.get("holder")
        if holder in by_player:
            by_player[holder]["possession_ms"] += max(0, int(seg["end_ms"]) - int(seg["start_ms"]))

    for e in ev:
        keys = e.get("track_keys") or []
        et = e["type"]
        if et == "pass" and keys:
            by_player.setdefault(keys[0], _empty_row(keys[0]))["passes"] += 1
        elif et == "shot" and keys:
            by_player.setdefault(keys[0], _empty_row(keys[0]))["shots"] += 1
        elif et == "goal" and keys:
            by_player.setdefault(keys[0], _empty_row(keys[0]))["goals"] += 1

    # Prefer jersey-labeled / longer tracks for the table
    rows = sorted(
        by_player.values(),
        key=lambda r: (r.get("jersey_number") is not None, r["possession_ms"], r["duration_ms"]),
        reverse=True,
    )
    totals = {
        "pass": sum(1 for e in ev if e["type"] == "pass"),
        "shot": sum(1 for e in ev if e["type"] == "shot"),
        "goal": sum(1 for e in ev if e["type"] == "goal"),
        "possession_segments": len(pos.get("segments") or []),
        "ball_samples": pos.get("ball_samples"),
        "ball_coverage_pct": summary.get("ball_coverage_pct"),
        "person_tracks_raw": summary.get("person_tracks"),
        "jersey_ocr_tracks": summary.get("jersey_ocr_tracks"),
    }
    stats = {
        "video": str(video),
        "summary": summary,
        "totals": totals,
        "by_player": rows[:32],
        "events": ev,
    }
    dump("stats.json", stats)

    # Human-readable sheet
    lines = [
        f"# CourtLens stats — {video.name}",
        f"Processed frames: {summary.get('frames_processed')} (stride {summary.get('frame_stride')})",
        f"Ball coverage: {summary.get('ball_coverage_pct')}%  |  OCR jersey tracks: {summary.get('jersey_ocr_tracks')}",
        f"Totals: {totals['pass']} passes, {totals['shot']} shots, {totals['goal']} goals",
        f"Possession segments: {totals['possession_segments']}  ball_samples={totals['ball_samples']}",
        "",
        "track_key | jersey | poss_s | passes | shots | goals | dur_s",
    ]
    for r in rows[:24]:
        lines.append(
            f"{r['track_key']:10} | {str(r.get('jersey_number') or '-'):>6} | "
            f"{r['possession_ms']/1000:6.1f} | {r['passes']:6} | {r['shots']:5} | "
            f"{r['goals']:5} | {r['duration_ms']/1000:6.1f}"
        )
    (out / "STATS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # GT score if fixture present
    gt = ROOT / "scripts/eval/fixtures/test_video_2_events.json"
    if gt.exists() and ev:
        import subprocess

        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/eval/score_events.py"),
                "--pred",
                str(out / "events.json"),
                "--gt",
                str(gt),
            ],
            capture_output=True,
            text=True,
        )
        (out / "score.json").write_text(r.stdout or r.stderr)
        print(r.stdout, flush=True)

    print(f"DONE wall={time.perf_counter()-t0:.0f}s out={out}", flush=True)
    (out / "DONE").write_text("ok\n")


def _empty_row(track_key: str) -> dict:
    return {
        "track_key": track_key,
        "jersey_number": None,
        "jersey_confidence": None,
        "thumb_path": None,
        "duration_ms": 0,
        "bbox_samples": 0,
        "passes": 0,
        "shots": 0,
        "goals": 0,
        "possession_ms": 0,
    }


if __name__ == "__main__":
    main()
