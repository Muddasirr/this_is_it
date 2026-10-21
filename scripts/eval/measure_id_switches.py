#!/usr/bin/env python3
"""Compare ByteTrack vs BoT-SORT(+OSNet) ID stability on probe clips.

Metrics (no human GT required):
  - person_tracks: unique track IDs
  - id_assignment_pct: % of person boxes that received a tracker ID
  - mean_track_duration_ms / median
  - short_tracks: tracks with < N bbox samples (fragment proxy)
  - reid_mergeable_pairs: pairs with ReID sim ≥ thresh and low time overlap
  - estimated_id_switches: short_tracks + (tracks - after_reid_merge estimate)

Usage:
  cd /path/to/paas && PYTHONPATH=backend python scripts/eval/measure_id_switches.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

OUT = ROOT / "data" / "eval" / "tracker_compare"
VIDEOS = [
    ("08fd33_4", ROOT / "testing_videos" / "08fd33_4.mp4"),
    (
        "bangladesh_pakistan_saff_2026",
        ROOT
        / "testing_videos"
        / "Bangladesh vs Pakistan Futsal Highlights _ Pakistan Wins 5-1 _SAFF Futsal Championship 2026.mp4",
    ),
]


def _track_stats(detections: list, reid_sim: float = 0.72, max_overlap_ms: int = 800) -> dict:
    import numpy as np
    from app.services.reid import cosine

    persons = [d for d in detections if d.class_name == "person"]
    durations = [max(0, d.end_ms - d.start_ms) for d in persons]
    samples = [len(d.bbox_series or []) for d in persons]
    short = sum(1 for n in samples if n < 5)

    # ReID mergeable pairs (proxy for recoverable ID switches)
    embs = []
    for d in persons:
        e = (d.features or {}).get("reid_emb")
        if e:
            embs.append((d, e))
    mergeable = 0
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            di, ei = embs[i]
            dj, ej = embs[j]
            sim = cosine(ei, ej)
            start = max(di.start_ms, dj.start_ms)
            end = min(di.end_ms, dj.end_ms)
            overlap = max(0, end - start)
            if sim >= reid_sim and overlap <= max_overlap_ms:
                mergeable += 1

    return {
        "person_tracks": len(persons),
        "mean_duration_ms": round(float(np.mean(durations)), 1) if durations else 0,
        "median_duration_ms": round(float(np.median(durations)), 1) if durations else 0,
        "mean_bbox_samples": round(float(np.mean(samples)), 2) if samples else 0,
        "short_tracks_lt5": short,
        "short_track_pct": round(100.0 * short / max(1, len(persons)), 1),
        "reid_embedded": len(embs),
        "reid_mergeable_pairs": mergeable,
        # Rough ID-switch proxy: short fragments + recoverable ReID pairs
        "id_switch_proxy": short + mergeable,
    }


def run_one(video: Path, tracker: str, reid: bool, max_frames: int, stride: int) -> dict:
    os.environ["VISION_TRACKER"] = tracker
    os.environ["VISION_REID"] = "1" if reid else "0"
    os.environ["VISION_FRAME_STRIDE"] = str(stride)
    os.environ["VISION_MAX_FRAMES"] = str(max_frames)
    os.environ["VISION_DEVICE"] = os.environ.get("VISION_DEVICE", "cpu")
    os.environ["YOLO_MODEL"] = str(ROOT / "yolov8n.pt")
    os.environ["VISION_JERSEY_OCR"] = "0"  # keep OCR out of this gate

    # Clear settings cache
    from app.core import config as cfg

    cfg.get_settings.cache_clear()

    from app.services.yolo_backend import YoloTrackBackend

    backend = YoloTrackBackend()
    t0 = time.perf_counter()
    dets = backend.detect_and_track(str(video), "A")
    elapsed = time.perf_counter() - t0
    stats = dict(backend.last_run_stats)
    stats["wall_sec"] = round(elapsed, 2)
    stats["tracker_requested"] = tracker
    stats["reid_requested"] = reid
    stats.update(_track_stats(dets))
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Short probes for CPU: full 08fd33 (750f), Bangladesh first ~20s worth of processed frames
    # stride=5 → 150 frames for 08fd33; bangladesh cap at 120 processed frames (~20s@30fps stride5)
    plans = [
        ("08fd33_4", VIDEOS[0][1], 0, 5),  # full 30s
        ("bangladesh_pakistan_saff_2026", VIDEOS[1][1], 120, 5),
    ]
    configs = [
        ("bytetrack", False),
        ("botsort", True),
    ]
    report: dict = {"runs": [], "notes": (
        "id_assignment_pct = person boxes with tracker ID. "
        "short_tracks_lt5 = fragment proxy. "
        "reid_mergeable_pairs = same-person candidates OSNet can merge. "
        "id_switch_proxy = short_tracks + mergeable (lower is better)."
    )}

    for slug, path, max_frames, stride in plans:
        if not path.exists():
            report["runs"].append({"slug": slug, "error": f"missing {path}"})
            continue
        for tracker, reid in configs:
            tag = f"{slug}__{tracker}{'_reid' if reid else ''}"
            print(f"RUN {tag} max_frames={max_frames} …", flush=True)
            try:
                stats = run_one(path, tracker, reid, max_frames, stride)
                stats["slug"] = slug
                stats["tag"] = tag
                report["runs"].append(stats)
                print(
                    f"  tracks={stats['person_tracks']} id%={stats.get('id_assignment_pct')} "
                    f"short={stats['short_tracks_lt5']} mergeable={stats['reid_mergeable_pairs']} "
                    f"proxy={stats['id_switch_proxy']} wall={stats['wall_sec']}s",
                    flush=True,
                )
            except Exception as e:
                report["runs"].append({"slug": slug, "tag": tag, "error": str(e)})
                print(f"  ERR {e}", flush=True)

    # Compact comparison table
    by_slug: dict[str, list] = {}
    for r in report["runs"]:
        if "error" in r and "person_tracks" not in r:
            continue
        by_slug.setdefault(r["slug"], []).append(r)
    compare = {}
    for slug, rows in by_slug.items():
        compare[slug] = {
            r["tag"]: {
                "person_tracks": r.get("person_tracks"),
                "id_assignment_pct": r.get("id_assignment_pct"),
                "short_tracks_lt5": r.get("short_tracks_lt5"),
                "reid_mergeable_pairs": r.get("reid_mergeable_pairs"),
                "id_switch_proxy": r.get("id_switch_proxy"),
                "mean_duration_ms": r.get("mean_duration_ms"),
            }
            for r in rows
        }
    report["compare"] = compare

    out_json = OUT / "id_switch_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    html = [
        "<!doctype html><html><head><meta charset=utf-8><title>Tracker ID compare</title>",
        "<style>body{font-family:system-ui;background:#111;color:#eee;margin:24px}",
        "table{border-collapse:collapse}td,th{border:1px solid #333;padding:8px}</style></head><body>",
        "<h1>ByteTrack vs BoT-SORT+OSNet</h1>",
        f"<p>{report['notes']}</p><table><tr><th>run</th><th>tracks</th><th>id%</th>"
        "<th>short&lt;5</th><th>reid mergeable</th><th>id_switch_proxy</th><th>mean dur ms</th></tr>",
    ]
    for r in report["runs"]:
        if "person_tracks" not in r:
            html.append(f"<tr><td>{r.get('tag')}</td><td colspan=6>{r.get('error')}</td></tr>")
            continue
        html.append(
            f"<tr><td>{r['tag']}</td><td>{r['person_tracks']}</td><td>{r.get('id_assignment_pct')}</td>"
            f"<td>{r['short_tracks_lt5']}</td><td>{r['reid_mergeable_pairs']}</td>"
            f"<td>{r['id_switch_proxy']}</td><td>{r['mean_duration_ms']}</td></tr>"
        )
    html.append("</table></body></html>")
    (OUT / "index.html").write_text("\n".join(html))
    print("WROTE", out_json)
    print(json.dumps(compare, indent=2))


if __name__ == "__main__":
    main()
