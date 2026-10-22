#!/usr/bin/env python3
"""Run CourtLens offline pipeline across testing_videos/ (smoke or full).

Examples:
  # Local smoke (~few minutes): first 240 processed frames, stride 15, OCR off
  python scripts/run_testing_videos.py --smoke

  # One clip only
  python scripts/run_testing_videos.py --match Portugal --smoke

  # Full GPU-style (stride 5, OCR on) for selected clips
  VISION_DEVICE=0 python scripts/run_testing_videos.py --match test_video_2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "scripts" / "kaggle" / "courtlens_offline"
VIDEOS = ROOT / "testing_videos"
DEFAULT_OUT = ROOT / "data" / "corpus_out"


def _slug(name: str) -> str:
    base = Path(name).stem
    s = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return (s[:80] or "clip")


def list_videos(match: str | None) -> list[Path]:
    vids = sorted(VIDEOS.glob("*.mp4"))
    if not vids:
        raise SystemExit(f"No mp4 under {VIDEOS}")
    if match:
        needle = match.lower()
        vids = [v for v in vids if needle in v.name.lower()]
        if not vids:
            raise SystemExit(f"No video matching {match!r} in {VIDEOS}")
    return vids


def run_one(video: Path, out_root: Path, *, smoke: bool) -> dict:
    if str(OFFLINE) not in sys.path:
        sys.path.insert(0, str(OFFLINE))

    slug = _slug(video.name)
    out_dir = out_root / slug
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    default_v5x = ROOT / "data" / "eval" / "football_analysis_model" / "best.pt"
    weights = os.environ.get("COURTLENS_WEIGHTS") or (
        str(default_v5x) if default_v5x.is_file() else str(ROOT / "yolov8n.pt")
    )
    os.environ["COURTLENS_VIDEO"] = str(video)
    os.environ["COURTLENS_WEIGHTS"] = weights
    os.environ["COURTLENS_OUT"] = str(out_dir)
    os.environ["COURTLENS_CAMERA"] = "A"

    if smoke:
        # Force (do not setdefault) — shell may already export VISION_* from prior runs
        os.environ["VISION_FRAME_STRIDE"] = os.environ.get("SMOKE_FRAME_STRIDE", "15")
        os.environ["VISION_MAX_FRAMES"] = os.environ.get("SMOKE_MAX_FRAMES", "240")
        os.environ["VISION_JERSEY_OCR"] = "0"
        os.environ.setdefault("VISION_IMGSZ", "640")
    else:
        os.environ.setdefault("VISION_FRAME_STRIDE", "5")
        os.environ.setdefault("VISION_MAX_FRAMES", "0")
        os.environ.setdefault("VISION_JERSEY_OCR", "1")

    # Prefer CUDA when available unless caller pinned device
    if "VISION_DEVICE" not in os.environ:
        try:
            import torch

            os.environ["VISION_DEVICE"] = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            os.environ["VISION_DEVICE"] = "cpu"

    import run as courtlens_run

    print("=" * 72)
    print(f"VIDEO {video.name}")
    print(f"OUT   {out_dir}")
    print(
        "ENV   stride=",
        os.environ.get("VISION_FRAME_STRIDE"),
        "max_frames=",
        os.environ.get("VISION_MAX_FRAMES"),
        "ocr=",
        os.environ.get("VISION_JERSEY_OCR"),
        "device=",
        os.environ.get("VISION_DEVICE"),
    )
    courtlens_run.main()

    table_path = out_dir / "player_table.json"
    totals_path = out_dir / "match_stats.json"
    summary = {
        "video": video.name,
        "slug": slug,
        "out": str(out_dir),
        "totals": {},
        "players": 0,
        "thumbs": 0,
    }
    if totals_path.exists():
        stats = json.loads(totals_path.read_text())
        summary["totals"] = stats.get("totals") or {}
        summary["players"] = len(stats.get("players") or [])
        summary["detection"] = stats.get("detection") or {}
    if table_path.exists():
        table = json.loads(table_path.read_text())
        summary["players"] = len(table)
    thumbs = out_dir / "thumbs"
    if thumbs.is_dir():
        summary["thumbs"] = len(list(thumbs.glob("*.jpg")))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", help="Substring filter on filename (case-insensitive)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Fast local pass: stride 15, max 240 frames, OCR off",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root (default {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List corpus videos and exit",
    )
    args = ap.parse_args()

    vids = list_videos(args.match)
    if args.list:
        for v in vids:
            print(v.name)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for v in vids:
        try:
            results.append(run_one(v, args.out, smoke=args.smoke))
        except Exception as e:
            results.append({"video": v.name, "error": str(e)})
            print("FAILED", v.name, e)

    index = args.out / ("corpus_smoke.json" if args.smoke else "corpus_index.json")
    index.write_text(json.dumps(results, indent=2))
    print("=" * 72)
    print("CORPUS SUMMARY")
    print(json.dumps(results, indent=2))
    print("wrote", index)


if __name__ == "__main__":
    main()
