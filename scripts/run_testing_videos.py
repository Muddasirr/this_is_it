#!/usr/bin/env python3
"""Run the real CourtLens pipeline (backend app.scripts.run_offline) across testing_videos/.

Examples:
  # Local smoke (~few minutes): first 240 processed frames, stride 15, OCR off
  python scripts/run_testing_videos.py --smoke
  # One clip only
  python scripts/run_testing_videos.py --match Portugal --smoke
  # Full GPU-style (stride 5, OCR on) — prefer the Kaggle kernel for this
  VISION_DEVICE=0 python scripts/run_testing_videos.py --match test_video_2

Byte-identical duplicates (the DFL "extra" downloads repeat clips) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
VIDEOS = ROOT / "testing_videos"
DEFAULT_OUT = ROOT / "data" / "corpus_out"


def _slug(name: str) -> str:
    base = Path(name).stem
    s = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return (s[:80] or "clip")


def _fingerprint(path: Path) -> tuple[int, str]:
    with path.open("rb") as f:
        head = f.read(1 << 20)
    return path.stat().st_size, hashlib.sha1(head).hexdigest()


def list_videos(match: str | None) -> list[Path]:
    vids: list[Path] = []
    seen: dict[tuple[int, str], Path] = {}
    for v in sorted(VIDEOS.glob("*.mp4")):
        fp = _fingerprint(v)
        if fp in seen:
            continue  # duplicate of seen[fp]
        seen[fp] = v
        vids.append(v)
    if not vids:
        raise SystemExit(f"No mp4 under {VIDEOS}")
    if match:
        needle = match.lower()
        vids = [v for v in vids if needle in v.name.lower()]
        if not vids:
            raise SystemExit(f"No video matching {match!r} in {VIDEOS}")
    return vids


def run_one(video: Path, out_root: Path, *, smoke: bool) -> dict:
    slug = _slug(video.name)
    out_dir = out_root / slug
    if out_dir.exists():
        shutil.rmtree(out_dir)

    env = dict(os.environ)
    default_v5x = ROOT / "data" / "eval" / "football_analysis_model" / "best.pt"
    env.setdefault("YOLO_MODEL", str(default_v5x) if default_v5x.is_file() else "yolov8n.pt")
    if smoke:
        # Force (do not setdefault) — shell may already export VISION_* from prior runs
        env["VISION_FRAME_STRIDE"] = os.environ.get("SMOKE_FRAME_STRIDE", "15")
        env["VISION_MAX_FRAMES"] = os.environ.get("SMOKE_MAX_FRAMES", "240")
        env["VISION_JERSEY_OCR"] = "0"
    else:
        env.setdefault("VISION_FRAME_STRIDE", "5")
        env.setdefault("VISION_MAX_FRAMES", "0")
        env.setdefault("VISION_JERSEY_OCR", "1")
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")

    print("=" * 72)
    print(f"VIDEO {video.name}")
    print(f"OUT   {out_dir}")
    cmd = [sys.executable, "-m", "app.scripts.run_offline", "--out", str(out_dir), "--camera", f"A={video}"]
    subprocess.run(cmd, check=True, cwd=BACKEND, env=env)

    summary = json.loads((out_dir / "summary.json").read_text())
    stats = json.loads((out_dir / "match_stats.json").read_text())
    return {
        "video": video.name,
        "slug": slug,
        "out": str(out_dir),
        "counts": summary.get("counts"),
        "wall_seconds": summary.get("wall_seconds"),
        "totals": stats.get("totals") or {},
    }


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
