#!/usr/bin/env python3
"""Load every run of a Kaggle GPU eval into the local app (one match per video).

  python scripts/import_kaggle_run.py data/kaggle_out/gpu_eval_v5/eval \\
      [--facility-name "CourtLens Kaggle eval"] [--only dfl_] [--api http://localhost:8001]

For each <slug>/summary.json: find the source video locally (testing_videos/,
scripts/kaggle/dataset/, or download the single DFL clip from Kaggle), stage the
JSON under backend/data/imports/ (the api container only mounts ./backend), then
run scripts/import_courtlens_out.py (create match → upload → inject → publish).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DFL_SLUG = "saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv"
DFL_CACHE = ROOT / "data" / "dfl_cache"
# Short names on our Kaggle dataset → local corpus files
LOCAL_NAMES = {
    "portugal_spain_euro_2026_final.mp4": "Portugal vs Spain Highlights _ Futsal Euro 2026 FINAL.mp4",
    "bangladesh_pakistan_saff_2026.mp4": "Bangladesh vs Pakistan Futsal Highlights _ Pakistan Wins 5-1 _SAFF Futsal Championship 2026.mp4",
    "spain_brazil_fifa_2012.mp4": "CRAZY ENDING! Spain vs Brazil _ 2012 FIFA Futsal World Cup.mp4",
    "spain_france_u19_2023.mp4": "Spain vs France _ Highlights _ U19 Euro Futsal 06-09-2023.mp4",
}


def local_video(kaggle_path: str) -> Path | None:
    name = Path(kaggle_path).name
    for cand in (
        ROOT / "testing_videos" / LOCAL_NAMES.get(name, name),
        ROOT / "scripts" / "kaggle" / "dataset" / name,
        DFL_CACHE / name,
    ):
        if cand.is_file():
            return cand
    marker = f"/{DFL_SLUG.split('/')[1]}/"
    if marker in kaggle_path:
        rel = kaggle_path.split(marker, 1)[1]
        DFL_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", DFL_SLUG, "-f", rel, "-p", str(DFL_CACHE), "--unzip", "-q"],
            check=False,
        )
        # The CLI may leave a URL-encoded .zip (e.g. "A%20(1).mp4.zip") instead of unzipping
        import zipfile
        from urllib.parse import unquote

        for z in DFL_CACHE.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                for member in zf.namelist():
                    target = DFL_CACHE / unquote(Path(member).name)
                    target.write_bytes(zf.read(member))
            z.unlink()
        hits = list(DFL_CACHE.rglob(name))
        if hits:
            return hits[0]
    return None


def title_for(slug: str, run_name: str) -> str:
    if slug.startswith("dfl_"):
        return f"DFL {slug[4:]} · {run_name}"
    return f"{slug.replace('_', ' ')} · {run_name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="…/eval folder from `kaggle kernels output`")
    ap.add_argument("--facility-name", default="CourtLens Kaggle eval")
    ap.add_argument("--email", default="corpus179@example.com")
    ap.add_argument("--password", default="Testpass123!")
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--only", default="", help="comma-separated slug substrings")
    args = ap.parse_args()

    run_name = args.run.resolve().parent.name  # e.g. gpu_eval_v5
    stage_root = ROOT / "backend" / "data" / "imports" / run_name
    results = []
    for summary_path in sorted(args.run.glob("*/summary.json")):
        slug = summary_path.parent.name
        if args.only and not any(o and o in slug for o in args.only.split(",")):
            continue
        if not (summary_path.parent / "detections.json").exists():
            print(f"SKIP {slug}: detections.json not downloaded")
            results.append({"slug": slug, "skipped": "no detections"})
            continue
        summary = json.loads(summary_path.read_text())
        src = (summary.get("cameras") or {}).get("A")
        video = local_video(src) if src else None
        if video is None:
            print(f"SKIP {slug}: source video not available locally ({src})")
            results.append({"slug": slug, "skipped": "no video"})
            continue
        stage = stage_root / slug
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("detections.json", "events.json", "summary.json"):
            shutil.copy2(summary_path.parent / name, stage / name)
        print(f"=== {slug} ← {video.name}")
        proc = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "import_courtlens_out.py"),
                "--api", args.api, "--out", str(stage), "--video", str(video),
                "--title", title_for(slug, run_name), "--facility-name", args.facility_name,
                "--email", args.email, "--password", args.password,
            ],
            capture_output=True, text=True,
        )
        ui = next((l.split("UI match:", 1)[1].strip() for l in proc.stdout.splitlines() if "UI match:" in l), None)
        print(proc.stdout[-600:] if proc.returncode else f"  ok → {ui}")
        if proc.returncode:
            print(proc.stderr[-1500:])
        results.append({"slug": slug, "rc": proc.returncode, "ui": ui})
    (stage_root / "import_index.json").parent.mkdir(parents=True, exist_ok=True)
    (stage_root / "import_index.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
