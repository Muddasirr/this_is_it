#!/usr/bin/env python3
"""After Kaggle corpus_v26 finishes: inject Spain-Brazil + Spain-France into UI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://localhost:8001"
EMAIL = "corpus179@example.com"
PASSWORD = "Testpass123!"
FACILITY_ID = "f387eef5-9780-4356-b4d3-467b8d08e620"

VIDEO_MAP = {
    "spain_brazil_fifa_2012": ROOT
    / "testing_videos/CRAZY ENDING! Spain vs Brazil _ 2012 FIFA Futsal World Cup.mp4",
    "spain_france_u19_2023": ROOT
    / "testing_videos/Spain vs France _ Highlights _ U19 Euro Futsal 06-09-2023.mp4",
}
TITLES = {
    "spain_brazil_fifa_2012": "Spain vs Brazil — 2012 FIFA Futsal World Cup highlights",
    "spain_france_u19_2023": "Spain vs France — U19 Euro Futsal 2023 highlights",
}


def main() -> None:
    root = ROOT / "data/kaggle_out/corpus_v26"
    outs = list(root.rglob("detections.json"))
    if not outs:
        raise SystemExit(f"No detections under {root} — wait for Kaggle pull")

    token = httpx.post(f"{API}/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD}).json()[
        "access_token"
    ]
    h = {"Authorization": f"Bearer {token}"}
    client = httpx.Client(base_url=API, timeout=120.0)

    reports = []
    for det in outs:
        slug = det.parent.name
        if slug not in VIDEO_MAP:
            continue
        video = VIDEO_MAP[slug]
        events = det.parent / "events.json"
        title = TITLES[slug]
        r = client.post(
            f"/api/v1/facilities/{FACILITY_ID}/matches",
            headers=h,
            json={"title": title},
        )
        r.raise_for_status()
        mid = r.json()["id"]
        size = video.stat().st_size
        up = client.post(
            f"/api/v1/facilities/{FACILITY_ID}/matches/{mid}/videos/upload-url",
            headers=h,
            json={
                "camera_label": "A",
                "filename": video.name,
                "content_type": "video/mp4",
                "byte_size": size,
            },
        )
        up.raise_for_status()
        with video.open("rb") as f:
            httpx.put(up.json()["upload_url"], content=f, headers={"Content-Type": "video/mp4"}, timeout=900.0).raise_for_status()

        dest = ROOT / "backend/data/corpus_inject/_live" / mid
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "detections.json").write_bytes(det.read_bytes())
        (dest / "events.json").write_bytes(events.read_bytes() if events.exists() else b"[]")
        # thumbs
        thumbs_src = det.parent / "thumbs"
        if thumbs_src.is_dir():
            tdest = dest / "thumbs"
            tdest.mkdir(exist_ok=True)
            for f in thumbs_src.glob("*.jpg"):
                (tdest / f.name).write_bytes(f.read_bytes())

        proc = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(ROOT / "docker-compose.yml"),
                "exec",
                "-T",
                "api",
                "python",
                "-m",
                "app.scripts.inject_detections",
                mid,
                f"/app/data/corpus_inject/_live/{mid}/detections.json",
                f"/app/data/corpus_inject/_live/{mid}/events.json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            continue
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        reports.append({"title": title, "share": f"http://localhost:5174{result.get('report_path')}"})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
