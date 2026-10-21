#!/usr/bin/env python3
"""Upload corpus videos + inject precomputed detections into the live UI."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://localhost:8001"
EMAIL = "corpus179@example.com"
PASSWORD = "Testpass123!"
FACILITY = "CourtLens Corpus"

# (title, video path, detections.json, events.json)
JOBS = [
    (
        "Facility CCTV — test_video_2",
        ROOT / "testing_videos/test_video_2.mp4",
        ROOT / "data/kaggle_out/v22/courtlens_out/detections.json",
        ROOT / "backend/data/corpus_inject/test_video_2/events.json",
    ),
    (
        "Portugal vs Spain — Euro 2026 FINAL highlights",
        ROOT / "testing_videos/Portugal vs Spain Highlights _ Futsal Euro 2026 FINAL.mp4",
        ROOT / "data/kaggle_out/portugal_v25/courtlens_out/detections.json",
        ROOT / "backend/data/corpus_inject/portugal_spain_euro_2026_final/events.json",
    ),
    (
        "Bangladesh vs Pakistan — SAFF 2026 highlights",
        ROOT / "testing_videos/Bangladesh vs Pakistan Futsal Highlights _ Pakistan Wins 5-1 _SAFF Futsal Championship 2026.mp4",
        ROOT / "data/kaggle_out/bangladesh_v23/courtlens_out/detections.json",
        ROOT / "backend/data/corpus_inject/bangladesh_pakistan_saff_2026/events.json",
    ),
]


def auth(client: httpx.Client) -> str:
    r = client.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
    if r.status_code != 200:
        client.post(
            "/api/v1/auth/register",
            json={"email": EMAIL, "full_name": "Corpus", "password": PASSWORD},
        )
        r = client.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> None:
    client = httpx.Client(base_url=API, timeout=120.0)
    token = auth(client)
    h = {"Authorization": f"Bearer {token}"}

    facs = client.get("/api/v1/facilities", headers=h).json()
    fac = next((f for f in facs if f["name"] == FACILITY), None)
    if not fac:
        fac = client.post("/api/v1/facilities", headers=h, json={"name": FACILITY}).json()
    facility_id = fac["id"]
    print("facility", facility_id)

    reports = []
    for title, video, dets, events in JOBS:
        if not dets.exists():
            print("SKIP missing detections", dets)
            continue
        if not video.exists():
            print("SKIP missing video", video)
            continue
        r = client.post(
            f"/api/v1/facilities/{facility_id}/matches",
            headers=h,
            json={"title": title},
        )
        r.raise_for_status()
        match = r.json()
        mid = match["id"]
        print("match", mid, title)

        size = video.stat().st_size
        up = client.post(
            f"/api/v1/facilities/{facility_id}/matches/{mid}/videos/upload-url",
            headers=h,
            json={
                "camera_label": "A",
                "filename": video.name,
                "content_type": "video/mp4",
                "byte_size": size,
            },
        )
        up.raise_for_status()
        upload_url = up.json()["upload_url"]
        print(f"  uploading {size/1e6:.0f}MB …")
        with video.open("rb") as f:
            put = httpx.put(upload_url, content=f, headers={"Content-Type": "video/mp4"}, timeout=900.0)
            put.raise_for_status()
        print("  upload ok")

        # Copy into backend mount so api container can read
        dest_dir = ROOT / "backend/data/corpus_inject" / "_live" / mid
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_dets = dest_dir / "detections.json"
        dest_ev = dest_dir / "events.json"
        dest_dets.write_bytes(dets.read_bytes())
        if events.exists():
            dest_ev.write_bytes(events.read_bytes())
        else:
            dest_ev.write_text("[]")

        cmd = [
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
        ]
        print("  inject…")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout[-800:] if proc.stdout else "")
        if proc.returncode != 0:
            print("STDERR", proc.stderr[-1200:])
            continue
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            # full stdout is json
            try:
                result = json.loads(proc.stdout)
            except Exception:
                result = {"raw": proc.stdout}
        token_s = result.get("share_token")
        url = f"http://localhost:5174/report/{token_s}" if token_s else None
        reports.append({"title": title, "match_id": mid, "share": url, "result": result})
        print("  report", url)

    out = ROOT / "data/eval/corpus_ui_reports.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print("wrote", out)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
