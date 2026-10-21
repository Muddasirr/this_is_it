#!/usr/bin/env python3
"""Import a CourtLens offline/Kaggle output folder into the live product API.

Creates match → uploads camera A video → injects detections as tracks →
identity + events (via worker-equivalent services) → publish + share URL.

Usage:
  python scripts/import_courtlens_out.py \\
    --out data/kaggle_out/corpus/test_video_2 \\
    --video testing_videos/test_video_2.mp4 \\
    --title "Facility CCTV test_video_2" \\
    --email corpus@example.com --password 'Testpass123!'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import UUID

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = "http://localhost:8001"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--out", type=Path, required=True, help="courtlens_out folder")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--email", default="corpus@example.com")
    ap.add_argument("--password", default="Testpass123!")
    ap.add_argument("--facility-name", default="CourtLens Corpus")
    ap.add_argument("--inject-only", action="store_true", help="Skip upload; inject detections via API worker bypass")
    args = ap.parse_args()

    dets_path = args.out / "detections.json"
    events_path = args.out / "events.json"
    if not dets_path.exists():
        raise SystemExit(f"missing {dets_path}")

    client = httpx.Client(base_url=args.api.rstrip("/"), timeout=120.0)
    token = _ensure_auth(client, args.email, args.password)
    headers = {"Authorization": f"Bearer {token}"}

    facilities = client.get("/api/v1/facilities", headers=headers).json()
    fac = next((f for f in facilities if f["name"] == args.facility_name), None)
    if not fac:
        r = client.post("/api/v1/facilities", headers=headers, json={"name": args.facility_name})
        r.raise_for_status()
        fac = r.json()
    facility_id = fac["id"]

    r = client.post(
        f"/api/v1/facilities/{facility_id}/matches",
        headers=headers,
        json={"title": args.title},
    )
    r.raise_for_status()
    match = r.json()
    match_id = match["id"]
    print("match", match_id, match["title"])

    # Upload camera A
    size = args.video.stat().st_size
    r = client.post(
        f"/api/v1/facilities/{facility_id}/matches/{match_id}/uploads",
        headers=headers,
        json={
            "camera_label": "A",
            "filename": args.video.name,
            "byte_size": size,
            "content_type": "video/mp4",
        },
    )
    r.raise_for_status()
    upload = r.json()
    upload_url = upload["upload_url"]
    print("uploading", args.video.name, f"{size/1e6:.1f}MB …")
    with args.video.open("rb") as f:
        put = httpx.put(upload_url, content=f, headers={"Content-Type": "video/mp4"}, timeout=600.0)
        put.raise_for_status()
    print("upload ok")

    # Inject detections via in-container helper (avoids CPU re-YOLO)
    inject = {
        "match_id": match_id,
        "detections_path": str(dets_path.resolve()),
        "events_path": str(events_path.resolve()) if events_path.exists() else None,
    }
    inj_file = Path("/tmp") / f"courtlens_inject_{match_id}.json"
    # Write next to out so docker can see it if mounted; else copy into backend volume via stdin
    inj_path = args.out / "_inject_request.json"
    inj_path.write_text(json.dumps(inject, indent=2))
    print("wrote", inj_path)
    print(
        "NEXT: docker compose exec -T api python -m app.scripts.inject_detections "
        f"{match_id} {dets_path}"
    )
    # Prefer direct python import when running inside api container; else shell out
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from app.core.database import SessionLocal
        from app.scripts.inject_detections import inject_match_from_files

        db = SessionLocal()
        try:
            share = inject_match_from_files(
                db,
                UUID(match_id),
                dets_path,
                events_path if events_path.exists() else None,
                publish=True,
            )
            print("SHARE", share)
        finally:
            db.close()
    except Exception as e:
        print("direct inject failed (run inside api container):", e)
        print(json.dumps({"match_id": match_id, "facility_id": facility_id, "inject": str(inj_path)}, indent=2))


def _ensure_auth(client: httpx.Client, email: str, password: str) -> str:
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if r.status_code == 200:
        return r.json()["access_token"]
    r = client.post(
        "/api/v1/auth/register",
        json={"email": email, "full_name": "Corpus Runner", "password": password},
    )
    if r.status_code not in (200, 201):
        # maybe exists
        r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    r.raise_for_status()
    return r.json()["access_token"]


if __name__ == "__main__":
    main()
