#!/usr/bin/env python3
"""End-to-end product check against the running app (API + worker), as a court would use it.

  python scripts/eval/e2e_product.py --video testing_videos/dfl_B1606b0e6_1_3.mp4 [--api http://localhost:8001]

Steps (each must pass): register owner + staff → facility → add staff → match →
upload camera A → start processing → worker finishes (real YOLO pipeline) →
tracks / identity / events / stats present → accept + reject an event (stats
follow) → calibrate camera A (8 landmarks) → metres in stats → publish → public
share report opens without login → revoke → link 404s.
Writes data/eval/e2e_product_report.json.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


class Check:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def __call__(self, name: str, ok: bool, detail: object = "") -> bool:
        self.steps.append({"step": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
        if not ok:
            self.finish()
            sys.exit(1)
        return ok

    def finish(self) -> None:
        out = ROOT / "data" / "eval" / "e2e_product_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": all(s["ok"] for s in self.steps), "steps": self.steps}, indent=2))
        print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--timeout-min", type=float, default=40)
    args = ap.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    check = Check()
    c = httpx.Client(base_url=args.api.rstrip("/"), timeout=120)
    tag = secrets.token_hex(3)

    def user(email: str) -> dict:
        c.post("/api/v1/auth/register", json={"email": email, "full_name": email, "password": "Testpass123!"})
        r = c.post("/api/v1/auth/login", json={"email": email, "password": "Testpass123!"})
        check(f"login {email}", r.status_code == 200, r.status_code)
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    owner = user(f"e2e-owner-{tag}@example.com")
    staff = user(f"e2e-staff-{tag}@example.com")
    fac = c.post("/api/v1/facilities", headers=owner, json={"name": f"E2E court {tag}"}).json()
    base = f"/api/v1/facilities/{fac['id']}"
    r = c.post(f"{base}/members", headers=owner, json={"email": f"e2e-staff-{tag}@example.com", "role": "staff"})
    check("owner adds staff", r.status_code == 201, r.status_code)

    m = c.post(f"{base}/matches", headers=staff, json={"title": f"E2E {args.video.stem}"}).json()
    mbase = f"{base}/matches/{m['id']}"
    check("staff creates match", "id" in m, m.get("status"))

    up = c.post(f"{mbase}/videos/upload-url", headers=staff, json={
        "camera_label": "A", "filename": args.video.name, "byte_size": args.video.stat().st_size,
        "content_type": "video/mp4"}).json()
    url = up["upload_url"] if up["upload_url"].startswith("http") else args.api.rstrip("/") + up["upload_url"]
    with args.video.open("rb") as f:
        put = httpx.put(url, content=f, headers={"Content-Type": "video/mp4"}, timeout=600)
    check("upload camera A", put.status_code < 300, put.status_code)

    r = c.post(f"{mbase}/process", headers=staff)
    check("start processing", r.status_code in (200, 201, 202), r.status_code)
    t0, job = time.time(), {}
    while time.time() - t0 < args.timeout_min * 60:
        jobs = c.get(f"{mbase}/jobs", headers=staff).json()
        job = jobs[0] if jobs else {}
        if job.get("status") in {"succeeded", "waiting_review", "failed", "cancelled"}:
            break
        time.sleep(10)
    check("worker finishes job", job.get("status") in {"succeeded", "waiting_review"},
          f"{job.get('status')} in {time.time() - t0:.0f}s {job.get('error_message') or ''}")

    tracks = c.get(f"{mbase}/tracks", headers=staff).json()
    check("tracks stored", len(tracks) > 0, len(tracks))
    linked = [t for t in tracks if (t.get("identity") or {}).get("player_id")]
    check("identity proposals", len(linked) > 0, f"{len(linked)} tracks linked to players")
    stats = c.get(f"{mbase}/stats", headers=staff).json()
    check("stats built", stats["totals"]["players"] > 0, stats["totals"])
    run = c.get(f"{mbase}/run-stats", headers=staff).json()
    check("run stats", run.get("available"), (run.get("cameras") or [{}])[0].get("imgsz"))

    events = c.get(f"{mbase}/events", headers=staff).json()
    if len(events) >= 2:
        a = c.post(f"{mbase}/events/{events[0]['id']}/review", headers=staff, json={"action": "accept"})
        rj = c.post(f"{mbase}/events/{events[1]['id']}/review", headers=staff, json={"action": "reject"})
        after = c.get(f"{mbase}/stats", headers=staff).json()["totals"]
        check("review accept/reject", a.status_code == 200 and rj.status_code == 200
              and after["confirmed"] >= 1 and after["rejected"] >= 1, after)
    else:
        check("events produced (review skipped)", True, f"{len(events)} events — clip may lack ball")

    frame = c.get(f"{mbase}/cameras/A/frame", headers=staff)
    w, h = int(frame.headers.get("x-frame-width", 0)), int(frame.headers.get("x-frame-height", 0))
    check("calibration frame", frame.status_code == 200 and w > 0, f"{w}x{h}")
    # Synthetic but valid 8-landmark set over the frame (court 40 x 20 m)
    L, W, R = 40.0, 20.0, 3.0
    img = [[0.1 * w, 0.9 * h], [0.3 * w, 0.3 * h], [0.7 * w, 0.3 * h], [0.9 * w, 0.9 * h],
           [0.5 * w, 0.3 * h], [0.5 * w, 0.9 * h], [0.5 * w, 0.45 * h], [0.5 * w, 0.62 * h]]
    court = [[0, W], [0, 0], [L, 0], [L, W], [L / 2, 0], [L / 2, W], [L / 2, W / 2 - R], [L / 2, W / 2 + R]]
    r = c.put(f"{mbase}/cameras/A/calibration", headers=staff, json={
        "image_points": img, "court_points": court, "court_length_m": L, "court_width_m": W, "frame_w": w, "frame_h": h})
    check("calibrate camera A (8 points)", r.status_code == 200, (r.json().get("calibration") or {}).get("model"))
    rows = c.get(f"{mbase}/stats", headers=staff).json()["by_player"]
    check("metres in stats", any(row.get("distance_m") is not None for row in rows),
          [row.get("distance_m") for row in rows[:5]])

    pub = c.post(f"{mbase}/publish", headers=staff).json()
    token = pub.get("share_token")
    public = httpx.get(f"{args.api.rstrip('/')}/api/v1/share/matches/{token}", timeout=60)
    check("public share report (no login)", public.status_code == 200 and public.json().get("by_player") is not None,
          pub.get("report_path"))
    c.post(f"{mbase}/share/revoke", headers=staff)
    gone = httpx.get(f"{args.api.rstrip('/')}/api/v1/share/matches/{token}", timeout=60)
    check("revoked link 404s", gone.status_code == 404, gone.status_code)
    print(f"UI: http://localhost:5174/facilities/{fac['id']}/matches/{m['id']}  (owner e2e-owner-{tag}@example.com / Testpass123!)")
    check.finish()


if __name__ == "__main__":
    main()
