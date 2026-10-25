#!/usr/bin/env python3
"""Draw what the pipeline saw onto the match video (for demos and visual QA).

Per frame: every person track (box + name in its team colour; tracks the identity
step left out — bystanders, keepers/refs, off-roster — in grey), the ball, the
player in possession (highlighted), event banners (pass / dribble / shot / goal with
who), and a running team panel (passes, possession share so far).

  python scripts/eval/render_overlay.py --run <run_offline out dir> --video <source.mp4> --out demo.mp4
         [--start 0 --seconds 120]

Run dir = app.scripts.run_offline output (detections.json, players.json,
canonical_events.json). Output is H.264 (imageio-ffmpeg's bundled ffmpeg) so it plays in a browser.
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

TEAM_BGR = {"Kit A": (246, 130, 59), "Kit B": (68, 68, 239)}  # UI blue / red
OTHER_BGR = (230, 200, 60)  # cyan-ish, thin: tracks not on a team (bystanders, keepers, refs)
BALL_BGR = (0, 255, 255)
EVENT_SHOW_MS = 1800
MAX_GAP_MS = 400  # interpolate boxes across gaps up to this; longer = track not visible


class Track:
    def __init__(self, series: list[dict]):
        pts = sorted((p for p in series if "x1" in p), key=lambda p: p["ts_ms"])
        self.t = [int(p["ts_ms"]) for p in pts]
        self.b = np.array([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in pts], dtype=float) if pts else np.zeros((0, 4))

    def at(self, ts: int):
        if not self.t or ts < self.t[0] or ts > self.t[-1]:
            return None
        i = bisect.bisect_left(self.t, ts)
        if i < len(self.t) and self.t[i] == ts:
            return self.b[i]
        a, c = i - 1, i
        if c >= len(self.t) or self.t[c] - self.t[a] > MAX_GAP_MS:
            return None
        w = (ts - self.t[a]) / max(1, self.t[c] - self.t[a])
        return self.b[a] * (1 - w) + self.b[c] * w


def _cross_team(e: dict, team) -> bool:
    roles = {p["role"]: p["track_key"] for p in e.get("participants") or []}
    a, b = team(roles.get("passer")), team(roles.get("receiver"))
    return a is not None and b is not None and a != b


def kit_colours(dets: list[dict], who: dict, video: Path) -> dict[str, tuple[int, int, int]]:
    """Box colour per team = median of the central torso pixels over up to 30 of its tracks,
    read from the video itself (the pipeline's stored torso colour mis-reads dark kits by
    picking their coloured numbers). Very dark kits are drawn near-black."""
    per_team: dict[str, list] = defaultdict(list)
    for d in sorted(dets, key=lambda d: -(d["end_ms"] - d["start_ms"])):
        t = who.get(d["track_key"], (None, None))[1]
        if t and len(per_team[t]) < 30 and d.get("bbox_series"):
            per_team[t].append(d["bbox_series"][len(d["bbox_series"]) // 2])
    cap = cv2.VideoCapture(str(video))
    out = dict(TEAM_BGR)
    for t, samples in per_team.items():
        px = []
        for b in samples:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(b["ts_ms"]))
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
            h, w = y2 - y1, x2 - x1
            torso = frame[max(0, y1 + int(0.2 * h)) : y1 + int(0.45 * h), max(0, x1 + int(0.3 * w)) : x1 + int(0.7 * w)]
            if torso.size:
                px.append(torso.reshape(-1, 3))
        if px:
            med = np.median(np.concatenate(px), axis=0)
            out[t] = (25, 25, 25) if med.mean() < 70 else tuple(int(v) for v in med)
    cap.release()
    return out


def label(img, text, org, bg, scale=0.5, fg=(255, 255, 255)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), bg, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = to the end")
    ap.add_argument("--camera", default="A")
    args = ap.parse_args()

    from app.services.possession import build_possession_timeline
    from app.services.vision import Detection

    dets = [d for d in json.loads((args.run / "detections.json").read_text()) if d.get("camera_label", "A") == args.camera]
    players = json.loads((args.run / "players.json").read_text())
    events = json.loads((args.run / "canonical_events.json").read_text())
    who = {k: (p["display_name"], p["team"]) for p in players for k in p.get("track_keys") or []}
    team_bgr = kit_colours(dets, who, args.video)
    # Same rule as the pipeline (drop_self_passes): a "pass" to the other team is a turnover
    def team(track_key):
        return who.get(track_key, (None, None))[1]

    events = [e for e in events if not (e["event_type"] == "pass" and _cross_team(e, team))]
    persons = {d["track_key"]: Track(d["bbox_series"]) for d in dets if d["class_name"] == "person"}
    balls = [Track(d["bbox_series"]) for d in dets if d["class_name"] == "sports_ball"]

    # Possession holder per ball sample — the same timeline the event engine uses
    poss = build_possession_timeline([Detection(**{**d, "camera_label": d.get("camera_label", "A")}) for d in dets])
    seg = sorted((int(s["start_ms"]), int(s["end_ms"]), s["holder"]) for s in poss.get("segments") or [])
    seg_starts = [s[0] for s in seg]

    def holder(ts):
        i = bisect.bisect_right(seg_starts, ts) - 1
        return seg[i][2] if i >= 0 and seg[i][0] <= ts <= seg[i][1] else None

    def name_of(track_key):
        return who.get(track_key, ("unassigned", None))[0] if track_key else "?"

    evs = sorted(events, key=lambda e: e["ts_ms"])
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0 if w <= 1280 else 1280 / w
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    start_f = int(args.start * fps)
    end_f = int((args.start + args.seconds) * fps) if args.seconds else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    import imageio_ffmpeg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(args.out)], stdin=subprocess.PIPE)
    passes, poss_ms = Counter(), Counter()
    ev_i, last_ts = 0, None
    while ev_i < len(evs) and evs[ev_i]["ts_ms"] < args.start * 1000:  # count events before the window
        e = evs[ev_i]
        if e["event_type"] == "pass":
            passer = next((p["track_key"] for p in e["participants"] if p["role"] == "passer"), None)
            passes[who.get(passer, (None, None))[1]] += 1
        ev_i += 1
    for f in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(f * 1000 / fps)
        hold = holder(ts)
        if last_ts is not None and hold:
            poss_ms[who.get(hold, (None, None))[1]] += ts - last_ts
        last_ts = ts
        for key, tr in persons.items():
            box = tr.at(ts)
            if box is None:
                continue
            name, team = who.get(key, (None, None))
            col = team_bgr.get(team, OTHER_BGR)
            x1, y1, x2, y2 = box.astype(int)
            thick = 4 if key == hold else (2 if team in team_bgr else 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
            label(frame, name or "other", (x1, y1), col, 0.45)
            if key == hold:
                cv2.circle(frame, ((x1 + x2) // 2, y2 + 6), 7, BALL_BGR, -1)
        for tr in balls:
            box = tr.at(ts)
            if box is not None:
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                cv2.circle(frame, (cx, cy), max(8, int((box[2] - box[0]) * 0.8)), BALL_BGR, 2)
        while ev_i < len(evs) and evs[ev_i]["ts_ms"] <= ts:
            e = evs[ev_i]
            if e["event_type"] == "pass":
                passer = next((p["track_key"] for p in e["participants"] if p["role"] == "passer"), None)
                passes[who.get(passer, (None, None))[1]] += 1
            ev_i += 1
        banners = [e for e in evs[max(0, ev_i - 6) : ev_i] if 0 <= ts - e["ts_ms"] <= EVENT_SHOW_MS]
        for j, e in enumerate(banners[-3:]):
            roles = {p["role"]: p["track_key"] for p in e["participants"]}
            if e["event_type"] == "pass":
                text = f"PASS  {name_of(roles.get('passer'))} -> {name_of(roles.get('receiver'))}"
            else:
                actor = next(iter(roles.values()), None)
                text = f"{e['event_type'].upper()}  {name_of(actor)}"
            label(frame, text + f"  ({e.get('confidence', 0):.2f})", (20, h - 30 - 34 * j), (40, 40, 40), 0.8)
        tot = sum(poss_ms[t] for t in team_bgr) or 1
        panel = [f"t {ts / 1000:6.1f}s"] + [f"{t}: passes {passes[t]:3d}  poss {100 * poss_ms[t] / tot:3.0f}%" for t in team_bgr]
        for j, line in enumerate(panel):
            label(frame, line, (20, 40 + 34 * j), (30, 30, 30) if j == 0 else team_bgr[list(team_bgr)[j - 1]], 0.75)
        if scale != 1.0:
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        enc.stdin.write(frame[:oh, :ow].tobytes())
    enc.stdin.close()
    enc.wait()
    cap.release()
    print("wrote", args.out, "teams", dict(passes), {k: round(v / 1000) for k, v in poss_ms.items()})


if __name__ == "__main__":
    main()
