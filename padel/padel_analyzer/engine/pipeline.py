"""Video (or cached detections) -> match report."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import ball as B
from . import court as C
from . import players as P
from . import rallies as R
from . import stats as ST
from .detect import Detections, Detector
from .events import HitModel
from .shots import CLASSES, ShotModel

REPLAY_HZ = 5
VERSION = "padel-0.1"


def sample_hists(video: str | Path, det: Detections, every_s: float = 0.5,
                 progress: Callable[[float], None] | None = None) -> dict[tuple[int, int], np.ndarray]:
    """Torso colour histograms of confident detections every `every_s` (for team/person linking)."""
    cap = cv2.VideoCapture(str(video))
    step = max(1, int(round(det.fps * every_s)))
    out: dict[tuple[int, int], np.ndarray] = {}
    for fv in range(det.n * det.step):
        if not cap.grab():
            break
        if fv % det.step:
            continue
        f = fv // det.step  # analysed frame index
        if f % step or not det.pose_done[f]:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        for j in np.where(det.pose_box[f, :, 4] >= P.MIN_CONF)[0]:
            h = P.torso_hist(frame, det.pose_kpt[f, j], det.pose_box[f, j, :4])
            if h is not None:
                out[(f, int(j))] = h
        if progress and f % (step * 50) == 0:
            progress(f / det.n)
    cap.release()
    return out


def analyze(det: Detections, calib: C.Calibration, video: str | Path | None = None,
            hit_model: HitModel | None = None, shot_model: ShotModel | None = None,
            progress: Callable[[str, float], None] | None = None) -> dict:
    prog = progress or (lambda stage, p: None)
    fps = det.fps
    t0 = time.time()
    hists = None
    if video is not None:
        hists = sample_hists(video, det, progress=lambda p: prog("identity", p))
    prog("players", 0)
    slot_tracks = P.track(det, calib, hists)
    tracks = slot_tracks.person_view()
    prog("ball", 0)
    bt = B.track(det.ball)  # precise: what the report shows
    bt_ev = B.track_for_events(det.ball)  # recall-oriented: what hit / stroke features see
    prog("hits", 0)
    hits, hit_score = (hit_model or HitModel()).detect(bt_ev, slot_tracks)
    strokes = (shot_model or ShotModel()).predict(hits, bt_ev, slot_tracks, fps, video, det.step)
    prog("rallies", 0)
    rallies = R.group(hits, fps, det.n, slot_tracks.segments, [s.kind == "serve" for s in strokes])
    rally_of = [0] * len(hits)
    for r in rallies:
        for i in r.hits:
            rally_of[i] = r.id

    person_of = [int(slot_tracks.person[h.frame, h.slot]) for h in hits]
    team_of = [0 if p < 2 else 1 for p in person_of]
    hit_pos = [tracks.pos_m[h.frame, p] for h, p in zip(hits, person_of)]
    speeds = ST.shot_ground_speed(hit_pos, [h.frame for h in hits], rally_of, fps)

    shots = []
    for i, (h, s) in enumerate(zip(hits, strokes)):
        pos = hit_pos[i]
        shots.append({
            "id": i, "frame": h.frame, "t": round(h.frame / fps, 3), "rally": rally_of[i],
            "player": person_of[i], "team": team_of[i], "type": s.kind, "type_conf": round(s.conf, 3),
            "type_probs": s.probs, "overhead": s.overhead, "volley": s.volley,
            "hit_score": round(h.score, 3), "speed_kmh": speeds[i],
            "pos_m": None if np.isnan(pos).any() else [round(float(pos[0]), 2), round(float(pos[1]), 2)],
            "ball_px": None if np.isnan(bt.xy[h.frame]).any() else [round(float(v), 1) for v in bt.xy[h.frame]],
            "label": "estimated",
        })

    players = []
    for p in range(4):
        st = ST.player_stats(tracks.pos_m[:, p], fps)
        mine = [s for s in shots if s["player"] == p]
        st["shots"] = len(mine)
        st["by_type"] = {c: sum(1 for s in mine if s["type"] == c) for c in CLASSES}
        st["volleys"] = sum(1 for s in mine if s["volley"])
        st["overheads"] = sum(1 for s in mine if s["overhead"])
        sp = [s["speed_kmh"] for s in mine if s["speed_kmh"]]
        st["avg_shot_speed_kmh"] = round(float(np.mean(sp)), 1) if sp else None
        players.append({"id": p, "name": f"Player {p + 1}", "team": 0 if p < 2 else 1, "stats": st,
                        "heatmap": ST.folded_heatmap(tracks.pos_m[:, p])})

    teams = []
    for t in (0, 1):
        a, b = (0, 1) if t == 0 else (2, 3)
        ts = ST.pair_stats(tracks.pos_m[:, a], tracks.pos_m[:, b])
        ts["shots"] = sum(1 for s in shots if s["team"] == t)
        ts["distance_m"] = round(players[a]["stats"]["distance_m"] + players[b]["stats"]["distance_m"], 1)
        ts["last_shot_in_rallies"] = sum(1 for r in rallies if r.hits and team_of[r.hits[-1]] == t)
        teams.append({"id": t, "name": f"Team {'AB'[t]}", "players": [a, b], "stats": ts})

    rally_out = []
    for r in rallies:
        rally_out.append({
            "id": r.id, "start_t": round(r.start / fps, 2), "end_t": round(r.end / fps, 2),
            "shots": r.hits, "n_shots": len(r.hits), "duration_s": round((r.end - r.start) / fps, 1),
            "first_type": shots[r.hits[0]]["type"] if r.hits else None,
            "last_player": person_of[r.hits[-1]] if r.hits else None,
            "last_team": team_of[r.hits[-1]] if r.hits else None,
            "winner_team": None,
        })
    lens = [r["n_shots"] for r in rally_out]
    match = {
        "version": VERSION, "fps": fps, "n_frames": det.n, "frame_step": det.step, "duration_s": round(det.n / fps, 1),
        "image_size": list(det.size), "rallies": len(rally_out),
        "avg_rally_shots": round(float(np.mean(lens)), 1) if lens else None,
        "max_rally_shots": max(lens) if lens else None,
        "avg_rally_s": round(float(np.mean([r["duration_s"] for r in rally_out])), 1) if rally_out else None,
        "hits": len(shots), "type_mix": {c: sum(1 for s in shots if s["type"] == c) for c in CLASSES},
    }
    quality = {
        "ball_observed_pct": round(float(bt.observed.mean() * 100), 1),
        "ball_track_pct": round(float((~np.isnan(bt.xy[:, 0])).mean() * 100), 1),
        "players_tracked_pct": round(float((~np.isnan(tracks.pos_m[:, :, 0])).mean() * 100), 1),
        "segments": len(slot_tracks.segments),
        "identity_linked": hists is not None,
        "calibration_reproj_m": calib.reproj_m,
        "analysis_s": round(time.time() - t0, 1),
    }
    step = max(1, int(round(fps / REPLAY_HZ)))
    replay = {
        "hz": fps / step,
        "players": [[None if np.isnan(tracks.pos_m[f, p, 0]) else [round(float(tracks.pos_m[f, p, 0]), 2),
                                                                     round(float(tracks.pos_m[f, p, 1]), 2)]
                     for p in range(4)] for f in range(0, det.n, step)],
        "ball_px": [None if np.isnan(bt.xy[f, 0]) else [round(float(bt.xy[f, 0])), round(float(bt.xy[f, 1]))]
                    for f in range(0, det.n, step)],
    }
    return {"match": match, "players": players, "teams": teams, "rallies": rally_out, "shots": shots,
            "quality": quality, "replay": replay, "calibration": calib.to_dict()}


def run(video: str | Path, calib: C.Calibration, detector: Detector | Callable[[], Detector] | None = None,
        max_frames: int = 0,
        progress: Callable[[str, float], None] | None = None, det_cache: Path | None = None) -> dict:
    prog = progress or (lambda stage, p: None)
    if det_cache and det_cache.exists():
        det = Detections.load(det_cache)
    else:
        if detector is None:
            detector = Detector()
        elif not isinstance(detector, Detector):
            detector = detector()  # factory: models load only when detection is needed
        det = detector.run(video, max_frames=max_frames, progress=lambda p: prog("detect", p))
        if det_cache:
            det.save(det_cache)
    return analyze(det, calib, video, progress=prog)
