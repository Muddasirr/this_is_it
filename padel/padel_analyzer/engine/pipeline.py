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
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
VERSION = "padel-0.1"


def sample_embeddings(video: str | Path, det: Detections, every_s: float = 0.5,
                      progress: Callable[[float], None] | None = None) -> dict[tuple[int, int], np.ndarray] | None:
    """DINOv2-S appearance embeddings of confident player crops every `every_s` (for pair / player identity
    across changes of ends; EVAL §11). None if the model cannot be loaded (then shirt histograms are used)."""
    try:
        import torch

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    except Exception:  # noqa: BLE001 - no internet / hub cache: fall back to shirt colours
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    cap = cv2.VideoCapture(str(video))
    step = max(1, int(round(det.fps * every_s)))
    keys: list[tuple[int, int]] = []
    crops: list[np.ndarray] = []
    out: dict[tuple[int, int], np.ndarray] = {}

    def flush() -> None:
        if not crops:
            return
        x = torch.from_numpy(np.stack(crops)).to(dev).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        with torch.no_grad():
            e = model(x.half() if dev == "cuda" else x).float()
        e = torch.nn.functional.normalize(e, dim=1).cpu().numpy()
        out.update(zip(keys, e))
        keys.clear()
        crops.clear()

    for fv in range(det.n * det.step):
        if not cap.grab():
            break
        if fv % det.step:
            continue
        f = fv // det.step
        if f % step or not det.pose_done[f]:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        for j in np.where(det.pose_box[f, :, 4] >= P.MIN_CONF)[0]:
            x1, y1, x2, y2 = det.pose_box[f, j, :4]
            w, h = x2 - x1, y2 - y1
            x1, x2 = int(max(0, x1 - 0.1 * w)), int(min(frame.shape[1], x2 + 0.1 * w))
            y1, y2 = int(max(0, y1 - 0.05 * h)), int(min(frame.shape[0], y2 + 0.05 * h))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            keys.append((f, int(j)))
            crops.append(np.ascontiguousarray(cv2.resize(frame[y1:y2, x1:x2], (112, 224))[:, :, ::-1]))
            if len(crops) == 256:
                flush()
        if progress and f % (step * 50) == 0:
            progress(f / det.n)
    flush()
    cap.release()
    return out or None


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


def audio_features(video: str | Path | None, det: Detections) -> tuple[np.ndarray | None, int | None]:
    """Racket-impact sound features per analysed frame (None without a model, a video or an audio track)."""
    from . import audio as A

    if video is None or not A.AudioHitModel.available():
        return None, None
    y = A.load_audio(video)
    if y is None or len(y) < A.SR:
        return None, None
    model = A.AudioHitModel()
    times, prob = model.probs(y)
    # det.fps is already the sampled rate: analysed frame i is at i / det.fps seconds (step must be 1 here)
    return (A.video_features(times, prob, det.n, det.fps, 1, model.threshold),
            int(len(A.peaks(times, prob, model.threshold))))


def analyze(det: Detections, calib: C.Calibration, video: str | Path | None = None,
            hit_model: HitModel | None = None, shot_model: ShotModel | None = None,
            progress: Callable[[str, float], None] | None = None) -> dict:
    prog = progress or (lambda stage, p: None)
    fps = det.fps
    t0 = time.time()
    hists = None
    if video is not None:
        hists = sample_embeddings(video, det, progress=lambda p: prog("identity", p))
        id_features = "dinov2" if hists else "shirt_colour"
        if hists is None:
            hists = sample_hists(video, det, progress=lambda p: prog("identity", p))
    prog("players", 0)
    slot_tracks = P.track(det, calib, hists)
    tracks = slot_tracks.person_view()
    prog("ball", 0)
    bt = B.track(det.ball)  # precise: what the report shows
    bt_ev = B.track_for_events(det.ball)  # recall-oriented: what hit / stroke features see
    prog("hits", 0)
    aud, n_audio_hits = audio_features(video, det)
    if hit_model is None:
        # audio + video model (modality dropout) when there is sound; the dedicated video-only model otherwise
        video_only = MODEL_DIR / "hit_lgbm_video.txt"
        hit_model = HitModel(video_only) if aud is None and video_only.exists() else HitModel()
    hits, hit_score = hit_model.detect(bt_ev, slot_tracks, audio=aud)
    # broadcast replays / close-ups: a hit can only be measured while the main court view is on screen
    # (≥ 3 players tracked around it); fixed facility cameras are never affected
    seen = (~np.isnan(slot_tracks.raw_m[:, :, 0])).sum(1) >= 3
    k = max(1, int(round(fps)))
    court_view = np.convolve(seen.astype(float), np.ones(2 * k + 1) / (2 * k + 1), "same") >= 0.6
    hits = [h for h in hits if court_view[min(h.frame, det.n - 1)]]
    strokes = (shot_model or ShotModel()).predict(hits, bt_ev, slot_tracks, fps, video, det.step)
    prog("rallies", 0)
    rallies = R.group(hits, fps, det.n, slot_tracks.segments, [s.kind == "serve" for s in strokes])
    person_of = [int(slot_tracks.person[h.frame, h.slot]) for h in hits]
    team_of = [0 if p < 2 else 1 for p in person_of]
    # touches between points are not shots: drop them, then regroup what is left
    live = sorted(i for r in rallies if not R.dead_ball(r, team_of) for i in r.hits)
    if len(live) < len(hits):
        hits, strokes = [hits[i] for i in live], [strokes[i] for i in live]
        person_of, team_of = [person_of[i] for i in live], [team_of[i] for i in live]
        rallies = R.group(hits, fps, det.n, slot_tracks.segments, [s.kind == "serve" for s in strokes])
    rally_of = [0] * len(hits)
    for r in rallies:
        for i in r.hits:
            rally_of[i] = r.id

    hit_pos = [tracks.pos_m[h.frame, p] for h, p in zip(hits, person_of)]

    from . import shot_types as SH

    padel_types = SH.classify(video, hits, slot_tracks, det.step) if (video is not None and hits and SH.available()) \
        else [(None, 0.0)] * len(hits)
    from . import ballspeed as BSP

    cam = BSP.camera_from_calibration(calib)
    speeds = []
    for i, h in enumerate(hits):
        end = min(hits[i + 1].frame - 1 if i + 1 < len(hits) else det.n, h.frame + int(2.5 * fps))
        r = BSP.shot_speed(cam, calib, bt.xy, bt.observed, h.frame, end, hit_pos[i], fps) if cam is not None else None
        speeds.append(r)
    shots = []
    for i, (h, s) in enumerate(zip(hits, strokes)):
        pos = hit_pos[i]
        shots.append({
            "id": i, "frame": h.frame, "t": round(h.frame / fps, 3), "rally": rally_of[i],
            "player": person_of[i], "team": team_of[i], "type": s.kind, "type_conf": round(s.conf, 3),
            "type_probs": s.probs, "overhead": s.overhead, "volley": s.volley,
            "hit_score": round(h.score, 3),
            "speed_kmh": speeds[i]["speed_kmh"] if speeds[i] else None,  # average, racket -> first bounce
            "bounce_m": speeds[i]["bounce_m"] if speeds[i] else None,
            "pos_m": None if np.isnan(pos).any() else [round(float(pos[0]), 2), round(float(pos[1]), 2)],
            "ball_px": None if np.isnan(bt.xy[h.frame]).any() else [round(float(v), 1) for v in bt.xy[h.frame]],
            "padel_type": padel_types[i][0], "padel_type_conf": padel_types[i][1],
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
        sp = {}
        for s in mine:
            if s["speed_kmh"]:
                sp.setdefault(s["type"], []).append(s["speed_kmh"])
        allsp = [v for vs in sp.values() for v in vs]
        st["ball_speed_kmh"] = {"avg": round(float(np.mean(allsp)), 1) if allsp else None,
                                "max": round(float(max(allsp)), 1) if allsp else None, "shots_measured": len(allsp),
                                "by_type": {k: round(float(np.median(v)), 1) for k, v in sp.items()}}
        pt = [s["padel_type"] for s in mine if s.get("padel_type")]
        st["padel_types"] = {k: pt.count(k) for k in sorted(set(pt))}
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
        "identity_features": id_features if hists is not None else None,
        "audio_used": aud is not None,
        "audio_hits": n_audio_hits,
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
    from . import advanced as ADV
    from . import points as PTS

    ADV.shot_context(shots, rally_out, tracks.pos_m)
    if PTS.PointModel.available() and rally_out:
        X = point_features(rally_out, shots, bt, tracks.pos_m, calib, fps, det.n, slot_tracks.segments)
        PTS.assign_winners(rally_out, shots, X, PTS.PointModel())
    extra, in_play = ADV.match_extra(rally_out, shots, det.n / fps, fps, det.n)
    match.update(extra)
    for pl in players:
        pl["stats"].update(ADV.player_extra(pl["id"], shots, rally_out, tracks.pos_m[:, pl["id"]], fps, in_play))
    for tm in teams:
        a, b = tm["players"]
        tm["stats"].update(ADV.pair_extra(tracks.pos_m[:, a], tracks.pos_m[:, b], shots, rally_out, tm["id"], fps))
    rep = {"match": match, "players": players, "teams": teams, "rallies": rally_out, "shots": shots,
           "quality": quality, "replay": replay, "calibration": calib.to_dict()}
    ADV.outcome_stats(rep)
    return rep


def point_features(rally_out: list[dict], shots: list[dict], bt, pos_m: np.ndarray, calib: C.Calibration,
                   fps: float, n: int, segments: list[tuple[int, int]]) -> np.ndarray:
    from . import points as PTS

    seg_end = [b for _, b in segments]
    rows = []
    for k, r in enumerate(rally_out):
        f = shots[r["shots"][-1]]["frame"] if r["shots"] else 0
        end = next((b for b in seg_end if b > f), n)
        nxt = shots[rally_out[k + 1]["shots"][0]]["frame"] if k + 1 < len(rally_out) and rally_out[k + 1]["shots"] else None
        rows.append(PTS.rally_features(r, shots, bt.xy, pos_m, calib, fps, end, nxt))
    return np.array(rows).reshape(len(rows), len(PTS.FEATURES))


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
