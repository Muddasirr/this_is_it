"""Stroke type of each hit from the hitter's pose window + ball context.

Classes follow PadelTracker100: serve, forehand, backhand, smash, other (dropshot folded into
other). Padel tags are derived on top (estimated): overhead (contact above the head: smash /
bandeja / víbora family), volley (contact in the net zone without a bounce on the hitter's side).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import court as C
from .ball import BallTrack
from .events import Hit
from .players import PlayerTracks

CLASSES = ["serve", "forehand", "backhand", "smash", "other"]
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
OFFSETS = (-8, -4, 0, 4)
KP = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # nose, shoulders, elbows, wrists, hips, knees


@dataclass
class Stroke:
    kind: str
    conf: float
    probs: dict[str, float]
    overhead: bool
    volley: bool


def _norm_pose(k: np.ndarray, flip: float) -> np.ndarray | None:
    """Keypoints relative to the hip centre, scaled by torso length, x mirrored for far-side players."""
    if k is None or np.isnan(k).any() or (k[[5, 6, 11, 12], 2] < 0.2).any():
        return None
    hip = (k[11, :2] + k[12, :2]) / 2
    sh = (k[5, :2] + k[6, :2]) / 2
    torso = max(np.linalg.norm(sh - hip), 1.0)
    p = (k[KP, :2] - hip) / torso
    p[:, 0] *= flip
    return p.ravel()


def handedness(hits: list[Hit], tracks: PlayerTracks) -> dict[int, float]:
    """Per person: +1 right-handed, -1 left-handed — the wrist that moves faster around their hits."""
    n = len(tracks.kpt)
    votes: dict[int, list[float]] = {}
    for h in hits:
        a, b = max(0, h.frame - 4), min(n - 1, h.frame + 4)
        ka, kb = tracks.kpt[a, h.slot], tracks.kpt[b, h.slot]
        if (ka[[9, 10, 5, 6, 11, 12], 2] < 0.2).any() or (kb[[9, 10], 2] < 0.2).any():
            continue
        torso = max(np.linalg.norm((ka[5, :2] + ka[6, :2]) / 2 - (ka[11, :2] + ka[12, :2]) / 2), 1.0)
        # left wrist = 9, right wrist = 10 (COCO, the person's own sides)
        dl = np.linalg.norm(kb[9, :2] - ka[9, :2]) / torso
        dr = np.linalg.norm(kb[10, :2] - ka[10, :2]) / torso
        votes.setdefault(int(tracks.person[h.frame, h.slot]), []).append(np.sign(dr - dl))
    return {p: (1.0 if np.mean(v) >= -0.1 else -1.0) for p, v in votes.items() if len(v) >= 5}


def _local_hand(tracks: PlayerTracks, f: int, s: int, w: int = 4) -> float:
    """Racket hand at this contact: the wrist that travels further over +-w frames (0 = unknown)."""
    n = len(tracks.kpt)
    a, b = max(0, f - w), min(n - 1, f + w)
    ka, kb = tracks.kpt[a, s], tracks.kpt[b, s]
    if (ka[[9, 10], 2] < 0.2).any() or (kb[[9, 10], 2] < 0.2).any():
        return 0.0
    dl = np.linalg.norm(kb[9, :2] - ka[9, :2])
    dr = np.linalg.norm(kb[10, :2] - ka[10, :2])
    return 1.0 if dr >= dl else -1.0


def feature_names() -> list[str]:
    names = []
    for o in OFFSETS:
        names += [f"k{o}_{i}_{a}" for i in KP for a in "xy"]
    return names + ["hand", "rwrist_rx", "rwrist_ry", "ball_rx", "ball_ry", "wrist_top", "side", "net_dist", "centre_dist", "since_prev_s",
                    "first_of_rally", "ball_speed_after", "ball_dy_after", "rally_pos"]


def features(hits: list[Hit], ball: BallTrack, tracks: PlayerTracks, fps: float,
             hands: dict[int, float] | None = None) -> np.ndarray:
    n = len(ball.xy)
    hands = handedness(hits, tracks) if hands is None else hands
    rows = []
    prev_f = -10**9
    rally_pos = 0
    for h in hits:
        s, f = h.slot, h.frame
        far = s < 2
        hand = _local_hand(tracks, f, s)
        if hand == 0:
            hand = hands.get(int(tracks.person[f, s]), 1.0)
        # far players face the camera; left-handers are mirrored so "forehand side" is always +x
        flip = (-1.0 if far else 1.0) * hand
        feats: list[float] = []
        for o in OFFSETS:
            g = min(max(f + o, 0), n - 1)
            p = _norm_pose(tracks.kpt[g, s].astype(float), flip)
            feats += list(p) if p is not None else [np.nan] * (2 * len(KP))
        k = tracks.kpt[f, s].astype(float)
        hip = (k[11, :2] + k[12, :2]) / 2
        sh = (k[5, :2] + k[6, :2]) / 2
        torso = max(np.linalg.norm(sh - hip), 1.0)
        bxy = ball.xy[f]
        rw = k[10 if hand > 0 else 9, :2]
        feats += [hand, flip * (rw[0] - sh[0]) / torso, (rw[1] - sh[1]) / torso]
        feats += [flip * (bxy[0] - sh[0]) / torso, (bxy[1] - k[0, 1]) / torso,
                  (k[0, 1] - min(k[9, 1], k[10, 1])) / torso]
        pos = tracks.pos_m[f, s]
        feats += [float(not far), abs(pos[1] - C.NET_Y) if not np.isnan(pos[1]) else np.nan,
                  abs(pos[0] - C.WIDTH / 2) if not np.isnan(pos[0]) else np.nan]
        since = (f - prev_f) / fps
        first = since > 3.0
        rally_pos = 0 if first else rally_pos + 1
        feats += [min(since, 10.0), float(first)]
        a = min(f + 5, n - 1)
        v = (ball.xy[a] - bxy) / max(a - f, 1)
        feats += [float(np.linalg.norm(v)) / torso, float(v[1]) / torso, float(min(rally_pos, 6))]
        rows.append(feats)
        prev_f = f
    return np.asarray(rows, np.float32).reshape(len(hits), len(feature_names()))


def tags(hits: list[Hit], tracks: PlayerTracks) -> list[tuple[bool, bool]]:
    out = []
    for h in hits:
        k = tracks.kpt[h.frame, h.slot]
        overhead = bool(k[0, 2] > 0.2 and min(k[9, 1] if k[9, 2] > 0.2 else 1e9, k[10, 1] if k[10, 2] > 0.2 else 1e9)
                        < k[0, 1])
        y = tracks.pos_m[h.frame, h.slot, 1]
        volley = bool(not np.isnan(y) and abs(y - C.NET_Y) <= C.NET_ZONE_M)
        out.append((overhead, volley))
    return out


class RGBStrokeModel:
    """R(2+1)D-18 on 16 frames of the hitter crop (kaggle/stroke/padel_stroke.py). Sees the racket,
    so it resolves forehand vs backhand where pose alone cannot tell which hand holds it."""

    def __init__(self, path: Path, device: str | None = None):
        import torch
        from torchvision.models.video import r2plus1d_18

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.size, self.t_before, self.n_fr, self.scale = ck["size"], ck["t_before"], ck["n_frames"], ck["crop_scale"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = r2plus1d_18(num_classes=len(self.classes))
        m.load_state_dict(ck["state_dict"])
        self.model = m.eval().to(self.device)

    def _clip(self, cap, f: int, box: np.ndarray, step: int = 1) -> np.ndarray | None:
        import cv2

        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0) * self.scale
        ax, ay = int(round((x0 + x1) / 2 - side / 2)), int(round((y0 + y1) / 2 - side / 2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, (f - self.t_before) * step))
        frames = []
        for _ in range(self.n_fr):
            ok, fr = cap.read()
            for _ in range(step - 1):
                cap.grab()
            if not ok:
                return None
            pad = cv2.copyMakeBorder(fr, 400, 400, 400, 400, cv2.BORDER_CONSTANT)
            c = pad[ay + 400:ay + 400 + int(side), ax + 400:ax + 400 + int(side)]
            frames.append(cv2.resize(c, (self.size, self.size))[:, :, ::-1])
        return np.stack(frames)

    def predict(self, video: str | Path, hits: list[Hit], tracks: PlayerTracks, step: int = 1) -> np.ndarray:
        """(len(hits), len(classes)) probabilities; rows of nan where no crop was possible."""
        import cv2
        import torch

        cap = cv2.VideoCapture(str(video))
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
        out = np.full((len(hits), len(self.classes)), np.nan)
        for i, h in enumerate(hits):
            box = tracks.box[h.frame, h.slot]
            if np.isnan(box).any():
                continue
            clip = self._clip(cap, h.frame, box, step)
            if clip is None:
                continue
            x = torch.from_numpy(clip).float().div(255).permute(3, 0, 1, 2)[None]
            x = ((x - mean) / std).to(self.device)
            with torch.no_grad():
                p = torch.softmax(self.model(x), 1) + torch.softmax(self.model(x.flip(-1)), 1)
            out[i] = (p / 2).cpu().numpy()[0]
        cap.release()
        return out


class ShotModel:
    def __init__(self, path: Path | str | None = None, rgb_path: Path | str | None = None):
        p = Path(path) if path else MODEL_DIR / "shot_lgbm.txt"
        self.booster = None
        if p.exists():
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=str(p))
        rp = Path(rgb_path) if rgb_path else MODEL_DIR / "stroke_rgb.pt"
        self.rgb = RGBStrokeModel(rp) if rp.exists() else None

    def predict(self, hits: list[Hit], ball: BallTrack, tracks: PlayerTracks, fps: float,
                video: str | Path | None = None, step: int = 1) -> list[Stroke]:
        if not hits:
            return []
        X = features(hits, ball, tracks, fps)
        tg = tags(hits, tracks)
        if self.booster is not None:
            P = self.booster.predict(X)
        else:
            P = np.array([_rule_probs(x, t) for x, t in zip(X, tg)])
        if self.rgb is not None and video is not None:
            R = self.rgb.predict(video, hits, tracks, step)
            ok = ~np.isnan(R[:, 0])
            # the clip model sees the racket: it leads; pose/context keeps serve + rally position
            P[ok] = 0.7 * R[ok] + 0.3 * P[ok]
        out = []
        for p, (overhead, volley) in zip(P, tg):
            i = int(np.argmax(p))
            out.append(Stroke(CLASSES[i], float(p[i]), {c: round(float(v), 3) for c, v in zip(CLASSES, p)},
                              overhead, volley and CLASSES[i] != "serve"))
        return out


def _rule_probs(x: np.ndarray, tag: tuple[bool, bool]) -> np.ndarray:
    names = feature_names()
    c = {k: i for i, k in enumerate(names)}
    p = np.full(len(CLASSES), 0.05)
    if x[c["first_of_rally"]] > 0:
        p[0] = 1
    elif tag[0]:
        p[3] = 1
    elif np.nan_to_num(x[c["ball_rx"]]) >= 0:
        p[1] = 1
    else:
        p[2] = 1
    return p / p.sum()


def save_meta(path: Path, meta: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
