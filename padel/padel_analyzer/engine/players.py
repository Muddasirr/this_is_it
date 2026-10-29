"""Four-player tracking on the court floor.

Padel structure does most of the work: exactly 4 players, a pair per net side, a fixed camera.
1. Court filter: feet (ankle midpoint, else box bottom) -> metres; keep detections on the floor.
2. Per side, up to 2 detections; slots 0,1 = far side, 2,3 = near side; Hungarian on metres
   against constant-velocity predictions.
3. Segments break at scene cuts / side changes; which pair plays which end is decided over all segments
   jointly from appearance embeddings (DINOv2), partners by individual appearance.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from . import court as C
from .detect import Detections

L_ANK, R_ANK = 15, 16
MIN_CONF = 0.3
FLOOR_MARGIN = (0.8, 1.5)  # metres outside the floor still accepted (x, y): runs through the door / behind glass
MAX_SPEED = 8.0  # m/s, faster steps are tracking glitches
CUT_PERSIST = 3  # frames the "everyone teleported" condition must hold before it counts as a cut


@dataclass
class PlayerTracks:
    pos_m: np.ndarray  # (n, 4, 2) smoothed floor metres, nan when unknown
    raw_m: np.ndarray  # (n, 4, 2) unsmoothed
    box: np.ndarray  # (n, 4, 4) xyxy px, nan when not detected this frame
    kpt: np.ndarray  # (n, 4, 17, 3)
    seen: np.ndarray  # (n, 4) bool detected (not interpolated)
    team: np.ndarray  # (n, 4) team id (0/1) of the person in each slot
    person: np.ndarray  # (n, 4) persistent person id 0..3 in each slot
    segments: list[tuple[int, int]]

    def person_view(self) -> "PlayerTracks":
        """Re-index slots -> persistent persons (slot order -> person 0..3)."""
        n = len(self.pos_m)
        rows = np.arange(n)[:, None]
        idx = np.argsort(self.person, axis=1)  # idx[f, p] = slot holding person p
        g = lambda a: a[rows, idx]  # noqa: E731
        return PlayerTracks(g(self.pos_m), g(self.raw_m), g(self.box), g(self.kpt), g(self.seen), g(self.team),
                            np.tile(np.arange(4), (n, 1)), self.segments)


def feet_px(box: np.ndarray, kpt: np.ndarray) -> np.ndarray:
    """Ankle midpoint where both ankles are confident, else bottom-centre of the box."""
    la, ra = kpt[..., L_ANK, :], kpt[..., R_ANK, :]
    ok = (la[..., 2] > 0.4) & (ra[..., 2] > 0.4)
    mid = (la[..., :2] + ra[..., :2]) / 2
    bottom = np.stack([(box[..., 0] + box[..., 2]) / 2, box[..., 3]], -1)
    return np.where(ok[..., None], mid, bottom)


def torso_hist(frame: np.ndarray, kpt: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """HS histogram of the shirt (shoulders -> hips quad, else the box's upper half)."""
    h, w = frame.shape[:2]
    pts = kpt[[5, 6, 12, 11], :2]
    if (kpt[[5, 6, 11, 12], 2] > 0.4).all():
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
    else:
        x0, y0, x1, y1 = box[0], box[1], box[2], box[1] + 0.5 * (box[3] - box[1])
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, None, 1.0, 0.0, cv2.NORM_L1).flatten()


def candidates(det: Detections, calib: C.Calibration, f: int) -> tuple[np.ndarray, np.ndarray]:
    """In-court detections of frame f: (indices, metres)."""
    b = det.pose_box[f]
    keep = np.where(b[:, 4] >= MIN_CONF)[0]
    if not len(keep):
        return keep, np.zeros((0, 2))
    m = calib.to_m(feet_px(b[keep, :4], det.pose_kpt[f, keep]))
    mx, my = FLOOR_MARGIN
    ok = (m[:, 0] > -mx) & (m[:, 0] < C.WIDTH + mx) & (m[:, 1] > -my) & (m[:, 1] < C.LENGTH + my)
    keep, m = keep[ok], m[ok]
    # duplicates (two boxes on one player): keep the more confident within 0.5 m
    order = np.argsort(-b[keep, 4])
    sel: list[int] = []
    for i in order:
        if all(np.linalg.norm(m[i] - m[j]) > 0.5 for j in sel):
            sel.append(i)
    return keep[sel], m[sel]


def track(det: Detections, calib: C.Calibration, hists: dict[tuple[int, int], np.ndarray] | None = None,
          max_gap_s: float = 1.0) -> PlayerTracks:
    """hists: optional {(frame, det_index): torso histogram} sampled by the caller from video frames."""
    n, fps = det.n, det.fps
    raw = np.full((n, 4, 2), np.nan)
    box = np.full((n, 4, 4), np.nan)
    kpt = np.zeros((n, 4, 17, 3), np.float32)
    seen = np.zeros((n, 4), bool)
    det_of = np.full((n, 4), -1)
    last = np.full((4, 2), np.nan)
    vel = np.zeros((4, 2))
    last_f = np.full(4, -10**9)
    seg_start, segments = 0, []
    jump_run, jump_first = 0, 0
    max_step = MAX_SPEED / fps
    for f in range(n):
        if not det.pose_done[f]:
            continue
        if det.cuts[f] and f > seg_start:
            segments.append((seg_start, f))
            seg_start = f
            last[:] = np.nan
        idx, m = candidates(det, calib, f)
        live = (f - last_f <= max_gap_s * fps) & ~np.isnan(last[:, 0])
        if len(m) >= 3 and live.sum() >= 3:
            # hard cut (broadcast camera switch / ends changed off-camera): most players "teleport"
            d = np.linalg.norm(m[:, None] - last[live][None], axis=2).min(1)
            if (d > 1.5).sum() >= 3:
                # a missed / spurious detection can fake one frame of it; a real cut persists
                jump_first = f if jump_run == 0 else jump_first
                jump_run += 1
                if jump_run >= CUT_PERSIST:
                    if jump_first > seg_start:
                        segments.append((seg_start, jump_first))
                    seg_start = jump_first
                    last[:] = np.nan
                    last_f[:] = -10**9
                    jump_run = 0
            else:
                jump_run = 0
        for s_id, slots in ((0, (0, 1)), (1, (2, 3))):
            on = np.where(C.side(m[:, 1]) == s_id)[0] if len(m) else np.zeros(0, int)
            on = on[:2] if len(on) > 2 else on  # already sorted by confidence
            if not len(on):
                continue
            pred = np.array([last[s] + vel[s] * min(f - last_f[s], fps) for s in slots])
            fresh = np.isnan(pred[:, 0]) | (f - last_f[list(slots)] > max_gap_s * fps)
            if fresh.all():
                # initialise left -> right
                order = on[np.argsort(m[on, 0])]
                pairs = list(zip(slots, order))
                if len(order) == 1:  # one player: slot by x relative to the centre line
                    pairs = [(slots[0] if m[order[0], 0] < C.WIDTH / 2 else slots[1], order[0])]
            else:
                pred = np.where(np.isnan(pred), np.array([[C.WIDTH / 2, 5.0 if s_id == 0 else 15.0]]), pred)
                cost = np.linalg.norm(pred[:, None] - m[on][None], axis=2)
                cost[fresh] = 3.0  # a stale slot takes whatever is left
                r, c = linear_sum_assignment(cost)
                pairs = [(slots[i], on[j]) for i, j in zip(r, c)]
            for s, j in pairs:
                gap = f - last_f[s]
                if not np.isnan(last[s, 0]) and gap <= max_gap_s * fps:
                    step = np.linalg.norm(m[j] - last[s])
                    if step > max_step * gap + 0.6:
                        continue  # implausible jump: leave the slot empty this frame
                    v = (m[j] - last[s]) / max(gap, 1)
                    vel[s] = 0.7 * vel[s] + 0.3 * v
                else:
                    vel[s] = 0
                last[s], last_f[s] = m[j], f
                raw[f, s] = m[j]
                box[f, s] = det.pose_box[f, idx[j], :4]
                kpt[f, s] = det.pose_kpt[f, idx[j]]
                seen[f, s] = True
                det_of[f, s] = idx[j]
    segments.append((seg_start, n))
    segments = _split_on_side_change(raw, segments, fps)
    pos = _smooth(raw, fps, max_fill=int(max_gap_s * fps))
    team, person = _link_identity(segments, det_of, hists, n)
    return PlayerTracks(pos, raw, box, kpt, seen, team, person, segments)


def _split_on_side_change(raw: np.ndarray, segments: list[tuple[int, int]], fps: float) -> list[tuple[int, int]]:
    """Teams change ends during breaks: a gap with no players for > 2 s starts a new segment."""
    present = (~np.isnan(raw[:, :, 0])).sum(1) >= 2
    out = []
    for a, b in segments:
        start = a
        f = a
        while f < b:
            if not present[f]:
                g = f
                while g < b and not present[g]:
                    g += 1
                if g - f > 2 * fps and f > start:
                    out.append((start, f))
                    start = g
                f = g
            else:
                f += 1
        out.append((start, b))
    return [s for s in out if s[1] > s[0]]


def _smooth(raw: np.ndarray, fps: float, max_fill: int) -> np.ndarray:
    out = raw.copy()
    n = len(raw)
    t = np.arange(n)
    win = max(5, int(fps * 0.5) | 1)
    for s in range(4):
        for d in range(2):
            x = raw[:, s, d]
            ok = ~np.isnan(x)
            if ok.sum() < 2:
                continue
            filled = np.interp(t, t[ok], x[ok])
            # only bridge short gaps
            gap_ok = np.zeros(n, bool)
            okf = np.where(ok)[0]
            for a, b in zip(okf[:-1], okf[1:]):
                if b - a <= max_fill:
                    gap_ok[a:b + 1] = True
            gap_ok[ok] = True
            y = filled.copy()
            # smooth each contiguous run
            runs = np.split(np.arange(n), np.where(np.diff(gap_ok.astype(int)) != 0)[0] + 1)
            for r in runs:
                if gap_ok[r[0]] and len(r) > win:
                    y[r] = savgol_filter(filled[r], win, 2)
            out[:, s, d] = np.where(gap_ok, y, np.nan)
    return out


def feature_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Shirt histograms (L1-normalised, all >= 0): Bhattacharyya distance; appearance embeddings: cosine."""
    if a.min() >= 0 and b.min() >= 0 and abs(a.sum() - 1) < 1e-3:
        return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


SWITCH_PENALTY = 0.3  # cost of a change of ends between consecutive segments (ends change every ~2 games)


def _orientation(seg: list[dict], lam: float = SWITCH_PENALTY) -> list[int]:
    """Per segment: 0 if the pair seen at the far end in the first usable segment is at the far end, else 1.

    Far and near players look different (size, lighting), so each end's mean feature is subtracted first;
    then S[i, j] = how much segments i and j agree on "same ends" (far~far, near~near) over "swapped"
    (far~near). The sign of S's leading eigenvector labels every segment jointly; a Viterbi pass over the
    segment chain with a penalty per change of ends removes isolated flips. PT100 finals, DINOv2 features:
    52/52 and 61/61 segments vs 30/52 and 61/61 for the old segment-by-segment shirt matching (EVAL §11).
    """
    ok = [g["F"] is not None and g["N"] is not None and g["w"] >= 2 for g in seg]
    idx = [k for k in range(len(seg)) if ok[k]]
    if len(idx) < 2:
        return [0] * len(seg)
    Fm = np.mean([seg[k]["F"] for k in idx], 0)
    Nm = np.mean([seg[k]["N"] for k in idx], 0)
    Fc = np.array([seg[k]["F"] - Fm for k in idx])
    Nc = np.array([seg[k]["N"] - Nm for k in idx])
    Fc /= np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-9
    Nc /= np.linalg.norm(Nc, axis=1, keepdims=True) + 1e-9
    S = Fc @ Fc.T + Nc @ Nc.T - Fc @ Nc.T - Nc @ Fc.T
    np.fill_diagonal(S, 0.0)
    u = np.linalg.eigh(S)[1][:, -1]
    unary = np.c_[-u, u] / (np.abs(u).max() + 1e-9)  # cost of state 0 / 1 per usable segment
    cost, back = unary[0].copy(), []
    for i in range(1, len(idx)):
        stay, switch = cost, cost[::-1] + lam
        back.append((switch < stay).astype(int))
        cost = np.minimum(stay, switch) + unary[i]
    st = int(np.argmin(cost))
    path = [st]
    for ch in reversed(back):
        st = 1 - st if ch[st] else st
        path.append(st)
    lab = dict(zip(idx, path[::-1]))
    flip = lab[idx[0]]  # the first usable segment's far pair is team 0
    o, cur = [], lab[idx[0]] ^ flip
    for k in range(len(seg)):
        cur = (lab[k] ^ flip) if k in lab else cur
        o.append(cur)
    return o


def _link_identity(segments, det_of, hists, n):
    """team[f, slot], person[f, slot]. hists: {(frame, det_index): feature} — DINOv2 appearance embeddings
    (production) or shirt histograms (fallback). Which pair plays which end: _orientation over all segments.
    Partners within a pair: individual features against running per-person prototypes from the same end."""
    team = np.zeros((n, 4), int)
    person = np.tile(np.arange(4), (n, 1))
    team[:, 2:] = 1
    if not hists or len(segments) < 2:
        return team, person

    def slot_feat(a, b, s):
        v = [hists[(f, det_of[f, s])] for f in range(a, b) if det_of[f, s] >= 0 and (f, det_of[f, s]) in hists]
        return (np.mean(v, 0), len(v)) if v else (None, 0)

    seg = []
    for a, b in segments:
        sl = [slot_feat(a, b, s) for s in range(4)]
        far = [v for v, _ in sl[:2] if v is not None]
        near = [v for v, _ in sl[2:] if v is not None]
        seg.append({"slots": [v for v, _ in sl], "F": np.mean(far, 0) if far else None,
                    "N": np.mean(near, 0) if near else None,
                    "w": min(sum(c for _, c in sl[:2]), sum(c for _, c in sl[2:]))})
    o = _orientation(seg)
    proto: dict[tuple[int, int], list] = {}  # (person, end 0 far / 1 near) -> features
    for k, (a, b) in enumerate(segments):
        far_people = [0, 1] if o[k] == 0 else [2, 3]
        near_people = [2, 3] if o[k] == 0 else [0, 1]
        for end, (slots, people) in enumerate((((0, 1), far_people), ((2, 3), near_people))):
            fs = [seg[k]["slots"][s] for s in slots]
            assigned = list(people)
            if all(v is not None for v in fs):
                refs = []
                for p in people:
                    got = proto.get((p, end)) or proto.get((p, 1 - end))
                    refs.append(np.mean(got[-20:], 0) if got else None)
                if all(r is not None for r in refs):
                    straight = feature_dist(fs[0], refs[0]) + feature_dist(fs[1], refs[1])
                    crossed = feature_dist(fs[0], refs[1]) + feature_dist(fs[1], refs[0])
                    if crossed < straight:
                        assigned = assigned[::-1]
                for v, p in zip(fs, assigned):
                    proto.setdefault((p, end), []).append(v)
            for s, p in zip(slots, assigned):
                person[a:b, s] = p
                team[a:b, s] = 0 if p < 2 else 1
    return team, person
