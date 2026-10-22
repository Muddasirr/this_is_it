"""KMeans jersey team assign (football_analysis idea, hardened for broadcast scale).

Per-player: torso band only (skip head / shorts), grass pixels masked out,
2-means in Lab on what remains → dominant cluster = jersey colour.
football_analysis's "top half, corner pixel = background" collapses on ~25px-wide
DFL crops (shirt + skin + grass edge average to beige), so it is not used.

Match-level: k-means (k up to 4) on jersey features. The two largest, clearly
different clusters are Kit A / Kit B; smaller clusters (refs, GKs, bibs) → Kit X.
No hard-coded "yellow = GK" rule — a yellow team is a team.
Sticky: once a track/meta gets a kit, keep it (`player_team_dict`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lab feature weights: lightness halved so shade doesn't split a team, but
# white vs black (pure L difference) still separates.
_L_WEIGHT = 0.5
# Two cluster centres are the same kit (sun vs shade) when chroma matches and
# only lightness differs: |Δab| < SAME_KIT_CHROMA and |ΔL·0.5| < SAME_KIT_LIGHT.
SAME_KIT_CHROMA = 14.0
SAME_KIT_LIGHT = 40.0
# Member farther than this from its kit centre is an outlier → X
MIN_OUTLIER_DIST = 30.0


def _kmeans(
    points: np.ndarray, k: int, n_init: int = 6, max_iter: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels [N], centres [k, D]). Pure numpy, k-means++ init."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    n = pts.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(max(1, n_init)):
        centres = [pts[int(rng.integers(0, n))]]
        for _c in range(1, k):
            d2 = np.min(
                np.stack([np.sum((pts - c) ** 2, axis=1) for c in centres]), axis=0
            )
            if d2.sum() <= 0:
                centres.append(pts[int(rng.integers(0, n))])
            else:
                centres.append(pts[int(rng.choice(n, p=d2 / d2.sum()))])
        cen = np.vstack(centres).astype(np.float64)
        labels = np.full(n, -1, dtype=np.int32)
        for _it in range(max_iter):
            dist = np.stack([np.sum((pts - c) ** 2, axis=1) for c in cen], axis=1)
            new_labels = np.argmin(dist, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    cen[j] = pts[mask].mean(axis=0)
        inertia = float(sum(np.sum((pts[labels == j] - cen[j]) ** 2) for j in range(k)))
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), cen.copy())
    assert best is not None
    return best[1], best[2]


def _kmeans_2(points: np.ndarray, n_init: int = 4, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
    return _kmeans(points, 2, n_init=n_init, max_iter=max_iter)


def _is_pitch_green_bgr(c: np.ndarray) -> bool:
    """Reject grass/pitch-like centres so they never become a 'team'."""
    b, g, r = float(c[0]), float(c[1]), float(c[2])
    return g > r + 25 and g > b + 15 and g > 80


def jersey_colour_from_crop(crop_bgr) -> tuple[np.ndarray, np.ndarray] | None:
    """(feature, bgr) of the dominant jersey colour, or None.

    feature = (L·0.5, a, b) in OpenCV 8-bit Lab; bgr is for display.
    """
    import cv2

    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 4:
        return None
    torso = crop_bgr[int(h * 0.15) : max(int(h * 0.15) + 2, int(h * 0.50)), int(w * 0.2) : max(int(w * 0.2) + 2, int(w * 0.8))]
    if torso.size == 0:
        torso = crop_bgr[: max(2, h // 2)]
    th, tw = torso.shape[:2]
    if th * tw > 3000:
        s = (3000.0 / (th * tw)) ** 0.5
        torso = cv2.resize(torso, (max(2, int(tw * s)), max(2, int(th * s))), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    grass = (hsv[:, 0] >= 28) & (hsv[:, 0] <= 95) & (hsv[:, 1] >= 40) & (hsv[:, 2] >= 30)
    bgr_all = torso.reshape(-1, 3)
    lab_all = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    # Pitch colour from this crop's own side borders (shaded / yellowed grass
    # escapes a fixed hue range and otherwise wins on small dark-kit crops)
    pitch = _border_pitch_lab(crop_bgr)
    if pitch is not None:
        near_pitch = np.linalg.norm(lab_all - pitch, axis=1) < 16.0
        grass = grass | near_pitch
    keep = ~grass
    if int(keep.sum()) < 12:
        return None
    bgr = bgr_all[keep]
    lab = lab_all[keep]
    feat = lab.copy()
    feat[:, 0] *= _L_WEIGHT
    if feat.shape[0] >= 24:
        labels, centres = _kmeans(feat, 2, n_init=2, max_iter=15)
        counts = np.bincount(labels, minlength=2)
        j = int(np.argmax(counts))
        # Near-tie: prefer the more saturated cluster (jersey over skin/shadow)
        if counts.min() >= 0.42 * counts.sum():
            chroma = np.hypot(centres[:, 1] - 128.0, centres[:, 2] - 128.0)
            j = int(np.argmax(chroma))
        mask = labels == j
        return centres[j], bgr[mask].astype(np.float64).mean(axis=0)
    return feat.mean(axis=0), bgr.astype(np.float64).mean(axis=0)


def _border_pitch_lab(crop_bgr) -> np.ndarray | None:
    """Median Lab of the crop's left/right edge columns when they look like pitch."""
    import cv2

    h, w = crop_bgr.shape[:2]
    bw = max(1, w // 8)
    edges = np.concatenate([crop_bgr[:, :bw].reshape(-1, 3), crop_bgr[:, w - bw :].reshape(-1, 3)])
    if edges.shape[0] < 8:
        return None
    hsv = cv2.cvtColor(edges.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    greenish = (hsv[:, 0] >= 25) & (hsv[:, 0] <= 95) & (hsv[:, 1] >= 30)
    if greenish.mean() < 0.4:
        return None
    lab = cv2.cvtColor(edges[greenish].reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab.astype(np.float64), axis=0)


def jersey_bgr_from_crop(crop_bgr) -> np.ndarray | None:
    """Dominant jersey BGR (display / legacy callers)."""
    out = jersey_colour_from_crop(crop_bgr)
    return None if out is None else out[1]


class TeamAssigner:
    """Fit 2 team colours from a set of player crops; sticky assign by id."""

    def __init__(self) -> None:
        self.team_colors: dict[str, np.ndarray] = {}  # "A"|"B" → BGR centre
        self.team_features: dict[str, np.ndarray] = {}
        self.team_radius: dict[str, float] = {}
        self._player_team: dict[Any, str] = {}
        self._fitted = False

    def fit(self, crops: list) -> bool:
        feats: list[np.ndarray] = []
        bgrs: list[np.ndarray] = []
        for crop in crops:
            out = jersey_colour_from_crop(crop)
            if out is None or _is_pitch_green_bgr(out[1]):
                continue
            feats.append(out[0])
            bgrs.append(out[1])
        n = len(feats)
        if n < 2:
            return False
        F = np.vstack(feats)
        B = np.vstack(bgrs)
        k = 2 if n < 8 else min(4, n // 2)
        labels, centres = _kmeans(F, k, n_init=8, max_iter=40)
        sizes = np.bincount(labels, minlength=k)
        order = [int(j) for j in np.argsort(-sizes) if sizes[j] > 0]

        # Kit 1 = biggest cluster; absorb clusters of the same colour (sun/shade);
        # Kit 2 = biggest cluster clearly different from Kit 1.
        def same_kit(p: np.ndarray, q: np.ndarray) -> bool:
            return (
                float(np.linalg.norm(p[1:] - q[1:])) < SAME_KIT_CHROMA
                and abs(float(p[0] - q[0])) < SAME_KIT_LIGHT
            )

        groups: list[list[int]] = []
        for j in order:
            for g in groups:
                if same_kit(centres[j], centres[g[0]]):
                    g.append(j)
                    break
            else:
                groups.append([j])
        groups.sort(key=lambda g: -int(sum(sizes[j] for j in g)))
        if len(groups) < 2:
            return False
        kits = []
        for g in groups[:2]:
            mask = np.isin(labels, g)
            fc = F[mask].mean(axis=0)
            d = np.linalg.norm(F[mask] - fc, axis=1)
            radius = max(MIN_OUTLIER_DIST, 2.5 * float(np.median(d)) if len(d) else 0.0)
            kits.append((fc, B[mask].mean(axis=0), radius))
        # Kit A = lighter kit (stable label)
        a_i = 0 if kits[0][0][0] >= kits[1][0][0] else 1
        for name, idx in (("A", a_i), ("B", 1 - a_i)):
            self.team_features[name] = kits[idx][0]
            self.team_colors[name] = kits[idx][1]
            self.team_radius[name] = kits[idx][2]
        self._fitted = True
        return True

    def predict(self, crop, sticky_id: Any = None) -> str | None:
        """Kit A / B, or X when the crop matches neither team colour."""
        if sticky_id is not None and sticky_id in self._player_team:
            return self._player_team[sticky_id]
        if not self._fitted:
            return None
        out = jersey_colour_from_crop(crop)
        if out is None:
            return None
        f = out[0]
        d_a = float(np.linalg.norm(f - self.team_features["A"]))
        d_b = float(np.linalg.norm(f - self.team_features["B"]))
        kit = "A" if d_a <= d_b else "B"
        if min(d_a, d_b) > self.team_radius[kit]:
            kit = "X"
        if sticky_id is not None:
            self._player_team[sticky_id] = kit
        return kit


def assign_kits_kmeans(metas: list[dict]) -> bool:
    """In-place Kit A/B/X using KMeans on crops. Returns True if applied.

    Expects metas with optional keys: img (BGR), sticky_id / src / track_key,
    class_name. Detector class 'referee' → Kit X before fitting.
    """
    import cv2

    if len(metas) < 2:
        return False

    crops_ready = 0
    for m in metas:
        if m.get("img") is not None and getattr(m["img"], "size", 0) > 0:
            crops_ready += 1
            continue
        path = m.get("thumb_path") or (m.get("jersey_hsv") or {}).get("thumb_path")
        if not path:
            feats = m.get("features") or {}
            path = feats.get("thumb_path")
        if path:
            img = cv2.imread(str(path))
            if img is not None:
                m["img"] = img
                crops_ready += 1

    if crops_ready < 2:
        return False

    players: list[dict] = []
    for m in metas:
        d = m.get("jersey_hsv") or {}
        cls = (m.get("class_name") or d.get("class_name") or "").lower()
        if cls == "referee":
            m["kit"] = "X"
            continue
        if m.get("img") is None:
            continue
        players.append(m)

    if len(players) < 2:
        return False

    assigner = TeamAssigner()
    if not assigner.fit([p["img"] for p in players]):
        return False

    for p in players:
        sid = p.get("sticky_id") or p.get("track_key") or p.get("src")
        kit = assigner.predict(p["img"], sticky_id=sid)
        p["kit"] = kit or "X"
        if p["kit"] in assigner.team_colors:
            p["team_color_bgr"] = assigner.team_colors[p["kit"]].tolist()

    for m in metas:
        if not m.get("kit"):
            m["kit"] = "X"
    return True
