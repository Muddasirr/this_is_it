"""Jersey colour features for Kit A/B — not raw upper-body histograms.

Fundamental fixes vs the old path:
  1. Mask green pitch + loose skin so grass/faces don't dominate the hist.
  2. Represent a kit as (hue, sat, val, black_frac) — black is low-V, white is
     high-V low-S; old "_kit_lightness" treated BOTH as "light" (low-S only).
  3. 2-means on that compact vector, not 64-D hist L2 (which collapses red≈black
     when both crops contain dark shadows/shorts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# OpenCV HSV: H∈[0,180), S,V∈[0,255]
_GREEN_LO = np.array([35, 40, 40], dtype=np.uint8)
_GREEN_HI = np.array([95, 255, 255], dtype=np.uint8)
# Indoor futsal courts (+ cyan ref kits) — same role as grass outdoors.
_BLUE_LO = np.array([85, 40, 40], dtype=np.uint8)
_BLUE_HI = np.array([140, 255, 255], dtype=np.uint8)
# Warm wooden indoor courts (SAFF etc.). Must stay BELOW jersey sat —
# S≤150 was deleting Spain red cloth (KitB_4: 67% of red pixels eaten).
_WOOD_LO = np.array([10, 25, 110], dtype=np.uint8)
_WOOD_HI = np.array([28, 75, 255], dtype=np.uint8)


def jersey_descriptor_bgr(crop_bgr) -> dict[str, float] | None:
    """Dominant jersey colour from a player crop (prefer upper body already)."""
    import cv2

    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 6:
        return None
    # Jersey torso centre: skip face/hair (esp. close-ups like Spain #10 where
    # y∈[0.12,0.55] was mostly face → red=0), skip shorts/legs.
    y0 = int(h * 0.28)
    y1 = int(h * 0.72)
    x0 = int(w * 0.18)
    x1 = int(w * 0.82)
    roi = crop_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = crop_bgr

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Official / neon-GK cues BEFORE pitch mask (cyan refs get eaten by blue mask,
    # then red ad boards inflate redness — Portugal KitA_4 bug).
    H_all = hsv[:, :, 0].astype(np.float64)
    S_all = hsv[:, :, 1].astype(np.float64)
    V_all = hsv[:, :, 2].astype(np.float64)
    cyan_frac = float(((H_all >= 85) & (H_all <= 110) & (S_all >= 80) & (V_all >= 80)).mean())
    neon_frac = float(((H_all >= 40) & (H_all <= 80) & (S_all >= 100) & (V_all >= 100)).mean())
    pitch = (
        cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
        | cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
        | cv2.inRange(hsv, _WOOD_LO, _WOOD_HI)
    )
    # Do NOT mask "skin" on the torso ROI. OpenCV skin ranges H∈[0,25]∪[160,180]
    # are exactly red/pink jersey hues — that ate KitB_7 (red cloth → red=0.20).
    # Head is already cut by y0; residual arms are minority vs jersey area.
    keep = pitch == 0
    pix = hsv[keep]
    if pix.shape[0] < 30:
        pix = hsv.reshape(-1, 3)

    H = pix[:, 0].astype(np.float64)
    S = pix[:, 1].astype(np.float64)
    V = pix[:, 2].astype(np.float64)

    dh = np.minimum(H, 180.0 - H)
    # Red/pink cloth — sat floor high enough to ignore warm wood leftovers
    red_m = (S >= 70) & (V >= 50) & (V < 250) & (dh <= 18)
    red_frac = float(red_m.mean())
    # Dark cloth that is NOT the red accent (black jersey body under a pink #)
    dark_body_m = (~red_m) & (V < 110)
    dark_body_frac = float(dark_body_m.mean())
    # Cream/white cloth (Spain home) — high V, very low S. Tighter than skin.
    light_body_m = (~red_m) & (V >= 150) & (S < 55)
    light_body_frac = float(light_body_m.mean())
    black_m = (V < 70) & (S < 100) & (~red_m)
    black_frac = float(black_m.mean())

    # Chromatic jersey pixels (coloured cloth) for display hue/sat/val
    chrom_m = (S >= 45) & (V >= 45) & (V < 250)
    chrom = pix[chrom_m]

    if chrom.shape[0] >= 25:
        Hc, Sc, Vc = chrom[:, 0].astype(np.float64), chrom[:, 1].astype(np.float64), chrom[:, 2].astype(np.float64)
        ang = Hc * (np.pi / 90.0)
        mh = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
        if mh < 0:
            mh += 2 * np.pi
        hue = mh * (90.0 / np.pi)  # back to [0,180)
        sat = float(Sc.mean() / 255.0)
        val = float(Vc.mean() / 255.0)
    else:
        # Mostly achromatic (black/white/grey kit)
        hue = float(H.mean())
        sat = float(S.mean() / 255.0)
        val = float(V.mean() / 255.0)

    out = {
        "hue": hue,
        "sat": sat,
        "val": val,
        "black_frac": black_frac,
        "red_frac": red_frac,
        "dark_body_frac": dark_body_frac,
        "light_body_frac": light_body_frac,
        "cyan_frac": cyan_frac,
        "neon_frac": neon_frac,
    }
    out["redness"] = _redness(out)
    out["darkness"] = _darkness(out)
    out["is_outlier"] = is_kit_outlier(out)
    return out


def is_kit_outlier(d: dict[str, float] | None) -> bool:
    """Ref / neon-GK / non-team colours — exclude from Kit A/B 2-means."""
    if not d:
        return False
    # Cyan refs often share the crop with red ad boards → redness looks high.
    # Require cyan signal and NOT a solid red jersey (redness≥0.65).
    cyan = float(d.get("cyan_frac") or 0)
    if cyan >= 0.12 and float(d.get("redness") or 0) < 0.65:
        return True
    if float(d.get("neon_frac") or 0) >= 0.22:
        return True
    return False


def _redness(d: dict[str, float]) -> float:
    """Net red-kit score: red cloth minus cream body and dark non-red body.

    - Solid red jersey → high red_frac, low light/dark body.
    - Black + pink number → red_frac ≈ dark_body → net ~0.
    - Cream + maroon panels → red_frac high-ish but light_body wins → net low.
    Soft dark penalty so underexposed red on a dark stand still scores.
    """
    if "red_frac" in d:
        rf = float(d["red_frac"])
        db = float(d.get("dark_body_frac") if d.get("dark_body_frac") is not None else d.get("black_frac") or 0.0)
        lb = float(d.get("light_body_frac") or 0.0)
        # Strong cream/white penalty — red court borders behind white kits
        # must not win (Spain/France KitA_8: rf=0.58 from red stripe, lb=0.33).
        return float(np.clip(rf - 0.55 * db - 1.25 * lb, 0.0, 1.0))
    h = float(d["hue"])
    s = float(d["sat"])
    b = float(d["black_frac"])
    dh = min(abs(h - 0.0), abs(h - 180.0))
    return float(s * np.exp(-((dh / 40.0) ** 2)) * max(0.0, 1.0 - 0.8 * b))


def _darkness(d: dict[str, float]) -> float:
    """How black/dark the jersey is (0–1)."""
    db = float(d.get("dark_body_frac") if d.get("dark_body_frac") is not None else d.get("black_frac") or 0.0)
    return float(0.55 * db + 0.45 * (1.0 - float(d["val"])))


def descriptor_to_vec(d: dict[str, float] | None) -> list[float]:
    """Cluster vector for ANY two kit colours (not red-only).

    Circular hue × sat separates blue/yellow/green/red; brightness/darkness
    separates white/black/cream. Soft redness keeps red-vs-black sharp.
    """
    if not d:
        return [0.0, 0.0, 0.3, 0.5, 0.0]
    if "redness" not in d:
        d = dict(d)
        d["redness"] = _redness(d)
        d["darkness"] = _darkness(d)
    ang = float(d["hue"]) * (np.pi / 90.0)
    s = float(d["sat"])
    return [
        float(1.6 * s * np.cos(ang)),
        float(1.6 * s * np.sin(ang)),
        float(1.3 * kit_brightness(d)),
        float(1.3 * d["darkness"]),
        float(1.0 * d["redness"]),
    ]


def descriptor_from_hist_fallback(hist: list[float] | None) -> dict[str, float] | None:
    """Best-effort from legacy 8×8 H×S hist (no V) — only used if no jersey_hsv."""
    if not hist or len(hist) < 64:
        return None
    bins = 8
    # Peak chromatic bin
    best_i, best_v = 0, -1.0
    for i, v in enumerate(hist[:64]):
        s_bin = i % bins
        if s_bin < 2:
            continue
        if float(v) > best_v:
            best_v = float(v)
            best_i = i
    hue = (best_i // bins) * (180.0 / bins) + (180.0 / bins) / 2
    sat_mass = [0.0] * bins
    for i, v in enumerate(hist[:64]):
        sat_mass[i % bins] += float(v)
    total = sum(sat_mass) or 1.0
    sat = sum(i * sat_mass[i] for i in range(bins)) / total / 7.0
    low_s = (sat_mass[0] + sat_mass[1]) / total
    # Without V we can't tell white from black — mark uncertain
    return {
        "hue": float(hue),
        "sat": float(sat),
        "val": 0.5,
        "black_frac": float(low_s * 0.5),  # weak prior
    }


def kit_brightness(d: dict[str, float] | None) -> float:
    """High = white/bright kit, low = black/dark kit. Uses V, not low-S alone."""
    if not d:
        return 0.5
    # White: high V, low S; Black: low V; Coloured: mid V high S → mid
    v = float(d["val"])
    s = float(d["sat"])
    b = float(d["black_frac"])
    return float(np.clip(0.65 * v + 0.20 * (1.0 - s) - 0.45 * b, 0.0, 1.0))


def _l2(a: list[float], b: list[float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def assign_kits_from_descriptors(metas: list[dict]) -> None:
    """In-place Kit A/B on metas that have jersey_hsv or color_hist.

    Outliers (cyan ref / neon GK) get kit='X' and are excluded from 2-means.
    """
    if not metas:
        return
    if len(metas) == 1:
        metas[0]["kit"] = "A"
        return

    for m in metas:
        d = m.get("jersey_hsv")
        if not d:
            d = descriptor_from_hist_fallback(m.get("color_hist"))
        if d is not None:
            d = dict(d)
            d["is_outlier"] = bool(d.get("is_outlier")) or is_kit_outlier(d)
        m["jersey_hsv"] = d

    players = [m for m in metas if not (m.get("jersey_hsv") or {}).get("is_outlier")]
    outliers = [m for m in metas if (m.get("jersey_hsv") or {}).get("is_outlier")]
    for m in outliers:
        m["kit"] = "X"

    if len(players) == 0:
        return
    if len(players) == 1:
        players[0]["kit"] = "A"
        return

    descs = [m.get("jersey_hsv") for m in players]
    vecs = [descriptor_to_vec(d) for d in descs]
    # Seed: two farthest points
    pair = max(
        ((i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs))),
        key=lambda ij: _l2(vecs[ij[0]], vecs[ij[1]]),
        default=(0, min(1, len(vecs) - 1)),
    )
    c0 = list(vecs[pair[0]])
    c1 = list(vecs[pair[1]])
    labels = [0] * len(vecs)
    for _ in range(16):
        groups: list[list[list[float]]] = [[], []]
        changed = False
        for i, v in enumerate(vecs):
            lab = 0 if _l2(v, c0) <= _l2(v, c1) else 1
            if lab != labels[i]:
                changed = True
            labels[i] = lab
            groups[lab].append(v)
        if not groups[0] or not groups[1]:
            break
        dim = len(vecs[0])
        c0 = [sum(x[k] for x in groups[0]) / len(groups[0]) for k in range(dim)]
        c1 = [sum(x[k] for x in groups[1]) / len(groups[1]) for k in range(dim)]
        if not changed:
            break

    if len(set(labels)) < 2:
        brights = [kit_brightness(d) for d in descs]
        med = sorted(brights)[len(brights) // 2]
        for m, b in zip(players, brights):
            m["kit"] = "A" if b >= med else "B"
        return

    # Kit A = brighter / less-dark / redder centre (stable label, any colours)
    # vec: [hue_x, hue_y, brightness, darkness, redness]
    def centre_a_score(c: list[float]) -> float:
        return float(c[2] - c[3] + 0.35 * c[4])

    a_label = 0 if centre_a_score(c0) >= centre_a_score(c1) else 1
    for m, lab in zip(players, labels):
        m["kit"] = "A" if lab == a_label else "B"

    # Soft red-vs-other rescue when redness centres clearly differ.
    red_sep = abs(c0[4] - c1[4])
    if red_sep >= 0.25:
        for m in players:
            d = m.get("jersey_hsv") or {}
            r = float(d.get("redness") or 0.0)
            lb = float(d.get("light_body_frac") or 0.0)
            rf = float(d.get("red_frac") or 0.0)
            db = float(d.get("dark_body_frac") or 0.0)
            if r >= 0.40 and lb < 0.18:
                m["kit"] = "A"
            elif lb >= 0.18 or (r < 0.12 and db >= rf):
                m["kit"] = "B"


def kit_purity(metas: list[dict]) -> dict[str, Any]:
    """Honest PASS/FAIL: strong reds must not sit in both kits; cream not in red kit."""
    by = {"A": [], "B": [], "X": []}
    for m in metas:
        by.setdefault(m.get("kit") or "?", []).append(m.get("jersey_hsv") or {})

    def _reds(ds: list) -> list:
        return [d for d in ds if float(d.get("redness") or 0) >= 0.25]

    def _creams(ds: list) -> list:
        return [d for d in ds if float(d.get("light_body_frac") or 0) >= 0.20]

    ra, rb = _reds(by["A"]), _reds(by["B"])
    ca = _creams(by["A"])
    issues: list[str] = []
    if ra and rb:
        issues.append(f"strong-red in BOTH kits (A={len(ra)} B={len(rb)})")
    if ca and ra:
        issues.append(f"cream mixed into red Kit A ({len(ca)})")
    if len(by["A"]) == 0 or len(by["B"]) == 0:
        issues.append("empty kit bucket")
    # Within-kit redness spread when A looks like a red team
    if ra and len(by["A"]) >= 2:
        weak = [d for d in by["A"] if float(d.get("redness") or 0) < 0.20]
        if len(weak) >= max(1, len(by["A"]) // 3):
            issues.append(f"Kit A polluted with {len(weak)} low-red snaps")
    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "A": len(by["A"]),
        "B": len(by["B"]),
        "X": len(by["X"]),
        "red_in_A": len(ra),
        "red_in_B": len(rb),
    }


def descriptor_from_frame_box(frame_bgr, xyxy: list[float]) -> dict[str, float] | None:
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return jersey_descriptor_bgr(frame_bgr[y1:y2, x1:x2])


def descriptor_from_thumb_path(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    import cv2
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return None
    img = cv2.imread(str(p))
    return jersey_descriptor_bgr(img)


def mean_descriptor(ds: list[dict[str, float] | None]) -> dict[str, float] | None:
    ok = [d for d in ds if d]
    if not ok:
        return None
    # Circular mean hue
    angs = [d["hue"] * (np.pi / 90.0) for d in ok]
    mh = float(np.arctan2(np.mean(np.sin(angs)), np.mean(np.cos(angs))))
    if mh < 0:
        mh += 2 * np.pi
    d = {
        "hue": mh * (90.0 / np.pi),
        "sat": float(np.mean([d["sat"] for d in ok])),
        "val": float(np.mean([d["val"] for d in ok])),
        "black_frac": float(np.mean([d["black_frac"] for d in ok])),
    }
    rfs = [x.get("red_frac") for x in ok if x.get("red_frac") is not None]
    if rfs:
        d["red_frac"] = float(np.mean(rfs))
    dbs = [x.get("dark_body_frac") for x in ok if x.get("dark_body_frac") is not None]
    if dbs:
        d["dark_body_frac"] = float(np.mean(dbs))
    lbs = [x.get("light_body_frac") for x in ok if x.get("light_body_frac") is not None]
    if lbs:
        d["light_body_frac"] = float(np.mean(lbs))
    d["redness"] = _redness(d)
    d["darkness"] = _darkness(d)
    return d
