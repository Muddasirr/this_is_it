"""Kit + jersey clustering without Postgres (same rules as identity.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jersey_ocr import prefer_jersey_number
from vision import Detection

COLOR_SIM_THRESHOLD = 0.72
CLOSEUP_BOX_FRAC = 0.18
MIN_PLAYER_BOX_FRAC = 0.0008
try:
    from court_format import max_auto_players as _max_auto

    MAX_AUTO_PLAYERS = _max_auto()
except Exception:
    MAX_AUTO_PLAYERS = 16


def median_box_frac(features: dict | None, bbox_series: list | None = None) -> float:
    series = list(bbox_series or []) or list((features or {}).get("bbox_series") or [])
    fracs: list[float] = []
    for pt in series:
        w = float(pt.get("frame_w") or 0) or 1280.0
        h = float(pt.get("frame_h") or 0) or 720.0
        bw = max(0.0, float(pt.get("x2", 0)) - float(pt.get("x1", 0)))
        bh = max(0.0, float(pt.get("y2", 0)) - float(pt.get("y1", 0)))
        fracs.append((bw * bh) / (w * h + 1e-6))
    if not fracs:
        return 0.0
    fracs.sort()
    return fracs[len(fracs) // 2]


def is_on_court_person(features: dict | None, bbox_series: list | None = None) -> bool:
    frac = median_box_frac(features, bbox_series)
    return MIN_PLAYER_BOX_FRAC <= frac <= CLOSEUP_BOX_FRAC


@dataclass
class TrackView:
    track_key: str
    start_ms: int
    end_ms: int
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)
    class_name: str = "person"
    bbox_series: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PlayerCluster:
    player_key: str
    display_name: str
    kit: str
    shirt_number: int | None
    jersey_confidence: float
    track_keys: list[str]
    color_hist: list[float] | None
    thumb_path: str | None = None


def detections_to_tracks(detections: list[Detection]) -> list[TrackView]:
    out: list[TrackView] = []
    for d in detections:
        feats = dict(d.features or {})
        feats.setdefault("class", d.class_name)
        feats["bbox_series"] = list(d.bbox_series or [])
        out.append(
            TrackView(
                track_key=d.track_key,
                start_ms=d.start_ms,
                end_ms=d.end_ms,
                confidence=d.confidence,
                features=feats,
                class_name=d.class_name,
                bbox_series=list(d.bbox_series or []),
            )
        )
    return out


def propose_players(tracks: list[TrackView]) -> tuple[list[PlayerCluster], dict[str, str]]:
    """Return player clusters + map track_key → player_key."""
    persons = [t for t in tracks if t.class_name == "person" or (t.features or {}).get("class") == "person"]
    on_court = [t for t in persons if is_on_court_person(t.features, t.bbox_series)]
    persons = on_court or persons
    clusters = _cluster_tracks(persons)
    clusters = sorted(clusters, key=lambda c: sum(t.end_ms - t.start_ms for t in c), reverse=True)[
        :MAX_AUTO_PLAYERS
    ]

    meta: list[dict] = []
    for cluster in clusters:
        number, nconf = _cluster_jersey(cluster)
        hist = _cluster_mean_hist(cluster)
        thumb = None
        best_score = -1.0
        for t in cluster:
            tp = (t.features or {}).get("thumb_path")
            sc = float((t.features or {}).get("thumb_score") or 0)
            # Prefer on-court crops over broadcast face close-ups for avatars
            frac = median_box_frac(t.features, t.bbox_series)
            if frac > CLOSEUP_BOX_FRAC:
                sc *= 0.05
            if tp and sc > best_score:
                best_score = sc
                thumb = tp
        meta.append(
            {
                "tracks": cluster,
                "jersey_number": number,
                "jersey_confidence": nconf,
                "has_color": hist is not None,
                "color_hist": hist,
                "duration": sum(t.end_ms - t.start_ms for t in cluster),
                "thumb_path": thumb,
                "kit": "A",
            }
        )
    meta = _merge_colour_fragments(meta)
    _assign_kits(meta)
    _reconcile_same_colour_kits(meta)
    _smooth_kits_by_neighbours(meta)

    # Unique shirt numbers within kit
    used: dict[str, set[int]] = {"A": set(), "B": set()}
    next_free = {"A": 1, "B": 1}
    players: list[PlayerCluster] = []
    track_to_player: dict[str, str] = {}

    for m in meta:
        kit = m.get("kit") or "A"
        num = m.get("jersey_number")
        if num is not None and int(num) not in used[kit]:
            shirt = int(num)
        else:
            shirt = next_free[kit]
            while shirt in used[kit]:
                shirt += 1
            next_free[kit] = shirt + 1
        used[kit].add(shirt)
        pkey = f"kit{kit}_{shirt}"
        display = f"Kit {kit} #{shirt}"
        cluster = m["tracks"]
        players.append(
            PlayerCluster(
                player_key=pkey,
                display_name=display,
                kit=kit,
                shirt_number=shirt,
                jersey_confidence=float(m.get("jersey_confidence") or 0),
                track_keys=[t.track_key for t in cluster],
                color_hist=m.get("color_hist"),
                thumb_path=m.get("thumb_path"),
            )
        )
        for t in cluster:
            track_to_player[t.track_key] = pkey

    return players, track_to_player


def _cluster_jersey(cluster: list[TrackView]) -> tuple[int | None, float]:
    votes: list[tuple[int, float]] = []
    for t in cluster:
        feats = t.features or {}
        num = feats.get("jersey_number")
        if num is None:
            continue
        conf = float(feats.get("jersey_confidence") or 0.5)
        votes.append((int(num), conf))
        for _ in range(max(0, int(feats.get("jersey_votes") or 1) - 1)):
            votes.append((int(num), conf))
    return prefer_jersey_number(votes)


def _cluster_mean_hist(cluster: list[TrackView]) -> list[float] | None:
    hists = []
    for t in cluster:
        h = (t.features or {}).get("color_hist")
        if h:
            hists.append(list(h))
    if not hists:
        return None
    return _mean_hist(hists)


def _dominant_hue_bin(hist: list[float] | None) -> float:
    """8×8 H×S hist → weighted hue bin (0–7). Prefer chromatic (higher-S) mass."""
    if not hist or len(hist) < 64:
        return 3.5
    bins = 8
    weights = [0.0] * bins
    for i, v in enumerate(hist[:64]):
        h_bin = i // bins
        s_bin = i % bins
        # Down-weight grey/white/black pixels — their hue is noise
        w = float(v) * (0.15 + 0.85 * (s_bin / 7.0))
        weights[h_bin] += w
    total = sum(weights) or 1.0
    return sum(i * w for i, w in enumerate(weights)) / total


def _kit_lightness(hist: list[float] | None) -> float:
    """0 = dark/saturated kit, 1 = light/grey/white kit.

    Uses low-saturation mass in the H×S hist (white/grey jerseys pile into
    low-S bins; coloured kits do not). Optional features['color_brightness']
    is folded in when present via hist metadata is not available here — callers
    that have it should prefer _kit_features().
    """
    if not hist or len(hist) < 64:
        return 0.5
    bins = 8
    sat = [0.0] * bins
    for i, v in enumerate(hist[:64]):
        sat[i % bins] += float(v)
    total = sum(sat) or 1.0
    # bins 0–2 ≈ unsaturated (white / black / grey cloth under CCTV)
    return min(1.0, max(0.0, (sat[0] + sat[1] + 0.5 * sat[2]) / total))


def _kit_features(meta: dict) -> tuple[float, float]:
    """(lightness, hue_bin) for a player cluster."""
    hist = meta.get("color_hist")
    # Prefer explicit brightness from detector when available on any track
    bright_vals: list[float] = []
    for t in meta.get("tracks") or []:
        feats = getattr(t, "features", None) or {}
        if feats.get("color_brightness") is not None:
            bright_vals.append(float(feats["color_brightness"]))
    lightness = sum(bright_vals) / len(bright_vals) if bright_vals else _kit_lightness(hist)
    return lightness, _dominant_hue_bin(hist)


def _two_means_1d(values: list[float]) -> tuple[float, float]:
    c0, c1 = min(values), max(values)
    if c0 == c1:
        return c0, c1
    for _ in range(10):
        g0, g1 = [], []
        for v in values:
            (g0 if abs(v - c0) <= abs(v - c1) else g1).append(v)
        if not g0 or not g1:
            break
        c0 = sum(g0) / len(g0)
        c1 = sum(g1) / len(g1)
    return c0, c1


def _circular_hue_dist(a: float, b: float, period: float = 8.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def _hist_vec(hist: list[float] | None) -> list[float]:
    if not hist or len(hist) < 64:
        return [1.0 / 64.0] * 64
    s = sum(float(x) for x in hist[:64]) or 1.0
    return [float(x) / s for x in hist[:64]]


def _hist_l2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# Fragments of one player (ByteTrack ID switches) share colour and rarely overlap.
FRAGMENT_MERGE_SIM = 0.90
FRAGMENT_MAX_OVERLAP_MS = 800
# After kit assign: only force same kit for *same-person* fragments (high sim +
# low overlap). Do NOT union all similar teammates — that collapses both teams.
SAME_COLOUR_KIT_SIM = 0.90


def _cluster_time_overlap_ms(a: list, b: list) -> int:
    """Total ms where any track in a overlaps any track in b."""
    total = 0
    for ta in a:
        for tb in b:
            start = max(ta.start_ms, tb.start_ms)
            end = min(ta.end_ms, tb.end_ms)
            if end > start:
                total += end - start
    return total


def _refresh_meta(meta: dict) -> None:
    tracks = meta["tracks"]
    meta["jersey_number"], meta["jersey_confidence"] = _cluster_jersey(tracks)
    hist = _cluster_mean_hist(tracks)
    meta["color_hist"] = hist
    meta["has_color"] = hist is not None
    meta["duration"] = sum(t.end_ms - t.start_ms for t in tracks)


def _merge_colour_fragments(cluster_meta: list[dict]) -> list[dict]:
    """Merge near-identical colour clusters that don't meaningfully co-exist.

    Same player split across track IDs → one roster row. Teammates with the
    same kit still overlap on court, so they stay separate.
    """
    if len(cluster_meta) < 2:
        return cluster_meta
    metas = [dict(m, tracks=list(m["tracks"])) for m in cluster_meta]
    changed = True
    while changed:
        changed = False
        metas.sort(key=lambda m: -int(m.get("duration") or 0))
        i = 0
        while i < len(metas):
            j = i + 1
            while j < len(metas):
                h1, h2 = metas[i].get("color_hist"), metas[j].get("color_hist")
                if not h1 or not h2:
                    j += 1
                    continue
                sim = _cosine(h1, h2)
                overlap = _cluster_time_overlap_ms(metas[i]["tracks"], metas[j]["tracks"])
                if sim >= FRAGMENT_MERGE_SIM and overlap <= FRAGMENT_MAX_OVERLAP_MS:
                    metas[i]["tracks"].extend(metas[j]["tracks"])
                    _refresh_meta(metas[i])
                    metas.pop(j)
                    changed = True
                    continue
                j += 1
            i += 1
    return metas


def _reconcile_same_colour_kits(cluster_meta: list[dict]) -> None:
    """Force same kit letter for same-person fragments split across A/B.

    MANAV #23 was Kit A #2 and Kit B #2 because ByteTrack fragments got
    opposite kit labels. Only pairs that look like one person (very similar
    colour AND little time overlap) are reconciled — teammates keep their kits.
    """
    n = len(cluster_meta)
    if n < 2:
        return
    # Iterate to convergence (short chains of fragments)
    for _ in range(4):
        changed = False
        for i in range(n):
            h1 = cluster_meta[i].get("color_hist")
            if not h1:
                continue
            for j in range(i + 1, n):
                h2 = cluster_meta[j].get("color_hist")
                if not h2:
                    continue
                if cluster_meta[i].get("kit") == cluster_meta[j].get("kit"):
                    continue
                if _cosine(h1, h2) < SAME_COLOUR_KIT_SIM:
                    continue
                overlap = _cluster_time_overlap_ms(
                    cluster_meta[i]["tracks"], cluster_meta[j]["tracks"]
                )
                if overlap > FRAGMENT_MAX_OVERLAP_MS:
                    continue  # likely two teammates on pitch together
                best = i if int(cluster_meta[i].get("duration") or 0) >= int(
                    cluster_meta[j].get("duration") or 0
                ) else j
                kit = cluster_meta[best].get("kit") or "A"
                if cluster_meta[i]["kit"] != kit or cluster_meta[j]["kit"] != kit:
                    cluster_meta[i]["kit"] = kit
                    cluster_meta[j]["kit"] = kit
                    changed = True
        if not changed:
            break


def _smooth_kits_by_neighbours(cluster_meta: list[dict], sim_thresh: float = 0.91) -> None:
    """Pull each cluster toward the majority kit of its colour-neighbours.

    Teammates in the same jersey (high cosine, even if overlapping on court)
    converge to one kit letter. Weakly related colours stay put.
    """
    n = len(cluster_meta)
    if n < 2:
        return
    for _ in range(6):
        changed = False
        for i, m in enumerate(cluster_meta):
            h1 = m.get("color_hist")
            if not h1:
                continue
            votes: dict[str, float] = {}
            for j, o in enumerate(cluster_meta):
                if i == j:
                    continue
                h2 = o.get("color_hist")
                if not h2:
                    continue
                if _cosine(h1, h2) < sim_thresh:
                    continue
                kit = o.get("kit") or "A"
                votes[kit] = votes.get(kit, 0.0) + float(o.get("duration") or 1)
            if not votes:
                continue
            self_kit = m.get("kit") or "A"
            votes[self_kit] = votes.get(self_kit, 0.0) + 0.25 * float(m.get("duration") or 1)
            best = max(votes, key=votes.get)
            if best != self_kit and votes[best] > votes.get(self_kit, 0.0):
                m["kit"] = best
                changed = True
        if not changed:
            break


def _assign_kits(cluster_meta: list[dict]) -> None:
    """Split clusters into Kit A / Kit B.

    Primary: 2-means on the full H×S colour histogram. That separates
    white vs red vs black/pink — lightness-only wrongly dumped red and
    dark/pink into the same bucket. Kit A = lighter mean cluster.
    """
    if not cluster_meta:
        return
    if len(cluster_meta) == 1:
        cluster_meta[0]["kit"] = "A"
        return

    vecs = [_hist_vec(m.get("color_hist")) for m in cluster_meta]
    # Seed with the two most distant histograms
    pair = max(
        ((i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs))),
        key=lambda ij: _hist_l2(vecs[ij[0]], vecs[ij[1]]),
        default=(0, 0),
    )
    c0 = list(vecs[pair[0]])
    c1 = list(vecs[pair[1]])
    labels = [0] * len(vecs)
    for _ in range(12):
        changed = False
        groups: list[list[list[float]]] = [[], []]
        for i, v in enumerate(vecs):
            lab = 0 if _hist_l2(v, c0) <= _hist_l2(v, c1) else 1
            if lab != labels[i]:
                changed = True
            labels[i] = lab
            groups[lab].append(v)
        if not groups[0] or not groups[1]:
            break
        c0 = _mean_hist(groups[0])
        c1 = _mean_hist(groups[1])
        if not changed:
            break

    if len(set(labels)) < 2:
        # Degenerate — fall back to lightness median
        lights = [_kit_lightness(m.get("color_hist")) for m in cluster_meta]
        med = sorted(lights)[len(lights) // 2]
        for m, light in zip(cluster_meta, lights):
            m["kit"] = "A" if light >= med else "B"
        return

    light0 = _kit_lightness(c0)
    light1 = _kit_lightness(c1)
    # Kit A = lighter centre (stable home/away convention)
    a_label = 0 if light0 >= light1 else 1
    for m, lab in zip(cluster_meta, labels):
        m["kit"] = "A" if lab == a_label else "B"


def _cluster_tracks(tracks: list[TrackView]) -> list[list[TrackView]]:
    if not tracks:
        return []
    ordered = sorted(tracks, key=lambda t: t.end_ms - t.start_ms, reverse=True)
    clusters: list[list[TrackView]] = []
    centroids: list[list[float] | None] = []
    for track in ordered:
        hist = (track.features or {}).get("color_hist")
        placed = False
        for i, cluster in enumerate(clusters):
            if _overlaps_any(track, cluster):
                continue
            cent = centroids[i]
            if hist and cent:
                if abs(_kit_lightness(hist) - _kit_lightness(cent)) > 0.32:
                    continue
                if _cosine(hist, cent) >= COLOR_SIM_THRESHOLD:
                    cluster.append(track)
                    centroids[i] = _mean_hist([cent, hist])
                    placed = True
                    break
            elif hist is None and cent is None:
                if _temporal_gap_ok(track, cluster):
                    cluster.append(track)
                    placed = True
                    break
        if not placed:
            clusters.append([track])
            centroids.append(list(hist) if hist else None)
    return clusters


def _overlaps_any(track: TrackView, cluster: list[TrackView]) -> bool:
    for other in cluster:
        if track.start_ms <= other.end_ms and other.start_ms <= track.end_ms:
            overlap = min(track.end_ms, other.end_ms) - max(track.start_ms, other.start_ms)
            if overlap > 400:
                return True
    return False


def _temporal_gap_ok(track: TrackView, cluster: list[TrackView], max_gap_ms: int = 8000) -> bool:
    gaps = []
    for other in cluster:
        if track.start_ms > other.end_ms:
            gaps.append(track.start_ms - other.end_ms)
        elif other.start_ms > track.end_ms:
            gaps.append(other.start_ms - track.end_ms)
    return bool(gaps) and min(gaps) <= max_gap_ms


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mean_hist(hists: list[list[float]]) -> list[float]:
    n = len(hists[0])
    acc = [0.0] * n
    for h in hists:
        for i, v in enumerate(h):
            acc[i] += v
    inv = 1.0 / len(hists)
    acc = [v * inv for v in acc]
    norm = sum(v * v for v in acc) ** 0.5 or 1.0
    return [v / norm for v in acc]
