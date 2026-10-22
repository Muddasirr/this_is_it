"""Identity: kit color (team) + jersey number → roster; staff confirms/corrects.

Same shirt number on opposite kits is allowed (Kit A #10 ≠ Kit B #10).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Match, MatchRoster, Player, PlayerTrack, Team, TrackIdentityLink
from app.models.entities import IdentityState
from app.services.court_format import max_auto_players

COLOR_SIM_THRESHOLD = 0.72
# Broadcast cutaways (manager/bench) fill a huge fraction of the frame.
# Facility CCTV players are typically ~0.2%–8% of the frame.
CLOSEUP_BOX_FRAC = 0.18
MIN_PLAYER_BOX_FRAC = 0.0008


def median_box_frac(features: dict | None) -> float:
    """Median person-box area as fraction of the frame (0–1)."""
    series = (features or {}).get("bbox_series") or []
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


def is_on_court_person(features: dict | None) -> bool:
    frac = median_box_frac(features)
    return MIN_PLAYER_BOX_FRAC <= frac <= CLOSEUP_BOX_FRAC


def propose_identity_links(db: Session, match_id: UUID, tracks: list[PlayerTrack]) -> list[TrackIdentityLink]:
    """Cluster by jersey color, fuse OCR shirt numbers, map/create roster players."""
    match = db.get(Match, match_id)
    if not match:
        return []

    person_all = [t for t in tracks if (t.features or {}).get("class", "person") == "person"]
    # Drop broadcast close-ups / sideline staff so they cannot top the leaderboard
    on_court = [t for t in person_all if is_on_court_person(t.features)]
    closeups = [t for t in person_all if t not in on_court]
    person_tracks = on_court if on_court else person_all  # fall back if filter wiped everyone
    skipped_closeups = closeups if on_court else []
    other_tracks = [
        t for t in tracks if (t.features or {}).get("class", "person") != "person"
    ] + skipped_closeups

    # Time-overlap rules ("two tracks at once ⇒ two people") only hold within
    # one camera, so cluster + fragment-merge per camera, then assign kits
    # jointly (Kit A must mean the same team on every angle), then link the
    # other cameras' clusters onto the reference camera's players.
    by_camera: dict[str, list[PlayerTrack]] = {}
    for t in person_tracks:
        by_camera.setdefault(_camera_of(t), []).append(t)
    metas_by_camera = {cam: _camera_cluster_meta(ts) for cam, ts in sorted(by_camera.items())}

    cluster_meta = [m for metas in metas_by_camera.values() for m in metas]
    _assign_kits(cluster_meta)
    for metas in metas_by_camera.values():
        _reconcile_same_colour_kits(metas)
    _smooth_kits_by_neighbours(cluster_meta)
    if len(metas_by_camera) > 1:
        cluster_meta = link_clusters_across_cameras(metas_by_camera)

    roster = _ensure_kit_roster(db, match, cluster_meta)
    roster_by_key = {(r.team_id, r.shirt_number): r for r in roster}

    used_roster_ids: set[UUID] = set()
    links: list[TrackIdentityLink] = []

    for meta in cluster_meta:
        cluster = meta["tracks"]
        number = meta["jersey_number"]
        team_id = meta.get("team_id")
        player_id = None
        state = IdentityState.unknown
        conf = 0.2
        source = "model_color"

        if number is not None and team_id is not None and (team_id, number) in roster_by_key:
            entry = roster_by_key[(team_id, number)]
            if entry.player_id not in used_roster_ids:
                player_id = entry.player_id
                used_roster_ids.add(entry.player_id)
                state = IdentityState.proposed
                conf = min(0.9, 0.45 + float(meta["jersey_confidence"] or 0) * 0.5)
                if meta["has_color"]:
                    conf = min(0.95, conf + 0.1)
                source = "model_kit_number"

        if player_id is None and roster:
            # Prefer unused roster slot on same kit team
            for entry in roster:
                if entry.player_id in used_roster_ids:
                    continue
                if team_id and entry.team_id != team_id:
                    continue
                player_id = entry.player_id
                used_roster_ids.add(entry.player_id)
                state = IdentityState.proposed
                conf = 0.55 if meta["has_color"] else 0.35
                source = "model_kit"
                break
            if player_id is None:
                for entry in roster:
                    if entry.player_id in used_roster_ids:
                        continue
                    player_id = entry.player_id
                    used_roster_ids.add(entry.player_id)
                    state = IdentityState.proposed
                    conf = 0.4
                    source = "model_color"
                    break

        for track in cluster:
            link = TrackIdentityLink(
                track_id=track.id,
                player_id=player_id,
                state=state,
                confidence=conf,
                source=source,
            )
            db.add(link)
            links.append(link)

    for track in other_tracks:
        link = TrackIdentityLink(
            track_id=track.id,
            player_id=None,
            state=IdentityState.unknown,
            confidence=0.0,
            source="model",
        )
        db.add(link)
        links.append(link)

    db.commit()
    return links


def _camera_of(track) -> str:
    label = getattr(track, "camera_label", None)
    return str(getattr(label, "value", label) or "A")


def _camera_cluster_meta(tracks: list[PlayerTrack]) -> list[dict]:
    """Cluster one camera's person tracks into candidate players."""
    clusters = _cluster_tracks(tracks)
    clusters = sorted(
        clusters,
        key=lambda c: sum(t.end_ms - t.start_ms for t in c),
        reverse=True,
    )[: max_auto_players()]

    cluster_meta: list[dict] = []
    for cluster in clusters:
        number, nconf = _cluster_jersey(cluster)
        hist = _cluster_mean_hist(cluster)
        cluster_meta.append(
            {
                "tracks": cluster,
                "camera": _camera_of(cluster[0]),
                "jersey_number": number,
                "jersey_confidence": nconf,
                "has_color": hist is not None,
                "color_hist": hist,
                "duration": sum(t.end_ms - t.start_ms for t in cluster),
                "kit": "A",  # filled by _assign_kits
            }
        )
    for meta in cluster_meta:
        meta["reid_emb"] = _mean_reid(meta["tracks"])

    # BoT-SORT/ByteTrack still fragment one player → many clusters.
    # Prefer OSNet ReID merge, then colour-hist fallback, then kits.
    try:
        from app.services.reid import merge_tracks_by_reid, reid_enabled

        if reid_enabled():
            cluster_meta = merge_tracks_by_reid(cluster_meta)
            for meta in cluster_meta:
                _refresh_meta(meta)
    except Exception:
        pass
    return _merge_colour_fragments(cluster_meta)


def _mean_reid(tracks: list) -> list[float] | None:
    embs = [e for t in tracks if (e := (getattr(t, "features", None) or {}).get("reid_emb"))]
    if not embs:
        return None
    import numpy as np

    m = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
    return (m / float(np.linalg.norm(m) + 1e-12)).tolist()


# Cross-camera link: OSNet cosine needed to call two angles the same player
# (views differ, so looser than the same-camera REID_MERGE_SIM).
CROSS_CAM_REID_SIM = 0.6
CROSS_CAM_COLOUR_SIM = 0.9


def _cross_camera_score(anchor: dict, other: dict) -> float | None:
    """Similarity of two clusters seen on different cameras; None = impossible."""
    kit_a, kit_o = anchor.get("kit"), other.get("kit")
    if kit_a != kit_o or kit_a not in {"A", "B"}:
        return None
    na, no = anchor.get("jersey_number"), other.get("jersey_number")
    conf_a = float(anchor.get("jersey_confidence") or 0)
    conf_o = float(other.get("jersey_confidence") or 0)
    numbers_known = na is not None and no is not None and conf_a > 0 and conf_o > 0
    if numbers_known and int(na) != int(no):
        return None  # readable, different shirts ⇒ different players
    from app.services.reid import cosine

    reid = cosine(anchor.get("reid_emb"), other.get("reid_emb"))
    if numbers_known:
        return 1.0 + reid  # same kit + same shirt beats any appearance match
    if anchor.get("reid_emb") is not None and other.get("reid_emb") is not None:
        return reid if reid >= CROSS_CAM_REID_SIM else None
    colour = _cosine(anchor.get("color_hist") or [], other.get("color_hist") or [])
    return colour - 0.5 if colour >= CROSS_CAM_COLOUR_SIM else None


def link_clusters_across_cameras(metas_by_camera: dict[str, list[dict]]) -> list[dict]:
    """Fold other cameras' clusters into the reference camera's players.

    Reference = camera with the most on-camera person time (usually A, the wide
    view). Each other camera is matched one-to-one (Hungarian on the scores
    above) so two players on one angle never collapse into one. Unmatched
    clusters stay separate players — staff can merge them in review.
    """
    import numpy as np

    from app.services.reid import linear_assignment_max

    cams = sorted(
        metas_by_camera,
        key=lambda c: (-sum(int(m.get("duration") or 0) for m in metas_by_camera[c]), c),
    )
    merged = [dict(m, tracks=list(m["tracks"]), cameras=[m["camera"]]) for m in metas_by_camera[cams[0]]]
    for cam in cams[1:]:
        others = metas_by_camera[cam]
        if not merged or not others:
            merged.extend(dict(m, tracks=list(m["tracks"]), cameras=[cam]) for m in others)
            continue
        score = np.full((len(merged), len(others)), -np.inf)
        for i, anchor in enumerate(merged):
            if cam in anchor["cameras"]:
                continue
            for j, other in enumerate(others):
                s = _cross_camera_score(anchor, other)
                if s is not None:
                    score[i, j] = s
        matched: set[int] = set()
        for i, j in linear_assignment_max(score):
            anchor, other = merged[i], others[j]
            anchor["tracks"].extend(other["tracks"])
            anchor["cameras"].append(cam)
            if anchor.get("jersey_number") is None and other.get("jersey_number") is not None:
                anchor["jersey_number"] = other["jersey_number"]
                anchor["jersey_confidence"] = other.get("jersey_confidence")
            anchor["duration"] = int(anchor.get("duration") or 0) + int(other.get("duration") or 0)
            matched.add(j)
        merged.extend(
            dict(m, tracks=list(m["tracks"]), cameras=[cam]) for j, m in enumerate(others) if j not in matched
        )
    return merged


def _cluster_jersey(cluster: list[PlayerTrack]) -> tuple[int | None, float]:
    from app.services.jersey_ocr import prefer_jersey_number

    votes: list[tuple[int, float]] = []
    for t in cluster:
        feats = getattr(t, "features", None) or {}
        num = feats.get("jersey_number")
        if num is None:
            continue
        conf = float(feats.get("jersey_confidence") or 0.5)
        votes.append((int(num), conf))
        for _ in range(max(0, int(feats.get("jersey_votes") or 1) - 1)):
            votes.append((int(num), conf))
    return prefer_jersey_number(votes)


def _cluster_mean_hist(cluster: list[PlayerTrack]) -> list[float] | None:
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
    """Legacy helper — prefer jersey_hsv + kit_color.kit_brightness.

    Old bug: low-S mass treated black and white the same (both unsaturated).
    Keep a weak sat-based prior for callers that only have hist.
    """
    from app.services.kit_color import descriptor_from_hist_fallback, kit_brightness

    d = descriptor_from_hist_fallback(hist)
    return kit_brightness(d)


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
    """Pull each cluster toward majority kit of near-identical jersey neighbours.

    Uses compact jersey descriptor distance (not hist cosine — that mixed red/black).
    """
    from app.services.kit_color import descriptor_to_vec

    n = len(cluster_meta)
    if n < 2:
        return
    vecs = [descriptor_to_vec(m.get("jersey_hsv")) for m in cluster_meta]
    # Same-kit neighbours are close on brightness/black; red↔black is far
    max_dist = 0.55

    def close(i: int, j: int) -> bool:
        a, b = vecs[i], vecs[j]
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 <= max_dist

    for _ in range(6):
        changed = False
        for i, m in enumerate(cluster_meta):
            votes: dict[str, float] = {}
            for j, o in enumerate(cluster_meta):
                if i == j or not close(i, j):
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
    """Split clusters into Kit A / Kit B (KMeans jersey crops → HSV fallback).

    Sticky kit via track_key / thumb. Refs (features.is_referee) → Kit X.
    """
    import cv2
    from app.services.kit_color import assign_kits_from_descriptors, descriptor_from_thumb_path

    if not cluster_meta:
        return
    # Prefer thumb recompute — corpus hists are often grass-polluted
    for m in cluster_meta:
        sticky = None
        for t in m.get("tracks") or []:
            feats = getattr(t, "features", None) or {}
            if feats.get("is_referee") or getattr(t, "class_name", "") == "referee":
                m["kit"] = "X"
                m["class_name"] = "referee"
                m["jersey_hsv"] = {**(m.get("jersey_hsv") or {}), "is_outlier": True}
            path = feats.get("thumb_path")
            if path and m.get("img") is None:
                img = cv2.imread(str(path))
                if img is not None:
                    m["img"] = img
                    m["thumb_path"] = path
            if sticky is None:
                sticky = getattr(t, "track_key", None)
            if feats.get("jersey_hsv") and not m.get("jersey_hsv"):
                m["jersey_hsv"] = feats["jersey_hsv"]
        m["sticky_id"] = sticky or m.get("src")
        if not m.get("jersey_hsv"):
            d = descriptor_from_thumb_path(m.get("thumb_path"))
            if d:
                m["jersey_hsv"] = d
    assign_kits_from_descriptors(cluster_meta)


def _ensure_kit_roster(db: Session, match: Match, cluster_meta: list[dict]) -> list[MatchRoster]:
    """Create Kit A / Kit B teams; allow same shirt number on both."""
    teams: dict[str, Team] = {}
    for kit, color in (("A", "#3b82f6"), ("B", "#ef4444")):
        name = f"Kit {kit}"
        team = (
            db.query(Team)
            .filter(Team.facility_id == match.facility_id, Team.name == name)
            .first()
        )
        if team is None:
            team = Team(facility_id=match.facility_id, name=name, primary_color=color)
            db.add(team)
            db.flush()
        teams[kit] = team

    # Assign shirt numbers unique within kit (skip refs / outliers → Kit X)
    used_by_kit: dict[str, set[int]] = {"A": set(), "B": set()}
    assignments: list[tuple[dict, str, int]] = []

    for meta in cluster_meta:
        kit = meta.get("kit") or "A"
        if kit not in {"A", "B"}:
            continue
        num = meta.get("jersey_number")
        if num is not None and int(num) not in used_by_kit[kit]:
            used_by_kit[kit].add(int(num))
            assignments.append((meta, kit, int(num)))
        else:
            assignments.append((meta, kit, 0))

    next_free = {"A": 1, "B": 1}
    filled: list[tuple[dict, str, int]] = []
    for meta, kit, num in assignments:
        if num:
            filled.append((meta, kit, num))
            continue
        n = next_free[kit]
        while n in used_by_kit[kit]:
            n += 1
        used_by_kit[kit].add(n)
        next_free[kit] = n + 1
        filled.append((meta, kit, n))
        # Keep OCR miss visible as assigned slot number
        meta["jersey_number"] = n

    # Players keyed by (team_id, shirt)
    players_by_key: dict[tuple[UUID, int], Player] = {}
    for kit, team in teams.items():
        existing = (
            db.query(Player)
            .filter(Player.facility_id == match.facility_id, Player.team_id == team.id)
            .all()
        )
        for p in existing:
            if p.default_shirt_number is not None:
                players_by_key[(team.id, p.default_shirt_number)] = p

    for meta, kit, num in filled:
        team = teams[kit]
        meta["team_id"] = team.id
        meta["jersey_number"] = num
        key = (team.id, num)
        if key in players_by_key:
            player = players_by_key[key]
            # Refresh auto names
            if player.display_name.startswith(("Player #", "Detected Player", "Kit A #", "Kit B #")):
                player.display_name = f"Kit {kit} #{num}"
                player.default_shirt_number = num
            continue
        player = Player(
            facility_id=match.facility_id,
            team_id=team.id,
            display_name=f"Kit {kit} #{num}",
            default_shirt_number=num,
            is_active=True,
        )
        db.add(player)
        db.flush()
        players_by_key[key] = player

    db.query(MatchRoster).filter(MatchRoster.match_id == match.id).delete(synchronize_session=False)
    roster: list[MatchRoster] = []
    for meta, kit, num in filled:
        team = teams[kit]
        player = players_by_key[(team.id, num)]
        entry = MatchRoster(
            match_id=match.id,
            player_id=player.id,
            team_id=team.id,
            shirt_number=num,
            is_goalkeeper=False,
        )
        db.add(entry)
        roster.append(entry)
    db.commit()
    return roster


def _cluster_tracks(tracks: list[PlayerTrack]) -> list[list[PlayerTrack]]:
    if not tracks:
        return []

    ordered = sorted(tracks, key=lambda t: t.end_ms - t.start_ms, reverse=True)
    clusters: list[list[PlayerTrack]] = []
    centroids: list[list[float] | None] = []

    for track in ordered:
        hist = (track.features or {}).get("color_hist")
        placed = False
        for i, cluster in enumerate(clusters):
            if _overlaps_any(track, cluster):
                continue
            cent = centroids[i]
            if hist and cent:
                # Never merge light (white/grey) with dark/coloured jerseys
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


def _overlaps_any(track: PlayerTrack, cluster: list[PlayerTrack]) -> bool:
    for other in cluster:
        if track.start_ms <= other.end_ms and other.start_ms <= track.end_ms:
            overlap = min(track.end_ms, other.end_ms) - max(track.start_ms, other.start_ms)
            if overlap > 400:
                return True
    return False


def _temporal_gap_ok(track: PlayerTrack, cluster: list[PlayerTrack], max_gap_ms: int = 8000) -> bool:
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


def set_track_identity(
    db: Session,
    track_id: UUID,
    player_id: UUID | None,
    user_id: UUID,
    state: IdentityState = IdentityState.confirmed,
) -> TrackIdentityLink:
    existing = (
        db.query(TrackIdentityLink)
        .filter(TrackIdentityLink.track_id == track_id)
        .order_by(TrackIdentityLink.created_at.desc())
        .first()
    )
    if existing and existing.state == IdentityState.confirmed and existing.player_id == player_id:
        return existing

    link = TrackIdentityLink(
        track_id=track_id,
        player_id=player_id,
        state=state if player_id else IdentityState.unknown,
        confidence=1.0 if state == IdentityState.confirmed else 0.0,
        source="staff",
        created_by=user_id,
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


def rename_player(db: Session, player_id: UUID, display_name: str) -> Player:
    player = db.get(Player, player_id)
    if not player:
        raise ValueError("Player not found")
    player.display_name = display_name.strip()[:120] or player.display_name
    db.commit()
    db.refresh(player)
    return player
