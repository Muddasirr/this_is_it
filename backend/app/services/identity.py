"""Identity: kit color (team) + jersey number → roster; staff confirms/corrects.

Same shirt number on opposite kits is allowed (Kit A #10 ≠ Kit B #10).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Match, MatchRoster, Player, PlayerTrack, Team, TrackIdentityLink
from app.models.entities import IdentityState

COLOR_SIM_THRESHOLD = 0.72
MAX_AUTO_PLAYERS = 16


def propose_identity_links(db: Session, match_id: UUID, tracks: list[PlayerTrack]) -> list[TrackIdentityLink]:
    """Cluster by jersey color, fuse OCR shirt numbers, map/create roster players."""
    match = db.get(Match, match_id)
    if not match:
        return []

    person_tracks = [t for t in tracks if (t.features or {}).get("class", "person") == "person"]
    other_tracks = [t for t in tracks if t not in person_tracks]

    clusters = _cluster_tracks(person_tracks)
    clusters = sorted(
        clusters,
        key=lambda c: sum(t.end_ms - t.start_ms for t in c),
        reverse=True,
    )[:MAX_AUTO_PLAYERS]

    cluster_meta: list[dict] = []
    for cluster in clusters:
        number, nconf = _cluster_jersey(cluster)
        hist = _cluster_mean_hist(cluster)
        cluster_meta.append(
            {
                "tracks": cluster,
                "jersey_number": number,
                "jersey_confidence": nconf,
                "has_color": hist is not None,
                "color_hist": hist,
                "duration": sum(t.end_ms - t.start_ms for t in cluster),
                "kit": "A",  # filled below
            }
        )

    _assign_kits(cluster_meta)

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


def _cluster_jersey(cluster: list[PlayerTrack]) -> tuple[int | None, float]:
    from app.services.jersey_ocr import prefer_jersey_number

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
    """8x8 HSV hist → weighted hue bin (0–7)."""
    if not hist or len(hist) < 64:
        return 3.5
    bins = 8
    weights = [0.0] * bins
    for i, v in enumerate(hist[:64]):
        weights[i // bins] += float(v)
    total = sum(weights) or 1.0
    return sum(i * w for i, w in enumerate(weights)) / total


def _assign_kits(cluster_meta: list[dict]) -> None:
    """Split clusters into Kit A / Kit B by kit-color hue."""
    if not cluster_meta:
        return
    hues = [_dominant_hue_bin(m.get("color_hist")) for m in cluster_meta]
    if len(cluster_meta) == 1 or max(hues) - min(hues) < 0.8:
        for m in cluster_meta:
            m["kit"] = "A"
        return

    # 2-means on hue
    c0, c1 = min(hues), max(hues)
    for _ in range(8):
        g0, g1 = [], []
        for h in hues:
            if abs(h - c0) <= abs(h - c1):
                g0.append(h)
            else:
                g1.append(h)
        if not g0 or not g1:
            break
        c0 = sum(g0) / len(g0)
        c1 = sum(g1) / len(g1)

    # Label lower-hue as A for stability
    if c0 > c1:
        c0, c1 = c1, c0
    for m, h in zip(cluster_meta, hues):
        m["kit"] = "A" if abs(h - c0) <= abs(h - c1) else "B"


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

    # Assign shirt numbers unique within kit
    used_by_kit: dict[str, set[int]] = {"A": set(), "B": set()}
    assignments: list[tuple[dict, str, int]] = []

    for meta in cluster_meta:
        kit = meta.get("kit") or "A"
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
