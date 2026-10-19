"""Identity: kit color + jersey number → roster; staff confirms/corrects."""

from __future__ import annotations

from collections import Counter, defaultdict
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Match, MatchRoster, Player, PlayerTrack, Team, TrackIdentityLink
from app.models.entities import IdentityState

COLOR_SIM_THRESHOLD = 0.72
MAX_AUTO_PLAYERS = 14


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

    # Enrich each cluster with dominant jersey number (OCR votes)
    cluster_meta: list[dict] = []
    for cluster in clusters:
        number, nconf = _cluster_jersey(cluster)
        has_color = any((t.features or {}).get("color_hist") for t in cluster)
        cluster_meta.append(
            {
                "tracks": cluster,
                "jersey_number": number,
                "jersey_confidence": nconf,
                "has_color": has_color,
                "duration": sum(t.end_ms - t.start_ms for t in cluster),
            }
        )

    roster = (
        db.query(MatchRoster).filter(MatchRoster.match_id == match_id).order_by(MatchRoster.shirt_number).all()
    )
    roster_by_number = {r.shirt_number: r for r in roster}

    # Ensure roster players exist for detected numbers / clusters
    if cluster_meta and (not roster or len(roster) < min(6, len(cluster_meta))):
        roster = _ensure_detected_roster(db, match, cluster_meta)
        roster_by_number = {r.shirt_number: r for r in roster}

    used_roster_ids: set[UUID] = set()
    links: list[TrackIdentityLink] = []

    for meta in cluster_meta:
        cluster = meta["tracks"]
        number = meta["jersey_number"]
        player_id = None
        state = IdentityState.unknown
        conf = 0.2
        source = "model_color"

        # 1) Prefer exact shirt-number match on roster
        if number is not None and number in roster_by_number:
            entry = roster_by_number[number]
            if entry.player_id not in used_roster_ids:
                player_id = entry.player_id
                used_roster_ids.add(entry.player_id)
                state = IdentityState.proposed
                conf = min(0.9, 0.45 + float(meta["jersey_confidence"] or 0) * 0.5)
                if meta["has_color"]:
                    conf = min(0.95, conf + 0.1)
                source = "model_color_number"
        # 2) Else assign next unused roster slot (color cluster only)
        if player_id is None and roster:
            for entry in roster:
                if entry.player_id in used_roster_ids:
                    continue
                player_id = entry.player_id
                used_roster_ids.add(entry.player_id)
                state = IdentityState.proposed
                conf = 0.55 if meta["has_color"] else 0.35
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
    votes: list[tuple[int, float]] = []
    for t in cluster:
        feats = t.features or {}
        num = feats.get("jersey_number")
        if num is None:
            continue
        conf = float(feats.get("jersey_confidence") or 0.5)
        votes.append((int(num), conf))
        # weight by vote count if present
        for _ in range(max(0, int(feats.get("jersey_votes") or 1) - 1)):
            votes.append((int(num), conf))
    if not votes:
        return None, 0.0
    scores: dict[int, float] = defaultdict(float)
    counts: Counter[int] = Counter()
    for n, c in votes:
        scores[n] += c
        counts[n] += 1
    best = max(scores.keys(), key=lambda n: (scores[n], counts[n]))
    return best, round(scores[best] / max(1, counts[best]), 3)


def _ensure_detected_roster(db: Session, match: Match, cluster_meta: list[dict]) -> list[MatchRoster]:
    """Create/reuse Auto-detected players named by jersey when OCR found one."""
    team = (
        db.query(Team)
        .filter(Team.facility_id == match.facility_id, Team.name == "Auto-detected")
        .first()
    )
    if team is None:
        team = Team(facility_id=match.facility_id, name="Auto-detected", primary_color=None)
        db.add(team)
        db.flush()

    existing_players = (
        db.query(Player)
        .filter(Player.facility_id == match.facility_id, Player.team_id == team.id)
        .order_by(Player.default_shirt_number)
        .all()
    )
    by_number = {p.default_shirt_number: p for p in existing_players if p.default_shirt_number is not None}

    # Choose shirt numbers: OCR number when unique, else free slots 1..N
    used_numbers: set[int] = set()
    assigned: list[tuple[dict, int]] = []
    for meta in cluster_meta:
        num = meta["jersey_number"]
        if num is not None and num not in used_numbers:
            used_numbers.add(num)
            assigned.append((meta, num))
        else:
            assigned.append((meta, 0))  # fill later

    next_free = 1
    filled: list[tuple[dict, int]] = []
    for meta, num in assigned:
        if num:
            filled.append((meta, num))
            continue
        while next_free in used_numbers:
            next_free += 1
        used_numbers.add(next_free)
        filled.append((meta, next_free))
        next_free += 1

    for _, num in filled:
        if num in by_number:
            continue
        name = f"Player #{num}"
        player = Player(
            facility_id=match.facility_id,
            team_id=team.id,
            display_name=name,
            default_shirt_number=num,
            is_active=True,
        )
        db.add(player)
        db.flush()
        by_number[num] = player
        existing_players.append(player)

    db.query(MatchRoster).filter(MatchRoster.match_id == match.id).delete(synchronize_session=False)
    roster: list[MatchRoster] = []
    for _, num in filled:
        player = by_number[num]
        # Keep display name synced to number for auto players
        if player.display_name.startswith("Detected Player") or player.display_name.startswith("Player #"):
            player.display_name = f"Player #{num}"
            player.default_shirt_number = num
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
