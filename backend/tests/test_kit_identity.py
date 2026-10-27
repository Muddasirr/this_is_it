"""Phase 2 gate: kit A/B must split different jersey colours.

Lock rule: this must pass before touching possession/events/UI again.
"""

from __future__ import annotations


def _enrich(d: dict) -> dict:
    from app.services.kit_color import _darkness, _redness

    d = dict(d)
    d["redness"] = _redness(d)
    d["darkness"] = _darkness(d)
    return d


def _white() -> dict:
    return _enrich(
        {
            "hue": 90.0,
            "sat": 0.08,
            "val": 0.88,
            "black_frac": 0.02,
            "red_frac": 0.02,
            "dark_body_frac": 0.05,
            "light_body_frac": 0.7,
        }
    )


def _red() -> dict:
    return _enrich(
        {
            "hue": 175.0,
            "sat": 0.62,
            "val": 0.72,
            "black_frac": 0.05,
            "red_frac": 0.65,
            "dark_body_frac": 0.08,
            "light_body_frac": 0.02,
        }
    )


def _black() -> dict:
    return _enrich(
        {
            "hue": 100.0,
            "sat": 0.22,
            "val": 0.28,
            "black_frac": 0.38,
            "red_frac": 0.04,
            "dark_body_frac": 0.55,
            "light_body_frac": 0.02,
        }
    )


def _black_pink() -> dict:
    # Black jersey with pink number accents — must stay with dark kit
    return _enrich(
        {
            "hue": 160.0,
            "sat": 0.35,
            "val": 0.40,
            "black_frac": 0.28,
            "red_frac": 0.12,
            "dark_body_frac": 0.48,
            "light_body_frac": 0.02,
        }
    )


def _blue() -> dict:
    return _enrich(
        {
            "hue": 110.0,
            "sat": 0.55,
            "val": 0.55,
            "black_frac": 0.08,
            "red_frac": 0.02,
            "dark_body_frac": 0.1,
            "light_body_frac": 0.05,
        }
    )


def test_brightness_white_gt_black():
    from app.services.kit_color import kit_brightness

    assert kit_brightness(_white()) > kit_brightness(_black()) + 0.25
    assert kit_brightness(_red()) > kit_brightness(_black())


def test_red_vs_black_not_same_kit():
    """The bug in the gallery: red and black must not share a kit letter."""
    from app.services.identity import _assign_kits

    meta = [
        {"jersey_hsv": _red(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _red(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _black(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _black_pink(), "tracks": [], "color_hist": None},
    ]
    _assign_kits(meta)
    red_kits = {meta[0]["kit"], meta[1]["kit"]}
    dark_kits = {meta[2]["kit"], meta[3]["kit"]}
    assert len(red_kits) == 1, f"reds split: {[m['kit'] for m in meta]}"
    assert len(dark_kits) == 1, f"darks split: {[m['kit'] for m in meta]}"
    assert red_kits != dark_kits, f"red and black SAME kit: {[m['kit'] for m in meta]}"


def test_white_vs_red_not_same_kit():
    from app.services.identity import _assign_kits

    meta = [
        {"jersey_hsv": _white(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _white(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _red(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _red(), "tracks": [], "color_hist": None},
    ]
    _assign_kits(meta)
    assert {m["kit"] for m in meta[:2]} != {m["kit"] for m in meta[2:]}


def test_blue_vs_yellow_any_colour_kits():
    """Kits can be any colours — not only red/black."""
    from app.services.identity import _assign_kits

    def _yellow():
        return _enrich({"hue": 25.0, "sat": 0.75, "val": 0.80, "black_frac": 0.02, "red_frac": 0.05, "dark_body_frac": 0.05, "light_body_frac": 0.1})

    meta = [
        {"jersey_hsv": _blue(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _blue(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _yellow(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _yellow(), "tracks": [], "color_hist": None},
    ]
    _assign_kits(meta)
    assert {m["kit"] for m in meta[:2]} != {m["kit"] for m in meta[2:]}, f"blue/yellow collapsed: {[m['kit'] for m in meta]}"


def test_cyan_ref_is_outlier_not_kit_a():
    from app.services.kit_color import assign_kits_from_descriptors, is_kit_outlier, kit_purity

    cyan = _enrich(
        {
            "hue": 95.0,
            "sat": 0.7,
            "val": 0.7,
            "black_frac": 0.05,
            "red_frac": 0.1,
            "dark_body_frac": 0.1,
            "light_body_frac": 0.05,
            "cyan_frac": 0.40,
            "neon_frac": 0.0,
        }
    )
    assert is_kit_outlier(cyan)
    meta = [
        {"jersey_hsv": _red(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _red(), "tracks": [], "color_hist": None},
        {"jersey_hsv": _white(), "tracks": [], "color_hist": None},
        {"jersey_hsv": cyan, "tracks": [], "color_hist": None},
    ]
    assign_kits_from_descriptors(meta)
    assert meta[3]["kit"] == "X"
    pur = kit_purity(meta)
    assert pur["X"] == 1
    assert meta[0]["kit"] != meta[2]["kit"]


def test_purity_fails_when_red_in_both_kits():
    from app.services.kit_color import kit_purity

    meta = [
        {"kit": "A", "jersey_hsv": _red()},
        {"kit": "B", "jersey_hsv": _red()},
        {"kit": "B", "jersey_hsv": _black()},
    ]
    pur = kit_purity(meta)
    assert pur["pass"] is False
    assert any("BOTH" in i for i in pur["issues"])


def test_gallery_snaps_red_vs_black(tmp_path=None):
    """Real test_video_2 snaps: red jerseys (incl. former KitB_7) vs black."""
    from pathlib import Path

    from app.services.kit_color import assign_kits_from_descriptors, jersey_descriptor_bgr
    import cv2

    root = Path(__file__).resolve().parents[2].parent
    gal = root / "data/eval/identity_gallery/test_video_2"
    if not gal.exists():
        return  # skip if gallery not present
    # After skin-mask fix: reds in KitA_*, blacks (incl. pink-number kits) in KitB_*.
    reds = ["KitA_1.jpg", "KitA_2.jpg", "KitA_3.jpg"]
    blacks = ["KitB_2.jpg", "KitB_3.jpg", "KitB_6.jpg", "KitB_7.jpg", "KitB_8.jpg"]
    metas = []
    for name in reds + blacks:
        p = gal / name
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        d = jersey_descriptor_bgr(img)
        metas.append({"jersey_hsv": d, "name": name, "tracks": [], "color_hist": None})
    if len(metas) < 4:
        return
    assign_kits_from_descriptors(metas)
    by = {m["name"]: m["kit"] for m in metas}
    red_names = [n for n in reds if n in by]
    black_names = [n for n in blacks if n in by]
    if not red_names or not black_names:
        return
    red_kit = {by[n] for n in red_names}
    black_kit = {by[n] for n in black_names}
    assert len(red_kit) == 1, f"reds not united: {by}"
    assert len(black_kit) == 1, f"blacks not united: {by}"
    assert red_kit != black_kit, f"red/black collapsed: {by}"


def test_red_jersey_not_eaten_by_skin_mask():
    """Regression: torso 'skin' H∈[0,25] must not strip red cloth (old KitB_7)."""
    from pathlib import Path

    import cv2

    from app.services.kit_color import jersey_descriptor_bgr

    gal = (
        Path(__file__).resolve().parents[2].parent
        / "data/eval/identity_gallery/test_video_2"
    )
    pred = gal / "KitA_1.jpg"
    pblack = gal / "KitB_3.jpg"  # black kit with pink number accents
    if not pred.exists():
        return
    d = jersey_descriptor_bgr(cv2.imread(str(pred)))
    assert d is not None
    assert d["redness"] >= 0.18, f"red jersey under-scored: {d}"
    assert float(d.get("red_frac") or 0) >= 0.25, f"red area too low: {d}"
    if pblack.exists():
        db = jersey_descriptor_bgr(cv2.imread(str(pblack)))
        assert db is not None
        assert d["redness"] > db["redness"] + 0.05, f"red not above black+# : {d} vs {db}"


def test_neighbour_smooth_does_not_pull_red_into_black():
    from types import SimpleNamespace

    from app.services.identity import _smooth_kits_by_neighbours

    meta = [
        {
            "jersey_hsv": _red(),
            "tracks": [SimpleNamespace(start_ms=0, end_ms=10000, features={})],
            "duration": 10000,
            "kit": "A",
            "color_hist": None,
        },
        {
            "jersey_hsv": _black(),
            "tracks": [SimpleNamespace(start_ms=0, end_ms=9000, features={})],
            "duration": 9000,
            "kit": "B",
            "color_hist": None,
        },
    ]
    _smooth_kits_by_neighbours(meta)
    assert meta[0]["kit"] != meta[1]["kit"], "smooth merged red into black"


def test_single_cluster_ok():
    from app.services.identity import _assign_kits

    meta = [{"jersey_hsv": _white(), "tracks": [], "color_hist": None}]
    _assign_kits(meta)
    assert meta[0]["kit"] == "A"


def test_same_player_not_both_kit_a_and_b():
    from types import SimpleNamespace

    from app.services.identity import _reconcile_same_colour_kits

    red_hist = [0.0] * 64
    for hue in (0, 1):
        for sat in (5, 6, 7):
            red_hist[hue * 8 + sat] = 4.0
    meta = [
        {
            "color_hist": red_hist,
            "jersey_hsv": _red(),
            "tracks": [SimpleNamespace(start_ms=0, end_ms=5000, features={})],
            "duration": 5000,
            "kit": "A",
        },
        {
            "color_hist": list(red_hist),
            "jersey_hsv": _red(),
            "tracks": [SimpleNamespace(start_ms=8000, end_ms=12000, features={})],
            "duration": 4000,
            "kit": "B",
        },
        {
            "color_hist": [1.0] * 64,
            "jersey_hsv": _black(),
            "tracks": [SimpleNamespace(start_ms=0, end_ms=9000, features={})],
            "duration": 9000,
            "kit": "B",
        },
    ]
    _reconcile_same_colour_kits(meta)
    assert meta[0]["kit"] == meta[1]["kit"]


def test_player_clusters_never_mix_kits_decided_per_track(monkeypatch):
    """Futsal phone clip regression: colour-hist clustering joined both teams' tracks
    (65 % pure players, 10 v 4 roster). Kits are now decided per track first."""
    from types import SimpleNamespace
    from uuid import uuid4

    from app.services import identity

    monkeypatch.setattr("app.services.reid.reid_enabled", lambda: False)
    grass_hist = [0.2, 0.6, 0.2]  # the polluted histogram both kits share
    tracks = []
    for i in range(12):
        red = i % 2 == 0
        tracks.append(SimpleNamespace(
            id=uuid4(), track_key=f"A-p{i}", start_ms=i * 1000, end_ms=i * 1000 + 900, camera_label="A",
            class_name="person",
            features={"class": "person", "color_hist": grass_hist,
                      "team_colour": [55.0, 60.0, 40.0] if red else [40.0, -5.0, -40.0],
                      "team_emb": [1.0, 0.0, 0.1] if red else [0.0, 1.0, 0.1]},
        ))
    red_ids = {t.id for t in tracks[::2]}
    metas = identity._camera_cluster_meta(tracks)
    for m in metas:
        ids = {t.id for t in m["tracks"]}
        assert ids <= red_ids or not (ids & red_ids)
        assert m["track_kit"] in {"A", "B"}
    assert {m["track_kit"] for m in metas} == {"A", "B"}


def test_keeper_clusters_never_take_kit_slots(client, register, monkeypatch):
    """Kit X (keepers / refs) used to take any free roster slot, pushing kit players into the other kit."""
    from uuid import UUID

    from app.core.database import get_db
    from app.main import app
    from app.models import PlayerTrack, ProcessingJob, ProcessingRun, Team, TrackIdentityLink
    from app.models.entities import CameraLabel, JobStage
    from app.services.identity import propose_identity_links

    monkeypatch.setattr("app.services.reid.reid_enabled", lambda: False)
    owner = register("roster@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "M"}).json()
    db = next(app.dependency_overrides[get_db]())
    job = ProcessingJob(match_id=UUID(match["id"]), idempotency_key="roster-test")
    db.add(job)
    db.flush()
    run = ProcessingRun(job_id=job.id, stage=JobStage.detect_track)
    db.add(run)
    db.flush()
    profiles = {"red": [1.0] + [0.0] * 10, "black": [0.0] * 3 + [1.0] + [0.0] * 7, "keeper": [0.0] * 7 + [1.0] + [0.0] * 3}
    tracks, colour_of = [], {}
    # keeper tracks are the longest (rank first), then 6 red and 6 black players at once
    plan = [("keeper", 0, 25000)] * 2 + [("red", 0, 20000)] * 6 + [("black", 0, 20000)] * 6  # keepers ~15 % of time
    for i, (colour, t0, t1) in enumerate(plan):
        x = 100 + 120 * i
        series = [{"ts_ms": t, "x1": x, "y1": 300, "x2": x + 40, "y2": 420, "frame_w": 1920, "frame_h": 1080}
                  for t in range(t0, t1 + 1, 500)]
        t = PlayerTrack(match_id=UUID(match["id"]), processing_run_id=run.id, camera_label=CameraLabel.A,
                        track_key=f"A-p{i}", start_ms=t0, end_ms=t1, confidence=0.9,
                        features={"class": "person", "bbox_series": series, "colour_profile": profiles[colour],
                                  "role_score": 0.1, "color_hist": [0.2, 0.6, 0.2]})
        db.add(t)
        tracks.append(t)
        colour_of[i] = colour
    db.commit()
    propose_identity_links(db, UUID(match["id"]), tracks)
    team_name = {t.id: t.name for t in db.query(Team).all()}
    from app.models import Player

    kit_of_player = {p.id: team_name.get(p.team_id) for p in db.query(Player).all()}
    kits = {}
    for i, t in enumerate(tracks):
        link = db.query(TrackIdentityLink).filter(TrackIdentityLink.track_id == t.id).first()
        kits.setdefault(colour_of[i], set()).add(kit_of_player.get(link.player_id) if link and link.player_id else None)
    assert len(kits["red"] - {None}) == 1 and len(kits["black"] - {None}) == 1
    assert kits["red"] - {None} != kits["black"] - {None}
    assert kits["keeper"] == {None}


def test_wide_shot_players_are_not_filtered_as_too_small():
    from types import SimpleNamespace

    from app.services.identity import MIN_PLAYER_BOX_FRAC, is_on_court_person, min_player_box_frac

    def track(w, h, fw=4096, fh=1080):
        return SimpleNamespace(features={"bbox_series": [
            {"x1": 0, "y1": 0, "x2": w, "y2": h, "frame_w": fw, "frame_h": fh}]})

    panorama = [track(21, 54) for _ in range(20)] + [track(6, 12)]  # players + a far-away speck
    m = min_player_box_frac(panorama)
    assert m < MIN_PLAYER_BOX_FRAC
    assert is_on_court_person(panorama[0].features, m) and not is_on_court_person(panorama[-1].features, m)
    phone = [track(80, 200, 1280, 720) for _ in range(10)]  # futsal: the fixed floor still applies
    assert min_player_box_frac(phone) == MIN_PLAYER_BOX_FRAC


def test_spectators_off_a_grass_pitch_are_filtered(tmp_path):
    """Wide broadcast shots show the stands: crops framed by seats, not grass, are not players.
    On a non-green floor (futsal) the filter stays off."""
    import uuid
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from app.services.identity import _off_grass

    def crop(name: str, bg: tuple[int, int, int]) -> str:
        im = np.full((80, 40, 3), bg, np.uint8)
        im[10:60, 12:28] = (40, 40, 200)  # the person
        path = tmp_path / f"{name}.jpg"
        cv2.imwrite(str(path), im)
        return str(path)

    grass, seats, floor = (40, 140, 40), (120, 120, 130), (60, 110, 170)
    players = [SimpleNamespace(id=uuid.uuid4(), features={"thumb_path": crop(f"p{i}", grass)}) for i in range(12)]
    fans = [SimpleNamespace(id=uuid.uuid4(), features={"thumb_path": crop(f"f{i}", seats)}) for i in range(4)]
    assert _off_grass(players + fans) == {t.id for t in fans}
    assert players[0].features["grass_frac"] > 0.9

    futsal = [SimpleNamespace(id=uuid.uuid4(), features={"thumb_path": crop(f"c{i}", floor)}) for i in range(12)]
    assert _off_grass(futsal + fans) == set()
