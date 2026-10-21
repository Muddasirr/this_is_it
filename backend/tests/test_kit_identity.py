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
