"""KMeans jersey team assign (football_analysis port)."""

from __future__ import annotations

import numpy as np


def _solid_crop(bgr, h=80, w=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def test_jersey_bgr_separates_red_and_white():
    from app.services.team_assigner import TeamAssigner, jersey_bgr_from_crop

    red = jersey_bgr_from_crop(_solid_crop((40, 40, 200)))
    white = jersey_bgr_from_crop(_solid_crop((220, 220, 220)))
    assert red is not None and white is not None
    assert red[2] > red[0]  # R channel dominant in BGR
    assert white.mean() > 180


def test_jersey_bgr_large_crop_does_not_crash():
    from app.services.team_assigner import jersey_bgr_from_crop

    c = jersey_bgr_from_crop(_solid_crop((40, 40, 200), h=188, w=94))
    assert c is not None and c[2] > c[0]


def test_assign_kits_kmeans_two_teams_sticky():
    from app.services.team_assigner import TeamAssigner, assign_kits_kmeans

    metas = [
        {"img": _solid_crop((30, 30, 210)), "sticky_id": "p1", "src": "a.jpg"},
        {"img": _solid_crop((25, 25, 200)), "sticky_id": "p2", "src": "b.jpg"},
        {"img": _solid_crop((210, 210, 210)), "sticky_id": "p3", "src": "c.jpg"},
        {"img": _solid_crop((200, 200, 205)), "sticky_id": "p4", "src": "d.jpg"},
    ]
    assert assign_kits_kmeans(metas)
    kits = {m["sticky_id"]: m["kit"] for m in metas}
    assert kits["p1"] == kits["p2"]
    assert kits["p3"] == kits["p4"]
    assert kits["p1"] != kits["p3"]

    # Sticky: same id keeps kit even if crop changes slightly
    ta = TeamAssigner()
    ta.fit([m["img"] for m in metas])
    k1 = ta.predict(metas[0]["img"], sticky_id="p1")
    k1b = ta.predict(_solid_crop((50, 50, 180)), sticky_id="p1")
    assert k1 == k1b


def test_referee_forced_to_x():
    from app.services.team_assigner import assign_kits_kmeans

    metas = [
        {"img": _solid_crop((30, 30, 210)), "sticky_id": "p1"},
        {"img": _solid_crop((210, 210, 210)), "sticky_id": "p2"},
        {
            "img": _solid_crop((20, 20, 20)),
            "sticky_id": "ref",
            "class_name": "referee",
        },
    ]
    assert assign_kits_kmeans(metas)
    assert metas[2]["kit"] == "X"


def test_interpolate_ball_fills_gap():
    from app.services.possession import interpolate_bbox_series

    pts = [
        {"ts_ms": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "confidence": 0.9},
        {"ts_ms": 200, "x1": 20, "y1": 20, "x2": 30, "y2": 30, "confidence": 0.8},
    ]
    out = interpolate_bbox_series(pts, max_gap_ms=500, step_ms=40)
    assert len(out) > 2
    assert any(p.get("interpolated") for p in out)
    mids = [p for p in out if p.get("interpolated")]
    assert abs(mids[0]["x1"] - 0) > 0
