"""Regressions found on the first Kaggle GPU run (not reachable without a GPU/video)."""

from app.services.reid import torch_device
from app.services.yolo_backend import _flat_series, _obs


def test_ball_series_interpolates_after_flattening():
    # Kaggle crash: interpolate read flat x1 from nested {"bbox": {...}} observations
    obs = [_obs(0, 0.9, [10, 10, 20, 20], 1080, 1920, "sports_ball"),
           _obs(200, 0.8, [30, 10, 40, 20], 1080, 1920, "sports_ball")]
    series = _flat_series(obs, interpolate=True)
    assert len(series) > 2 and all("x1" in p and "bbox" not in p for p in series)
    mid = [p for p in series if p.get("interpolated")]
    assert mid and 10 < mid[0]["x1"] < 30


def test_torch_device_accepts_ultralytics_style_ids():
    # Kaggle: "Invalid device string: '0'" disabled ReID on every track
    assert torch_device("cpu") == "cpu"
    d = torch_device("0")
    assert d in {"cpu", "cuda:0"}  # cpu when no GPU in the test env


def test_reset_tracker_drops_persisted_state():
    from types import SimpleNamespace

    from app.services.yolo_backend import reset_tracker

    model = SimpleNamespace(predictor=SimpleNamespace(trackers=["camera A tracker"]))
    reset_tracker(model)
    assert not hasattr(model.predictor, "trackers")
    reset_tracker(SimpleNamespace(predictor=None))  # first video: nothing to reset


def test_auto_imgsz_keeps_small_players_visible():
    from app.services.yolo_backend import auto_imgsz

    assert auto_imgsz(4096, 1080) == 1920  # SoccerTrack panorama → capped
    assert auto_imgsz(1920, 1080) == 1920  # facility CCTV at native width
    assert auto_imgsz(1280, 720) == 1280
    assert auto_imgsz(640, 360) == 640
    assert auto_imgsz(0, 0) == 640  # unknown size → safe default


def test_ball_tiles_cover_wide_panorama_at_native_resolution():
    from app.services.yolo_backend import ball_tiles

    assert ball_tiles(1920, 1080, 1920) == [(0, 0, 1920, 1080)]  # fits: no tiling
    tiles = ball_tiles(4096, 1080, 1920)
    assert all(x1 - x0 == 1920 for x0, _, x1, _ in tiles)
    assert tiles[0][0] == 0 and tiles[-1][2] == 4096  # full width covered
    assert all(b[0] < a[2] for a, b in zip(tiles, tiles[1:]))  # overlapping, no gaps
