"""TrackIndex must answer exactly like the naive per-track scan it replaces."""

import random

from app.services.possession import PERSON_TIME_WINDOW_MS, TrackIndex, person_at


def test_track_index_matches_naive_scan():
    rng = random.Random(3)
    tracks = {}
    for k in range(60):
        ts, series = rng.randint(0, 50_000), []
        for i in range(rng.randint(1, 80)):  # time-ordered like real tracks, with repeats
            ts += rng.choice([0, 40, 80, 200])
            series.append({"ts_ms": ts, "x1": i})
        tracks[f"p{k}"] = series
    idx = TrackIndex(tracks)
    for ts in [rng.randint(-3000, 70_000) for _ in range(500)]:
        naive = [(k, person_at(s, ts)) for k, s in tracks.items()]
        naive = [(k, p) for k, p in naive if p is not None]
        fast = [(k, idx.at(k, ts)) for k in idx.active(ts)]
        fast = [(k, p) for k, p in fast if p is not None]
        assert fast == naive, ts


def test_tie_goes_to_earlier_sample_like_min():
    idx = TrackIndex({"a": [{"ts_ms": 0, "i": 0}, {"ts_ms": 200, "i": 1}]})
    assert idx.at("a", 100)["i"] == 0
    assert idx.at("a", 200 + PERSON_TIME_WINDOW_MS + 1) is None
