"""Multi-camera (A–D): sync offsets, cross-camera fuse, cross-camera identity."""

import numpy as np

from app.services.fuse import fuse_event_candidates, group_event_candidates
from app.services.identity import link_clusters_across_cameras
from app.services.media import xcorr_lag
from app.services.vision import EventCandidate


def _evt(et: str, cam: str, ts: int, keys: list[str], conf: float = 0.5) -> EventCandidate:
    return EventCandidate(event_type=et, camera_label=cam, camera_ts_ms=ts, confidence=conf, track_keys=keys)


# --- sync -------------------------------------------------------------------


def test_xcorr_lag_recovers_known_offset_and_sign():
    rng = np.random.default_rng(1)
    ref = rng.random(3000)  # 300 s at 10 Hz
    # Camera B started recording 4.2 s after A: B's t=0 is A's t=4.2 s
    other = ref[42:]
    lag_ms, conf = xcorr_lag(ref, other, 10.0)
    assert lag_ms == 4200  # canonical (A) = B time + 4200
    assert conf > 0.3


def test_xcorr_lag_flat_signal_has_zero_confidence():
    lag_ms, conf = xcorr_lag(np.ones(500), np.ones(500), 10.0)
    assert (lag_ms, conf) == (0, 0.0)


# --- fuse -------------------------------------------------------------------


def test_same_goal_on_two_cameras_fuses_after_offset():
    # B lags A by 3 s: the goal at A 60.0 s is at B 57.1 s
    cands = [_evt("goal", "A", 60_000, ["A-p3"], 0.6), _evt("goal", "B", 57_100, ["B-p9"], 0.8)]
    groups = group_event_candidates(cands, offsets_ms={"A": 0, "B": 3000})
    assert len(groups) == 1 and len(groups[0]) == 2
    assert fuse_event_candidates(cands, offsets_ms={"B": 3000})[0].camera_label == "B"  # best conf


def test_unsynced_cameras_do_not_fuse():
    cands = [_evt("goal", "A", 60_000, ["A-p3"]), _evt("goal", "B", 57_100, ["B-p9"])]
    assert len(group_event_candidates(cands)) == 2


def test_raw_track_ids_are_not_compared_across_cameras():
    # Old fuse keyed on "p12" with the camera stripped — unrelated tracks collided
    cands = [_evt("shot", "A", 10_000, ["A-p12"]), _evt("shot", "B", 10_200, ["B-p40"])]
    assert len(group_event_candidates(cands)) == 1


def test_two_passes_on_one_camera_stay_separate():
    cands = [_evt("pass", "A", 10_000, ["A-p1"]), _evt("pass", "A", 10_800, ["A-p2"])]
    assert len(group_event_candidates(cands)) == 2


def test_different_players_on_two_cameras_stay_separate():
    players = {"A-p1": "kitA#7", "B-p5": "kitB#4"}
    cands = [_evt("pass", "A", 10_000, ["A-p1"]), _evt("pass", "B", 10_300, ["B-p5"])]
    assert len(group_event_candidates(cands, player_of=players.get)) == 2


def test_four_cameras_one_event():
    cands = [_evt("goal", c, 30_000 - off, [f"{c}-p1"]) for c, off in zip("ABCD", (0, 500, 1000, 200))]
    groups = group_event_candidates(cands, offsets_ms={"A": 0, "B": 500, "C": 1000, "D": 200})
    assert len(groups) == 1 and {e.camera_label for e in groups[0]} == set("ABCD")


# --- identity ---------------------------------------------------------------


class _T:
    def __init__(self, key: str, cam: str, start: int, end: int):
        self.track_key, self.camera_label, self.start_ms, self.end_ms = key, cam, start, end
        self.features = {}


def _meta(cam: str, kit: str, num, conf: float, emb, dur: int = 10_000) -> dict:
    return {
        "tracks": [_T(f"{cam}-{kit}{num}", cam, 0, dur)],
        "camera": cam,
        "kit": kit,
        "jersey_number": num,
        "jersey_confidence": conf,
        "reid_emb": emb,
        "color_hist": None,
        "duration": dur,
    }


def test_cross_camera_identity_links_by_kit_and_shirt():
    a = [_meta("A", "A", 7, 0.9, None, 60_000), _meta("A", "B", 7, 0.9, None, 60_000)]
    b = [_meta("B", "B", 7, 0.8, None), _meta("B", "A", 7, 0.8, None)]
    merged = link_clusters_across_cameras({"A": a, "B": b})
    assert len(merged) == 2  # Kit A #7 and Kit B #7 — not four players
    for m in merged:
        assert sorted(m["cameras"]) == ["A", "B"]
        assert {t.track_key[2] for t in m["tracks"]} == {m["kit"]}


def test_cross_camera_identity_uses_reid_without_numbers():
    e1, e2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    a = [_meta("A", "A", None, 0.0, e1, 60_000), _meta("A", "A", None, 0.0, e2, 60_000)]
    b = [_meta("B", "A", None, 0.0, [0.0, 0.98, 0.1]), _meta("B", "A", None, 0.0, [0.97, 0.05, 0.0])]
    merged = link_clusters_across_cameras({"A": a, "B": b})
    assert len(merged) == 2
    by_emb = {tuple(m["reid_emb"]): m for m in merged}
    assert [t.camera_label for t in by_emb[tuple(e1)]["tracks"]] == ["A", "B"]


def test_conflicting_shirts_never_link():
    a = [_meta("A", "A", 7, 0.9, [1.0, 0.0], 60_000)]
    b = [_meta("B", "A", 9, 0.9, [1.0, 0.0])]
    assert len(link_clusters_across_cameras({"A": a, "B": b})) == 2
