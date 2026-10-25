from app.services.track_stitch import stitch


def _walk(t0, t1, x0, vx, y=500, step=100, h=80):
    return [(t, [x0 + vx * (t - t0) - 20, y - h, x0 + vx * (t - t0) + 20, y]) for t in range(t0, t1 + 1, step)]


def test_links_fragment_that_continues_the_motion():
    # one player walking right, lost for 600 ms, re-detected where he should be
    tracks = {1: _walk(0, 2000, 100, 0.1), 2: _walk(2600, 4000, 100 + 0.1 * 2600, 0.1)}
    assert stitch(tracks) == {1: 1, 2: 1}


def test_does_not_link_across_the_court_or_other_team():
    tracks = {1: _walk(0, 2000, 100, 0.1), 2: _walk(2600, 4000, 1500, 0.1)}
    assert stitch(tracks)[2] == 2  # 1,300 px away after 0.6 s: someone else
    near = {1: _walk(0, 2000, 100, 0.1), 2: _walk(2600, 4000, 360, 0.1)}
    assert stitch(near, team={1: "A", 2: "B"})[2] == 2


def test_never_joins_chains_visible_at_the_same_time():
    # 1 → 2 is a good link, but 3 overlaps 2 in time, so 3 may not also join 1's chain
    tracks = {
        1: _walk(0, 1000, 100, 0.1),
        2: _walk(1300, 3000, 230, 0.1),
        3: _walk(1400, 3000, 240, 0.1),
    }
    out = stitch(tracks)
    assert out[2] == 1 and out[3] == 3


def test_appearance_gate_blocks_a_different_shirt():
    tracks = {1: _walk(0, 2000, 100, 0.1), 2: _walk(2600, 4000, 360, 0.1)}
    assert stitch(tracks, emb={1: [1.0, 0.0], 2: [0.0, 1.0]})[2] == 2
    assert stitch(tracks, emb={1: [1.0, 0.0], 2: [0.98, 0.05]})[2] == 1
