"""MOT scorer: detection recall/precision and ID switches on synthetic tracks."""

from app.scripts.bench_mot import score


def _gt():
    # two players over 4 frames, GT format: frame -> [(id, [x, y, w, h])]
    return {f: [(1, [100 + 5 * f, 100, 20, 50]), (2, [400, 200, 20, 50])] for f in range(4)}


def test_perfect_tracking():
    log = [(f, tid, x, y, x + w, y + h, 0.9) for f, boxes in _gt().items() for tid, (x, y, w, h) in boxes]
    m = score(_gt(), log, list(range(4)))
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["num_switches"] == 0 and m["idf1"] == 1.0


def test_missed_player_and_id_switch():
    log = []
    for f, boxes in _gt().items():
        (_, (x, y, w, h)), _ = boxes
        tid = 7 if f < 2 else 8  # tracker switches ID halfway; player 2 never detected
        log.append((f, tid, x, y, x + w, y + h, 0.9))
    m = score(_gt(), log, list(range(4)))
    assert m["recall"] == 0.5 and m["precision"] == 1.0 and m["num_switches"] == 1


def test_team_accuracy_uses_best_letter_mapping():
    from app.scripts.bench_mot import team_accuracy

    teams = {"white": [1], "blue": [2], "goalkeeper": []}
    log = [(f, tid, x, y, x + w, y + h, 0.9) for f, boxes in _gt().items() for tid, (x, y, w, h) in boxes]
    # Predicted kits are flipped (white→B, blue→A) — still a perfect split
    assert team_accuracy(_gt(), log, list(range(4)), {1: "B", 2: "A"}, teams)["team_accuracy"] == 1.0
    # Both players on one kit → only half the detections can be right
    assert team_accuracy(_gt(), log, list(range(4)), {1: "A", 2: "A"}, teams)["team_accuracy"] == 0.5
