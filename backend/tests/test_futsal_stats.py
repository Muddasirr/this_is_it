from app.services.match_stats import _aggregate_teams, futsal_event_stats


def ev(type_, ts, parts=(), **prov):
    return {"type": type_, "ts_ms": ts, "provenance": prov, "participants": list(parts)}


TEAM = {"a1": "A", "a2": "A", "b1": "B", "gk_b": "B"}


def test_on_target_interceptions_cards_and_fouls():
    events = [
        ev("shot", 1000, [("a1", "shooter")]),
        ev("save", 2500, [("gk_b", "goalkeeper")]),  # → the shot was on target
        ev("shot", 10_000, [("a1", "shooter")]),  # off target: nothing follows
        ev("shot", 20_000, [("a2", "shooter")], on_target=True),  # staff call
        ev("goal", 30_000, [("a2", "scorer")]),  # no shot before it → its own shot on target
        ev("interception", 40_000, [("b1", "won_by"), ("a1", "lost_by")], from_pass=True),
        ev("interception", 41_000, [("a1", "won_by"), ("b1", "lost_by")]),  # staff-tagged tackle
        ev("foul", 60_000, [("b1", "primary")]),
        ev("foul", 25 * 60_000, [], team_id="B"),  # second half, player unknown
        ev("yellow_card", 60_500, [("b1", "primary")]),
        ev("corner", 70_000, [], team_id="A"),
        ev("own_goal", 80_000, [("b1", "scorer")]),
    ]
    players, teams = futsal_event_stats(events, TEAM)
    assert players["a1"]["shot_on_target"] == 1
    assert players["a2"]["shot_on_target"] == 2
    assert players["gk_b"]["save"] == 1
    assert players["b1"]["interception"] == 1 and players["a1"]["interception"] == 1
    assert players["a1"]["ball_lost"] == 1 and players["a1"]["pass_failed"] == 1
    assert players["b1"]["ball_lost"] == 1 and players["b1"]["pass_failed"] == 0
    assert players["b1"]["foul"] == 1 and players["b1"]["yellow_card"] == 1 and players["b1"]["own_goal"] == 1
    assert teams["B"]["foul"] == 2 and teams["B"]["fouls_by_half"] == [1, 1]
    assert teams["A"]["corner"] == 1 and teams["A"]["own_goal_for"] == 1


def test_team_score_and_pass_accuracy():
    rows = [
        {"player_id": "a1", "team_id": "A", "goal": 1, "pass": 6, "pass_failed": 2, "shot": 2, "foul": 0},
        {"player_id": "b1", "team_id": "B", "goal": 1, "pass": 3, "pass_failed": 0, "shot": 1, "foul": 1},
    ]
    teams = {t["team_id"]: t for t in _aggregate_teams(rows, {"A": {"own_goal_for": 1, "fouls_by_half": [0, 0]}})}
    assert teams["A"]["score"] == 2 and teams["B"]["score"] == 1
    assert teams["A"]["pass_accuracy_pct"] == 75.0 and teams["B"]["pass_accuracy_pct"] == 100.0
    assert teams["A"]["conversion_pct"] == 50.0


def test_pass_network_ranks_pairs():
    from app.services.match_stats import pass_network

    passes = [ev("pass", i, [("a1", "passer"), ("a2", "receiver")]) for i in range(3)]
    passes.append(ev("pass", 9, [("a2", "passer"), ("a1", "receiver")]))
    passes.append(ev("shot", 10, [("a1", "shooter")]))
    assert pass_network(passes) == [{"from": "a1", "to": "a2", "passes": 3}, {"from": "a2", "to": "a1", "passes": 1}]
