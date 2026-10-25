from app.scripts.bench_teams import score


def test_best_mapping_and_bystanders():
    players = [{"team": "Kit B", "track_keys": ["a", "b"]}, {"team": "Kit A", "track_keys": ["c", "o1"]}]
    labels = {"a": "R", "b": "R", "c": "W", "d": "W", "o1": "O", "o2": "O", "g": "G"}
    r = score(players, labels)
    assert r["team_accuracy"] == 1.0  # R ↔ Kit B, W ↔ Kit A
    assert r["team_assigned"] == "3/4" and r["bystanders_kept"] == "1/2"
