"""Jersey consensus rule (what the product accepts as a shirt number)."""

from app.scripts.bench_jersey import accept, score_rules


def test_consensus_rule():
    assert accept({"7": 3, "1": 1}, min_reads=3, min_share=0.6) == "7"
    assert accept({"7": 2, "1": 2}, min_reads=2, min_share=0.6) is None  # no majority
    assert accept({"7": 1}, min_reads=2, min_share=0.0) is None  # single read
    assert accept({}, min_reads=1, min_share=0.0) is None


def test_score_rules_counts_correct_answers():
    tracks = [{"truth": "7", "reads": {"r": {"7": 3}}}, {"truth": "9", "reads": {"r": {"1": 1}}}]
    strict = next(r for r in score_rules(tracks, "r") if r["min_reads"] == 2 and r["min_share"] == 0.5)
    assert strict["answered"] == 1 and strict["correct"] == 1 and strict["precision"] == 1.0
