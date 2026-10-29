"""Broadcast scoreboard states (padel-scoreboard OCR output) -> one-point transitions: who won, who served."""
from __future__ import annotations

import re

PTS = {"0": 0, "O": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


def parse_row(tokens: list[str]) -> tuple[list[int], int] | None:
    """[name tokens..., set games..., points] -> (games per set, point index)."""
    nums = [t.replace("O", "0") if re.fullmatch(r"[0-9O]+", t) else t for t in tokens]
    nums = [t for t in nums if re.fullmatch(r"[0-9]+|AD|A", t)]
    if not nums or nums[-1] not in PTS:
        return None
    games = []
    for t in nums[:-1]:
        if t.isdigit() and len(t) == 1:
            games.append(int(t))
    return games, PTS[nums[-1]]


def transitions(states: list[dict]) -> list[dict]:
    """Consecutive readable states that differ by exactly one point -> {start, end (frames), winner_row, server_row}."""
    rd = []
    for s in states:
        rows = (s.get("p") or {}).get("rows") or []
        if len(rows) != 2:
            continue
        a, b = parse_row(rows[0]), parse_row(rows[1])
        if a is None or b is None or len(a[0]) != len(b[0]):  # no games column yet at 0-0 in the first game
            continue
        dot = s.get("dot") or [0, 0]
        srv = 0 if dot[0] > max(30, 2 * dot[1]) else 1 if dot[1] > max(30, 2 * dot[0]) else None
        rd.append({"start": s["start"], "end": s["end"], "games": (a[0], b[0]), "pts": (a[1], b[1]), "srv": srv})
    out = []
    for x, y in zip(rd, rd[1:]):
        if x["games"] == y["games"]:
            d = (y["pts"][0] - x["pts"][0], y["pts"][1] - x["pts"][1])
            win = 0 if d == (1, 0) else 1 if d == (0, 1) else None
            # deuce back from advantage: AD-40 -> 40-40 means the other row won
            if x["pts"] == (4, 3) and y["pts"] == (3, 3):
                win = 1
            if x["pts"] == (3, 4) and y["pts"] == (3, 3):
                win = 0
        else:
            ga, gb = x["games"], y["games"]
            dg = [sum(gb[0]) - sum(ga[0]), sum(gb[1]) - sum(ga[1])]
            win = 0 if dg == [1, 0] and y["pts"] == (0, 0) and x["pts"][0] >= 3 else \
                1 if dg == [0, 1] and y["pts"] == (0, 0) and x["pts"][1] >= 3 else None
        if win is not None:
            out.append({"from": x["start"], "to": y["start"], "winner_row": win, "server_row": x["srv"]})
    return out
