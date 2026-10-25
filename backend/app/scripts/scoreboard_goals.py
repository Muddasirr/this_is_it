"""Goal timestamps from a broadcast scoreboard overlay (futsal ground truth without labelling).

A broadcast shows the score on screen; every lasting change of the score is a
goal. OCR the score box every `--step` seconds, keep readings that parse as
"H-A", require a new score to hold for `--hold` consecutive readings (replays,
wipes and graphics flicker drop the overlay or garble it), and emit one goal per
increment at the time the new score first appeared. The actual shot happens a
few seconds *before* the scoreboard updates — scorers should match predictions
in [goal_ts - lead, goal_ts].

  python -m app.scripts.scoreboard_goals --video data/futsal_gt/videos/portugal_spain.mp4 \\
      --score-box 355,50,425,80 --clock-box 205,50,290,80 --out portugal_goals.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCORE_RE = re.compile(r"(\d{1,2})\s*[-–:]\s*(\d{1,2})")
CLOCK_RE = re.compile(r"(\d{1,2})[:.](\d{2})")


def _ocr(img, whitelist: str) -> str:
    import cv2
    import pytesseract

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:  # make text dark on light
        bw = 255 - bw
    bw = 255 - bw
    return pytesseract.image_to_string(bw, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}").strip()


def read_series(video: Path, score_box, clock_box, step_s: float):
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(1, int(round(step_s * fps)))
    out = []
    for f in range(0, n, every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = score_box
        m = SCORE_RE.search(_ocr(frame[y1:y2, x1:x2], "0123456789-"))
        score = (int(m.group(1)), int(m.group(2))) if m else None
        clock = None
        if clock_box:
            cx1, cy1, cx2, cy2 = clock_box
            c = CLOCK_RE.search(_ocr(frame[cy1:cy2, cx1:cx2], "0123456789:"))
            clock = f"{int(c.group(1)):02d}:{c.group(2)}" if c else None
        out.append({"ts_ms": int(f * 1000 / fps), "score": score, "clock": clock})
    cap.release()
    return out


def goals_from_series(series: list[dict], hold: int = 3) -> list[dict]:
    """One goal per lasting score increment (each +1 on either side)."""
    goals = []
    current = None
    run_score, run_start, run_len = None, None, 0
    for r in series:
        s = r["score"]
        if s is None:
            continue
        if s == run_score:
            run_len += 1
        else:
            run_score, run_start, run_len = s, r, 1
        if run_len == hold and run_score != current:
            if current is None:
                current = run_score  # first stable reading = starting score
                continue
            dh, da = run_score[0] - current[0], run_score[1] - current[1]
            if dh >= 0 and da >= 0 and 0 < dh + da <= 2:  # ignore OCR jumps / resets
                for side, d in (("home", dh), ("away", da)):
                    for _ in range(d):
                        goals.append({"ts_ms": run_start["ts_ms"], "type": "goal", "side": side,
                                      "score_after": list(run_score), "clock": run_start.get("clock")})
                current = run_score
    return goals


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--score-box", required=True, help="x1,y1,x2,y2 of the 'H-A' score")
    ap.add_argument("--clock-box", default="", help="x1,y1,x2,y2 of the match clock (optional)")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    box = [int(v) for v in args.score_box.split(",")]
    cbox = [int(v) for v in args.clock_box.split(",")] if args.clock_box else None
    series = read_series(args.video, box, cbox, args.step)
    goals = goals_from_series(series, args.hold)
    readable = sum(1 for r in series if r["score"] is not None)
    final = next((r["score"] for r in reversed(series) if r["score"] is not None), None)
    result = {"video": str(args.video), "source": "scoreboard OCR", "tolerance_ms": 15000,
              "readable_frac": round(readable / max(1, len(series)), 3), "final_score_read": final,
              "events": goals}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "events"}), len(goals), "goals")
    return result


if __name__ == "__main__":
    main()
