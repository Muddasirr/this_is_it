"""Kernel: broadcast scoreboard states -> point labels for PadelTracker100.

PT100 videos are the rallies of each final joined together, so the on-screen score changes once per point.
For every stable scoreboard state: first / last frame, the crop (saved for checking by eye), EasyOCR text.
Output: /kaggle/working/out/<match>/states.json, crops/<frame>.png, sheet_<k>.jpg
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working/out")
BOX = (40, 60, 760, 210)  # x0, y0, x1, y1 in 1920x1080: WPT 2022 score bug (top-left, grows with sets)
EVERY = 5  # frames
DIFF = 2.0  # mean abs grey change that triggers a new OCR read


def parse(tokens: list) -> dict | None:
    """EasyOCR tokens -> {"rows": [[names, numbers...], [...]]}; rows split at the box's vertical middle."""
    if not tokens:
        return None
    ys = sorted(t[2][1] for t in tokens)
    mid = (ys[0] + ys[-1]) / 2
    rows = [[], []]
    for txt, p, (x, y) in sorted(tokens, key=lambda t: t[2][0]):
        rows[0 if y < mid else 1].append(txt.strip().upper())
    return {"rows": rows}


def serve_dot(img) -> list[int]:
    """Yellow pixels per scoreboard row (the serving pair has a yellow ball next to its names)."""
    import numpy as np

    hsv = __import__("cv2").cvtColor(img, __import__("cv2").COLOR_BGR2HSV)
    yellow = (hsv[..., 0] > 22) & (hsv[..., 0] < 38) & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)
    h = img.shape[0]
    return [int(yellow[: h // 2, :300].sum()), int(yellow[h // 2:, :300].sum())]


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=False)
    import cv2
    import easyocr
    import numpy as np

    reader = easyocr.Reader(["en"], gpu=True)
    for v in sorted(glob.glob("/kaggle/input/**/videos/*.mp4", recursive=True)):
        name = Path(v).stem
        d = OUT / name
        (d / "crops").mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(v)
        x0, y0, x1, y1 = BOX
        reads, last_g, last_txt, f = [], None, None, 0
        while cap.grab():
            if f % EVERY == 0:
                _, img = cap.retrieve()
                c = img[y0:y1, x0:x1]
                g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if last_g is None or np.abs(g - last_g).mean() > DIFF:
                    big = cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    toks = [[t, float(p), [int(min(q[0] for q in b)), int(min(q[1] for q in b))]]
                            for b, t, p in reader.readtext(big, detail=1) if p > 0.3]
                    last_txt = {"p": parse(toks), "dot": serve_dot(c)}
                    last_g = g
                    if last_txt["p"]:
                        cv2.imwrite(str(d / "crops" / f"{f}.jpg"), c)
                reads.append({"f": f, **(last_txt or {})})
            f += 1
        # states: runs of the same OCR rows lasting >= 0.5 s
        states, cur = [], None
        for r in reads:
            key = json.dumps(r.get("p"))
            if cur and cur["key"] == key:
                cur["end"] = r["f"]
                continue
            if cur and cur["p"] and cur["end"] - cur["start"] >= 15:
                states.append(cur)
            cur = {"key": key, "start": r["f"], "end": r["f"], "p": r.get("p"), "dot": r.get("dot")}
        if cur and cur["p"] and cur["end"] - cur["start"] >= 15:
            states.append(cur)
        for s_ in states:
            s_.pop("key")
        (d / "states.json").write_text(json.dumps(states, indent=1))
        tiles = []
        for s_ in states:
            fp = max((int(p.stem) for p in (d / "crops").glob("*.jpg") if int(p.stem) <= s_["start"]), default=None)
            if fp is None:
                continue
            t = cv2.imread(str(d / "crops" / f"{fp}.jpg"))
            t = cv2.resize(t, (360, 75))
            cv2.rectangle(t, (0, 55), (110, 75), (0, 0, 0), -1)
            cv2.putText(t, str(s_["start"]), (3, 71), 0, 0.5, (0, 255, 255), 1)
            tiles.append(t)
        while tiles and len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rws = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        for k in range(0, len(rws), 14):
            cv2.imwrite(str(d / f"sheet_{k // 14}.jpg"), np.vstack(rws[k:k + 14]))
        print(name, len(reads), "samples", len(states), "states", flush=True)


if __name__ == "__main__":
    main()
