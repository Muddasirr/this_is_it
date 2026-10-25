"""Jersey-number accuracy vs SoccerNet Game State Reconstruction labels.

Ground truth: Hugging Face SoccerNet/SN-GSR-2024 (public) — every player box in
1920×1080 broadcast clips carries `jersey` (when visible), `team` and `role`.
Only the files needed are read from the remote zip (HTTP range requests).

For each labelled player track: crop the true boxes on sampled frames, run the
reader, majority-vote per track, compare with the true number. Reports
track-level accuracy (SoccerNet jersey-challenge style) and how often the reader
answers (coverage) / is right when it answers (precision).

  python -m app.scripts.bench_jersey --readers tesseract,trocr --sequences 6 --out bench_jersey.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GSR_URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2024/resolve/main/{split}.zip"


def _reader(name: str):
    if name == "tesseract":
        import os

        os.environ["VISION_JERSEY_OCR"] = "1"
        from app.services.jersey_ocr import read_jersey_number

        return read_jersey_number
    if name == "classifier":
        from app.services.jersey_cls import read_jersey_number_cls

        return read_jersey_number_cls
    if name == "parseq":
        from app.services.jersey_ocr import read_jersey_number_parseq

        return read_jersey_number_parseq
    if name == "trocr":
        from app.services.jersey_ocr import read_jersey_number_trocr

        return read_jersey_number_trocr
    raise SystemExit(f"unknown reader {name}")


def load_tracks(z, seq: str, frame_step: int, max_per_track: int):
    """{track_id: (true_jersey, [(file_name, xyxy), ...])} for labelled player tracks."""
    labels = json.loads(z.read(f"{seq}/Labels-GameState.json"))
    files = {img["image_id"]: img["file_name"] for img in labels["images"]}
    by_track: dict[int, dict] = defaultdict(lambda: {"jerseys": Counter(), "boxes": []})
    for a in labels["annotations"]:
        attr = a.get("attributes") or {}
        if attr.get("role") not in {"player", "goalkeeper"} or not attr.get("jersey"):
            continue
        frame_no = int(str(a["image_id"])[-6:])
        b = a["bbox_image"]
        t = by_track[a["track_id"]]
        t["jerseys"][str(attr["jersey"])] += 1
        if frame_no % frame_step == 0:
            t["boxes"].append((files[a["image_id"]], [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]))
    out = {}
    for tid, t in by_track.items():
        boxes = sorted(t["boxes"], key=lambda fb: (fb[1][3] - fb[1][1]), reverse=True)[:max_per_track]
        if boxes:
            out[tid] = (t["jerseys"].most_common(1)[0][0], boxes)
    return out


def accept(reads: dict[str, int], min_reads: int, min_share: float) -> str | None:
    """Consensus rule: the top number needs ≥ min_reads agreeing reads and ≥ min_share of all reads."""
    if not reads:
        return None
    best, n = max(reads.items(), key=lambda kv: kv[1])
    return best if n >= min_reads and n / sum(reads.values()) >= min_share else None


def score_rules(per_track: list[dict], reader: str) -> list[dict]:
    rows = []
    for min_reads in (1, 2, 3, 4):
        for share in (0.0, 0.5, 0.6, 0.75):
            got = [(accept(t["reads"][reader], min_reads, share), t["truth"]) for t in per_track]
            ans = [(g, tr) for g, tr in got if g is not None]
            ok = sum(1 for g, tr in ans if g == tr)
            rows.append({"min_reads": min_reads, "min_share": share, "answered": len(ans), "correct": ok,
                         "precision": round(ok / len(ans), 3) if ans else None,
                         "accuracy": round(ok / len(per_track), 3) if per_track else None})
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--sequences", type=int, default=6)
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--max-per-track", type=int, default=10, help="largest boxes per track (most legible)")
    ap.add_argument("--readers", default="tesseract")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import cv2
    from remotezip import RemoteZip

    readers = {n: _reader(n) for n in args.readers.split(",") if n}
    stats = {n: {"tracks": 0, "answered": 0, "correct": 0, "boxes": 0, "box_answers": 0, "box_correct": 0}
             for n in readers}
    per_track: list[dict] = []  # every track's votes per reader → acceptance rules scored offline
    t0 = time.time()
    with RemoteZip(GSR_URL.format(split=args.split)) as z:
        seqs = sorted({n.split("/")[0] for n in z.namelist() if n.endswith("Labels-GameState.json")})
        step = max(1, len(seqs) // args.sequences)
        for seq in seqs[::step][: args.sequences]:
            tracks = load_tracks(z, seq, args.frame_step, args.max_per_track)
            frames: dict[str, np.ndarray] = {}
            for truth, boxes in tracks.values():
                votes = {n: Counter() for n in readers}
                reads = {n: Counter() for n in readers}  # unweighted read counts
                for fname, xyxy in boxes:
                    if fname not in frames:
                        frames[fname] = cv2.imdecode(np.frombuffer(z.read(f"{seq}/img1/{fname}"), np.uint8),
                                                     cv2.IMREAD_COLOR)
                    for n, read in readers.items():
                        hit = read(frames[fname], xyxy)
                        s = stats[n]
                        s["boxes"] += 1
                        if hit:
                            s["box_answers"] += 1
                            s["box_correct"] += int(str(hit["number"]) == truth)
                            votes[n][str(hit["number"])] += float(hit.get("confidence") or 1.0)
                            reads[n][str(hit["number"])] += 1
                per_track.append({"seq": seq, "truth": truth, "crops": len(boxes),
                                  "reads": {n: dict(v) for n, v in reads.items()}})
                for n in readers:
                    s = stats[n]
                    s["tracks"] += 1
                    if votes[n]:
                        s["answered"] += 1
                        s["correct"] += int(votes[n].most_common(1)[0][0] == truth)
            print(seq, {n: (s["correct"], s["tracks"]) for n, s in stats.items()}, f"{time.time() - t0:.0f}s", flush=True)

    result = {"split": args.split, "sequences": args.sequences, "seconds": round(time.time() - t0), "readers": {}}
    for n, s in stats.items():
        result["readers"][n] = {
            **s,
            "track_accuracy": round(s["correct"] / s["tracks"], 3) if s["tracks"] else None,
            "track_precision": round(s["correct"] / s["answered"], 3) if s["answered"] else None,
            "track_coverage": round(s["answered"] / s["tracks"], 3) if s["tracks"] else None,
            "box_precision": round(s["box_correct"] / s["box_answers"], 3) if s["box_answers"] else None,
        }
    result["rules"] = {n: score_rules(per_track, n) for n in readers}
    result["tracks_detail"] = per_track
    print(json.dumps({k: v for k, v in result.items() if k != "tracks_detail"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
