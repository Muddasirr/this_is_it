"""Train a dedicated ball detector (roadmap W2) — runs on any GPU box (Modal, Kaggle, local).

Every YOLO-format dataset under --data (a dir holding data.yaml) contributes its
ball class(es): any class whose name contains "ball" ("ball", "Location-of-ball",
"ball_in"/"ball_out"); every other class is dropped, so images without a ball are
negatives. Checkpoints go to --out/runs/ball/weights/{last,best}.pt every epoch;
re-running the same command resumes from last.pt (copy the --out dir to move a run
to another machine).

Eval: per source dataset's held-out test split, trained model vs COCO yolov8m
sports_ball, same centre-in-box metric at the pipeline's 0.12 threshold.

  python -m app.scripts.train_ball --data /data/ball --out /data/ball_run --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _ball_ids(names) -> set[int]:
    names = names if isinstance(names, dict) else dict(enumerate(names))
    return {int(k) for k, v in names.items() if "ball" in str(v).lower() and "player" not in str(v).lower()}


def build_dataset(data_root: Path, ds_dir: Path) -> dict[str, dict[str, int]]:
    """Symlinked images + ball-only labels per source; returns {source: {split: n}}."""
    sources = {}
    for y in sorted(data_root.rglob("data.yaml")):
        ids = _ball_ids(yaml.safe_load(y.read_text())["names"])
        if not ids:
            continue
        tag = "_".join(y.parent.relative_to(data_root).parts) or y.parent.name
        counts = {}
        for split in ("train", "valid", "test"):
            img_dir = y.parent / split / "images"
            if not img_dir.is_dir():
                continue
            dst_i, dst_l = ds_dir / tag / split / "images", ds_dir / tag / split / "labels"
            dst_i.mkdir(parents=True, exist_ok=True)
            dst_l.mkdir(parents=True, exist_ok=True)
            n = 0
            for img in img_dir.iterdir():
                lab = y.parent / split / "labels" / f"{img.stem}.txt"
                rows = []
                if lab.exists():
                    for line in lab.read_text().splitlines():
                        p = line.split()
                        if p and int(p[0]) in ids:
                            rows.append(" ".join(["0", *p[1:5]]))
                link = dst_i / img.name
                if not link.exists():
                    link.symlink_to(img.resolve())
                (dst_l / f"{img.stem}.txt").write_text("\n".join(rows))
                n += 1
            counts[split] = n
        sources[tag] = counts
    return sources


def centre_match(model, test: Path, labels: Path, classes=None, device=0) -> dict:
    """Ball found when a predicted centre lies inside the labelled box (conf 0.12)."""
    tp = fp = fn = 0
    for img in sorted(test.iterdir()):
        lab = labels / f"{img.stem}.txt"
        gt = [list(map(float, l.split()[1:5])) for l in lab.read_text().splitlines() if l.strip()]
        r = model.predict(str(img), imgsz=1280, conf=0.12, classes=classes, device=device, verbose=False)[0]
        pred = r.boxes.xywhn.tolist() if r.boxes is not None else []
        used = set()
        for g in gt:
            hit = next((i for i, p in enumerate(pred) if i not in used
                        and abs(p[0] - g[0]) <= max(g[2], 0.01) and abs(p[1] - g[1]) <= max(g[3], 0.01)), None)
            if hit is None:
                fn += 1
            else:
                used.add(hit)
                tp += 1
        fp += len(pred) - len(used)
    return {"precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
            "balls": tp + fn, "false_balls": fp}


def main(argv: list[str] | None = None, on_epoch=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    ds = args.out / "ds"
    sources = build_dataset(args.data, ds)
    print("sources", json.dumps(sources))
    if not sources:
        raise SystemExit(f"no ball datasets under {args.data}")
    lists = lambda split: [str(ds / t / split / "images") for t in sources if (ds / t / split / "images").is_dir()]  # noqa: E731
    data_yaml = ds / "ball.yaml"
    data_yaml.write_text(yaml.safe_dump({"train": lists("train"), "val": lists("valid") or lists("test"),
                                         "names": {0: "ball"}}))
    run_dir = args.out / "runs" / "ball"
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        print("resuming from", last)
        model = YOLO(str(last))
        if on_epoch:
            model.add_callback("on_fit_epoch_end", lambda _t: on_epoch())
        model.train(resume=True)
    else:
        model = YOLO(args.base)
        if on_epoch:
            model.add_callback("on_fit_epoch_end", lambda _t: on_epoch())
        model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=12,
                    device=args.device, project=str(run_dir.parent), name="ball", exist_ok=True, mosaic=1.0,
                    scale=0.5, close_mosaic=5, save_period=1, verbose=False, plots=False)
    best = run_dir / "weights" / "best.pt"
    trained, coco = YOLO(str(best)), YOLO(args.base)
    evals = {}
    for tag in sources:
        test = ds / tag / "test" / "images"
        if test.is_dir():
            evals[tag] = {"trained": centre_match(trained, test, ds / tag / "test" / "labels", device=args.device),
                          "coco_sports_ball": centre_match(coco, test, ds / tag / "test" / "labels", [32], args.device)}
    result = {"sources": sources, "weights": str(best), "test": evals}
    (args.out / "ball_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(evals, indent=1))
    return result


if __name__ == "__main__":
    main()
