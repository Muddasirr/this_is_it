#!/usr/bin/env python3
"""Formal metrics for Roboflow football-players-detection v12 + Bangladesh transfer.

Reports:
  - Ultralytics val/test: Precision, Recall, mAP50, mAP50-95 (overall + per-class)
  - Bangladesh probe: IoU-matched P/R/F1 vs COCO person boxes (domain-shift proxy)
  - Detection density / class counts on Bangladesh frames

Usage:
  python scripts/eval/eval_football_v12.py
  python scripts/eval/eval_football_v12.py --skip-train --weights path/to/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "eval" / "roboflow_football_players_v12"
YAML = DS / "data.yaml"
BD_FRAMES = DS / "bangladesh_probe" / "frames"
OUT = DS / "metrics"
COCO_WEIGHTS = ROOT / "yolov8n.pt"
FUTSAL_WEIGHTS = ROOT / "backend" / "models" / "futsal_yolov8n.pt"


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """xyxy IoU."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy IoU match. Returns TP, FP, FN."""
    if not gts and not preds:
        return 0, 0, 0
    if not gts:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(gts)
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou = _box_iou(p, g)
            if iou >= iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0  # Jaccard / detection accuracy
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "accuracy_jaccard": round(acc, 4),
        "support_gt": tp + fn,
        "support_pred": tp + fp,
    }


def ultralytics_val(weights: Path, split: str, imgsz: int, batch: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # Ultralytics uses data.yaml; override split via 'split' arg
    r = model.val(
        data=str(YAML),
        split=split,
        imgsz=imgsz,
        batch=batch,
        plots=False,
        verbose=False,
        device="cpu",
    )
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    per_class = {}
    for i, name in names.items():
        try:
            per_class[str(name)] = {
                "precision": round(float(r.box.p[i]), 4),
                "recall": round(float(r.box.r[i]), 4),
                "mAP50": round(float(r.box.ap50[i]), 4),
                "mAP50-95": round(float(r.box.ap[i]), 4),
            }
        except Exception as e:
            per_class[str(name)] = {"error": str(e)}
    return {
        "weights": str(weights),
        "split": split,
        "imgsz": imgsz,
        "images": int(getattr(r, "metrics", r).box if False else 0) or None,
        "overall": {
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
        },
        "per_class": per_class,
        "speed_ms": {
            "preprocess": round(float(r.speed.get("preprocess", 0)), 2),
            "inference": round(float(r.speed.get("inference", 0)), 2),
            "postprocess": round(float(r.speed.get("postprocess", 0)), 2),
        },
    }


def predict_boxes(model, frame_bgr, conf: float, person_like: bool) -> tuple[list[np.ndarray], Counter]:
    """Return xyxy boxes + class counts. person_like maps player/gk(/ref) or COCO person."""
    r = model.predict(frame_bgr, conf=conf, verbose=False)[0]
    names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
    boxes: list[np.ndarray] = []
    counts: Counter = Counter()
    if r.boxes is None:
        return boxes, counts
    for b in r.boxes:
        cls = int(b.cls.item())
        lab = str(names[cls])
        counts[lab] += 1
        if person_like:
            if lab in {"person", "player", "goalkeeper", "gk"}:
                boxes.append(b.xyxy[0].cpu().numpy().astype(float))
            # optionally include referee as distractor FP if we only want players —
            # for transfer vs COCO person, include referee too (COCO labels refs as person)
            elif lab == "referee":
                boxes.append(b.xyxy[0].cpu().numpy().astype(float))
        else:
            boxes.append(b.xyxy[0].cpu().numpy().astype(float))
    return boxes, counts


def bangladesh_transfer(
    weights: Path,
    tag: str,
    conf: float = 0.25,
    iou_thr: float = 0.5,
) -> dict:
    from ultralytics import YOLO

    coco = YOLO(str(COCO_WEIGHTS))
    model = YOLO(str(weights))
    frames = sorted(BD_FRAMES.glob("*.jpg"))
    tot_tp = tot_fp = tot_fn = 0
    per_frame = []
    class_totals = Counter()
    for fp in frames:
        img = cv2.imread(str(fp))
        gt_boxes, _ = predict_boxes(coco, img, conf=conf, person_like=True)
        pred_boxes, counts = predict_boxes(model, img, conf=conf, person_like=True)
        class_totals.update(counts)
        tp, fp_n, fn = match_boxes(pred_boxes, gt_boxes, iou_thr=iou_thr)
        tot_tp += tp
        tot_fp += fp_n
        tot_fn += fn
        per_frame.append(
            {
                "frame": fp.name,
                "gt_persons_coco": len(gt_boxes),
                "pred_persons": len(pred_boxes),
                "pred_classes": dict(counts),
                **prf(tp, fp_n, fn),
            }
        )
    overall = prf(tot_tp, tot_fp, tot_fn)
    return {
        "tag": tag,
        "weights": str(weights),
        "reference": "COCO yolov8n person boxes (pseudo-GT; no human labels on Bangladesh)",
        "conf": conf,
        "iou_thr": iou_thr,
        "n_frames": len(frames),
        "overall": overall,
        "pred_class_totals": dict(class_totals),
        "per_frame": per_frame,
    }


def train_short(epochs: int, imgsz: int, batch: int) -> Path:
    from ultralytics import YOLO

    out = DS / "runs" / "yolov8n_football_v12_metrics"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(COCO_WEIGHTS))
    model.train(
        data=str(YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        project=str(DS / "runs"),
        name="yolov8n_football_v12_metrics",
        exist_ok=True,
        patience=max(5, epochs // 2),
        workers=2,
        verbose=True,
        plots=True,
    )
    best = DS / "runs" / "yolov8n_football_v12_metrics" / "weights" / "best.pt"
    # Ultralytics sometimes nests under runs/detect/
    if not best.exists():
        cands = list((ROOT / "runs").rglob("yolov8n_football_v12_metrics/weights/best.pt"))
        cands += list(DS.rglob("yolov8n_football_v12_metrics/weights/best.pt"))
        if not cands:
            raise FileNotFoundError("trained best.pt not found")
        best = cands[0]
    dest = DS / "yolov8n_football_v12_best.pt"
    dest.write_bytes(best.read_bytes())
    return dest


def write_html(report: dict, path: Path) -> None:
    rows = []
    for row in report.get("dataset_val", []):
        o = row["overall"]
        rows.append(
            "<tr>"
            f"<td>{row['label']}</td><td>{row['split']}</td><td>{row['imgsz']}</td>"
            f"<td>{o['precision']:.3f}</td><td>{o['recall']:.3f}</td>"
            f"<td>{o['mAP50']:.3f}</td><td>{o['mAP50-95']:.3f}</td>"
            "</tr>"
        )
    bd_rows = []
    for row in report.get("bangladesh_transfer", []):
        o = row["overall"]
        bd_rows.append(
            "<tr>"
            f"<td>{row['tag']}</td>"
            f"<td>{o['precision']:.3f}</td><td>{o['recall']:.3f}</td>"
            f"<td>{o['f1']:.3f}</td><td>{o['accuracy_jaccard']:.3f}</td>"
            f"<td>{o['tp']}/{o['fp']}/{o['fn']}</td>"
            f"<td>{row.get('pred_class_totals', {})}</td>"
            "</tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Football v12 metrics</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#eee;margin:24px;line-height:1.45}}
table{{border-collapse:collapse;margin:12px 0 28px;width:100%;max-width:1100px}}
th,td{{border:1px solid #333;padding:8px 10px;text-align:left;font-size:13px}}
th{{background:#1a1a1a}}
.meta{{opacity:.85;max-width:900px}}
code{{background:#1a1a1a;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>Football-players v12 — formal metrics</h1>
<p class="meta">{report.get("notes","")}</p>
<h2>Dataset validation (labeled GT)</h2>
<table>
<tr><th>Model</th><th>Split</th><th>imgsz</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr>
{''.join(rows)}
</table>
<h2>Bangladesh transfer (vs COCO person pseudo-GT, IoU≥0.5)</h2>
<p class="meta">No human boxes on Bangladesh — COCO person detections are the reference.
Precision/recall here measure agreement with COCO, not absolute truth.</p>
<table>
<tr><th>Model</th><th>P</th><th>R</th><th>F1</th><th>Acc (Jaccard)</th><th>TP/FP/FN</th><th>Pred classes</th></tr>
{''.join(bd_rows)}
</table>
<p class="meta">Full JSON: <code>{report.get("json_path","")}</code></p>
</body></html>"""
    path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--train-imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", type=str, default="", help="Use existing trained weights")
    ap.add_argument("--skip-1280", action="store_true", help="Skip slow imgsz=1280 vals")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {
        "dataset": {
            "url": "https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/12",
            "path": str(DS),
            "classes": ["ball", "goalkeeper", "player", "referee"],
            "splits": {"train": 298, "valid": 49, "test": 25},
        },
        "notes": (
            "Labeled metrics from Ultralytics val on Roboflow v12. "
            "Bangladesh uses IoU match vs COCO person boxes (pseudo-GT). "
            "CPU-only training/eval."
        ),
        "dataset_val": [],
        "bangladesh_transfer": [],
    }

    trained = Path(args.weights) if args.weights else None
    if trained and not trained.exists():
        raise SystemExit(f"weights not found: {trained}")
    if not args.skip_train and trained is None:
        print(f"TRAIN epochs={args.epochs} imgsz={args.train_imgsz} batch={args.batch}")
        trained = train_short(args.epochs, args.train_imgsz, args.batch)
        report["train"] = {
            "epochs": args.epochs,
            "imgsz": args.train_imgsz,
            "batch": args.batch,
            "best_weights": str(trained),
        }
        print("TRAINED", trained)
    elif trained is None:
        # prefer prior best, else ep1
        for cand in [
            DS / "yolov8n_football_v12_best.pt",
            DS / "yolov8n_football_v12_ep1.pt",
        ]:
            if cand.exists():
                trained = cand
                break
        if trained is None:
            raise SystemExit("No trained weights; run without --skip-train")
        report["train"] = {"skipped": True, "best_weights": str(trained)}

    models = [
        ("futsal_yolov8n", FUTSAL_WEIGHTS),
        ("local_v12_trained", trained),
    ]
    imgszs = [640] if args.skip_1280 else [640, 1280]
    batches = {640: 8, 1280: 2}

    for label, w in models:
        if not w.exists():
            print("MISSING", w)
            continue
        for split in ("val", "test"):
            for imgsz in imgszs:
                print(f"VAL {label} split={split} imgsz={imgsz}")
                try:
                    metrics = ultralytics_val(w, split=split, imgsz=imgsz, batch=batches[imgsz])
                    metrics["label"] = label
                    report["dataset_val"].append(metrics)
                    print(" ", metrics["overall"])
                except Exception as e:
                    report["dataset_val"].append(
                        {"label": label, "split": split, "imgsz": imgsz, "error": str(e)}
                    )
                    print("  ERR", e)

    if BD_FRAMES.exists():
        for label, w in models:
            if not w.exists():
                continue
            print(f"BANGLADESH transfer {label}")
            bd = bangladesh_transfer(w, tag=label)
            report["bangladesh_transfer"].append(bd)
            print(" ", bd["overall"])
        # also report COCO self-match (sanity ≈ P=R=1)
        print("BANGLADESH transfer coco_self")
        bd_coco = bangladesh_transfer(COCO_WEIGHTS, tag="coco_yolov8n_self")
        report["bangladesh_transfer"].append(bd_coco)
        print(" ", bd_coco["overall"])

    report["elapsed_sec"] = round(time.time() - t0, 1)
    json_path = OUT / "EVAL_METRICS.json"
    report["json_path"] = str(json_path)
    json_path.write_text(json.dumps(report, indent=2))
    html_path = OUT / "index.html"
    write_html(report, html_path)
    print("WROTE", json_path)
    print("WROTE", html_path)
    print(json.dumps({"elapsed_sec": report["elapsed_sec"], "n_val_rows": len(report["dataset_val"])}, indent=2))


if __name__ == "__main__":
    main()
