"""Metrics against PadelTracker100 ground truth."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def ball_pr(pred: np.ndarray, gt: np.ndarray, tol_px: float = 10.0, frames: np.ndarray | None = None) -> dict:
    """Precision / recall of a ball track. pred, gt: (n, 2) px with nan = none."""
    if frames is not None:
        pred, gt = pred[frames], gt[frames]
    hp, hg = ~np.isnan(pred[:, 0]), ~np.isnan(gt[:, 0])
    close = np.zeros(len(gt), bool)
    both = hp & hg
    close[both] = np.linalg.norm(pred[both] - gt[both], axis=1) <= tol_px
    tp = int(close.sum())
    p = tp / max(1, int(hp.sum()))
    r = tp / max(1, int(hg.sum()))
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4),
            "gt_frames": int(hg.sum()), "pred_frames": int(hp.sum())}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def player_det_pr(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thr: float = 0.5,
                  frames: np.ndarray | None = None) -> dict:
    """pred_boxes: (n, K, 4) nan padded; gt_boxes: (n, G, 4) nan padded."""
    rng = frames if frames is not None else np.arange(len(gt_boxes))
    tp = fp = fn = 0
    for f in rng:
        p = pred_boxes[f][~np.isnan(pred_boxes[f, :, 0])]
        g = gt_boxes[f][~np.isnan(gt_boxes[f, :, 0])]
        if not len(p) or not len(g):
            fp += len(p)
            fn += len(g)
            continue
        iou = _iou(p, g)
        r, c = linear_sum_assignment(-iou)
        m = int((iou[r, c] >= iou_thr).sum())
        tp += m
        fp += len(p) - m
        fn += len(g) - m
    p, rcl = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(p, 4), "recall": round(rcl, 4), "f1": round(2 * p * rcl / max(1e-9, p + rcl), 4)}


def position_error(pred_m: np.ndarray, gt_m: np.ndarray, frames: np.ndarray | None = None) -> dict:
    """Per-frame Hungarian between predicted and GT floor positions (metres)."""
    rng = frames if frames is not None else np.arange(len(gt_m))
    errs = []
    found = total = 0
    for f in rng:
        p = pred_m[f][~np.isnan(pred_m[f, :, 0])]
        g = gt_m[f][~np.isnan(gt_m[f, :, 0])]
        total += len(g)
        if not len(p) or not len(g):
            continue
        d = np.linalg.norm(p[:, None] - g[None], axis=2)
        r, c = linear_sum_assignment(d)
        ok = d[r, c] < 1.5
        found += int(ok.sum())
        errs.extend(d[r, c][ok].tolist())
    e = np.asarray(errs)
    return {"median_m": round(float(np.median(e)), 3) if len(e) else None,
            "p90_m": round(float(np.percentile(e, 90)), 3) if len(e) else None,
            "coverage": round(found / max(1, total), 4)}


def event_f1(pred_frames: list[int], gt_frames: list[int], tol: int = 7) -> dict:
    """Greedy one-to-one matching of event frames within +-tol."""
    pred = sorted(pred_frames)
    gt = sorted(gt_frames)
    used = set()
    tp = 0
    pairs = []
    for p in pred:
        best, bd = None, tol + 1
        for i, g in enumerate(gt):
            if i in used:
                continue
            d = abs(g - p)
            if d < bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            tp += 1
            pairs.append((p, gt[best]))
    prec, rec = tp / max(1, len(pred)), tp / max(1, len(gt))
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4), "tp": tp, "pred": len(pred), "gt": len(gt),
            "pairs": pairs}
