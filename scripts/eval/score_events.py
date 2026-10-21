#!/usr/bin/env python3
"""Precision/recall for pass|shot|goal vs GT JSON (time tolerance)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_pred(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "events" in raw:
        raw = raw["events"]
    out = []
    for e in raw:
        et = e.get("event_type") or e.get("type")
        ts = e.get("camera_ts_ms", e.get("ts_ms"))
        if et and ts is not None:
            out.append({"type": et, "ts_ms": int(ts)})
    return out


def load_gt(path: Path) -> tuple[list[dict], int]:
    data = json.loads(path.read_text())
    tol = int(data.get("tolerance_ms", 3000))
    events = [
        {"type": e["type"], "ts_ms": int(e["ts_ms"])}
        for e in data.get("events", [])
        if e.get("type") and e.get("ts_ms") is not None
    ]
    return events, tol


def match_class(pred: list[dict], gt: list[dict], etype: str, tol: int) -> dict:
    p = sorted([e for e in pred if e["type"] == etype], key=lambda x: x["ts_ms"])
    g = sorted([e for e in gt if e["type"] == etype], key=lambda x: x["ts_ms"])
    used = set()
    tp = 0
    for pe in p:
        for i, ge in enumerate(g):
            if i in used:
                continue
            if abs(pe["ts_ms"] - ge["ts_ms"]) <= tol:
                used.add(i)
                tp += 1
                break
    fp = len(p) - tp
    fn = len(g) - tp
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else (1.0 if not p else 0.0)
    return {
        "type": etype,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred": len(p),
        "gt": len(g),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    args = ap.parse_args()
    pred = load_pred(args.pred)
    gt, tol = load_gt(args.gt)
    report = {
        "tolerance_ms": tol,
        "pred_total": len(pred),
        "gt_total": len(gt),
        "by_class": [match_class(pred, gt, t, tol) for t in ("pass", "shot", "goal")],
        "pred_counts": defaultdict(int),
    }
    for e in pred:
        report["pred_counts"][e["type"]] += 1
    report["pred_counts"] = dict(report["pred_counts"])
    print(json.dumps(report, indent=2))
    # Soft gate: refuse flood
    if report["pred_counts"].get("shot", 0) > 40:
        raise SystemExit("FAIL: shot flood (>40) — event engine gate failed")
    print("GATE: score_events ok (no shot flood)")


if __name__ == "__main__":
    main()
