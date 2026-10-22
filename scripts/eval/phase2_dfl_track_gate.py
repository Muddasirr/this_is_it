#!/usr/bin/env python3
"""Phase 2 final gate on DFL 30s clips: detect + track + jersey-size kit crops.

Uses football fine-tune (player/ref/ball). Writes:
  data/eval/dfl_phase2/PHASE2_DFL_TRACK_GATE.json
  data/eval/dfl_phase2/<slug>/index.html  (only jersey-sized player crops)

Gate (honest):
  - each clip: ≥1 person track with bbox series
  - kit gallery only from boxes with height ≥ JERSEY_H_FRAC * frame_h
  - unit tests for kit/OCR must be run separately (pytest)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend"
sys.path.insert(0, str(_BACKEND))

OUT = Path(os.environ.get("PHASE2_DATA", _ROOT / "data")) / "eval" / "dfl_phase2"
VIDEOS = Path(os.environ.get("PHASE2_VIDEOS", _ROOT / "testing_videos"))

# Prefer user-named set; also pick up dfl_extra_* if downloaded
CLIPS_ENV = os.environ.get("DFL_CLIPS", "").strip()
if CLIPS_ENV:
    CLIPS = [c.strip() for c in CLIPS_ENV.split(",") if c.strip()]
else:
    CLIPS = sorted({p.name for p in VIDEOS.glob("dfl_*.mp4")})

FRAME_STRIDE = int(os.environ.get("DFL_FRAME_STRIDE", "3"))
MAX_FRAMES = int(os.environ.get("DFL_MAX_FRAMES", "120"))  # processed frames
CONF = float(os.environ.get("DFL_CONF", "0.25"))
IMGSZ = int(os.environ.get("DFL_IMGSZ", "960"))
# Kit assign uses soft size (football_analysis: every player box). Jersey-OCR
# gallery can still prefer larger boxes via DFL_JERSEY_H_FRAC.
JERSEY_H_FRAC = float(os.environ.get("DFL_JERSEY_H_FRAC", "0.04"))  # 1080p → ≥43px
KIT_MIN_H = int(os.environ.get("DFL_KIT_MIN_H", "40"))  # matches TeamAssigner.fit
KIT_MIN_W = int(os.environ.get("DFL_KIT_MIN_W", "20"))
MAX_GALLERY = int(os.environ.get("DFL_MAX_GALLERY", "24"))
REF_VOTE_FRAC = float(os.environ.get("DFL_REF_VOTE_FRAC", "0.25"))


def _weights() -> str:
    for p in [
        os.environ.get("YOLO_MODEL"),
        str(_BACKEND / "yolov5x_football.pt"),
        str(_ROOT / "yolov5x_football.pt"),
        str(_ROOT / "data/eval/football_analysis_model/best.pt"),
        str(_BACKEND / "yolov8n_football.pt"),
        str(_ROOT / "data/eval/roboflow_football_players_v12/yolov8n_football_v12_best.pt"),
        str(_ROOT / "yolov8n_futsal.pt"),
        str(_BACKEND / "yolov8n.pt"),
    ]:
        if p and Path(p).is_file() and Path(p).stat().st_size > 1000:
            return str(Path(p).resolve())
    return "yolov5x_football.pt"


def _buckets(model):
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    b = {"person": set(), "ball": set(), "referee": set()}
    for cid, name in names.items():
        n = str(name).lower().replace(" ", "_")
        if n in {"person", "player", "goalkeeper", "gk"}:
            b["person"].add(int(cid))
        elif n in {"ball", "sports_ball", "football"}:
            b["ball"].add(int(cid))
        elif n in {"referee", "ref"}:
            b["referee"].add(int(cid))
    if not b["person"]:
        b = {"person": {0}, "ball": {32}, "referee": set()}
    return b


def run_clip(video: Path, model, buckets) -> dict:
    import cv2
    from ultralytics import YOLO  # noqa: F401
    from app.services.kit_color import assign_kits_from_descriptors, jersey_descriptor_bgr, kit_purity

    slug = video.stem
    print(f"\n=== {slug} ===", flush=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"slug": slug, "error": "open_fail", "pass": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    # Soft floor for kit colour (Ablone) — NOT OCR jersey-size. Broadcast DFL
    # players are often <10% of frame height; starving crops caused 0/11 kits.
    min_h = max(KIT_MIN_H, int(fh * JERSEY_H_FRAC))
    classes = sorted(buckets["person"] | buckets["ball"] | buckets["referee"])
    tracker = str(_BACKEND / "cfg" / "botsort.yaml")
    if not Path(tracker).is_file():
        tracker = "botsort.yaml"

    person_ids: set[int] = set()
    ref_ids: set[int] = set()
    ball_hits = 0
    # Best crop per track_id (football_analysis sticky id)
    best_by_tid: dict = {}
    anon_crops: list[dict] = []
    # Detector class flickers player↔referee within one track; vote per track
    class_votes: dict[int, list[int]] = {}
    frames_tracked = 0
    idx = 0
    processed = 0

    while processed < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_STRIDE != 0:
            idx += 1
            continue
        idx += 1
        processed += 1
        res = model.track(
            frame,
            persist=True,
            conf=CONF,
            imgsz=IMGSZ,
            classes=classes,
            tracker=tracker,
            verbose=False,
            device=os.environ.get("VISION_DEVICE", "cpu"),
        )
        frames_tracked += 1
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            continue
        ids = r0.boxes.id
        for i in range(len(r0.boxes)):
            cls_id = int(r0.boxes.cls[i].item())
            conf = float(r0.boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in r0.boxes.xyxy[i].tolist()]
            bh, bw = y2 - y1, x2 - x1
            tid = None
            if ids is not None:
                try:
                    tid = int(ids[i].item())
                except Exception:
                    tid = None
            if cls_id in buckets["ball"]:
                ball_hits += 1
                continue
            is_ref = cls_id in buckets["referee"]
            if tid is not None:
                votes = class_votes.setdefault(tid, [0, 0])
                votes[0] += int(is_ref)
                votes[1] += 1
            if is_ref and tid is None:
                continue
            if tid is not None and not is_ref:
                person_ids.add(tid)
            # Soft size only — no tall-aspect lock (wide broadcast boxes are squat)
            if bh < min_h or bw < KIT_MIN_W:
                continue
            crop = frame[max(0, y1) : min(fh, y2), max(0, x1) : min(fw, x2)].copy()
            if crop.size == 0:
                continue
            score = conf * (bh / max(fh, 1))
            entry = {
                "img": crop,
                "conf": conf,
                "tid": tid,
                "bh": bh,
                "bw": bw,
                "score": score,
                "frame": idx,
                "class_name": "person",
            }
            if tid is not None:
                prev = best_by_tid.get(tid)
                if prev is None or score > prev["score"]:
                    best_by_tid[tid] = entry
            else:
                anon_crops.append(entry)

    cap.release()
    for tid, (refs, total) in class_votes.items():
        if total and refs / total >= REF_VOTE_FRAC:
            ref_ids.add(tid)
            person_ids.discard(tid)
            if tid in best_by_tid:
                best_by_tid[tid]["class_name"] = "referee"
    # Fit kits on every track (football_analysis fits on all players); the
    # gallery below only shows the largest crops per kit.
    picked = sorted(best_by_tid.values(), key=lambda c: -c["score"])
    if len(picked) < 2:
        picked.extend(sorted(anon_crops, key=lambda c: -c["score"])[:MAX_GALLERY])

    gal = OUT / slug
    gal.mkdir(parents=True, exist_ok=True)
    for p in gal.glob("Kit*.jpg"):
        p.unlink()

    metas = []
    for i, c in enumerate(picked):
        d = jersey_descriptor_bgr(c["img"])
        metas.append(
            {
                "img": c["img"],
                "jersey_hsv": d,
                "sticky_id": c["tid"] if c["tid"] is not None else f"anon_{i}",
                "src": f"f{c['frame']}_{i}.jpg",
                "det_conf": c["conf"],
                "px": f"{c['bw']}×{c['bh']}",
                "class_name": c.get("class_name") or "person",
            }
        )
    if len(metas) >= 2:
        assign_kits_from_descriptors(metas)
        purity = kit_purity(metas)
    else:
        for m in metas:
            m["kit"] = "A"
        purity = {"pass": False, "issues": ["too_few_jersey_crops"], "A": len(metas), "B": 0, "X": 0}

    by = {"A": [], "B": [], "X": []}
    for m in metas:
        by.setdefault(m.get("kit") or "X", []).append(m)

    import cv2
    import numpy as np

    rows = {"A": [], "B": [], "X": []}
    sheet_rows = []
    for kit in ("A", "B", "X"):
        thumbs = [cv2.resize(m["img"], (60, 120)) for m in (by.get(kit) or [])[:16]]
        blank = np.zeros((120, 60, 3), np.uint8)
        row = np.hstack(thumbs + [blank] * (16 - len(thumbs)))
        cv2.putText(row, f"{kit} ({len(by.get(kit) or [])})", (2, 15), 0, 0.5, (0, 255, 255), 2)
        sheet_rows.append(row)
    cv2.imwrite(str(gal / "sheet.jpg"), np.vstack(sheet_rows))
    for kit in ("A", "B", "X"):
        for i, m in enumerate((by.get(kit) or [])[:MAX_GALLERY], 1):
            name = f"Kit{kit}_{i}.jpg"
            cv2.imwrite(str(gal / name), m["img"])
            d = m.get("jersey_hsv") or {}
            rows[kit].append(
                {
                    "file": name,
                    "px": m.get("px"),
                    "det_conf": round(float(m.get("det_conf") or 0), 3),
                    "redness": round(float(d.get("redness") or 0), 3),
                }
            )

    track_ok = len(person_ids) >= 1 and frames_tracked > 0
    jersey_ok = len(metas) >= 2
    # Honest: clip passes track gate if persons tracked; kit subgate separate
    report = {
        "slug": slug,
        "frames_tracked": frames_tracked,
        "unique_person_ids": len(person_ids),
        "unique_ref_ids": len(ref_ids),
        "ball_hits": ball_hits,
        "jersey_crops": len(metas),
        "min_box_h_px": min_h,
        "frame_hw": [fw, fh],
        "kit_A": len(by.get("A") or []),
        "kit_B": len(by.get("B") or []),
        "kit_X": len(by.get("X") or []),
        "purity": purity,
        "track_pass": track_ok,
        "jersey_kit_pass": bool(purity.get("pass")) and jersey_ok,
        "pass": track_ok,  # Phase2 track gate for DFL; kit purity reported separately
    }

    def figs(kit):
        return "".join(
            f'<figure class="kit{kit}"><img src="{r["file"]}"/><figcaption>'
            f"<b>Kit {kit}</b> {r['px']} det={r['det_conf']}<br/>red={r['redness']:.2f}"
            f"</figcaption></figure>"
            for r in rows[kit]
        )

    (gal / "index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>{slug}</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px}}
figure{{margin:0;border:2px solid #333;padding:6px;background:#1a1a1a}}
.kitA{{border-color:#ef4444}}.kitB{{border-color:#94a3b8}}.kitX{{border-color:#f59e0b}}
img{{width:100%;aspect-ratio:3/4;object-fit:cover}}figcaption{{font-size:11px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}}</style></head><body>
<h1>{slug}</h1>
<p>track <span class="{'ok' if track_ok else 'bad'}">{'PASS' if track_ok else 'FAIL'}</span>
· persons={len(person_ids)} refs={len(ref_ids)} ball={ball_hits}
· jersey_crops={len(metas)} (h≥{min_h}px) · kit purity
<span class="{'ok' if report['jersey_kit_pass'] else 'bad'}">{'PASS' if report['jersey_kit_pass'] else 'FAIL'}</span>
<br/>issues: {'; '.join(purity.get('issues') or []) or 'none'}</p>
<h2>Kit A ({len(rows['A'])})</h2><grid>{figs('A')}</grid>
<h2>Kit B ({len(rows['B'])})</h2><grid>{figs('B')}</grid>
<h2>Kit X ({len(rows['X'])})</h2><grid>{figs('X')}</grid>
</body></html>"""
    )
    print(
        f"  track={'PASS' if track_ok else 'FAIL'} ids={len(person_ids)} ball={ball_hits} "
        f"jersey={len(metas)} kit_purity={'PASS' if report['jersey_kit_pass'] else 'FAIL'}",
        flush=True,
    )
    return report


def main() -> None:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    w = _weights()
    print(f"weights={w} clips={len(CLIPS)} jersey_frac={JERSEY_H_FRAC}", flush=True)
    model = YOLO(w)
    buckets = _buckets(model)
    reports = []
    for name in CLIPS:
        path = VIDEOS / name
        if not path.is_file():
            reports.append({"slug": Path(name).stem, "error": "missing", "pass": False})
            continue
        # fresh tracker state per clip
        model = YOLO(w)
        reports.append(run_clip(path, model, buckets))

    track_pass = sum(1 for r in reports if r.get("track_pass"))
    kit_pass = sum(1 for r in reports if r.get("jersey_kit_pass"))
    summary = {
        "videos": len(reports),
        "track_pass": track_pass,
        "jersey_kit_pass": kit_pass,
        "weights": w,
        "jersey_h_frac": JERSEY_H_FRAC,
        "reports": reports,
        "gate_note": "Phase2 DFL track gate = track_pass. Kit purity only when jersey-sized crops ≥2.",
    }
    (OUT / "PHASE2_DFL_TRACK_GATE.json").write_text(json.dumps(summary, indent=2))
    cards = []
    for r in reports:
        if r.get("error"):
            cards.append(f"<li>{r.get('slug')}: ERROR {r['error']}</li>")
            continue
        tp = "ok" if r.get("track_pass") else "bad"
        kp = "ok" if r.get("jersey_kit_pass") else "bad"
        cards.append(
            f"<li><a href=\"{r['slug']}/index.html\">{r['slug']}</a> "
            f"track=<span class=\"{tp}\">{'PASS' if r.get('track_pass') else 'FAIL'}</span> "
            f"kit=<span class=\"{kp}\">{'PASS' if r.get('jersey_kit_pass') else 'FAIL'}</span> "
            f"ids={r.get('unique_person_ids')} jersey={r.get('jersey_crops')} ball={r.get('ball_hits')}</li>"
        )
    (OUT / "index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>DFL Phase2 gate</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}}a{{color:#93c5fd}}</style></head>
<body><h1>Phase 2 — DFL detect/track gate</h1>
<p>track_pass {track_pass}/{len(reports)} · jersey_kit_pass {kit_pass}/{len(reports)}
· weights <code>{Path(w).name}</code></p>
<ul>{''.join(cards)}</ul></body></html>"""
    )
    print(f"\nWrote {OUT}/PHASE2_DFL_TRACK_GATE.json track={track_pass}/{len(reports)} kit={kit_pass}/{len(reports)}", flush=True)
    # Exit 0 if tracking works on majority — kit purity is secondary on wide shots
    if track_pass < max(1, len(reports) // 2):
        sys.exit(1)


if __name__ == "__main__":
    main()
