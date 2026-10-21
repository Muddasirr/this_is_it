#!/usr/bin/env python3
"""Phase 2 kit+OCR probe on DFL soccer clips (no prior corpus thumbs).

  docker run --rm \
    -v $PWD/backend:/app -v $PWD/data:/host_data \
    -v $PWD/testing_videos:/videos -v $PWD/scripts:/scripts \
    -e PHASE2_DATA=/host_data -w /app paas-api \
    python -u /scripts/eval/phase2_dfl_videos.py

Samples frames → YOLO person crops → kit A/B (+X outliers) → OCR → gallery.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_BACKEND = Path("/app") if Path("/app/app/services").is_dir() else Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(_BACKEND))

_DATA = Path(os.environ["PHASE2_DATA"]) if "PHASE2_DATA" in os.environ else (
    Path("/host_data") if Path("/host_data/eval").is_dir() else Path(__file__).resolve().parents[2] / "data"
)
VIDEOS_DIR = Path(os.environ.get("PHASE2_VIDEOS", "/videos"))
OUT = _DATA / "eval" / "identity_gallery"

CLIPS_DEFAULT = [
    "dfl_A1606b0e6_0_43.mp4",
    "dfl_A1606b0e6_0_58.mp4",
    "dfl_A1606b0e6_0_66.mp4",
    "dfl_B1606b0e6_1_2.mp4",
    "dfl_B1606b0e6_1_3.mp4",
    "dfl_D35bd9041_1_11.mp4",
    "dfl_E3c993bd2_0_5.mp4",
    "dfl_E3c993bd2_0_6.mp4",
    "dfl_E3c993bd2_0_8.mp4",
    "dfl_test_3.mp4",
    "dfl_test_14.mp4",
]
# Optional: DFL_CLIPS="a.mp4,b.mp4" to run a subset
_CLIPS_ENV = os.environ.get("DFL_CLIPS", "").strip()
CLIPS = [c.strip() for c in _CLIPS_ENV.split(",") if c.strip()] if _CLIPS_ENV else CLIPS_DEFAULT

# CPU-friendly sampling (DFL broadcast: distant players → lower conf / min height)
FRAME_STRIDE = int(os.environ.get("DFL_FRAME_STRIDE", "20"))
MAX_FRAMES = int(os.environ.get("DFL_MAX_FRAMES", "60"))  # processed frames
MAX_CROPS = int(os.environ.get("DFL_MAX_CROPS", "24"))
MIN_BOX_H = int(os.environ.get("DFL_MIN_BOX_H", "32"))
CONF = float(os.environ.get("DFL_CONF", "0.18"))
IMGSZ = int(os.environ.get("DFL_IMGSZ", "960"))


def _slug(name: str) -> str:
    return Path(name).stem


def _collect_crops(video: Path) -> list[tuple]:
    """Return list of (bgr_crop, frame_idx, conf)."""
    import cv2
    from ultralytics import YOLO

    weights = _BACKEND / "yolov8n.pt"
    if not weights.is_file():
        weights = Path("/app/yolov8n.pt")
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    crops: list[tuple] = []
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
        h, w = frame.shape[:2]
        res = model.predict(frame, conf=CONF, classes=[0], verbose=False, imgsz=IMGSZ)
        if not res:
            continue
        boxes = res[0].boxes
        if boxes is None:
            continue
        for b in boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            if bh < MIN_BOX_H or bw < 20:
                continue
            # expand slightly, clamp
            pad = int(0.05 * bh)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append((crop, idx, conf))

    cap.release()
    # Keep largest / highest-conf crops
    crops.sort(key=lambda t: (t[0].shape[0] * t[0].shape[1], t[2]), reverse=True)
    return crops[:MAX_CROPS]


def run_one(video: Path) -> dict:
    import cv2
    from app.services.kit_color import (
        assign_kits_from_descriptors,
        jersey_descriptor_bgr,
        kit_purity,
    )
    from app.services.jersey_ocr import jersey_ocr_enabled, read_jersey_number_bgr

    slug = _slug(video.name)
    print(f"\n=== {slug} ===", flush=True)
    crops = _collect_crops(video)
    print(f"  crops={len(crops)} ocr={jersey_ocr_enabled()}", flush=True)
    if len(crops) < 2:
        return {"slug": slug, "error": "too_few_crops", "crops": len(crops), "verdict": "FAIL"}

    metas = []
    for i, (crop, fidx, conf) in enumerate(crops):
        d = jersey_descriptor_bgr(crop)
        ocr = read_jersey_number_bgr(crop) if jersey_ocr_enabled() else None
        metas.append(
            {
                "jersey_hsv": d,
                "img": crop,
                "src": f"f{fidx}_{i}.jpg",
                "ocr": ocr,
                "det_conf": conf,
            }
        )

    assign_kits_from_descriptors(metas)
    purity = kit_purity(metas)

    gal = OUT / slug
    gal.mkdir(parents=True, exist_ok=True)
    for p in gal.glob("Kit*.jpg"):
        p.unlink()

    by: dict[str, list] = {"A": [], "B": [], "X": []}
    for m in metas:
        by.setdefault(m.get("kit") or "?", []).append(m)
    for k in by:
        by[k].sort(key=lambda m: (-(m["jersey_hsv"] or {}).get("redness", 0), -(m.get("det_conf") or 0)))

    rows: dict[str, list] = {"A": [], "B": [], "X": []}
    ocr_hits = 0
    two_digit = 0
    used: dict[str, set[int]] = {"A": set(), "B": set(), "X": set()}

    for kit, items in by.items():
        if kit not in rows:
            continue
        for i, m in enumerate(items, 1):
            d = m["jersey_hsv"] or {}
            shirt = None
            jconf = 0.0
            ocr = m.get("ocr")
            if ocr and float(ocr.get("confidence") or 0) >= 0.4:
                num = int(ocr["number"])
                # Prefer keeping 2-digit; skip duplicate
                if num not in used[kit]:
                    # Drop lone tens-digit if we later... just take best once
                    shirt = num
                    jconf = float(ocr["confidence"])
                    used[kit].add(num)
                    ocr_hits += 1
                    if num >= 10:
                        two_digit += 1
            name = f"Kit{kit}_{i}.jpg"
            cv2.imwrite(str(gal / name), m["img"])
            rows[kit].append(
                {
                    "file": name,
                    "shirt": shirt,
                    "ocr_conf": round(jconf, 3) if shirt else None,
                    "redness": round(float(d.get("redness") or 0), 3),
                    "darkness": round(float(d.get("darkness") or 0), 3),
                    "hue": round(float(d.get("hue") or 0), 1),
                    "sat": round(float(d.get("sat") or 0), 3),
                    "cyan": round(float(d.get("cyan_frac") or 0), 3),
                    "neon": round(float(d.get("neon_frac") or 0), 3),
                }
            )

    na, nb, nx = len(rows["A"]), len(rows["B"]), len(rows["X"])
    verdict = "PASS" if purity.get("pass") else "FAIL"
    issues = "; ".join(purity.get("issues") or []) or "none"

    def fig(kit: str, r: dict) -> str:
        cls = {"A": "kitA", "B": "kitB", "X": "kitX"}.get(kit, "kitB")
        shirt = f"#{r['shirt']}" if r["shirt"] is not None else "(no #)"
        ocr = f" ocr={r['ocr_conf']}" if r["ocr_conf"] else ""
        return (
            f'<figure class="{cls}"><img src="{r["file"]}"/>'
            f"<figcaption><b>Kit {kit} {shirt}</b>{ocr}<br/>"
            f"red={r['redness']:.2f} dark={r['darkness']:.2f}<br/>"
            f"hue={r['hue']:.0f} sat={r['sat']:.2f}</figcaption></figure>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Phase2 DFL — {slug}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#eee;margin:24px}}
.meta{{opacity:.85;max-width:960px;line-height:1.45}}
grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-top:12px}}
figure{{margin:0;background:#1a1a1a;border-radius:8px;padding:8px;border:2px solid #333}}
figure.kitA{{border-color:#ef4444}}figure.kitB{{border-color:#94a3b8}}figure.kitX{{border-color:#f59e0b}}
img{{width:100%;aspect-ratio:3/4;object-fit:cover}}
figcaption{{font-size:11px;margin-top:6px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}}
</style></head><body>
<h1>Phase 2 DFL — {slug}</h1>
<p class="meta">Sampled YOLO crops · purity gate · outliers→X.<br/>
A={na} B={nb} X={nx} · crops={len(crops)} · OCR={ocr_hits} (2dig={two_digit}) ·
<span class="{'ok' if verdict=='PASS' else 'bad'}">{verdict}</span><br/>issues: {issues}</p>
<h2>Kit A ({na})</h2><grid>{''.join(fig('A', r) for r in rows['A'])}</grid>
<h2>Kit B ({nb})</h2><grid>{''.join(fig('B', r) for r in rows['B'])}</grid>
<h2>Kit X — outliers ({nx})</h2><grid>{''.join(fig('X', r) for r in rows['X'])}</grid>
</body></html>"""
    (gal / "index.html").write_text(html)
    (gal / "players.json").write_text(
        json.dumps(
            [
                {
                    "kit": kit,
                    "shirt": r["shirt"],
                    "snapshot": r["file"],
                    "ocr_likely": r["shirt"] is not None,
                }
                for kit in ("A", "B", "X")
                for r in rows[kit]
            ],
            indent=2,
        )
    )

    report = {
        "slug": slug,
        "A": na,
        "B": nb,
        "X": nx,
        "crops": len(crops),
        "ocr_hits": ocr_hits,
        "ocr_two_digit": two_digit,
        "verdict": verdict,
        "purity": purity,
    }
    print(f"  → A={na} B={nb} X={nx} ocr={ocr_hits} {verdict} ({issues})", flush=True)
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in CLIPS:
        path = VIDEOS_DIR / name
        if not path.is_file():
            # fallback host relative
            alt = Path(__file__).resolve().parents[2] / "testing_videos" / name
            path = alt if alt.is_file() else path
        if not path.is_file():
            print(f"MISSING {name}", flush=True)
            reports.append({"slug": _slug(name), "error": "missing", "verdict": "FAIL"})
            continue
        reports.append(run_one(path))

    # Merge into prior report by slug (so subset runs keep earlier clips)
    report_path = OUT / "PHASE2_DFL_REPORT.json"
    by_slug: dict[str, dict] = {}
    if report_path.is_file():
        try:
            prev = json.loads(report_path.read_text())
            for r in prev.get("reports") or []:
                if r.get("slug"):
                    by_slug[r["slug"]] = r
        except Exception:
            pass
    for r in reports:
        if r.get("slug"):
            by_slug[r["slug"]] = r
    # Stable order: default clip order, then any extras
    order = [_slug(c) for c in CLIPS_DEFAULT]
    merged = [by_slug[s] for s in order if s in by_slug]
    for s, r in by_slug.items():
        if s not in {x["slug"] for x in merged}:
            merged.append(r)

    passed = sum(1 for r in merged if r.get("verdict") == "PASS")
    summary = {"videos": len(merged), "kit_pass": passed, "reports": merged}
    report_path.write_text(json.dumps(summary, indent=2))

    cards = []
    for r in merged:
        cls = "ok" if r.get("verdict") == "PASS" else "bad"
        if r.get("error"):
            cards.append(f"<li>{r.get('slug')}: ERROR {r['error']}</li>")
        else:
            cards.append(
                f"<li><a href=\"{r['slug']}/index.html\">{r['slug']}</a> — "
                f"<span class=\"{cls}\">{r['verdict']}</span> "
                f"A={r['A']} B={r['B']} X={r['X']} OCR={r['ocr_hits']}</li>"
            )
    (OUT / "dfl_index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Phase2 DFL</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}}a{{color:#93c5fd}}</style></head>
<body><h1>Phase 2 — DFL clips</h1>
<p>kit_pass {passed}/{len(merged)}</p><ul>{''.join(cards)}</ul></body></html>"""
    )
    print(f"\nWrote {OUT}/PHASE2_DFL_REPORT.json  kit_pass={passed}/{len(merged)}", flush=True)
    this_pass = sum(1 for r in reports if r.get("verdict") == "PASS")
    if this_pass < len(reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
