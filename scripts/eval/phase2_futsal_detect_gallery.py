#!/usr/bin/env python3
"""Detect players on futsal highlight clips → KMeans Kit A/B → HTML gallery.

Samples frames evenly across the whole video (highlights are long), runs
football/futsal YOLO (player/ref/ball), assigns teams via KMeans, optional OCR.

  PYTHONPATH=backend PHASE2_DATA=data \\
    python -u scripts/eval/phase2_futsal_detect_gallery.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = Path("/app") if Path("/app/app/services").is_dir() else _ROOT / "backend"
sys.path.insert(0, str(_BACKEND))

_DATA = Path(os.environ["PHASE2_DATA"]) if "PHASE2_DATA" in os.environ else (
    Path("/host_data") if Path("/host_data/eval").is_dir() else _ROOT / "data"
)
VIDEOS_DIR = Path(os.environ.get("PHASE2_VIDEOS", str(_ROOT / "testing_videos")))
OUT = _DATA / "eval" / "identity_gallery"

CLIPS = [
    (
        "bangladesh_pakistan_saff_2026",
        "Bangladesh vs Pakistan Futsal Highlights _ Pakistan Wins 5-1 _SAFF Futsal Championship 2026.mp4",
    ),
    (
        "portugal_spain_euro_2026_final",
        "Portugal vs Spain Highlights _ Futsal Euro 2026 FINAL.mp4",
    ),
    (
        "spain_france_u19_2023",
        "Spain vs France _ Highlights _ U19 Euro Futsal 06-09-2023.mp4",
    ),
]

# Quality gates — kit ID needs a visible TORSO, not socks / seats / ads.
MAX_FRAMES = int(os.environ.get("FUTSAL_MAX_FRAMES", "80"))
MAX_CROPS = int(os.environ.get("FUTSAL_MAX_CROPS", "24"))
# Absolute px floor AND fraction of frame height (720p → ≥108px at 0.15)
MIN_BOX_H = int(os.environ.get("FUTSAL_MIN_BOX_H", "100"))
MIN_BOX_H_FRAC = float(os.environ.get("FUTSAL_MIN_BOX_H_FRAC", "0.16"))
MIN_ASPECT = float(os.environ.get("FUTSAL_MIN_ASPECT", "1.4"))  # h/w — standing person
MAX_ASPECT = float(os.environ.get("FUTSAL_MAX_ASPECT", "4.5"))
CONF = float(os.environ.get("FUTSAL_CONF", "0.40"))
IMGSZ = int(os.environ.get("FUTSAL_IMGSZ", "1280"))
MIN_CROP_PX = int(os.environ.get("FUTSAL_MIN_CROP_PX", "80"))  # min side after crop


def _weights() -> str:
    env = os.environ.get("YOLO_MODEL") or os.environ.get("OVERLAY_WEIGHTS")
    cands = []
    if env:
        cands += [Path(env), _BACKEND / env, _ROOT / env]
    cands += [
        _BACKEND / "yolov5x_football.pt",
        _ROOT / "yolov5x_football.pt",
        _ROOT / "data/eval/football_analysis_model/best.pt",
        _BACKEND / "yolov8n_futsal.pt",
        _ROOT / "yolov8n_futsal.pt",
        _BACKEND / "yolov8n_football.pt",
        _ROOT / "data/eval/roboflow_football_players_v12/yolov8n_football_v12_best.pt",
        _BACKEND / "yolov8n.pt",
    ]
    for p in cands:
        if p.is_file() and p.stat().st_size > 1000:
            return str(p.resolve())
    return "yolov8n.pt"


def _class_buckets(model) -> dict[str, set[int]]:
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    out = {"person": set(), "ball": set(), "referee": set()}
    for cid, name in names.items():
        n = str(name).strip().lower().replace(" ", "_")
        if n in {"person", "player", "goalkeeper", "gk"}:
            out["person"].add(int(cid))
        elif n in {"ball", "sports_ball", "football", "soccer_ball"}:
            out["ball"].add(int(cid))
        elif n in {"referee", "ref", "official"}:
            out["referee"].add(int(cid))
    if not out["person"]:
        out["person"].add(0)
        out["ball"].add(32)
    return out


def _frame_indices(n_frames: int, max_n: int) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames <= max_n:
        return list(range(n_frames))
    # even sample across full highlight
    return [int(i * (n_frames - 1) / (max_n - 1)) for i in range(max_n)]


def _accept_person_box(bh: int, bw: int, frame_h: int, conf: float) -> bool:
    """Only keep boxes large enough that a jersey torso is actually visible."""
    if conf < CONF:
        return False
    if bw < 28 or bh < MIN_BOX_H:
        return False
    if bh < int(frame_h * MIN_BOX_H_FRAC):
        return False
    aspect = bh / max(1, bw)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False  # rejects wide seat/ad slabs and tiny foot blobs
    return True


def collect_crops(video: Path, model, buckets: dict) -> list[dict]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = _frame_indices(n, MAX_FRAMES)
    # Detect players+refs only (ball irrelevant for kit gallery)
    track_classes = sorted(buckets["person"] | buckets["referee"]) or sorted(buckets["person"])
    crops: list[dict] = []
    rejected = {"small": 0, "aspect": 0, "conf": 0, "kept": 0}

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = model.predict(frame, conf=min(CONF, 0.25), classes=track_classes, verbose=False, imgsz=IMGSZ)
        if not res or res[0].boxes is None:
            continue
        boxes = res[0].boxes
        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bh, bw = y2 - y1, x2 - x1
            if cls_id in buckets.get("ball", set()):
                continue
            if conf < CONF:
                rejected["conf"] += 1
                continue
            if bh < MIN_BOX_H or bh < int(h * MIN_BOX_H_FRAC) or bw < 28:
                rejected["small"] += 1
                continue
            aspect = bh / max(1, bw)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                rejected["aspect"] += 1
                continue
            kind = "referee" if cls_id in buckets["referee"] else "player"
            # Slight pad; keep full person for jersey top-half KMeans
            pad = int(0.03 * bh)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < MIN_CROP_PX:
                rejected["small"] += 1
                continue
            rejected["kept"] += 1
            crops.append(
                {
                    "img": crop,
                    "conf": conf,
                    "kind": kind,
                    "frame": fi,
                    "bh": bh,
                    "bw": bw,
                    # Prefer tall confident players (jersey-readable), not junk area
                    "score": conf * (bh / h) * min(aspect / 2.0, 1.5),
                }
            )
    cap.release()
    print(f"  filter kept={rejected['kept']} small={rejected['small']} aspect={rejected['aspect']} conf<{CONF}={rejected['conf']}", flush=True)
    crops.sort(key=lambda t: (t["kind"] != "player", -t["score"]))
    # Dedup near-identical frames: keep top score per ~2s window per kind
    players = [c for c in crops if c["kind"] == "player"][:MAX_CROPS]
    refs = [c for c in crops if c["kind"] == "referee"][:4]
    if players:
        sizes = [f"{c['bw']}x{c['bh']}" for c in players[:5]]
        print(f"  top player crop sizes (wxh px): {sizes}", flush=True)
    return players + refs


def run_one(slug: str, video: Path, model, buckets) -> dict:
    import cv2
    from app.services.kit_color import jersey_descriptor_bgr, kit_purity
    from app.services.jersey_ocr import jersey_ocr_enabled, read_jersey_number_bgr
    from app.services.team_assigner import assign_kits_kmeans

    print(f"\n=== {slug} ===", flush=True)
    print(f"  video={video.name}", flush=True)
    crops = collect_crops(video, model, buckets)
    print(f"  crops={len(crops)} (players+refs) ocr={jersey_ocr_enabled()}", flush=True)
    if len(crops) < 2:
        return {"slug": slug, "error": "too_few_crops", "crops": len(crops), "verdict": "FAIL"}

    metas = []
    for i, c in enumerate(crops):
        d = jersey_descriptor_bgr(c["img"])
        if c["kind"] == "referee" and d is not None:
            d = dict(d)
            d["is_outlier"] = True
        ocr = None
        if c["kind"] == "player" and jersey_ocr_enabled():
            ocr = read_jersey_number_bgr(c["img"])
        metas.append(
            {
                "img": c["img"],
                "jersey_hsv": d,
                "src": f"f{c['frame']}_{i}.jpg",
                "sticky_id": f"{slug}_{i}",
                "ocr": ocr,
                "det_conf": c["conf"],
                "class_name": c["kind"] if c["kind"] == "referee" else "person",
                "kind": c["kind"],
            }
        )

    # Prefer KMeans; kit_color.assign also tries KMeans then HSV
    from app.services.kit_color import assign_kits_from_descriptors

    assign_kits_from_descriptors(metas)
    # Force refs to X
    for m in metas:
        if m.get("kind") == "referee":
            m["kit"] = "X"
    purity = kit_purity(metas)

    gal = OUT / slug
    gal.mkdir(parents=True, exist_ok=True)
    for p in gal.glob("Kit*.jpg"):
        p.unlink()

    by: dict[str, list] = {"A": [], "B": [], "X": []}
    for m in metas:
        by.setdefault(m.get("kit") or "X", []).append(m)

    rows: dict[str, list] = {"A": [], "B": [], "X": []}
    ocr_hits = 0
    two_digit = 0
    used: dict[str, set] = {"A": set(), "B": set(), "X": set()}

    for kit in ("A", "B", "X"):
        items = by.get(kit) or []
        items.sort(key=lambda m: (-(m.get("det_conf") or 0), m["src"]))
        for i, m in enumerate(items, 1):
            d = m.get("jersey_hsv") or {}
            shirt = None
            jconf = 0.0
            ocr = m.get("ocr")
            if ocr and float(ocr.get("confidence") or 0) >= 0.4:
                num = int(ocr["number"])
                if num not in used[kit]:
                    shirt = num
                    jconf = float(ocr["confidence"])
                    used[kit].add(num)
                    ocr_hits += 1
                    if num >= 10:
                        two_digit += 1
            name = f"Kit{kit}_{i}.jpg"
            cv2.imwrite(str(gal / name), m["img"])
            img = m["img"]
            ih, iw = img.shape[:2]
            rows[kit].append(
                {
                    "file": name,
                    "shirt": shirt,
                    "ocr_conf": round(jconf, 3) if shirt else None,
                    "redness": round(float(d.get("redness") or 0), 3),
                    "darkness": round(float(d.get("darkness") or 0), 3),
                    "hue": round(float(d.get("hue") or 0), 1),
                    "sat": round(float(d.get("sat") or 0), 3),
                    "det_conf": round(float(m.get("det_conf") or 0), 3),
                    "kind": m.get("kind") or "player",
                    "px": f"{iw}×{ih}",
                }
            )

    na, nb, nx = len(rows["A"]), len(rows["B"]), len(rows["X"])
    verdict = "PASS" if purity.get("pass") else "FAIL"
    issues = "; ".join(purity.get("issues") or []) or "none"

    def fig(kit: str, r: dict) -> str:
        cls = {"A": "kitA", "B": "kitB", "X": "kitX"}.get(kit, "kitB")
        shirt = f"#{r['shirt']}" if r["shirt"] is not None else "(no #)"
        ocr = f" ocr={r['ocr_conf']}" if r["ocr_conf"] else ""
        kind = r.get("kind") or "player"
        return (
            f'<figure class="{cls}"><img src="{r["file"]}"/>'
            f"<figcaption><b>Kit {kit} {shirt}</b>{ocr}<br/>"
            f"{kind} {r.get('px','?')} det={r['det_conf']:.2f}<br/>"
            f"red={r['redness']:.2f} dark={r['darkness']:.2f}<br/>"
            f"hue={r['hue']:.0f} sat={r['sat']:.2f}</figcaption></figure>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Detect+Team — {slug}</title>
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
<h1>Detect + team — {slug}</h1>
<p class="meta">Even-frame YOLO (player/ref) · KMeans jersey teams · OCR.<br/>
A={na} B={nb} X={nx} · crops={len(crops)} · OCR={ocr_hits} (2dig={two_digit}) ·
<span class="{'ok' if verdict=='PASS' else 'bad'}">{verdict}</span><br/>
issues: {issues}</p>
<h2>Kit A — team 1 ({na})</h2><grid>{''.join(fig('A', r) for r in rows['A'])}</grid>
<h2>Kit B — team 2 ({nb})</h2><grid>{''.join(fig('B', r) for r in rows['B'])}</grid>
<h2>Kit X — ref / outlier ({nx})</h2><grid>{''.join(fig('X', r) for r in rows['X'])}</grid>
</body></html>"""
    (gal / "index.html").write_text(html)
    (gal / "players.json").write_text(
        json.dumps(
            [
                {
                    "kit": kit,
                    "shirt": r["shirt"],
                    "snapshot": r["file"],
                    "kind": r["kind"],
                    "det_conf": r["det_conf"],
                }
                for kit in ("A", "B", "X")
                for r in rows[kit]
            ],
            indent=2,
        )
    )
    print(f"  → A={na} B={nb} X={nx} ocr={ocr_hits} {verdict} ({issues})", flush=True)
    print(f"  gallery={gal / 'index.html'}", flush=True)
    return {
        "slug": slug,
        "A": na,
        "B": nb,
        "X": nx,
        "crops": len(crops),
        "ocr_hits": ocr_hits,
        "ocr_two_digit": two_digit,
        "verdict": verdict,
        "purity": purity,
        "gallery": str(gal / "index.html"),
    }


def main() -> None:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    w = _weights()
    print(f"weights={w}", flush=True)
    model = YOLO(w)
    buckets = _class_buckets(model)
    print(f"classes person={buckets['person']} ref={buckets['referee']} ball={buckets['ball']}", flush=True)

    reports = []
    for slug, name in CLIPS:
        path = VIDEOS_DIR / name
        if not path.is_file():
            print(f"MISSING {name}", flush=True)
            reports.append({"slug": slug, "error": "missing", "verdict": "FAIL"})
            continue
        reports.append(run_one(slug, path, model, buckets))

    passed = sum(1 for r in reports if r.get("verdict") == "PASS")
    summary = {"videos": len(reports), "kit_pass": passed, "weights": w, "reports": reports}
    (OUT / "PHASE2_FUTSAL_DETECT_REPORT.json").write_text(json.dumps(summary, indent=2))

    cards = []
    for r in reports:
        if r.get("error"):
            cards.append(f"<li>{r.get('slug')}: ERROR {r['error']}</li>")
        else:
            cls = "ok" if r.get("verdict") == "PASS" else "bad"
            cards.append(
                f"<li><a href=\"{r['slug']}/index.html\">{r['slug']}</a> — "
                f"<span class=\"{cls}\">{r['verdict']}</span> "
                f"A={r['A']} B={r['B']} X={r['X']} crops={r['crops']} OCR={r['ocr_hits']}</li>"
            )
    index = OUT / "futsal_detect_index.html"
    index.write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Futsal detect+team</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}}a{{color:#93c5fd}}</style></head>
<body><h1>Futsal — detect + team division</h1>
<p>kit_pass {passed}/{len(reports)} · weights <code>{Path(w).name}</code></p>
<ul>{''.join(cards)}</ul></body></html>"""
    )
    print(f"\nIndex: {index}", flush=True)
    print(f"Report: {OUT / 'PHASE2_FUTSAL_DETECT_REPORT.json'}  kit_pass={passed}/{len(reports)}", flush=True)


if __name__ == "__main__":
    main()
