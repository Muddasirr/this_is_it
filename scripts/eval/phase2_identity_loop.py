#!/usr/bin/env python3
"""Phase 2 gate loop: rebuild kit galleries + OCR for all corpus highlight videos.

  PYTHONPATH=backend python scripts/eval/phase2_identity_loop.py
  # or inside api container (has tesseract):
  docker exec -w /app paas-api-1 python scripts/eval/phase2_identity_loop.py

Writes:
  data/eval/identity_gallery/<slug>/index.html
  data/eval/identity_gallery/PHASE2_REPORT.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Host checkout or docker: -v paas/backend:/app -v paas/data:/host_data
_DATA = Path(os.environ["PHASE2_DATA"]) if "PHASE2_DATA" in os.environ else (
    Path("/host_data") if Path("/host_data/kaggle_out").is_dir() else (ROOT / "data")
)
_BACKEND = Path("/app") if Path("/app/app/services").is_dir() else (ROOT / "backend")
sys.path.insert(0, str(_BACKEND))

CORPUS = _DATA / "kaggle_out" / "corpus_v26" / "corpus_out"
OUT = _DATA / "eval" / "identity_gallery"

# Highlight / facility videos under testing_videos that have corpus thumbs
SLUGS_DEFAULT = [
    "test_video_2",
    "08fd33_4",
    "bangladesh_pakistan_saff_2026",
    "portugal_spain_euro_2026_final",
    "spain_france_u19_2023",
    "spain_brazil_fifa_2012",
]
_SLUGS_ENV = os.environ.get("PHASE2_SLUGS", "").strip()
SLUGS = [s.strip() for s in _SLUGS_ENV.split(",") if s.strip()] if _SLUGS_ENV else SLUGS_DEFAULT


def _rebuild_slug(slug: str) -> dict:
    import cv2
    from app.services.kit_color import (
        assign_kits_from_descriptors,
        jersey_descriptor_bgr,
        kit_purity,
    )
    from app.services.jersey_ocr import jersey_ocr_enabled, read_jersey_number_bgr

    thumbs = CORPUS / slug / "thumbs"
    gal = OUT / slug
    gal.mkdir(parents=True, exist_ok=True)

    srcs = sorted(thumbs.glob("*.jpg")) if thumbs.is_dir() else []
    if not srcs:
        # fall back to existing gallery snaps
        srcs = sorted(gal.glob("Kit*.jpg"))
    if len(srcs) < 2:
        return {"slug": slug, "error": "too_few_thumbs", "n": len(srcs)}

    # Corpus prior shirt numbers keyed by thumb filename
    corpus_num: dict[str, tuple[int, float]] = {}
    players_path = CORPUS / slug / "players.json"
    if players_path.is_file():
        for row in json.loads(players_path.read_text()):
            thumb = row.get("thumb_path") or ""
            name = Path(thumb).name if thumb else ""
            if not name:
                # offline thumbs often kitA_1.jpg
                kit = row.get("kit") or "A"
                sn = row.get("shirt_number")
                if sn is not None:
                    name = f"kit{kit}_{sn}.jpg"
            sn = row.get("shirt_number")
            conf = float(row.get("jersey_confidence") or 0)
            if name and sn is not None and conf >= 0.4:
                corpus_num[name.lower()] = (int(sn), conf)
                corpus_num[Path(name).name.lower()] = (int(sn), conf)

    metas = []
    for p in srcs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        d = jersey_descriptor_bgr(img)
        ocr = read_jersey_number_bgr(img) if jersey_ocr_enabled() else None
        prior = corpus_num.get(p.name.lower())
        metas.append(
            {"jersey_hsv": d, "img": img, "src": p.name, "ocr": ocr, "prior": prior}
        )

    assign_kits_from_descriptors(metas)
    purity = kit_purity(metas)
    by = {"A": [], "B": [], "X": []}
    for m in metas:
        by.setdefault(m.get("kit") or "?", []).append(m)
    for k in by:
        by[k].sort(key=lambda m: (-(m["jersey_hsv"] or {}).get("redness", 0), m["src"]))

    for p in gal.glob("Kit*.jpg"):
        p.unlink()

    rows: dict[str, list] = {"A": [], "B": [], "X": []}
    ocr_hits = 0
    two_digit = 0
    used_nums: dict[str, set[int]] = {"A": set(), "B": set(), "X": set()}

    for kit, items in by.items():
        if kit not in rows:
            continue
        for i, m in enumerate(items, 1):
            d = m["jersey_hsv"] or {}
            shirt = None
            jconf = 0.0
            ocr = m.get("ocr")
            prior = m.get("prior")
            cand: list[tuple[int, float, str]] = []
            if ocr and float(ocr.get("confidence") or 0) >= 0.4:
                cand.append((int(ocr["number"]), float(ocr["confidence"]), "ocr"))
            # Corpus priors: only trust 2-digit (legacy often stored truncated #21→2)
            if prior and int(prior[0]) >= 10:
                cand.append((int(prior[0]), float(prior[1]), "corpus"))
            if cand:
                # 2-digit ALWAYS beats 1-digit
                cand.sort(key=lambda t: (1 if t[0] >= 10 else 0, t[1]), reverse=True)
                num, conf, _src = cand[0]
                if num not in used_nums.get(kit, set()):
                    shirt = num
                    jconf = conf
                    used_nums.setdefault(kit, set()).add(num)
                    ocr_hits += 1
                    if num >= 10:
                        two_digit += 1
            name = f"Kit{kit}_{i}.jpg"
            cv2.imwrite(str(gal / name), m["img"])
            rows[kit].append(
                {
                    "file": name,
                    "src": m["src"],
                    "shirt": shirt,
                    "ocr_conf": round(jconf, 3) if shirt else None,
                    "redness": round(float(d.get("redness") or 0), 3),
                    "darkness": round(float(d.get("darkness") or 0), 3),
                    "hue": round(float(d.get("hue") or 0), 1),
                    "sat": round(float(d.get("sat") or 0), 3),
                    "val": round(float(d.get("val") or 0), 3),
                }
            )

    na, nb, nx = len(rows["A"]), len(rows["B"]), len(rows["X"])
    ma = sum(r["redness"] for r in rows["A"]) / max(na, 1)
    mb = sum(r["redness"] for r in rows["B"]) / max(nb, 1)
    # Honest gate: purity (not mean-sep). Sep kept for info only.
    import numpy as np

    def _mean_vec(kit_rows):
        if not kit_rows:
            return None
        hs = [r["hue"] * (np.pi / 90.0) for r in kit_rows]
        s = float(sum(r["sat"] for r in kit_rows) / len(kit_rows))
        br = float(sum(r["val"] for r in kit_rows) / len(kit_rows))
        dk = float(sum(r["darkness"] for r in kit_rows) / len(kit_rows))
        return [s * float(np.cos(sum(hs) / len(hs))), s * float(np.sin(sum(hs) / len(hs))), br, dk]

    va, vb = _mean_vec(rows["A"]), _mean_vec(rows["B"])
    sep = 0.0
    if va and vb:
        sep = sum((a - b) ** 2 for a, b in zip(va, vb)) ** 0.5

    verdict = "PASS" if purity.get("pass") else "FAIL"

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

    issues = "; ".join(purity.get("issues") or []) or "none"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Phase2 — {slug}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#eee;margin:24px}}
.meta{{opacity:.85;max-width:960px;line-height:1.45}}
grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}}
figure{{margin:0;background:#1a1a1a;border-radius:8px;padding:8px;border:2px solid #333}}
figure.kitA{{border-color:#ef4444}}figure.kitB{{border-color:#94a3b8}}figure.kitX{{border-color:#f59e0b}}
img{{width:100%;aspect-ratio:3/4;object-fit:cover}}
figcaption{{font-size:11px;margin-top:6px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}}
</style></head><body>
<h1>Phase 2 — {slug}</h1>
<p class="meta">Purity gate (not mean-sep). Outliers (ref/neon GK) → Kit X.<br/>
A={na} B={nb} X={nx} · sep={sep:.3f} · OCR={ocr_hits} (2-digit={two_digit}) ·
<span class="{'ok' if verdict=='PASS' else 'bad'}">{verdict}</span><br/>
issues: {issues}</p>
<h2>Kit A ({na})</h2><grid>{''.join(fig('A', r) for r in rows['A'])}</grid>
<h2>Kit B ({nb})</h2><grid>{''.join(fig('B', r) for r in rows['B'])}</grid>
<h2>Kit X — outliers / ref / neon ({nx})</h2><grid>{''.join(fig('X', r) for r in rows['X'])}</grid>
</body></html>"""
    (gal / "index.html").write_text(html)
    players = []
    for kit in ("A", "B", "X"):
        for r in rows[kit]:
            players.append(
                {
                    "kit": kit,
                    "shirt": r["shirt"],
                    "snapshot": r["file"],
                    "jersey_confidence": r["ocr_conf"] or 0.0,
                    "ocr_likely": r["shirt"] is not None,
                    "label": f"Kit {kit} #{r['shirt']}" if r["shirt"] is not None else f"Kit {kit}",
                }
            )
    (gal / "players.json").write_text(json.dumps(players, indent=2))

    report = {
        "slug": slug,
        "A": na,
        "B": nb,
        "X": nx,
        "sep": round(sep, 3),
        "mean_redness_A": round(ma, 3),
        "mean_redness_B": round(mb, 3),
        "ocr_hits": ocr_hits,
        "ocr_two_digit": two_digit,
        "ocr_enabled": jersey_ocr_enabled(),
        "verdict": verdict,
        "purity": purity,
        "thumbs": len(metas),
    }
    print(
        f"  {slug}: A={na} B={nb} X={nx} ocr={ocr_hits} "
        f"2dig={two_digit} {verdict} {issues}"
    )
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    print("Phase 2 identity loop")
    for slug in SLUGS:
        reports.append(_rebuild_slug(slug))

    passed = sum(1 for r in reports if r.get("verdict") == "PASS")
    ocr_on = any(r.get("ocr_enabled") for r in reports)
    summary = {
        "videos": len(reports),
        "kit_pass": passed,
        "ocr_enabled": ocr_on,
        "reports": reports,
    }
    (OUT / "PHASE2_REPORT.json").write_text(json.dumps(summary, indent=2))

    # index
    cards = []
    for r in reports:
        if r.get("error"):
            cards.append(f"<li>{r['slug']}: ERROR {r['error']}</li>")
            continue
        cls = "ok" if r["verdict"] == "PASS" else "bad"
        cards.append(
            f"<li><a href=\"{r['slug']}/index.html\">{r['slug']}</a> — "
            f"<span class=\"{cls}\">{r['verdict']}</span> "
            f"A={r['A']} B={r['B']} sep={r['sep']} OCR={r['ocr_hits']}</li>"
        )
    (OUT / "index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Phase 2 galleries</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
.ok{{color:#86efac}}.bad{{color:#fca5a5}} a{{color:#93c5fd}}</style></head>
<body><h1>Phase 2 identity galleries</h1>
<p>Kit pass {passed}/{len(reports)} · OCR {'ON' if ocr_on else 'OFF'}</p>
<ul>{''.join(cards)}</ul></body></html>"""
    )
    print(f"\nWrote {OUT}/PHASE2_REPORT.json  kit_pass={passed}/{len(reports)}")
    if passed < len(reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
