#!/usr/bin/env python3
"""Phase 2 identity gallery from run_offline outputs (Kaggle or local).

Shows what the product's identity stage proposed — one card per player with its
best crop, kit, shirt and the cameras it was linked across — for manual kit/number
confirmation. No identity logic lives here; re-run identity with
`python -m app.scripts.run_offline --replay <run>` after changing it.

  python scripts/eval/build_identity_gallery.py data/kaggle_out/gpu_eval/eval
  # → data/eval/identity_gallery/<slug>/index.html, index.html, summary.json
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "identity_gallery"

CSS = """body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}
h1{font-size:1.2rem}.meta{opacity:.8;margin-bottom:1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}
figure{margin:0;background:#1c1c1c;border-radius:8px;overflow:hidden;padding:8px}
img{width:100%;aspect-ratio:1;object-fit:cover;background:#000}
figcaption{font-size:12px;margin-top:6px;line-height:1.35}
.missing{height:160px;display:grid;place-items:center;opacity:.5}"""


def _best_thumb(run: Path, keys: list[str], dets: dict[str, dict]) -> Path | None:
    best, score = None, -1.0
    for k in keys:
        feats = (dets.get(k) or {}).get("features") or {}
        path = feats.get("thumb_path")
        if not path:
            continue
        p = Path(path) if Path(path).is_absolute() else run / path
        s = float(feats.get("thumb_score") or 0)
        if p.is_file() and s > score:
            best, score = p, s
    return best


def run_one(run: Path) -> dict:
    slug = run.name
    players = json.loads((run / "players.json").read_text())
    dets = {d["track_key"]: d for d in json.loads((run / "detections.json").read_text())}
    dest = OUT / slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    cards, kits = [], Counter()
    for i, p in enumerate(sorted(players, key=lambda p: (p.get("team") or "", p.get("shirt_number") or 0))):
        kits[p.get("team") or "?"] += 1
        thumb = _best_thumb(run, p["track_keys"], dets)
        img = '<div class="missing">no crop</div>'
        if thumb:
            name = f"{i:02d}_{thumb.name}"
            shutil.copy2(thumb, dest / name)
            img = f'<img src="{name}" alt="" />'
        ocr = [dets[k]["features"].get("jersey_number") for k in p["track_keys"] if k in dets]
        ocr_hits = sum(1 for n in ocr if n is not None)
        cards.append(
            f"<figure>{img}<figcaption><b>{html.escape(p['display_name'])}</b><br/>"
            f"cams {'+'.join(p['cameras'])} · tracks {len(p['track_keys'])} · OCR tracks {ocr_hits}"
            "</figcaption></figure>"
        )
    both = kits.get("Kit A", 0) >= 1 and kits.get("Kit B", 0) >= 1
    (dest / "index.html").write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Identity — {slug}</title>"
        f"<style>{CSS}</style></head><body><h1>{slug}</h1>"
        f"<p class='meta'>{' · '.join(f'{k}: {v}' for k, v in sorted(kits.items()))}<br/>"
        "Confirm: one kit colour per Kit letter; visible shirt numbers match labels; "
        "multi-cam players show more than one camera.</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    return {"slug": slug, "kits": dict(kits), "players": len(players), "both_kits": both,
            "gallery": str(dest / "index.html")}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    runs = sorted(p.parent for p in root.glob("*/players.json"))
    if not runs:
        raise SystemExit(f"no */players.json under {root} — pull a Kaggle run first")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = [run_one(r) for r in runs]
    for s in summaries:
        print(f"{s['slug']:40s} {s['kits']} {'OK' if s['both_kits'] else 'FAIL single-kit'}")
    (OUT / "index.html").write_text(
        "<!DOCTYPE html><html><body style='font-family:system-ui;background:#111;color:#eee;padding:24px'>"
        "<h1>CourtLens identity gallery</h1><ul>"
        + "".join(
            f"<li><a href='{s['slug']}/index.html'>{s['slug']}</a> {s['kits']} "
            f"{'OK' if s['both_kits'] else 'FAIL single-kit'}</li>"
            for s in summaries
        )
        + "</ul></body></html>"
    )
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("INDEX", OUT / "index.html")
    if not all(s["both_kits"] for s in summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
