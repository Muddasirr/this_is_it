#!/usr/bin/env python3
"""Phase 2 identity-only run on one corpus detections folder.

  python scripts/eval/run_identity_one.py bangladesh_pakistan_saff_2026
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = ROOT / "scripts" / "kaggle" / "courtlens_offline"
CORPUS = ROOT / "data" / "kaggle_out" / "corpus_v26" / "corpus_out"
OUT_ROOT = ROOT / "data" / "eval" / "identity_gallery"

sys.path.insert(0, str(OFFLINE))

from identity_offline import (  # noqa: E402
    _kit_lightness,
    detections_to_tracks,
    propose_players,
)
from vision import Detection  # noqa: E402


def load_dets(path: Path) -> list[Detection]:
    raw = json.loads(path.read_text())
    out = []
    for d in raw:
        feats = dict(d.get("features") or {})
        feats.setdefault("class", d.get("class_name", "person"))
        series = d.get("bbox_series") or feats.get("bbox_series") or []
        feats["bbox_series"] = series
        out.append(
            Detection(
                track_key=d["track_key"],
                camera_label=d.get("camera_label") or "A",
                start_ms=int(d["start_ms"]),
                end_ms=int(d["end_ms"]),
                confidence=float(d.get("confidence") or 0.5),
                class_name=d.get("class_name") or "person",
                features=feats,
                bbox_series=series,
            )
        )
    return out


def dominant_hue_sat(hist: list[float] | None) -> tuple[float, float]:
    if not hist or len(hist) < 64:
        return 3.5, 0.0
    # Find peak H×S bin among mid/high sat
    best_i, best_v = 0, -1.0
    for i, v in enumerate(hist[:64]):
        s_bin = i % 8
        if s_bin < 2:
            continue
        if float(v) > best_v:
            best_v = float(v)
            best_i = i
    return best_i / 8.0, best_i % 8


def main() -> None:
    slug = sys.argv[1] if len(sys.argv) > 1 else "bangladesh_pakistan_saff_2026"
    src = CORPUS / slug
    if not (src / "detections.json").exists():
        raise SystemExit(f"missing {src}/detections.json")

    dets = load_dets(src / "detections.json")
    tracks = detections_to_tracks(dets)
    players, _ = propose_players(tracks)

    dest = OUT_ROOT / f"{slug}_fresh"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    thumbs_src = src / "thumbs"

    rows = []
    for p in players:
        light = _kit_lightness(p.color_hist)
        hue, sat = dominant_hue_sat(p.color_hist)
        src_thumb = None
        if p.thumb_path and Path(p.thumb_path).is_file():
            src_thumb = Path(p.thumb_path)
        elif thumbs_src.is_dir():
            cand = thumbs_src / f"{p.player_key}.jpg"
            if cand.is_file():
                src_thumb = cand
            # also try kitA_1 style
            alt = thumbs_src / f"kit{p.kit}_{p.shirt_number}.jpg"
            if not src_thumb and alt.is_file():
                src_thumb = alt
        snap_name = f"Kit{p.kit}_{p.shirt_number}.jpg"
        snap = dest / snap_name
        if src_thumb:
            shutil.copy2(src_thumb, snap)
        rows.append(
            {
                "label": p.display_name,
                "kit": p.kit,
                "shirt": p.shirt_number,
                "jersey_confidence": round(float(p.jersey_confidence or 0), 3),
                "tracks": len(p.track_keys),
                "lightness": round(light, 3),
                "peak_hue_bin": round(hue, 2),
                "peak_sat_bin": int(sat),
                "snapshot": snap_name if snap.exists() else None,
                "ocr_likely": float(p.jersey_confidence or 0) >= 0.45,
            }
        )

    kits = Counter(r["kit"] for r in rows)
    cards = []
    for r in sorted(rows, key=lambda x: (x["kit"], x["shirt"] or 0)):
        img = (
            f'<img src="{r["snapshot"]}" alt="{r["label"]}"/>'
            if r["snapshot"]
            else '<div class="missing">no snap</div>'
        )
        ocr = "OCR" if r["ocr_likely"] else "slot"
        cards.append(
            f'<figure class="kit{r["kit"]}">{img}'
            f'<figcaption><b>{r["label"]}</b><br/>{ocr} conf={r["jersey_confidence"]} '
            f'tracks={r["tracks"]}<br/>light={r["lightness"]} hue={r["peak_hue_bin"]} '
            f'sat={r["peak_sat_bin"]}</figcaption></figure>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Identity — {slug}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#eee;margin:24px}}
.meta{{opacity:.85;max-width:720px;line-height:1.45}}
grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-top:16px}}
figure{{margin:0;background:#1a1a1a;border-radius:8px;padding:8px;border:2px solid #333}}
figure.kitA{{border-color:#3b82f6}}
figure.kitB{{border-color:#ef4444}}
img{{width:100%;aspect-ratio:1;object-fit:cover;background:#000}}
figcaption{{font-size:12px;margin-top:6px;line-height:1.35}}
.missing{{height:160px;display:grid;place-items:center;opacity:.4}}
h2{{margin-top:28px;font-size:1rem}}
</style></head><body>
<h1>Identity test — {slug}</h1>
<p class="meta">
Method: YOLO person → upper-body 8×8 HSV hist → cluster tracks → 2-means → Kit A/B.<br/>
<b>Kit A: {kits.get('A',0)}</b> (blue border) · <b>Kit B: {kits.get('B',0)}</b> (red border) ·
OCR-likely: {sum(1 for r in rows if r['ocr_likely'])}/{len(rows)}
</p>
<p class="meta">Check: all same-colour jerseys should share one border colour.
Jackets/refs should not dominate a kit.</p>
<h2>Kit A</h2>
<grid>{''.join(c for c,r in zip(cards, sorted(rows, key=lambda x: (x['kit'], x['shirt'] or 0))) if r['kit']=='A')}</grid>
<h2>Kit B</h2>
<grid>{''.join(c for c,r in zip(cards, sorted(rows, key=lambda x: (x['kit'], x['shirt'] or 0))) if r['kit']=='B')}</grid>
</body></html>
"""
    # Fix cards zip - rebuild properly
    by_kit = {"A": [], "B": []}
    for r in sorted(rows, key=lambda x: (x["kit"], x["shirt"] or 0)):
        img = (
            f'<img src="{r["snapshot"]}" alt="{r["label"]}"/>'
            if r["snapshot"]
            else '<div class="missing">no snap</div>'
        )
        ocr = "OCR" if r["ocr_likely"] else "slot"
        card = (
            f'<figure class="kit{r["kit"]}">{img}'
            f'<figcaption><b>{r["label"]}</b><br/>{ocr} conf={r["jersey_confidence"]} '
            f'tracks={r["tracks"]}<br/>light={r["lightness"]} hue={r["peak_hue_bin"]} '
            f'sat={r["peak_sat_bin"]}</figcaption></figure>'
        )
        by_kit[r["kit"]].append(card)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Identity — {slug}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#eee;margin:24px}}
.meta{{opacity:.85;max-width:760px;line-height:1.45}}
grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-top:12px}}
figure{{margin:0;background:#1a1a1a;border-radius:8px;padding:8px;border:2px solid #333}}
figure.kitA{{border-color:#3b82f6}}
figure.kitB{{border-color:#ef4444}}
img{{width:100%;aspect-ratio:1;object-fit:cover;background:#000}}
figcaption{{font-size:12px;margin-top:6px;line-height:1.35}}
.missing{{height:160px;display:grid;place-items:center;opacity:.4}}
h2{{margin-top:28px;font-size:1.05rem}}
</style></head><body>
<h1>Identity test — {slug}</h1>
<p class="meta">
Method: YOLO person → upper-body 8×8 HSV hist → cluster tracks → 2-means → Kit A/B.<br/>
<b>Kit A: {kits.get('A',0)}</b> (blue border) · <b>Kit B: {kits.get('B',0)}</b> (red border) ·
OCR-likely: {sum(1 for r in rows if r['ocr_likely'])}/{len(rows)}
</p>
<p class="meta">Pass if: one kit letter = one jersey colour. Fail if green and white (or red/green)
mix inside the same section.</p>
<h2>Kit A ({kits.get('A',0)})</h2>
<grid>{''.join(by_kit['A'])}</grid>
<h2>Kit B ({kits.get('B',0)})</h2>
<grid>{''.join(by_kit['B'])}</grid>
</body></html>
"""
    (dest / "index.html").write_text(html)
    (dest / "players.json").write_text(json.dumps(rows, indent=2))
    report = {
        "slug": slug,
        "kit_a": kits.get("A", 0),
        "kit_b": kits.get("B", 0),
        "players": len(rows),
        "ocr_likely": sum(1 for r in rows if r["ocr_likely"]),
        "gallery": str(dest / "index.html"),
        "by_player": rows,
    }
    (dest / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "by_player"}, indent=2))
    print("GALLERY", f"file://{dest / 'index.html'}")
    print("--- players ---")
    for r in sorted(rows, key=lambda x: (x["kit"], x["shirt"] or 0)):
        print(
            f"  {r['label']:12s} light={r['lightness']:.2f} hue={r['peak_hue_bin']:.1f} "
            f"sat={r['peak_sat_bin']} tracks={r['tracks']:3d} "
            f"{'OCR' if r['ocr_likely'] else 'slot'} snap={bool(r['snapshot'])}"
        )


if __name__ == "__main__":
    main()
