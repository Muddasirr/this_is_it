#!/usr/bin/env python3
"""Phase 2: rebuild kit/number identity for every corpus video + snapshot gallery.

Writes:
  data/eval/identity_gallery/<slug>/
    KitA_17.jpg ...
    index.html   (open in browser to manually confirm)
    players.json

Usage:
  python scripts/eval/build_identity_gallery.py
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
OUT = ROOT / "data" / "eval" / "identity_gallery"

sys.path.insert(0, str(OFFLINE))

from identity_offline import detections_to_tracks, propose_players  # noqa: E402
from jersey_ocr import read_jersey_number_bgr  # noqa: E402
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


def run_one(slug: str, det_path: Path) -> dict:
    dets = load_dets(det_path)
    tracks = detections_to_tracks(dets)
    players, _ttp = propose_players(tracks)
    dest = OUT / slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    thumbs_src = det_path.parent / "thumbs"

    rows = []
    ocr_n = 0
    used_shirts: dict[str, set[int]] = {"A": set(), "B": set()}
    # First pass: resolve thumbs + optional OCR-on-thumb to catch visible numbers (e.g. 17)
    enriched = []
    for p in players:
        src = None
        if p.thumb_path and Path(p.thumb_path).is_file():
            src = Path(p.thumb_path)
        elif thumbs_src.is_dir():
            cand = thumbs_src / f"{p.player_key}.jpg"
            if cand.is_file():
                src = cand
        shirt = p.shirt_number
        jconf = float(p.jersey_confidence or 0)
        ocr_from_thumb = False
        if src and src.is_file() and jconf < 0.45:
            try:
                import cv2

                img = cv2.imread(str(src))
                hit = read_jersey_number_bgr(img) if img is not None else None
                if hit and float(hit["confidence"]) >= 0.5:
                    cand_num = int(hit["number"])
                    if cand_num not in used_shirts[p.kit]:
                        shirt = cand_num
                        jconf = float(hit["confidence"])
                        ocr_from_thumb = True
            except Exception:
                pass
        if shirt is not None:
            used_shirts[p.kit].add(int(shirt))
        enriched.append((p, src, shirt, jconf, ocr_from_thumb))

    for p, src, shirt, jconf, ocr_from_thumb in enriched:
        label = f"Kit{p.kit}_{shirt}"
        snap = dest / f"{label}.jpg"
        if src and src.is_file():
            shutil.copy2(src, snap)
        ocr_likely = ocr_from_thumb or jconf >= 0.45
        if ocr_likely:
            ocr_n += 1
        rows.append(
            {
                "label": f"Kit {p.kit} #{shirt}",
                "kit": p.kit,
                "shirt": shirt,
                "jersey_confidence": round(jconf, 3),
                "tracks": len(p.track_keys),
                "snapshot": snap.name if snap.exists() else None,
                "ocr_likely": ocr_likely,
            }
        )

    # HTML gallery
    cards = []
    for r in rows:
        img = (
            f'<img src="{r["snapshot"]}" alt="{r["label"]}" />'
            if r["snapshot"]
            else '<div class="missing">no snap</div>'
        )
        ocr = "OCR" if r["ocr_likely"] else "slot"
        cards.append(
            f'<figure><a href="{r["snapshot"] or "#"}">{img}</a>'
            f'<figcaption><b>{r["label"]}</b><br/>{ocr} conf={r["jersey_confidence"]} '
            f'tracks={r["tracks"]}</figcaption></figure>'
        )
    kits = Counter(r["kit"] for r in rows)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Identity gallery — {slug}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}}
h1{{font-size:1.2rem}} .meta{{opacity:.8;margin-bottom:1rem}}
grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;border-radius:8px;overflow:hidden;padding:8px}}
img{{width:100%;aspect-ratio:1;object-fit:cover;background:#000}}
figcaption{{font-size:12px;margin-top:6px;line-height:1.35}}
.missing{{height:160px;display:grid;place-items:center;opacity:.5}}
</style></head><body>
<h1>{slug}</h1>
<p class="meta">Kit A: {kits.get('A',0)} · Kit B: {kits.get('B',0)} · OCR-likely numbers: {ocr_n}/{len(rows)}
<br/>Open snapshots and confirm: same kit colour → same Kit letter; visible shirt # → label.</p>
<grid>
{''.join(cards)}
</grid>
</body></html>
"""
    (dest / "index.html").write_text(html)
    (dest / "players.json").write_text(json.dumps(rows, indent=2))
    return {
        "slug": slug,
        "kit_a": kits.get("A", 0),
        "kit_b": kits.get("B", 0),
        "players": len(rows),
        "ocr_likely": ocr_n,
        "gallery": str(dest / "index.html"),
        "both_kits": kits.get("A", 0) >= 1 and kits.get("B", 0) >= 1,
    }


def main() -> None:
    if not CORPUS.is_dir():
        raise SystemExit(f"missing {CORPUS} — pull Kaggle corpus first")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for det in sorted(CORPUS.glob("*/detections.json")):
        slug = det.parent.name
        print("===", slug)
        s = run_one(slug, det)
        summaries.append(s)
        print(
            f"  A={s['kit_a']} B={s['kit_b']} ocr={s['ocr_likely']}/{s['players']} "
            f"both_kits={s['both_kits']}"
        )
        print(f"  gallery: file://{s['gallery']}")

    index = OUT / "index.html"
    links = "".join(
        f'<li><a href="{s["slug"]}/index.html">{s["slug"]}</a> '
        f'A={s["kit_a"]} B={s["kit_b"]} OCR={s["ocr_likely"]}/{s["players"]} '
        f'{"OK" if s["both_kits"] else "FAIL single-kit"}</li>'
        for s in summaries
    )
    index.write_text(
        f"<!DOCTYPE html><html><body style='font-family:system-ui;background:#111;color:#eee;padding:24px'>"
        f"<h1>CourtLens identity gallery</h1><ul>{links}</ul></body></html>"
    )
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("INDEX", index)
    failed = [s for s in summaries if not s["both_kits"]]
    if failed:
        print("PHASE2_KIT_GATE FAIL single-kit:", [s["slug"] for s in failed])
        sys.exit(1)
    print("PHASE2_KIT_GATE PASS (both kits on every video)")


if __name__ == "__main__":
    main()
