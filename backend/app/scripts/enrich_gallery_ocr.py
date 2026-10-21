"""OCR player thumbs in identity gallery (run inside api container — has tesseract)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from app.services.jersey_ocr import jersey_ocr_enabled, read_jersey_number_bgr


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/data/eval/identity_gallery")
    print("ocr_enabled", jersey_ocr_enabled(), "root", root)
    import cv2

    summary = []
    for slug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        players_path = slug_dir / "players.json"
        if not players_path.exists():
            continue
        rows = json.loads(players_path.read_text())
        used = {"A": set(), "B": set()}
        for r in rows:
            if r.get("shirt") is not None and r.get("ocr_likely"):
                used[r["kit"]].add(int(r["shirt"]))
        ocr_n = 0
        new_rows = []
        for r in rows:
            kit = r["kit"]
            shirt = r.get("shirt")
            snap = slug_dir / (r.get("snapshot") or "")
            jconf = float(r.get("jersey_confidence") or 0)
            ocr_likely = bool(r.get("ocr_likely"))
            if snap.is_file() and not ocr_likely:
                img = cv2.imread(str(snap))
                hit = read_jersey_number_bgr(img) if img is not None else None
                if hit and float(hit["confidence"]) >= 0.5:
                    num = int(hit["number"])
                    if num not in used[kit]:
                        # rename snapshot file to new number
                        new_name = f"Kit{kit}_{num}.jpg"
                        new_path = slug_dir / new_name
                        if snap.name != new_name:
                            shutil.move(str(snap), str(new_path))
                            snap = new_path
                        shirt = num
                        jconf = float(hit["confidence"])
                        ocr_likely = True
                        used[kit].add(num)
                        print(f"  {slug_dir.name}: {r['label']} → Kit {kit} #{num}")
            if ocr_likely:
                ocr_n += 1
            new_rows.append(
                {
                    **r,
                    "label": f"Kit {kit} #{shirt}",
                    "shirt": shirt,
                    "jersey_confidence": round(jconf, 3),
                    "snapshot": snap.name if snap.is_file() else r.get("snapshot"),
                    "ocr_likely": ocr_likely,
                }
            )
        players_path.write_text(json.dumps(new_rows, indent=2))
        # rewrite index.html cards lightly
        summary.append({"slug": slug_dir.name, "ocr": ocr_n, "players": len(new_rows)})
        print(f"{slug_dir.name}: OCR {ocr_n}/{len(new_rows)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
