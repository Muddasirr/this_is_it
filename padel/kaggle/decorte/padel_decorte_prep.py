"""CPU kernel: fetch the Decorte et al. (CVPRW 2024) padel dataset — 99 rallies, 11 tournaments, 2,377 audio-
annotated hits — from its public imec Nextcloud share, streaming the (on-the-fly, Zip64, data-descriptor) zip.

Keeps rallies + annotations; files > 1.5 GB (the unprocessed full commentary matches) are listed, not kept.
Output: /kaggle/working/decorte/<path in zip>, listing.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

URL = "https://cloud.ilabt.imec.be/index.php/s/TFimLDWno6W9ED3/download"
OUT = Path("/kaggle/working/decorte")
MAX_KEEP = 1.5e9


def main() -> None:
    import shutil
    import urllib.request
    import zipfile

    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/decorte.zip")
    t0 = time.time()
    with urllib.request.urlopen(URL, timeout=600) as r, open(tmp, "wb") as f:
        got = 0
        while True:
            c = r.read(16 * 1024 * 1024)
            if not c:
                break
            f.write(c)
            got += len(c)
            if got % (1024 * 1024 * 1024) < len(c):
                print(f"{got / 1e9:.1f} GB {time.time() - t0:.0f}s", flush=True)
    print("zip", f"{tmp.stat().st_size / 1e9:.2f} GB", f"{time.time() - t0:.0f}s", flush=True)
    listing = []
    with zipfile.ZipFile(tmp) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            keep = i.file_size <= MAX_KEEP
            listing.append({"name": i.filename, "bytes": i.file_size, "kept": keep})
            print("kept" if keep else "SKIP", i.filename, f"{i.file_size / 1e6:.1f} MB", flush=True)
            if keep:
                dst = OUT / i.filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 16 * 1024 * 1024)
    (OUT.parent / "listing.json").write_text(json.dumps(listing, indent=1))
    tmp.unlink()
    print("done", len(listing), "files", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
