"""PadelTracker100 data prep (CPU): download match videos + labels, build YOLO datasets.

Runs as the CPU Kaggle kernel `padel-pt100-prep` (output is mounted by the GPU kernel)
and locally:  python kaggle/prep/padel_prep.py --out data/pt100_prep [--skip-frames]

Output layout (under --out):
  videos/<match>.mp4                       full 1920x1080@30 broadcast main view
  labels/<match>_{ball,pose,homography}.json, <match>_shots.csv
  yolo/images/<match>/<frame>.jpg          sampled frames (shared by ball + pose sets)
  yolo/ball_labels/<match>/<frame>.txt     class 0 = ball (min 12 px box)
  yolo/pose_labels/<match>/<frame>.txt     class 0 = player, 17 COCO keypoints
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

V1_ZIP = "https://zenodo.org/records/14653706/files/padel-data-labels.zip?download=1"
V2_LABELS = "https://zenodo.org/records/17020011/files/labels.zip?download=1"
MATCHES = ["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"]
CHUNK = 32 * 1024 * 1024


def _pip(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "padel-prep/1.0"
    return s


def _get_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    import time

    for attempt in range(retries):
        try:
            r = _session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            if len(r.content) == end - start + 1:
                return r.content
        except Exception as exc:  # noqa: BLE001 - retried
            print(f"  range {start}-{end} attempt {attempt}: {exc}", flush=True)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed")


def fetch_member(url: str, info: zipfile.ZipInfo, dest: Path, workers: int = 12) -> None:
    """Parallel HTTP-range download of one deflated zip member, inflated on the fly."""
    if dest.exists() and dest.stat().st_size == info.file_size:
        print(f"have {dest.name}", flush=True)
        return
    hdr = _get_range(url, info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len
    ranges = [
        (s, min(s + CHUNK, info.compress_size) - 1)
        for s in range(0, info.compress_size, CHUNK)
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    inflate = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    with open(tmp, "wb") as out, cf.ThreadPoolExecutor(workers) as pool:
        # keep a bounded window of in-flight chunks so memory stays ~workers*CHUNK
        futs: dict[int, cf.Future] = {}
        nxt = 0
        for i in range(len(ranges)):
            while nxt < len(ranges) and nxt < i + workers * 2:
                a, b = ranges[nxt]
                futs[nxt] = pool.submit(_get_range, url, data_start + a, data_start + b)
                nxt += 1
            raw = futs.pop(i).result()
            out.write(inflate.decompress(raw) if inflate else raw)
            if i % 10 == 0:
                print(f"  {dest.name}: {100 * (i + 1) // len(ranges)}%", flush=True)
        if inflate:
            out.write(inflate.flush())
    tmp.rename(dest)
    assert dest.stat().st_size == info.file_size, "size mismatch"
    print(f"got {dest.name} ({info.file_size / 1e9:.2f} GB)", flush=True)


def download(out: Path) -> None:
    from remotezip import RemoteZip

    with RemoteZip(V1_ZIP) as z:
        infos = {i.filename: i for i in z.infolist()}
    for m in MATCHES:
        fetch_member(V1_ZIP, infos[f"{m}.mp4"], out / "videos" / f"{m}.mp4")
    lab = out / "labels"
    if not (lab / f"{MATCHES[1]}_homography.json").exists():
        buf = io.BytesIO(_session().get(V2_LABELS, timeout=600).content)
        with zipfile.ZipFile(buf) as z:
            for n in z.namelist():
                if n.endswith((".json", ".csv")):
                    (lab / Path(n).name).parent.mkdir(parents=True, exist_ok=True)
                    (lab / Path(n).name).write_bytes(z.read(n))
    print("labels:", sorted(p.name for p in lab.iterdir()), flush=True)


def _frame_idx(file_name: str) -> int:
    return int(file_name.split("_")[1].split(".")[0])


def build_yolo(out: Path, ball_step: dict[str, int], neg_every: int, pose_every: int) -> None:
    import cv2

    for m in MATCHES:
        ball = json.loads((out / "labels" / f"{m}_ball.json").read_text())
        pose = json.loads((out / "labels" / f"{m}_pose.json").read_text())
        W, H = 1920.0, 1080.0
        img2f = {im["id"]: _frame_idx(im["file_name"]) for im in ball["images"]}
        balls = {img2f[a["image_id"]]: a["bbox"] for a in ball["annotations"] if a["category_id"] == 1}
        pimg2f = {im["id"]: _frame_idx(im["file_name"]) for im in pose["images"]}
        players: dict[int, list] = {}
        for a in pose["annotations"]:
            players.setdefault(pimg2f[a["image_id"]], []).append(a)
        n_frames = len(ball["images"])
        ball_frames = sorted(balls)[:: ball_step[m]]
        neg = [f for f in range(0, n_frames, neg_every) if f not in balls]
        pose_set = set(ball_frames[::pose_every]) | set(neg[::2])
        want = sorted(set(ball_frames) | set(neg))
        print(f"{m}: {len(ball_frames)} ball + {len(neg)} neg frames, {len(pose_set)} pose", flush=True)

        idir = out / "yolo" / "images" / m
        bdir = out / "yolo" / "ball_labels" / m
        pdir = out / "yolo" / "pose_labels" / m
        for d in (idir, bdir, pdir):
            d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(out / "videos" / f"{m}.mp4"))
        wanted = set(want)
        f = 0
        last = want[-1]
        while f <= last:
            ok = cap.grab()
            if not ok:
                break
            if f in wanted:
                _, frame = cap.retrieve()
                stem = f"{m}_{f:06d}"
                cv2.imwrite(str(idir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                if f in balls:
                    x, y, w, h = balls[f]
                    cx, cy = x + w / 2, y + h / 2
                    w, h = max(w, 12.0), max(h, 12.0)
                    lines.append(f"0 {cx / W:.6f} {cy / H:.6f} {w / W:.6f} {h / H:.6f}")
                (bdir / f"{stem}.txt").write_text("\n".join(lines))
                if f in pose_set:
                    pl = []
                    for a in players.get(f, []):
                        x, y, w, h = a["bbox"]
                        kp = a["keypoints"]
                        k = " ".join(
                            f"{kp[i] / W:.6f} {kp[i + 1] / H:.6f} {int(kp[i + 2])}" for i in range(0, 51, 3)
                        )
                        pl.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f} {k}")
                    (pdir / f"{stem}.txt").write_text("\n".join(pl))
            if f % 5000 == 0:
                print(f"  {m} frame {f}", flush=True)
            f += 1
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/pt100")
    ap.add_argument("--skip-frames", action="store_true")
    args = ap.parse_args()
    _pip("remotezip", "requests")
    out = Path(args.out)
    download(out)
    if not args.skip_frames:
        # F: ball visible in ~82% of frames, M: ~36% -> different strides for similar set sizes
        build_yolo(out, {MATCHES[0]: 4, MATCHES[1]: 3}, neg_every=40, pose_every=2)
    os.system(f"du -sh {out}/* {out}/yolo/*")


if __name__ == "__main__":
    main()
