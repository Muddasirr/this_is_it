"""Run the Kaggle CLI / API as the padel Kaggle account (KAGGLE_API_KEY_LSECOND in ../.env).

The futsal pipeline uses the primary account in ~/.kaggle; padel never touches it:
an empty KAGGLE_CONFIG_DIR + KAGGLE_API_TOKEN pins every call to the second account.

  python kaggle/kaggle_cli.py push prep          # push kaggle/prep as a kernel
  python kaggle/kaggle_cli.py push gpu
  python kaggle/kaggle_cli.py status gpu
  python kaggle/kaggle_cli.py output gpu data/kaggle_out/gpu
  python kaggle/kaggle_cli.py code                # upload padel_analyzer/ + scripts/ as dataset padel-code
  python kaggle/kaggle_cli.py push pose           # pose models + held-out caches + downstream eval
  python kaggle/kaggle_cli.py raw kernels list --mine
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ENV = ROOT.parent.parent / ".env"
KEY_NAMES = ("KAGGLE_API_KEY_LSECOND", "KAGGLE_API_KEY_SECOND")
USER = "hadivenix"

KERNELS = {
    "prep": {
        "slug": "padel-pt100-prep",
        "code_file": "padel_prep.py",
        "enable_gpu": False,
        "kernel_sources": [],
    },
    "gpu": {
        "slug": "padel-gpu-train",
        "code_file": "padel_gpu.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep"],
    },
    "pose": {
        "slug": "padel-gpu-pose",
        "code_file": "padel_pose.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-train"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "stroke": {
        "slug": "padel-stroke-rgb",
        "code_file": "padel_stroke.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "infer": {
        "slug": "padel-infer",
        "code_file": "padel_infer.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-gpu-train", f"{USER}/padel-gpu-pose", f"{USER}/padel-stroke-rgb",
                           f"{USER}/padel-eval", f"{USER}/padel-venues", f"{USER}/padel-fusion",
                           f"{USER}/padel-audio", f"{USER}/padel-points"],
        "dataset_sources": [f"{USER}/padel-code", f"{USER}/padel-jobs"],
    },
    "eval": {
        "slug": "padel-eval",
        "code_file": "padel_eval.py",
        "enable_gpu": False,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-pose", f"{USER}/padel-stroke-rgb"],
        "dataset_sources": [f"{USER}/padel-code"],
        "dir": "evalk",
    },
    "venues": {
        "slug": "padel-venues",
        "code_file": "padel_venues.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-train", f"{USER}/padel-gpu-pose"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "decorte": {
        "slug": "padel-decorte-prep",
        "code_file": "padel_decorte_prep.py",
        "enable_gpu": False,
        "kernel_sources": [],
    },
    "audio": {
        "slug": "padel-audio",
        "code_file": "padel_audio.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-decorte-prep", f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-pose",
                           f"{USER}/padel-venues"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "fusion": {
        "slug": "padel-fusion",
        "code_file": "padel_fusion.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-audio", f"{USER}/padel-decorte-prep", f"{USER}/padel-pt100-prep",
                           f"{USER}/padel-gpu-pose", f"{USER}/padel-venues"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "hitter": {
        "slug": "padel-hitter",
        "code_file": "padel_hitter.py",
        "enable_gpu": False,
        "kernel_sources": [f"{USER}/padel-audio", f"{USER}/padel-decorte-prep", f"{USER}/padel-pt100-prep"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "rallytest": {
        "slug": "padel-rallytest",
        "code_file": "padel_rallytest.py",
        "enable_gpu": False,
        "kernel_sources": [f"{USER}/padel-audio", f"{USER}/padel-decorte-prep", f"{USER}/padel-pt100-prep"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "scoreboard": {
        "slug": "padel-scoreboard",
        "code_file": "padel_scoreboard.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep"],
    },
    "points": {
        "slug": "padel-points",
        "code_file": "padel_points.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-train", f"{USER}/padel-gpu-pose",
                           f"{USER}/padel-stroke-rgb", f"{USER}/padel-eval", f"{USER}/padel-fusion",
                           f"{USER}/padel-scoreboard"],
        "dataset_sources": [f"{USER}/padel-code", f"{USER}/padel-kuwait-small"],
    },
    "tests": {
        "slug": "padel-tests",
        "code_file": "padel_tests.py",
        "enable_gpu": False,
        "kernel_sources": [],
        "dataset_sources": [f"{USER}/padel-code"],
        "dir": "testsk",
    },
    "ballspeed": {
        "slug": "padel-ballspeed",
        "code_file": "padel_ballspeed.py",
        "enable_gpu": False,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-pose"],
        "dataset_sources": [f"{USER}/padel-code", f"{USER}/padel-kuwait-small"],
        "dir": "ballspeedk",
    },
    "identity": {
        "slug": "padel-identity",
        "code_file": "padel_identity.py",
        "enable_gpu": True,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-pose", f"{USER}/padel-stroke-rgb",
                           f"{USER}/padel-eval", f"{USER}/padel-fusion", f"{USER}/padel-scoreboard"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
    "idlab": {
        "slug": "padel-idlab",
        "code_file": "padel_idlab.py",
        "enable_gpu": False,
        "kernel_sources": [f"{USER}/padel-pt100-prep", f"{USER}/padel-gpu-pose", f"{USER}/padel-identity"],
        "dataset_sources": [f"{USER}/padel-code"],
    },
}
CODE_DATASET = "padel-code"
JOBS_DATASET = "padel-jobs"


def padel_env() -> dict[str, str]:
    token = next((os.environ[k] for k in KEY_NAMES if os.environ.get(k)), None)
    for line in (REPO_ENV.read_text().splitlines() if REPO_ENV.exists() and not token else []):
        k, _, v = line.partition("=")
        if k.strip() in KEY_NAMES and v.strip():
            token = v.strip()
    if not token:
        sys.exit(f"no {KEY_NAMES[0]} in {REPO_ENV}")
    cfg = Path(os.environ.get("PADEL_KAGGLE_CFG", ROOT / ".kaggle_cfg"))
    cfg.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(KAGGLE_API_TOKEN=token, KAGGLE_CONFIG_DIR=str(cfg))
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    return env


def write_metadata(name: str) -> Path:
    k = KERNELS[name]
    d = ROOT / k.get("dir", name)
    meta = {
        "id": f"{USER}/{k['slug']}",
        "title": k["slug"],
        "code_file": k["code_file"],
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": k["enable_gpu"],
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": k.get("dataset_sources", []),
        "competition_sources": [],
        "kernel_sources": k["kernel_sources"],
        "model_sources": [],
    }
    if k["enable_gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return d


def push_code() -> int:
    """Bundle the analyzer code (no data, no weights) and create / version the padel-code dataset."""
    pkg = ROOT.parent
    out = ROOT / ".code_bundle"
    if out.exists():
        shutil.rmtree(out)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "data", "*.pt")
    for d in ("padel_analyzer", "scripts", "tests"):
        shutil.copytree(pkg / d, out / d, ignore=ignore)
    (out / "kaggle" / "gpu").mkdir(parents=True)
    shutil.copy(ROOT / "gpu" / "padel_gpu.py", out / "kaggle" / "gpu" / "padel_gpu.py")
    (out / "kaggle" / "audio").mkdir(parents=True)
    shutil.copy(ROOT / "audio" / "padel_audio.py", out / "kaggle" / "audio" / "padel_audio.py")
    links = ROOT / "venues" / "roboflow_links.json"
    if links.exists():
        (out / "kaggle" / "venues").mkdir(parents=True)
        shutil.copy(links, out / "kaggle" / "venues" / links.name)
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": CODE_DATASET, "id": f"{USER}/{CODE_DATASET}", "licenses": [{"name": "CC0-1.0"}]}))
    exists = subprocess.call(["kaggle", "datasets", "status", f"{USER}/{CODE_DATASET}"], env=padel_env(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if exists:
        return kaggle("datasets", "version", "-p", str(out), "-m", "update", "-r", "zip", "-q")
    return kaggle("datasets", "create", "-p", str(out), "-r", "zip", "-q")


def push_jobs(folder: Path) -> int:
    """Create / version the padel-jobs dataset from a folder holding job.json + videos."""
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": JOBS_DATASET, "id": f"{USER}/{JOBS_DATASET}", "licenses": [{"name": "other"}]}))
    exists = subprocess.call(["kaggle", "datasets", "status", f"{USER}/{JOBS_DATASET}"], env=padel_env(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if exists:
        return kaggle("datasets", "version", "-p", str(folder), "-m", "job", "-q")
    return kaggle("datasets", "create", "-p", str(folder), "-q")


def dataset_ready(name: str) -> bool:
    r = subprocess.run(["kaggle", "datasets", "status", f"{USER}/{name}"], env=padel_env(), capture_output=True,
                       text=True)
    return "ready" in r.stdout


def dataset_has_file(name: str, filename: str) -> bool:
    """True once the dataset's *current* version lists filename ("ready" can still describe the old version)."""
    r = subprocess.run(["kaggle", "datasets", "files", f"{USER}/{name}", "--page-size", "200"], env=padel_env(),
                       capture_output=True, text=True)
    return filename in r.stdout


def kernel_status(name: str) -> str:
    r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/{KERNELS[name]['slug']}"], env=padel_env(),
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out:
            return s.lower()
    return "unknown"


def push_folder_dataset(name: str, folder: Path) -> int:
    """Create / version a private dataset from a folder (e.g. the Roboflow exports)."""
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": name, "id": f"{USER}/{name}", "licenses": [{"name": "CC-BY-4.0"}]}))
    exists = subprocess.call(["kaggle", "datasets", "status", f"{USER}/{name}"], env=padel_env(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if exists:
        return kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-r", "zip", "-q")
    return kaggle("datasets", "create", "-p", str(folder), "-r", "zip", "-q")


def kaggle(*args: str) -> int:
    return subprocess.call(["kaggle", *args], env=padel_env())


def download_output(name: str, dest: str, pattern: str | None = None) -> None:
    os.environ.update(padel_env())
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    Path(dest).mkdir(parents=True, exist_ok=True)
    # CLI only fetches 20 files; the API pages through everything
    kw = {"file_pattern": pattern} if pattern else {}
    api.kernels_output(f"{USER}/{KERNELS[name]['slug']}", path=dest, force=True, quiet=True, page_size=200, **kw)


def main() -> None:
    cmd, *rest = sys.argv[1:] or ["help"]
    if cmd == "push":
        sys.exit(kaggle("kernels", "push", "-p", str(write_metadata(rest[0]))))
    if cmd == "status":
        sys.exit(kaggle("kernels", "status", f"{USER}/{KERNELS[rest[0]]['slug']}"))
    if cmd == "output":
        download_output(rest[0], rest[1], rest[2] if len(rest) > 2 else None)
        return
    if cmd == "dataset":
        sys.exit(push_folder_dataset(rest[0], Path(rest[1])))
    if cmd == "code":
        sys.exit(push_code())
    if cmd == "raw":
        sys.exit(kaggle(*rest))
    print(__doc__)


if __name__ == "__main__":
    main()
