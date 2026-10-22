"""Run the real CourtLens pipeline on 1–4 camera videos without Postgres/Redis/API.

Same `process_job` the production worker runs (ingest → sync → detect/track →
identity → events → fuse → stats), backed by a throwaway SQLite DB and local
storage under --out. Used for Kaggle GPU batches so eval numbers come from the
product code, not a fork.

  python -m app.scripts.run_offline --out /kaggle/working/out/test_video_2 \\
      --camera A=/kaggle/input/.../test_video_2.mp4 [--camera B=...] [--title ...]

  # GPU-free re-run of identity → events → fuse → stats on a saved run
  # (tune kit split / events locally against real Kaggle detections):
  python -m app.scripts.run_offline --replay data/kaggle_out/gpu_eval/eval/test_video_2 --out /tmp/re

Outputs (consumed by scripts/import_courtlens_out.py and scripts/eval/*):
  summary.json        vision stats per camera, sync offsets, timings, settings
  detections.json     tracks (bbox_series + features) — importer format
  events.json         per-camera observations of canonical events — importer format
  canonical_events.json  fused events with cameras + participants
  players.json        proposed players (team, shirt, tracks, cameras)
  match_stats.json    build_match_stats output (same as the API)
  thumbs/<camera>/    player crops (paths in detections.json are relative to --out)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_cameras(values: list[str]) -> dict[str, Path]:
    cams: dict[str, Path] = {}
    for raw in values:
        label, sep, path = raw.partition("=")
        if not sep:
            label, path = "ABCD"[len(cams)], raw
        label = label.strip().upper()
        if label not in {"A", "B", "C", "D"}:
            raise SystemExit(f"camera label must be A–D, got {label!r}")
        p = Path(path).expanduser()
        if not p.is_file():
            raise SystemExit(f"camera {label}: {p} not found")
        cams[label] = p
    if not cams:
        raise SystemExit("pass at least one --camera A=/path/video.mp4")
    return cams


def _configure_env(out: Path) -> None:
    """Point settings at SQLite + local storage before any app module is imported."""
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "courtlens.db"
    if db_path.exists():
        db_path.unlink()
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["LOCAL_STORAGE_PATH"] = str(out / "storage")
    os.environ["VISION_THUMB_DIR"] = str(out / "thumbs")
    os.environ.setdefault("PIPELINE_PROXIES", "false")
    os.environ.setdefault("PIPELINE_MAX_CLIPS", "0")
    os.environ.setdefault("MIN_VIDEO_DURATION_S", "20")
    try:
        import torch

        os.environ.setdefault("VISION_DEVICE", "0" if torch.cuda.is_available() else "cpu")
    except ImportError:
        os.environ.setdefault("VISION_DEVICE", "cpu")


def _install_replay(src: Path, out: Path) -> tuple[dict[str, Path], dict]:
    """Swap vision + media for a saved run so only post-vision stages execute.

    Detections come from src/detections.json; probes and sync offsets from
    src/summary.json. Identity, events, fuse and stats are the live code.
    """
    from app.scripts.inject_detections import _load_detections
    from app.services.events import infer_event_candidates
    from app.services.vision import VisionBackend
    from app.workers import pipeline

    summary = json.loads((src / "summary.json").read_text())
    dets = _load_detections(src / "detections.json")
    for d in dets:
        thumb = (d.features or {}).get("thumb_path")
        if thumb and not Path(thumb).is_absolute():
            d.features["thumb_path"] = str(src / thumb)
    probes = (summary.get("stages") or {}).get("ingest", {}).get("probes") or {}
    offsets = {s["camera"]: s for s in summary.get("sync") or []}

    class ReplayBackend(VisionBackend):
        name = "replay"
        version = str(src)

        def __init__(self) -> None:
            self.last_run_stats = {"backend": "replay", "source": str(src)}

        def detect_and_track(self, video_path: str, camera_label: str):
            return [d for d in dets if d.camera_label == camera_label]

        def infer_events(self, detections, camera_label: str):
            return infer_event_candidates(detections, camera_label)

    def probe(path):
        return dict(probes[Path(path).stem])

    def offset(_ref, other, **_kw):
        s = offsets.get(Path(other).stem) or {}
        return int(s.get("offset_ms", 0)), float(s.get("confidence") or 0.0), "replay"

    swaps = {"get_vision_backend": ReplayBackend, "probe_video": probe, "estimate_offset_ms": offset}
    originals = {name: getattr(pipeline, name) for name in swaps}
    for name, fn in swaps.items():
        setattr(pipeline, name, fn)

    stub_dir = out / "_replay"
    stub_dir.mkdir(exist_ok=True)
    cams = {}
    for label in sorted(summary["cameras"]):
        stub = stub_dir / f"{label}.mp4"
        stub.write_bytes(b"")
        cams[label] = stub
    return cams, originals


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", action="append", default=[], help="LABEL=path (A is the reference)")
    ap.add_argument("--replay", type=Path, default=None, help="saved run dir: skip vision, re-run the rest")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    out = args.out.resolve()
    _configure_env(out)
    restore: dict = {}
    if args.replay:
        cams, restore = _install_replay(args.replay.resolve(), out)
    else:
        cams = _parse_cameras(args.camera)
    try:
        return _run(cams, out, args.title)
    finally:
        if restore:
            from app.workers import pipeline

            for name, fn in restore.items():
                setattr(pipeline, name, fn)


def _run(cams: dict[str, Path], out: Path, title: str | None) -> dict:

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.config import get_settings
    from app.core.database import Base
    from app.models import (
        CameraSource,
        CameraSync,
        EventObservation,
        EventParticipant,
        Facility,
        Match,
        MatchEvent,
        Player,
        PlayerTrack,
        ProcessingJob,
        ProcessingRun,
        Team,
        TrackIdentityLink,
        User,
        VideoAsset,
    )
    from app.models.entities import CameraLabel, JobStatus, MatchStatus, VideoKind
    from app.services.match_stats import build_match_stats
    from app.services.storage import object_key_for_original, upload_file
    from app.workers.pipeline import process_job

    get_settings.cache_clear()
    settings = get_settings()
    # Own engine: app.core.database may already be bound to Postgres if imported first
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
    started = time.monotonic()
    try:
        user = User(email="offline@courtlens.local", full_name="Offline runner", hashed_password="!")
        db.add(user)
        db.flush()
        facility = Facility(name="Offline eval")
        db.add(facility)
        db.flush()
        match = Match(
            facility_id=facility.id,
            title=title or cams["A" if "A" in cams else sorted(cams)[0]].stem,
            status=MatchStatus.validating,
            created_by=user.id,
        )
        db.add(match)
        db.flush()
        for label, path in sorted(cams.items()):
            cam = CameraSource(match_id=match.id, label=CameraLabel(label), description=f"Camera {label}")
            db.add(cam)
            db.flush()
            key = object_key_for_original(facility.id, match.id, label, path.name)
            upload_file(path, key)
            db.add(
                VideoAsset(
                    match_id=match.id,
                    camera_source_id=cam.id,
                    kind=VideoKind.original,
                    object_key=key,
                    content_type="video/mp4",
                    byte_size=path.stat().st_size,
                    validation_status="pending",
                )
            )
        job = ProcessingJob(match_id=match.id, status=JobStatus.queued, idempotency_key=f"offline-{match.id}")
        db.add(job)
        db.commit()

        process_job(db, job.id)
        db.refresh(job)
        if job.status == JobStatus.failed:
            raise SystemExit(f"pipeline failed: {job.error_message}")

        tracks = db.query(PlayerTrack).filter(PlayerTrack.match_id == match.id).all()
        track_key_by_id = {t.id: t.track_key for t in tracks}
        links = db.query(TrackIdentityLink).filter(TrackIdentityLink.track_id.in_(list(track_key_by_id))).all()
        player_by_track = {track_key_by_id[lk.track_id]: lk.player_id for lk in links}
        players = {p.id: p for p in db.query(Player).filter(Player.facility_id == facility.id).all()}
        teams = {t.id: t.name for t in db.query(Team).filter(Team.facility_id == facility.id).all()}

        det_dump = []
        for t in tracks:
            feats = dict(t.features or {})
            series = feats.pop("bbox_series", None)
            thumb = feats.get("thumb_path")
            if thumb and Path(thumb).is_relative_to(out):
                feats["thumb_path"] = str(Path(thumb).relative_to(out))
            det_dump.append(
                {
                    "track_key": t.track_key,
                    "camera_label": t.camera_label.value,
                    "start_ms": t.start_ms,
                    "end_ms": t.end_ms,
                    "confidence": round(float(t.confidence or 0), 4),
                    "class_name": feats.get("class", "person"),
                    "features": feats,
                    "bbox_series": series,
                }
            )

        events = db.query(MatchEvent).filter(MatchEvent.match_id == match.id).order_by(MatchEvent.canonical_ts_ms).all()
        obs_dump, canonical_dump = [], []
        for e in events:
            parts = db.query(EventParticipant).filter(EventParticipant.event_id == e.id).all()
            part_keys = [track_key_by_id.get(p.track_id) for p in parts if p.track_id in track_key_by_id]
            etype = "pass" if e.event_type.name == "pass_event" else e.event_type.value
            observations = db.query(EventObservation).filter(EventObservation.event_id == e.id).all()
            for o in observations:
                cam = o.camera_label.value
                obs_dump.append(
                    {
                        "event_type": etype,
                        "camera_label": cam,
                        "camera_ts_ms": o.camera_ts_ms,
                        "confidence": round(float(o.confidence or 0), 4),
                        "track_keys": [k for k in part_keys if k and k.startswith(f"{cam}-")],
                        "evidence": o.payload,
                    }
                )
            canonical_dump.append(
                {
                    "event_type": etype,
                    "ts_ms": e.canonical_ts_ms,
                    "confidence": round(float(e.confidence or 0), 4),
                    "cameras": sorted({o.camera_label.value for o in observations}),
                    "participants": [
                        {
                            "role": p.role,
                            "track_key": track_key_by_id.get(p.track_id),
                            "player": players[p.player_id].display_name if p.player_id in players else None,
                        }
                        for p in parts
                    ],
                }
            )

        players_dump = []
        for pid, p in players.items():
            keys = sorted(k for k, v in player_by_track.items() if v == pid)
            if not keys:
                continue
            players_dump.append(
                {
                    "player_id": str(pid),
                    "display_name": p.display_name,
                    "team": teams.get(p.team_id),
                    "shirt_number": p.default_shirt_number,
                    "track_keys": keys,
                    "cameras": sorted({k.split("-", 1)[0] for k in keys}),
                }
            )

        runs = db.query(ProcessingRun).filter(ProcessingRun.job_id == job.id).all()
        checkpoints = {r.stage.value: r.checkpoint for r in runs if r.checkpoint}
        syncs = db.query(CameraSync).filter(CameraSync.match_id == match.id).all()
        stats = build_match_stats(db, match.id)
        summary = {
            "cameras": {k: str(v) for k, v in sorted(cams.items())},
            "job_status": job.status.value,
            "wall_seconds": round(time.monotonic() - started, 1),
            "sync": [
                {"camera": s.camera_label.value, "offset_ms": s.offset_ms, "confidence": s.confidence, "method": s.method}
                for s in syncs
            ],
            "stages": checkpoints,
            "counts": {
                "tracks": len(tracks),
                "players": len(players_dump),
                "canonical_events": len(canonical_dump),
                "multi_camera_events": sum(1 for e in canonical_dump if len(e["cameras"]) > 1),
            },
            "settings": {
                "yolo_model": settings.yolo_model,
                "vision_device": settings.vision_device,
                "vision_frame_stride": settings.vision_frame_stride,
                "vision_tracker": settings.vision_tracker,
                "vision_reid": settings.vision_reid,
                "court_format": settings.court_format,
            },
        }

        def dump(name: str, data) -> None:
            (out / name).write_text(json.dumps(data, indent=None if name == "detections.json" else 2, default=str))

        dump("summary.json", summary)
        dump("detections.json", det_dump)
        dump("events.json", obs_dump)
        dump("canonical_events.json", canonical_dump)
        dump("players.json", players_dump)
        dump("match_stats.json", stats)
        print(json.dumps(summary["counts"] | {"wall_seconds": summary["wall_seconds"]}, indent=2))
        return summary
    finally:
        db.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
