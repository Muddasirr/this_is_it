from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py → repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def resolve_yolo_weights(name: str | None = None) -> str:
    """Resolve a YOLO weights name to an existing file when possible."""
    raw = (name or "yolov5x_football.pt").strip() or "yolov5x_football.pt"
    given = Path(raw)
    if given.is_file() and given.stat().st_size > 1000:
        return str(given.resolve())

    candidates = [
        Path.cwd() / raw,
        _BACKEND_ROOT / raw,
        _REPO_ROOT / raw,
        _BACKEND_ROOT / given.name,
        _REPO_ROOT / given.name,
    ]
    stem = given.name.lower()
    if "yolov5x" in stem or stem in {"best.pt", "yolov5x_football.pt"}:
        candidates.extend(
            [
                _BACKEND_ROOT / "yolov5x_football.pt",
                _REPO_ROOT / "yolov5x_football.pt",
                _REPO_ROOT / "data/eval/football_analysis_model/best.pt",
            ]
        )
    if stem in {"yolov8n.pt", "yolo11n.pt", "yolov8s.pt"}:
        candidates.extend(
            [
                _BACKEND_ROOT / stem,
                _REPO_ROOT / stem,
                _REPO_ROOT / "scripts/kaggle/dataset" / stem,
            ]
        )
    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size > 1000:
                return str(cand.resolve())
        except OSError:
            continue
    return raw


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Futsal Analytics"
    database_url: str = "postgresql+psycopg://futsal:futsal@localhost:5432/futsal"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint: str = "http://localhost:9000"
    s3_public_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "futsal-videos"
    s3_region: str = "us-east-1"
    storage_backend: str = "local"  # local | s3
    local_storage_path: str = "./data/videos"
    api_public_url: str = "http://localhost:8000"
    jwt_secret: str = "dev-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    cors_origins: str = "http://localhost:5173"
    signed_url_expire_seconds: int = 3600
    default_retention_days: int = 30

    # Vision / Phase 0
    vision_backend: str = "yolo"  # yolo | stub | modal (GPU on Modal, app.services.modal_backend)
    # General COCO detector: Kaggle corpus run (5 futsal videos) found 4–80× more players
    # per frame than the football fine-tune (yolov5x_football.pt), which misses futsal players.
    yolo_model: str = "yolov8m.pt"
    vision_device: str = "cpu"  # cpu | 0 | cuda
    vision_frame_stride: int = 5  # process every Nth frame (when vision_target_fps = 0)
    vision_target_fps: float = 10.0  # processed frames per video second; stride = fps / target
    vision_team_embed: bool = True  # SigLIP crop embeddings → team split (falls back to colour)
    vision_role_filter: bool = True  # zero-shot SigLIP: drop spectators / coaches before the team split
    vision_conf: float = 0.25
    # 0 = auto: frame's long side, capped at VISION_IMGSZ_MAX. SoccerTrack v2 MOT
    # (4096×1080 fixed panorama): recall 16 % @640 → 74 % @1280 → 89 % @1920.
    vision_imgsz: int = 0
    vision_imgsz_max: int = 1920
    vision_max_frames: int = 0  # 0 = no cap; use e.g. 300 for quick online-clip tests
    # Dedicated ball detector weights ("" = the primary model's sports_ball class). When set,
    # the primary's ball boxes (COCO: flags, cones, boards) are ignored and every processed
    # frame gets a ball pass with this model.
    vision_ball_model: str = ""
    # Tracker: botsort (default, GMC) | bytetrack (legacy A/B)
    vision_tracker: str = "botsort"
    # BoT-SORT global motion compensation: auto = off when the camera is static (facility
    # CCTV — GMC's per-frame optical flow is most of the tracker time), on when it pans
    vision_gmc: str = "auto"  # auto | on | off
    # OSNet appearance ReID — embed crops + merge track fragments post-track
    vision_reid: bool = True
    vision_reid_model: str = "osnet_x0_25"
    vision_reid_every: int = 3  # embed every N processed frames (cost control)
    vision_thumb_dir: str = ""  # player crops; "" = <tmp>/courtlens_thumbs
    court_format: str = "5v5"  # 5v5 | 6v6 | 7v7
    allow_single_camera: bool = True  # pilot: Camera A only; production courts add B–D
    # Pipeline knobs (batch runs on Kaggle skip review-only work)
    min_video_duration_s: int = 60  # 30 for DFL 30 s eval clips
    pipeline_proxies: bool = True  # review proxies (not needed for offline eval)
    pipeline_max_clips: int = 25  # auto evidence clips per job

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
