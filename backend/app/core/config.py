from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    vision_backend: str = "yolo"  # yolo | stub
    yolo_model: str = "yolov8n.pt"
    vision_device: str = "cpu"  # cpu | 0 | cuda
    vision_frame_stride: int = 5  # process every Nth frame
    vision_conf: float = 0.25
    vision_imgsz: int = 640
    vision_max_frames: int = 0  # 0 = no cap; use e.g. 300 for quick online-clip tests
    # Tracker: botsort (default, GMC) | bytetrack (legacy A/B)
    vision_tracker: str = "botsort"
    # OSNet appearance ReID — embed crops + merge track fragments post-track
    vision_reid: bool = True
    vision_reid_model: str = "osnet_x0_25"
    vision_reid_every: int = 3  # embed every N processed frames (cost control)
    court_format: str = "5v5"  # 5v5 | 6v6 | 7v7
    allow_single_camera: bool = True  # Phase 0 public-clip testing

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
