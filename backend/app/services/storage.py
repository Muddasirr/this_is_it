from __future__ import annotations

import hashlib
import hmac
import time
from functools import lru_cache
from pathlib import Path
from uuid import UUID
from urllib.parse import quote

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import get_settings


class StorageError(Exception):
    pass


@lru_cache
def get_s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )


def use_local_storage() -> bool:
    return get_settings().storage_backend == "local"


def local_root() -> Path:
    root = Path(get_settings().local_storage_path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def object_key_for_original(facility_id: UUID, match_id: UUID, camera_label: str, filename: str) -> str:
    safe = filename.replace("/", "_").replace("\\", "_")
    return f"facilities/{facility_id}/matches/{match_id}/cameras/{camera_label}/original/{safe}"


def object_key_for_proxy(facility_id: UUID, match_id: UUID, camera_label: str) -> str:
    return f"facilities/{facility_id}/matches/{match_id}/cameras/{camera_label}/proxy/review.mp4"


def _sign(object_key: str, expires_at: int) -> str:
    settings = get_settings()
    msg = f"{object_key}:{expires_at}".encode()
    return hmac.new(settings.jwt_secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_local_signature(object_key: str, expires_at: int, signature: str) -> bool:
    if expires_at < int(time.time()):
        return False
    expected = _sign(object_key, expires_at)
    return hmac.compare_digest(expected, signature)


def presign_upload(object_key: str, content_type: str, expires: int | None = None) -> str:
    settings = get_settings()
    ttl = expires or settings.signed_url_expire_seconds
    if use_local_storage():
        expires_at = int(time.time()) + ttl
        sig = _sign(object_key, expires_at)
        return (
            f"{settings.api_public_url}/api/v1/storage/upload"
            f"?key={quote(object_key)}&expires={expires_at}&signature={sig}"
            f"&content_type={quote(content_type)}"
        )

    client = get_s3_client()
    url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.s3_bucket,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=ttl,
    )
    if settings.s3_endpoint != settings.s3_public_endpoint:
        url = url.replace(settings.s3_endpoint, settings.s3_public_endpoint)
    return url


def presign_download(object_key: str, expires: int | None = None) -> str:
    settings = get_settings()
    ttl = expires or settings.signed_url_expire_seconds
    if use_local_storage():
        expires_at = int(time.time()) + ttl
        sig = _sign(object_key, expires_at)
        return (
            f"{settings.api_public_url}/api/v1/storage/download"
            f"?key={quote(object_key)}&expires={expires_at}&signature={sig}"
        )

    client = get_s3_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": object_key},
        ExpiresIn=ttl,
    )
    if settings.s3_endpoint != settings.s3_public_endpoint:
        url = url.replace(settings.s3_endpoint, settings.s3_public_endpoint)
    return url


def save_local_bytes(object_key: str, data: bytes) -> Path:
    path = local_root() / object_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def local_path(object_key: str) -> Path:
    return local_root() / object_key


def download_to_path(object_key: str, dest: Path) -> None:
    if use_local_storage():
        src = local_path(object_key)
        if not src.exists():
            raise StorageError(f"Missing object {object_key}")
        dest.write_bytes(src.read_bytes())
        return
    settings = get_settings()
    get_s3_client().download_file(settings.s3_bucket, object_key, str(dest))


def upload_file(path: Path, object_key: str, content_type: str = "video/mp4") -> None:
    if use_local_storage():
        dest = local_path(object_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        return
    settings = get_settings()
    get_s3_client().upload_file(
        str(path),
        settings.s3_bucket,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )


def delete_object(object_key: str) -> None:
    if use_local_storage():
        path = local_path(object_key)
        if path.exists():
            path.unlink()
        return
    settings = get_settings()
    get_s3_client().delete_object(Bucket=settings.s3_bucket, Key=object_key)


def ensure_bucket() -> None:
    if use_local_storage():
        local_root()
        return
    settings = get_settings()
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except (ClientError, BotoCoreError):
        client.create_bucket(Bucket=settings.s3_bucket)
