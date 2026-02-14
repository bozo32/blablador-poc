"""S3 (MinIO) wrapper for storing Work PDFs and artifacts."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from botocore.exceptions import ClientError  # noqa: F401

from backend.settings import settings


@lru_cache(maxsize=1)
def _client():
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is required for S3 features; install requirements/app.in"
        ) from exc

    # MinIO commonly requires path-style addressing.
    config = Config(s3={"addressing_style": "path"})
    return boto3.client(
        "s3",
        endpoint_url=str(settings.S3_ENDPOINT_URL),
        aws_access_key_id=str(settings.S3_ACCESS_KEY),
        aws_secret_access_key=str(settings.S3_SECRET_KEY),
        region_name=str(settings.S3_REGION),
        use_ssl=bool(settings.S3_USE_SSL),
        config=config,
    )


@lru_cache(maxsize=1)
def ensure_bucket() -> None:
    """Ensure the works bucket exists.

    Compose runs `minio-init` to create the bucket, but host-run/dev can race
    bucket creation. This helper makes `put_bytes` robust.
    """
    bucket = str(settings.S3_BUCKET_WORKS)
    if not bucket:
        raise ValueError("S3_BUCKET_WORKS is required")

    client = _client()
    try:
        client.head_bucket(Bucket=bucket)
        return
    except Exception:
        pass

    try:
        client.create_bucket(Bucket=bucket)
    except Exception as exc:
        # If the bucket was created concurrently, treat it as success.
        response = getattr(exc, "response", None) or {}
        code = str((response.get("Error", {}) or {}).get("Code", "")).lower()
        if code in {"bucketalreadyownedbyyou", "bucketalreadyexists"}:
            return
        raise


def put_bytes(
    object_key: str, data: bytes, content_type: str = "application/octet-stream"
) -> None:
    key = str(object_key or "").lstrip("/")
    if not key:
        raise ValueError("object_key is required")
    ensure_bucket()
    _client().put_object(
        Bucket=str(settings.S3_BUCKET_WORKS),
        Key=key,
        Body=data,
        ContentType=str(content_type or "application/octet-stream"),
    )


def get_bytes(object_key: str) -> bytes:
    key = str(object_key or "").lstrip("/")
    if not key:
        raise ValueError("object_key is required")
    ensure_bucket()
    resp = _client().get_object(Bucket=str(settings.S3_BUCKET_WORKS), Key=key)
    body = resp.get("Body")
    if body is None:
        raise RuntimeError("S3 response missing Body")
    return body.read()


def exists(object_key: str) -> bool:
    key = str(object_key or "").lstrip("/")
    if not key:
        raise ValueError("object_key is required")
    ensure_bucket()
    try:
        _client().head_object(Bucket=str(settings.S3_BUCKET_WORKS), Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", None) or {}
        code = str((response.get("Error", {}) or {}).get("Code", "")).lower()
        if code in {"404", "nosuchkey", "notfound"}:
            return False
        raise
