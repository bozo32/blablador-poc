"""S3 (MinIO) wrapper for storing Work PDFs and artifacts."""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.config import Config

from backend.settings import settings


@lru_cache(maxsize=1)
def _client():
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


def put_bytes(
    object_key: str, data: bytes, content_type: str = "application/octet-stream"
) -> None:
    key = str(object_key or "").lstrip("/")
    if not key:
        raise ValueError("object_key is required")
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
    resp = _client().get_object(Bucket=str(settings.S3_BUCKET_WORKS), Key=key)
    body = resp.get("Body")
    if body is None:
        raise RuntimeError("S3 response missing Body")
    return body.read()
