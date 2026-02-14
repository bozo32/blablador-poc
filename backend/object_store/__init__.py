"""S3-compatible object store wrappers."""

from .s3 import get_bytes, put_bytes

__all__ = ["get_bytes", "put_bytes"]
