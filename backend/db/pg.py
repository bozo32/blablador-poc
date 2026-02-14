"""Minimal Postgres connection wrapper.

Keep this lightweight: no pooling yet, just a single connect helper so higher
layers can evolve without importing psycopg directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from backend.settings import settings

if TYPE_CHECKING:  # pragma: no cover
    import psycopg

    Connection = psycopg.Connection
else:
    Connection = Any


@contextmanager
def connect(*, autocommit: bool = False) -> Iterator[Connection]:
    try:
        import psycopg  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "psycopg is required for Postgres features; install requirements/app.in"
        ) from exc

    conn = psycopg.connect(settings.POSTGRES_DSN)
    try:
        conn.autocommit = bool(autocommit)
        yield conn
    finally:
        conn.close()
