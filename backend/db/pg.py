"""Minimal Postgres connection wrapper.

Keep this lightweight: no pooling yet, just a single connect helper so higher
layers can evolve without importing psycopg directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

from backend.settings import settings


@contextmanager
def connect(*, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(settings.POSTGRES_DSN)
    try:
        conn.autocommit = bool(autocommit)
        yield conn
    finally:
        conn.close()
