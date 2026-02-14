"""Postgres helpers for Work rows (V2 ingestion spine)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.db import connect


_WORK_COLUMNS = (
    "work_id",
    "created_at",
    "filename",
    "sha256",
    "size_bytes",
    "pdf_object_key",
    "active_attempt_id",
    "tags",
)


def get_work(work_id: str) -> Optional[Dict[str, Any]]:
    wid = str(work_id or "").strip()
    if not wid:
        raise ValueError("work_id is required")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT work_id, created_at, filename, sha256, size_bytes,
                       pdf_object_key, active_attempt_id, tags
                  FROM works
                 WHERE work_id = %s
                """,
                (wid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return dict(zip(_WORK_COLUMNS, row))


def upsert_work_from_pdf(
    work_id: str,
    filename: str,
    sha256: str,
    size_bytes: int,
    pdf_object_key: str,
) -> Dict[str, Any]:
    wid = str(work_id or "").strip()
    if not wid:
        raise ValueError("work_id is required")
    fname = str(filename or "").strip() or "document.pdf"
    digest = str(sha256 or "").strip().lower()
    if not digest:
        raise ValueError("sha256 is required")
    bytes_n = int(size_bytes)
    if bytes_n < 0:
        raise ValueError("size_bytes must be >= 0")
    obj_key = str(pdf_object_key or "").lstrip("/")
    if not obj_key:
        raise ValueError("pdf_object_key is required")

    existing = get_work(wid)
    if existing is not None and str(existing.get("sha256") or "").lower() != digest:
        raise ValueError(
            "work_id already exists with different sha256; refusing to overwrite"
        )

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO works (
                    work_id,
                    filename,
                    sha256,
                    size_bytes,
                    pdf_object_key
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (work_id)
                DO UPDATE SET
                  filename = EXCLUDED.filename,
                  sha256 = EXCLUDED.sha256,
                  size_bytes = EXCLUDED.size_bytes,
                  pdf_object_key = EXCLUDED.pdf_object_key
                RETURNING work_id, created_at, filename, sha256, size_bytes,
                          pdf_object_key, active_attempt_id, tags
                """,
                (wid, fname, digest, bytes_n, obj_key),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("upsert_work_from_pdf did not return a row")
        conn.commit()
    return dict(zip(_WORK_COLUMNS, row))
