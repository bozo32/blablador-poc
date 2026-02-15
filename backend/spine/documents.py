"""Postgres helpers for Documents and DocumentVersions.

This is scaffolding for the identity split:

- Work (bibliographic, global) [later]
- Document (global)
- DocumentVersion (sha256 bytes/version, global)
- Project membership tracked separately

For now (transition), callers may treat document_id == document_version_id == doc_id.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.db import connect


_DOCUMENT_COLUMNS = (
    "document_id",
    "created_at",
    "created_by_user_id",
    "biblio_work_id",
)


def get_or_create_document(
    document_id: str,
    *,
    created_by_user_id: str,
    biblio_work_id: Optional[str] = None,
) -> Dict[str, Any]:
    did = str(document_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    bid = str(biblio_work_id or "").strip() or None
    if not did:
        raise ValueError("document_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, created_at, created_by_user_id, biblio_work_id
                  FROM documents
                 WHERE document_id = %s
                 """,
                (did,),
            )
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                return {str(k): v for k, v in zip(_DOCUMENT_COLUMNS, row)}

            cur.execute(
                """
                INSERT INTO documents (document_id, created_by_user_id, biblio_work_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (document_id) DO NOTHING
                RETURNING document_id, created_at, created_by_user_id, biblio_work_id
                """,
                (did, uid, bid),
            )
            created = cur.fetchone()
        conn.commit()

    if created is not None:
        return {str(k): v for k, v in zip(_DOCUMENT_COLUMNS, created)}

    # Concurrent insert: re-read.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, created_at, created_by_user_id, biblio_work_id
                  FROM documents
                 WHERE document_id = %s
                 """,
                (did,),
            )
            row2 = cur.fetchone()
            if row2 is None:
                raise RuntimeError("document upsert did not create a row")
            return {str(k): v for k, v in zip(_DOCUMENT_COLUMNS, row2)}


_VERSION_COLUMNS = (
    "document_version_id",
    "document_id",
    "sha256",
    "size_bytes",
    "pdf_object_key",
    "filename",
    "created_at",
    "created_by_user_id",
)


def upsert_document_version(
    document_version_id: str,
    *,
    document_id: str,
    sha256: str,
    size_bytes: int,
    pdf_object_key: str,
    filename: str,
    created_by_user_id: str,
) -> Dict[str, Any]:
    vid = str(document_version_id or "").strip()
    did = str(document_id or "").strip()
    digest = str(sha256 or "").strip().lower()
    obj_key = str(pdf_object_key or "").lstrip("/")
    fname = str(filename or "").strip() or "document.pdf"
    uid = str(created_by_user_id or "").strip()
    if not vid:
        raise ValueError("document_version_id is required")
    if not did:
        raise ValueError("document_id is required")
    if not digest:
        raise ValueError("sha256 is required")
    if not obj_key:
        raise ValueError("pdf_object_key is required")
    if not uid:
        raise ValueError("created_by_user_id is required")
    bytes_n = int(size_bytes)
    if bytes_n < 0:
        raise ValueError("size_bytes must be >= 0")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_versions (
                  document_version_id,
                  document_id,
                  sha256,
                  size_bytes,
                  pdf_object_key,
                  filename,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_version_id)
                DO UPDATE SET
                  document_id = EXCLUDED.document_id,
                  sha256 = EXCLUDED.sha256,
                  size_bytes = EXCLUDED.size_bytes,
                  pdf_object_key = EXCLUDED.pdf_object_key,
                  filename = EXCLUDED.filename,
                  created_by_user_id = EXCLUDED.created_by_user_id
                RETURNING document_version_id, document_id, sha256, size_bytes,
                          pdf_object_key, filename, created_at, created_by_user_id
                """,
                (vid, did, digest, bytes_n, obj_key, fname, uid),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("document_version upsert did not return a row")
        conn.commit()

    return {str(k): v for k, v in zip(_VERSION_COLUMNS, row)}


def is_document_in_project(*, project_id: str, document_id: str) -> bool:
    pid = str(project_id or "").strip()
    did = str(document_id or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not did:
        raise ValueError("document_id is required")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                  FROM project_documents
                 WHERE project_id = %s
                   AND document_id = %s
                 LIMIT 1
                """,
                (pid, did),
            )
            row = cur.fetchone()
            return row is not None


def list_project_documents(
    *, project_id: str, limit: int = 200
) -> list[Dict[str, Any]]:
    pid = str(project_id or "").strip()
    n = int(limit)
    if not pid:
        raise ValueError("project_id is required")
    if n <= 0:
        return []

    cols = (
        "project_id",
        "document_id",
        "added_at",
        "added_by_user_id",
        "tags",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT project_id, document_id, added_at, added_by_user_id, tags
                  FROM project_documents
                 WHERE project_id = %s
                 ORDER BY added_at DESC
                 LIMIT %s
                """,
                (pid, n),
            )
            rows = cur.fetchall() or []
            return [{str(k): v for k, v in zip(cols, row)} for row in rows]


def get_latest_document_version(*, document_id: str) -> Optional[Dict[str, Any]]:
    did = str(document_id or "").strip()
    if not did:
        raise ValueError("document_id is required")

    cols = _VERSION_COLUMNS
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_version_id, document_id, sha256, size_bytes,
                       pdf_object_key, filename, created_at, created_by_user_id
                  FROM document_versions
                 WHERE document_id = %s
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (did,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {str(k): v for k, v in zip(cols, row)}


def ensure_project_document(
    project_id: str,
    *,
    document_id: str,
    added_by_user_id: str,
) -> None:
    pid = str(project_id or "").strip()
    did = str(document_id or "").strip()
    uid = str(added_by_user_id or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not did:
        raise ValueError("document_id is required")
    if not uid:
        raise ValueError("added_by_user_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO project_documents (
                  project_id,
                  document_id,
                  added_by_user_id
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id, document_id) DO NOTHING
                """,
                (pid, did, uid),
            )
        conn.commit()
