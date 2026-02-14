"""Identifier helpers for the V2 ingestion spine.

Migration invariant (Phase 09.1): legacy `doc_id` == V2 `work_id`.
"""

from __future__ import annotations

import re


_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def work_id_from_doc_id(doc_id: str) -> str:
    wid = str(doc_id or "").strip()
    if not wid:
        raise ValueError("doc_id is required")
    return wid


def doc_id_from_work_id(work_id: str) -> str:
    did = str(work_id or "").strip()
    if not did:
        raise ValueError("work_id is required")
    return did


def normalize_sha256(sha256: str) -> str:
    value = str(sha256 or "").strip().lower()
    if not value:
        raise ValueError("sha256 is required")
    if not _SHA256_RE.match(value):
        raise ValueError("sha256 must be a 64-char hex string")
    return value


def pdf_object_key_for_work_pdf(work_id: str, sha256: str) -> str:
    wid = str(work_id or "").strip().strip("/")
    if not wid:
        raise ValueError("work_id is required")
    digest = normalize_sha256(sha256)
    return f"pdf/{wid}/{digest}.pdf"
