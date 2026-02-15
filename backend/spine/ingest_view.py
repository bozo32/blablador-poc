"""Spine-backed view over the legacy ingest document payload.

Goal: allow API/UI to read ingest state from Postgres + S3 (MinIO) when present,
while keeping legacy filesystem ingestion as fallback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.object_store import s3 as object_store_s3
from backend.spine.attempts import get_latest_attempt_for_work
from backend.spine.artifacts import get_artifact_for_attempt
from backend.spine.documents import (
    get_latest_document_version,
    is_document_in_project,
    list_project_documents,
)


def _iso(dt: Any) -> str:
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _attempt_to_stage(attempt: Optional[dict]) -> dict:
    if not attempt:
        return {"status": "pending", "extracted_at": None, "data": None}
    state = str(attempt.get("state") or "").strip()
    finished_at = attempt.get("finished_at")
    if state in {"queued"}:
        return {"status": "pending", "extracted_at": None, "data": None}
    if state in {"running"}:
        return {"status": "running", "extracted_at": None, "data": None}
    if state in {"succeeded", "partial"}:
        return {"status": "complete", "extracted_at": _iso(finished_at), "data": None}
    if state in {"failed"}:
        return {
            "status": "error",
            "extracted_at": _iso(finished_at) if finished_at else None,
            "data": None,
            "error": str(
                attempt.get("failure_detail") or attempt.get("failure_reason") or ""
            ),
        }
    return {"status": state or "unknown", "extracted_at": None, "data": None}


def build_ingested_document_from_spine(
    *,
    work_id: str,
    project_id: str,
    include_extraction_data: bool,
) -> Optional[Dict[str, Any]]:
    # Back-compat wrapper; new list/get flows should prefer membership-based
    # helpers that are driven by project_documents + document_versions.
    wid = str(work_id or "").strip()
    pid = str(project_id or "").strip()
    if not wid or not pid:
        return None

    if not is_document_in_project(project_id=pid, document_id=wid):
        return None

    version = get_latest_document_version(document_id=wid)
    if version is None:
        return None

    attempt = get_latest_attempt_for_work(project_id=pid, work_id=wid, kind="primary")
    extraction = _attempt_to_stage(attempt)

    resolution = {"status": "pending", "resolved_at": None, "data": None}
    if attempt:
        artifact = get_artifact_for_attempt(
            attempt_id=str(attempt.get("attempt_id")),
            artifact_type="resolution.json",
        )
        if artifact and artifact.get("object_key"):
            resolution = {
                "status": "complete",
                "resolved_at": _iso(artifact.get("created_at")),
                "data": None,
            }
            if include_extraction_data:
                try:
                    raw = object_store_s3.get_bytes(str(artifact["object_key"]))
                    resolution["data"] = json.loads(raw.decode("utf-8"))
                except Exception:
                    resolution["data"] = None

    if include_extraction_data and attempt and extraction.get("status") == "complete":
        artifact = get_artifact_for_attempt(
            attempt_id=str(attempt.get("attempt_id")),
            artifact_type="extraction.json",
        )
        if artifact and artifact.get("object_key"):
            try:
                raw = object_store_s3.get_bytes(str(artifact["object_key"]))
                extraction["data"] = json.loads(raw.decode("utf-8"))
            except Exception:
                extraction["data"] = None

    uploaded_at = _iso(version.get("created_at"))
    # Keep response compatible with schemas.IngestedDocument.
    return {
        "id": wid,
        "project_id": pid,
        "filename": str(version.get("filename") or "document.pdf"),
        "size_bytes": int(version.get("size_bytes") or 0),
        "sha256": str(version.get("sha256") or ""),
        "uploaded_at": uploaded_at,
        "status": "uploaded",
        "extraction": extraction,
        "body_extraction": {"status": "pending", "extracted_at": None, "data": None},
        "resolution": resolution,
    }


def list_ingests_from_spine(
    *, project_id: str, limit: int = 200
) -> list[Dict[str, Any]]:
    pid = str(project_id or "").strip()
    if not pid:
        raise ValueError("project_id is required")

    rows = list_project_documents(project_id=pid, limit=int(limit))
    out: list[Dict[str, Any]] = []
    for row in rows:
        did = str(row.get("document_id") or "").strip()
        if not did:
            continue
        version = get_latest_document_version(document_id=did)
        if version is None:
            continue

        attempt = get_latest_attempt_for_work(
            project_id=pid,
            work_id=did,
            kind="primary",
        )
        extraction = _attempt_to_stage(attempt)

        resolution = {"status": "pending", "resolved_at": None, "data": None}
        if attempt:
            artifact = get_artifact_for_attempt(
                attempt_id=str(attempt.get("attempt_id")),
                artifact_type="resolution.json",
            )
            if artifact and artifact.get("object_key"):
                resolution = {
                    "status": "complete",
                    "resolved_at": _iso(artifact.get("created_at")),
                    "data": None,
                }

        out.append(
            {
                "id": did,
                "project_id": pid,
                "filename": str(version.get("filename") or "document.pdf"),
                "size_bytes": int(version.get("size_bytes") or 0),
                "sha256": str(version.get("sha256") or ""),
                "uploaded_at": _iso(version.get("created_at")),
                "status": "uploaded",
                "extraction": extraction,
                "body_extraction": {
                    "status": "pending",
                    "extracted_at": None,
                    "data": None,
                },
                "resolution": resolution,
            }
        )

    return out


def get_tei_xml_from_spine(*, work_id: str, project_id: str) -> Optional[str]:
    wid = str(work_id or "").strip()
    pid = str(project_id or "").strip()
    if not wid or not pid:
        return None
    attempt = get_latest_attempt_for_work(project_id=pid, work_id=wid, kind="primary")
    if not attempt:
        return None
    artifact = get_artifact_for_attempt(
        attempt_id=str(attempt.get("attempt_id")),
        artifact_type="tei.xml",
    )
    if not artifact or not artifact.get("object_key"):
        return None
    try:
        raw = object_store_s3.get_bytes(str(artifact["object_key"]))
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None


def get_resolution_from_spine(*, work_id: str, project_id: str) -> Optional[list]:
    wid = str(work_id or "").strip()
    pid = str(project_id or "").strip()
    if not wid or not pid:
        return None
    attempt = get_latest_attempt_for_work(project_id=pid, work_id=wid, kind="primary")
    if not attempt:
        return None
    artifact = get_artifact_for_attempt(
        attempt_id=str(attempt.get("attempt_id")),
        artifact_type="resolution.json",
    )
    if not artifact or not artifact.get("object_key"):
        return None
    try:
        raw = object_store_s3.get_bytes(str(artifact["object_key"]))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, list) else None
    except Exception:
        return None
