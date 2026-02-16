from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.settings import settings

DEFAULT_INGESTION_DIR = settings.INGESTION_DIR


def _spine_mode() -> str:
    mode = str(os.environ.get("SPINE_MODE", "spine")).strip().lower()
    return mode if mode in {"legacy", "dual", "spine"} else "spine"


def ensure_ingestion_dir(ingestion_dir: Path) -> Path:
    if _spine_mode() != "spine":
        ingestion_dir.mkdir(parents=True, exist_ok=True)
    return ingestion_dir


def _document_dir(ingestion_dir: Path, doc_id: str) -> Path:
    return ingestion_dir / doc_id


def _metadata_path(ingestion_dir: Path, doc_id: str) -> Path:
    return _document_dir(ingestion_dir, doc_id) / "metadata.json"


def _default_stage(timestamp_key: str) -> Dict[str, Any]:
    return {"status": "pending", timestamp_key: None, "data": None}


def _extraction_dir(ingestion_dir: Path, doc_id: str) -> Path:
    return _document_dir(ingestion_dir, doc_id) / "extraction"


def _tei_path(ingestion_dir: Path, doc_id: str) -> Path:
    return _extraction_dir(ingestion_dir, doc_id) / "tei.xml"


def get_tei_xml(doc_id: str, ingestion_dir: Optional[Path] = None) -> str:
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    tei_path = _tei_path(target_dir, doc_id)
    if not tei_path.exists():
        raise FileNotFoundError(
            f"TEI XML not found for document {doc_id}. Run extraction first."
        )
    return tei_path.read_text(encoding="utf-8", errors="ignore")


def create_ingested_document(
    file_bytes: bytes,
    filename: str,
    ingestion_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if _spine_mode() == "spine":
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "id": str(uuid4()),
            "project_id": str(settings.DEFAULT_PROJECT_ID),
            "filename": filename,
            "size_bytes": len(file_bytes),
            "sha256": sha256,
            "uploaded_at": uploaded_at,
            "status": "uploaded",
            "aliases": [],
            "extraction": _default_stage("extracted_at"),
            "body_extraction": _default_stage("body_extracted_at"),
            "resolution": _default_stage("resolved_at"),
        }

    target_dir = ensure_ingestion_dir(ingestion_dir or DEFAULT_INGESTION_DIR)

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    # De-dupe uploads by content hash. If a PDF with the same sha256 already
    # exists, re-use that document id and record the new filename as an alias.
    for doc_dir in target_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        meta_path = doc_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            existing = json.loads(meta_path.read_text())
        except Exception:
            continue
        if str(existing.get("sha256") or "").strip().lower() != sha256.lower():
            continue
        aliases = list(existing.get("aliases") or [])
        if (
            filename
            and filename not in aliases
            and filename != existing.get("filename")
        ):
            aliases.append(filename)
        if aliases:
            existing["aliases"] = aliases
            meta_path.write_text(json.dumps(existing, indent=2, sort_keys=True))

        if not str(existing.get("project_id") or "").strip():
            existing["project_id"] = str(settings.DEFAULT_PROJECT_ID)
            meta_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
        return existing

    doc_id = str(uuid4())
    doc_dir = _document_dir(target_dir, doc_id)
    doc_dir.mkdir(parents=True, exist_ok=False)

    source_path = doc_dir / "source.pdf"
    source_path.write_bytes(file_bytes)

    uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    metadata: Dict[str, Any] = {
        "id": doc_id,
        "project_id": str(settings.DEFAULT_PROJECT_ID),
        "filename": filename,
        "size_bytes": len(file_bytes),
        "sha256": sha256,
        "uploaded_at": uploaded_at,
        "status": "uploaded",
        "aliases": [],
        "extraction": _default_stage("extracted_at"),
        "body_extraction": _default_stage("body_extracted_at"),
        "resolution": _default_stage("resolved_at"),
    }

    _metadata_path(target_dir, doc_id).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return metadata


def get_ingested_document(
    doc_id: str, ingestion_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    if _spine_mode() == "spine":
        return None
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    metadata_path = _metadata_path(target_dir, doc_id)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def list_ingested_documents(
    ingestion_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if _spine_mode() == "spine":
        return []
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    if not target_dir.exists():
        return []

    documents: List[Dict[str, Any]] = []
    for doc_dir in target_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        metadata_path = doc_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        documents.append(json.loads(metadata_path.read_text()))

    documents.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
    return documents


def get_document_source_path(doc_id: str, ingestion_dir: Optional[Path] = None) -> Path:
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    source_path = _document_dir(target_dir, doc_id) / "source.pdf"
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF not found for document {doc_id}")
    return source_path


def store_extraction(
    doc_id: str,
    tei_xml: str,
    extraction_data: Dict[str, Any],
    ingestion_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    doc_dir = _document_dir(target_dir, doc_id)
    if not doc_dir.exists():
        raise FileNotFoundError(f"Document {doc_id} not found")

    extraction_dir = _extraction_dir(target_dir, doc_id)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    _tei_path(target_dir, doc_id).write_text(tei_xml, encoding="utf-8")

    extracted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    extraction_payload = {
        "status": "complete",
        "extracted_at": extracted_at,
        "data": extraction_data,
    }
    return update_ingested_document(
        doc_id, {"extraction": extraction_payload}, target_dir
    )


def store_resolution(
    doc_id: str,
    resolution_data: Any,
    ingestion_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    resolved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    resolution_payload = {
        "status": "complete",
        "resolved_at": resolved_at,
        "data": resolution_data,
    }
    return update_ingested_document(
        doc_id, {"resolution": resolution_payload}, target_dir
    )


def update_ingested_document(
    doc_id: str,
    updates: Dict[str, Any],
    ingestion_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if _spine_mode() == "spine":
        raise FileNotFoundError("Legacy ingestion store disabled (SPINE_MODE=spine)")
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    existing = get_ingested_document(doc_id, target_dir)
    if existing is None:
        raise FileNotFoundError(f"Document {doc_id} not found")

    merged = dict(existing)
    for key, value in updates.items():
        if key in (
            "extraction",
            "resolution",
            "body_extraction",
            "spine",
        ) and isinstance(value, dict):
            nested = dict(merged.get(key, {}))
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value

    _metadata_path(target_dir, doc_id).write_text(
        json.dumps(merged, indent=2, sort_keys=True)
    )
    return merged
