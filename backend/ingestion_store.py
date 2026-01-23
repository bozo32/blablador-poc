from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.settings import settings

DEFAULT_INGESTION_DIR = settings.INGESTION_DIR


def ensure_ingestion_dir(ingestion_dir: Path) -> Path:
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    return ingestion_dir


def _document_dir(ingestion_dir: Path, doc_id: str) -> Path:
    return ingestion_dir / doc_id


def _metadata_path(ingestion_dir: Path, doc_id: str) -> Path:
    return _document_dir(ingestion_dir, doc_id) / "metadata.json"


def _default_stage() -> Dict[str, Any]:
    return {"status": "pending", "payload": None}


def create_ingested_document(
    file_bytes: bytes,
    filename: str,
    ingestion_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    target_dir = ensure_ingestion_dir(ingestion_dir or DEFAULT_INGESTION_DIR)
    doc_id = str(uuid4())
    doc_dir = _document_dir(target_dir, doc_id)
    doc_dir.mkdir(parents=True, exist_ok=False)

    source_path = doc_dir / "source.pdf"
    source_path.write_bytes(file_bytes)

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    uploaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    metadata: Dict[str, Any] = {
        "id": doc_id,
        "filename": filename,
        "size_bytes": len(file_bytes),
        "sha256": sha256,
        "uploaded_at": uploaded_at,
        "status": "uploaded",
        "extraction": _default_stage(),
        "resolution": _default_stage(),
    }

    _metadata_path(target_dir, doc_id).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return metadata


def get_ingested_document(
    doc_id: str, ingestion_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    metadata_path = _metadata_path(target_dir, doc_id)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def list_ingested_documents(
    ingestion_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
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


def update_ingested_document(
    doc_id: str,
    updates: Dict[str, Any],
    ingestion_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    target_dir = ingestion_dir or DEFAULT_INGESTION_DIR
    existing = get_ingested_document(doc_id, target_dir)
    if existing is None:
        raise FileNotFoundError(f"Document {doc_id} not found")

    merged = dict(existing)
    for key, value in updates.items():
        if key in ("extraction", "resolution") and isinstance(value, dict):
            nested = dict(merged.get(key, {}))
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value

    _metadata_path(target_dir, doc_id).write_text(
        json.dumps(merged, indent=2, sort_keys=True)
    )
    return merged
