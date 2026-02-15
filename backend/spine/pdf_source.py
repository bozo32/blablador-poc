"""Resolve Work PDFs from the ingestion spine (S3/MinIO).

Primary extraction should read PDF bytes from the object store using spine
pointers, not from legacy `data/ingestion/**/source.pdf`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from backend.ingestion_store import get_ingested_document
from backend.object_store import s3 as object_store_s3
from backend.settings import settings
from backend.spine.works import get_work


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lstrip("/")


def resolve_pdf_object_key(
    *,
    doc_id: Optional[str] = None,
    work_id: Optional[str] = None,
    project_id: Optional[str] = None,
    document: Optional[dict] = None,
    ingestion_dir: Optional[Path] = None,
) -> str:
    """Resolve the S3 object key for a PDF.

    Preference order:
    1) `document["spine"]["pdf_object_key"]` (if provided)
    2) `data/ingestion/**/metadata.json` spine fields (if doc_id provided)
    3) Postgres `works.pdf_object_key` (requires project_id)
    """
    doc_id2 = str(doc_id or "").strip() or None
    work_id2 = str(work_id or "").strip() or None

    # (1) Caller-provided document payload.
    if isinstance(document, dict):
        spine = document.get("spine")
        if isinstance(spine, dict):
            key = _normalize_key(spine.get("pdf_object_key"))
            if key:
                return key
            if not work_id2:
                wid = str(spine.get("work_id") or "").strip()
                if wid:
                    work_id2 = wid
        if not project_id:
            pid = str(document.get("project_id") or "").strip()
            if pid:
                project_id = pid

    # (2) Legacy metadata file (still useful for transition).
    if doc_id2:
        meta = get_ingested_document(doc_id2, ingestion_dir)
        if isinstance(meta, dict):
            spine = meta.get("spine")
            if isinstance(spine, dict):
                key = _normalize_key(spine.get("pdf_object_key"))
                if key:
                    return key
                if not work_id2:
                    wid = str(spine.get("work_id") or "").strip()
                    if wid:
                        work_id2 = wid
            if not project_id:
                pid = str(meta.get("project_id") or "").strip()
                if pid:
                    project_id = pid

    # Default: during 09.1 transition, work_id == doc_id.
    if not work_id2 and doc_id2:
        work_id2 = doc_id2

    pid2 = str(project_id or "").strip() or str(settings.DEFAULT_PROJECT_ID)
    if not work_id2:
        raise ValueError("work_id or doc_id is required")

    # (3) Postgres works table.
    work = get_work(work_id2, project_id=pid2)
    if isinstance(work, dict):
        key = _normalize_key(work.get("pdf_object_key"))
        if key:
            return key

    raise FileNotFoundError(
        f"Unable to resolve spine PDF object key for work_id={work_id2}"
    )


def get_pdf_temp_path(
    *,
    doc_id: Optional[str] = None,
    work_id: Optional[str] = None,
    project_id: Optional[str] = None,
    document: Optional[dict] = None,
    ingestion_dir: Optional[Path] = None,
) -> Path:
    """Download the spine PDF to a temp path and return the path."""
    key = resolve_pdf_object_key(
        doc_id=doc_id,
        work_id=work_id,
        project_id=project_id,
        document=document,
        ingestion_dir=ingestion_dir,
    )

    # `grobid_client` needs a filesystem path.
    tmp = tempfile.NamedTemporaryFile(
        prefix="blablador_spine_pdf_",
        suffix=".pdf",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        object_store_s3.download_to_path(key, tmp_path)
        return tmp_path
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise


def cleanup_temp_path(path: Optional[Path]) -> None:
    p = Path(path) if path else None
    if not p:
        return
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass
