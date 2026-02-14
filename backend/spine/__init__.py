"""V2 ingestion spine helpers.

This package is intentionally small. During migration we treat legacy `doc_id`
as the same identifier as the V2 `work_id`.
"""

from .ids import doc_id_from_work_id, pdf_object_key_for_work_pdf, work_id_from_doc_id
from .works import get_work, upsert_work_from_pdf

__all__ = [
    "doc_id_from_work_id",
    "pdf_object_key_for_work_pdf",
    "work_id_from_doc_id",
    "get_work",
    "upsert_work_from_pdf",
]
