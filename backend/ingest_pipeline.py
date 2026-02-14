from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend import extraction, grobid_client
from backend.graph_store import GraphStore
from backend.ingestion_store import (
    get_document_source_path,
    get_ingested_document,
    store_extraction,
    store_resolution,
    update_ingested_document,
)
from backend.reference_resolver import resolve_references


logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _set_ingest_stage_error(
    doc_id: str,
    *,
    stage: str,
    error: str,
    ingestion_dir: Optional[Path] = None,
) -> None:
    try:
        patch = {stage: {"status": "error", "error": str(error), "data": None}}
        update_ingested_document(doc_id, patch, ingestion_dir)
    except Exception:
        logger.exception("Unable to persist %s error for %s", stage, doc_id)


def run_full_ingest_pipeline(
    doc_id: str,
    *,
    graph_store: GraphStore,
    ingestion_dir: Optional[Path] = None,
) -> None:
    """Run extraction + resolution for an ingested document.

    Idempotent: stages are skipped when they already have data.
    """
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return

    document = get_ingested_document(doc_id, ingestion_dir)
    if document is None:
        return

    # --- Extraction ---
    extraction_stage = document.get("extraction") or {}
    extraction_data = (
        (extraction_stage.get("data") or {})
        if isinstance(extraction_stage, dict)
        else {}
    )
    if not extraction_data:
        try:
            try:
                update_ingested_document(
                    doc_id,
                    {"extraction": {"status": "running", "error": None}},
                    ingestion_dir,
                )
            except Exception:
                pass

            pdf_path = get_document_source_path(doc_id, ingestion_dir)
            mode = "fulltext"
            try:
                tei_xml = grobid_client.extract_tei_fulltext(pdf_path)
            except grobid_client.GrobidError:
                mode = "header+references"
                header_xml = grobid_client.extract_tei_header(pdf_path)
                refs_xml = grobid_client.extract_tei_references(pdf_path)
                tei_xml = grobid_client.merge_header_and_references_tei(
                    header_xml=header_xml,
                    references_xml=refs_xml,
                )

            extraction_payload = extraction.parse_tei(tei_xml)
            stored = store_extraction(
                doc_id, tei_xml, extraction_payload, ingestion_dir
            )
            document = stored

            try:
                update_ingested_document(
                    doc_id,
                    {
                        "body_extraction": {
                            "status": "complete" if mode == "fulltext" else "error",
                            "body_extracted_at": _iso_now(),
                            "error": None
                            if mode == "fulltext"
                            else (
                                "Fulltext TEI failed; used header+references fallback."
                            ),
                        }
                    },
                    ingestion_dir,
                )
            except Exception:
                pass

            try:
                graph_store.index_extraction(
                    ingest_meta=stored,
                    extraction_data=extraction_payload,
                )
            except Exception:
                logger.exception("Graph index failed for extraction")
        except Exception as exc:
            logger.exception("Extraction failed for %s", doc_id)
            _set_ingest_stage_error(
                doc_id,
                stage="extraction",
                error=str(exc),
                ingestion_dir=ingestion_dir,
            )
            return

    # --- Resolution ---
    resolution_stage = document.get("resolution") or {}
    resolution_data = (
        (resolution_stage.get("data") or [])
        if isinstance(resolution_stage, dict)
        else []
    )
    if not resolution_data:
        try:
            try:
                update_ingested_document(
                    doc_id,
                    {"resolution": {"status": "running", "error": None}},
                    ingestion_dir,
                )
            except Exception:
                pass

            extraction_data2 = (document.get("extraction") or {}).get("data") or {}
            references = extraction_data2.get("references")
            if references:
                resolved = resolve_references(references)
                stored2 = store_resolution(doc_id, resolved, ingestion_dir)
                try:
                    graph_store.index_resolution(
                        ingest_meta=stored2,
                        resolution_data=resolved,
                    )
                except Exception:
                    logger.exception("Graph index failed for resolution")
        except Exception as exc:
            logger.exception("Resolution failed for %s", doc_id)
            _set_ingest_stage_error(
                doc_id,
                stage="resolution",
                error=str(exc),
                ingestion_dir=ingestion_dir,
            )
            return


class IngestWorkerPool:
    def __init__(
        self,
        *,
        graph_db_path: Path,
        max_workers: int,
        max_queue: int,
        ingestion_dir: Optional[Path] = None,
    ) -> None:
        """Background worker pool for full ingest processing."""
        self._graph_db_path = Path(graph_db_path)
        self._ingestion_dir = ingestion_dir
        self._max_workers = max(1, int(max_workers or 1))
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max(1, int(max_queue or 1)))
        self._threads: list[threading.Thread] = []
        self._started = False
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            for i in range(self._max_workers):
                t = threading.Thread(
                    target=self._worker,
                    daemon=True,
                    name=f"ingest-worker-{i+1}",
                )
                t.start()
                self._threads.append(t)
            self._started = True

    def _worker(self) -> None:
        store = GraphStore(self._graph_db_path)
        while True:
            doc_id = self._queue.get()
            try:
                run_full_ingest_pipeline(
                    doc_id,
                    graph_store=store,
                    ingestion_dir=self._ingestion_dir,
                )
            except Exception:
                logger.exception("Unhandled ingest worker error for %s", doc_id)
            finally:
                self._queue.task_done()

    def enqueue(self, doc_id: str) -> bool:
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return False
        self._ensure_started()
        try:
            self._queue.put_nowait(doc_id)
        except queue.Full:
            return False
        return True
