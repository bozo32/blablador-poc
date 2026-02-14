from __future__ import annotations

import json
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
from backend.object_store import s3 as object_store_s3
from backend.spine.attempts import (
    create_or_get_attempt,
    mark_attempt_failed,
    mark_attempt_partial,
    mark_attempt_running,
    mark_attempt_succeeded,
)
from backend.spine.artifacts import create_artifact
from backend.spine.jobs import create_job, set_job_state
from backend.reference_resolver import resolve_references
from backend.settings import settings as app_settings


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
        spine_meta = document.get("spine") if isinstance(document, dict) else None
        spine_meta = spine_meta if isinstance(spine_meta, dict) else {}
        work_id = str(spine_meta.get("work_id") or "").strip() or str(doc_id)

        project_id = str(document.get("project_id") or "").strip() or str(
            app_settings.DEFAULT_PROJECT_ID
        )
        user_id = str(app_settings.DEFAULT_USER_ID)

        attempt_id: Optional[str] = None
        job_id: Optional[str] = None
        try:
            attempt_id, _state = create_or_get_attempt(
                project_id,
                user_id,
                work_id,
                "primary",
                settings_json={},
            )
            try:
                update_ingested_document(
                    doc_id,
                    {"spine": {"work_id": work_id, "active_attempt_id": attempt_id}},
                    ingestion_dir,
                )
            except Exception:
                pass

            if attempt_id:
                try:
                    mark_attempt_running(attempt_id)
                except Exception:
                    pass
                try:
                    job_id = create_job(
                        project_id,
                        user_id,
                        attempt_id,
                        worker="grobid",
                        state="running",
                        progress_json={"stage": "extract"},
                    )
                except Exception:
                    job_id = None

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

            if attempt_id:
                tei_key = f"extract/{work_id}/attempts/{attempt_id}/primary/tei.xml"
                tei_bytes = tei_xml.encode("utf-8", errors="ignore")
                object_store_s3.put_bytes(
                    tei_key,
                    tei_bytes,
                    content_type="application/xml",
                )
                create_artifact(
                    project_id,
                    user_id,
                    attempt_id,
                    "tei.xml",
                    tei_key,
                    len(tei_bytes),
                    "application/xml",
                )

                extraction_key = (
                    f"extract/{work_id}/attempts/{attempt_id}/primary/extraction.json"
                )
                extraction_bytes = json.dumps(
                    extraction_payload,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                object_store_s3.put_bytes(
                    extraction_key,
                    extraction_bytes,
                    content_type="application/json",
                )
                create_artifact(
                    project_id,
                    user_id,
                    attempt_id,
                    "extraction.json",
                    extraction_key,
                    len(extraction_bytes),
                    "application/json",
                )

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

            if attempt_id:
                try:
                    if mode == "fulltext":
                        mark_attempt_succeeded(attempt_id)
                    else:
                        mark_attempt_partial(attempt_id)
                except Exception:
                    pass
            if job_id:
                try:
                    set_job_state(
                        job_id,
                        "succeeded" if mode == "fulltext" else "partial",
                        progress_json={"stage": "done", "mode": mode},
                    )
                except Exception:
                    pass
        except Exception as exc:
            logger.exception("Extraction failed for %s", doc_id)

            if attempt_id:
                try:
                    mark_attempt_failed(
                        attempt_id,
                        failure_reason=type(exc).__name__,
                        failure_detail=str(exc),
                    )
                except Exception:
                    pass
            if job_id:
                try:
                    set_job_state(
                        job_id,
                        "failed",
                        progress_json={"stage": "error", "error": str(exc)},
                    )
                except Exception:
                    pass

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
