from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

from backend import extraction, fallback_text, grobid_client
from backend.object_store import s3 as object_store_s3
from backend.graph_store import GraphStore
from backend.spine.attempts import (
    create_or_get_attempt,
    get_attempt,
    mark_attempt_cancelled,
    mark_attempt_failed,
    mark_attempt_partial,
    mark_attempt_running,
    mark_attempt_succeeded,
    merge_attempt_quality_json,
)
from backend.spine.artifacts import create_artifact
from backend.spine.ingest_view import build_ingested_document_from_spine
from backend.settings import settings

import json
from typing import Any, Dict

from backend.spine.jobs import create_job, set_job_state, update_job_progress
from backend.spine.pdf_source import cleanup_temp_path, get_pdf_temp_path
from backend.spine.pdf_source import resolve_pdf_object_key

import requests


@dataclass(frozen=True)
class ExtractionTask:
    doc_id: str
    project_id: str
    user_id: str
    force_fallback: bool
    job_id: str
    attempt_id: str


def _should_cancel(attempt_id: str) -> bool:
    try:
        a = get_attempt(str(attempt_id))
        return bool(a and str(a.get("state")) == "cancelled")
    except Exception:
        return False


def _write_fallback_artifacts(
    *, work_id: str, attempt_id: str, project_id: str, user_id: str, pdf_path: Path
) -> dict:
    # Prefer external OCR-capable fallback worker when configured.
    worker_url = str(getattr(settings, "FALLBACK_WORKER_URL", "") or "").strip()
    token = str(getattr(settings, "INTERNAL_SERVICE_TOKEN", "") or "").strip()

    if worker_url:
        # Resolve PDF object key from the spine so the worker can download it.
        pdf_object_key = resolve_pdf_object_key(
            work_id=work_id,
            project_id=project_id,
        )
        url = worker_url.rstrip("/") + "/v1/fallback"
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = {
            "work_id": work_id,
            "attempt_id": attempt_id,
            "pdf_object_key": pdf_object_key,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=900)
        if resp.status_code == 409 and ("attempt_cancelled" in (resp.text or "")):
            raise RuntimeError("attempt_cancelled")
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json() if resp.content else {}

        pages_key = str(data.get("pages_object_key") or "").strip()
        body_key = str(data.get("body_object_key") or "").strip()
        refs_key = str(data.get("refs_object_key") or "").strip()
        if not pages_key or not body_key or not refs_key:
            raise RuntimeError("fallback worker returned missing object keys")

        create_artifact(
            project_id,
            user_id,
            attempt_id,
            "fallback.pages.jsonl",
            pages_key,
            int(data.get("pages_bytes") or 0),
            "application/x-ndjson",
        )
        create_artifact(
            project_id,
            user_id,
            attempt_id,
            "fallback.body.txt",
            body_key,
            int(data.get("body_bytes") or 0),
            "text/plain",
        )
        create_artifact(
            project_id,
            user_id,
            attempt_id,
            "fallback.refs.json",
            refs_key,
            int(data.get("refs_bytes") or 0),
            "application/json",
        )

        qp = (
            data.get("quality_patch")
            if isinstance(data.get("quality_patch"), dict)
            else {}
        )
        return qp

    # Local deterministic fallback (text-layer only).
    pages = fallback_text.extract_fallback_pages(Path(pdf_path))
    pages_jsonl = fallback_text.pages_to_jsonl(pages)
    body_txt = fallback_text.pages_to_body_text(pages)
    refs_json = b"[]\n"

    fb_pages_key = f"extract/{work_id}/attempts/{attempt_id}/fallback/pages.jsonl"
    object_store_s3.put_bytes(
        fb_pages_key,
        pages_jsonl,
        content_type="application/x-ndjson",
    )
    create_artifact(
        project_id,
        user_id,
        attempt_id,
        "fallback.pages.jsonl",
        fb_pages_key,
        len(pages_jsonl),
        "application/x-ndjson",
    )

    fb_body_key = f"extract/{work_id}/attempts/{attempt_id}/fallback/body.txt"
    object_store_s3.put_bytes(
        fb_body_key,
        body_txt,
        content_type="text/plain",
    )
    create_artifact(
        project_id,
        user_id,
        attempt_id,
        "fallback.body.txt",
        fb_body_key,
        len(body_txt),
        "text/plain",
    )

    fb_refs_key = f"extract/{work_id}/attempts/{attempt_id}/fallback/refs.json"
    object_store_s3.put_bytes(
        fb_refs_key,
        refs_json,
        content_type="application/json",
    )
    create_artifact(
        project_id,
        user_id,
        attempt_id,
        "fallback.refs.json",
        fb_refs_key,
        len(refs_json),
        "application/json",
    )

    return {**fallback_text.fallback_quality(pages)}


def _run_task(task: ExtractionTask) -> None:
    work_id = task.doc_id
    attempt_id = task.attempt_id
    job_id = task.job_id

    try:
        set_job_state(job_id, "running", progress_json={"stage": "start"})
    except Exception:
        pass

    try:
        try:
            mark_attempt_running(attempt_id)
        except Exception:
            pass

        if _should_cancel(attempt_id):
            try:
                set_job_state(job_id, "cancelled", progress_json={"stage": "cancelled"})
            except Exception:
                pass
            return

        try:
            update_job_progress(job_id, {"stage": "download_pdf"})
        except Exception:
            pass

        tmp_pdf_path: Optional[Path] = None
        try:
            tmp_pdf_path = get_pdf_temp_path(
                doc_id=task.doc_id,
                work_id=work_id,
                project_id=task.project_id,
                document=None,
            )

            if _should_cancel(attempt_id):
                try:
                    mark_attempt_cancelled(attempt_id)
                except Exception:
                    pass
                try:
                    set_job_state(
                        job_id, "cancelled", progress_json={"stage": "cancelled"}
                    )
                except Exception:
                    pass
                return

            mode = "fulltext"
            tei_xml: Optional[str] = None
            quality_patch: dict = {}

            grobid_error: Optional[str] = None
            if task.force_fallback:
                mode = "fallback"
            else:
                try:
                    update_job_progress(job_id, {"stage": "grobid_fulltext"})
                except Exception:
                    pass

                try:
                    tei_xml = grobid_client.extract_tei_fulltext(tmp_pdf_path)
                except grobid_client.GrobidError as exc:
                    grobid_error = str(exc)
                    header_xml = None
                    try:
                        update_job_progress(job_id, {"stage": "grobid_header"})
                    except Exception:
                        pass
                    try:
                        header_xml = grobid_client.extract_tei_header(tmp_pdf_path)
                    except grobid_client.GrobidError as exc2:
                        grobid_error = str(exc2)
                        quality_patch = {"primary": {"header_failed": True}}
                        header_xml = None

                    try:
                        update_job_progress(job_id, {"stage": "grobid_refs"})
                    except Exception:
                        pass

                    refs_xml = None
                    try:
                        refs_xml = grobid_client.extract_tei_references(tmp_pdf_path)
                    except grobid_client.GrobidError as exc3:
                        grobid_error = str(exc3)
                        mode = "fallback"

                    if mode != "fallback" and refs_xml is not None:
                        if header_xml:
                            mode = "header+references"
                            tei_xml = grobid_client.merge_header_and_references_tei(
                                header_xml=header_xml,
                                references_xml=refs_xml,
                            )
                        else:
                            mode = "refs-only"
                            quality_patch = {
                                **(quality_patch or {}),
                                "primary": {
                                    **((quality_patch or {}).get("primary") or {}),
                                    "refs_only": True,
                                },
                            }
                            tei_xml = refs_xml

            extraction_payload = None
            if mode != "fallback" and tei_xml:
                try:
                    update_job_progress(job_id, {"stage": "parse_tei", "mode": mode})
                except Exception:
                    pass
                extraction_payload = extraction.parse_tei(tei_xml)

                # Persist TEI + extraction JSON
                tei_key = f"extract/{work_id}/attempts/{attempt_id}/primary/tei.xml"
                tei_bytes = tei_xml.encode("utf-8", errors="ignore")
                object_store_s3.put_bytes(
                    tei_key, tei_bytes, content_type="application/xml"
                )
                create_artifact(
                    task.project_id,
                    task.user_id,
                    attempt_id,
                    "tei.xml",
                    tei_key,
                    len(tei_bytes),
                    "application/xml",
                )

            if mode == "fallback":
                try:
                    stage = (
                        "fallback_worker"
                        if str(
                            getattr(settings, "FALLBACK_WORKER_URL", "") or ""
                        ).strip()
                        else "fallback_text_layer"
                    )
                    update_job_progress(job_id, {"stage": stage})
                except Exception:
                    pass

                try:
                    quality_patch = {
                        **(quality_patch or {}),
                        **_write_fallback_artifacts(
                            work_id=work_id,
                            attempt_id=attempt_id,
                            project_id=task.project_id,
                            user_id=task.user_id,
                            pdf_path=tmp_pdf_path,
                        ),
                    }
                except RuntimeError as exc:
                    if "attempt_cancelled" in str(exc):
                        try:
                            mark_attempt_cancelled(attempt_id)
                        except Exception:
                            pass
                        try:
                            set_job_state(
                                job_id,
                                "cancelled",
                                progress_json={"stage": "cancelled"},
                            )
                        except Exception:
                            pass
                        return
                    raise
                extraction_payload = {
                    "extraction_version": "fallback-v1",
                    "metadata": {
                        "title": None,
                        "authors": [],
                        "year": None,
                        "journal": None,
                        "container": None,
                        "doi": None,
                        "url": None,
                    },
                    "citations": [],
                    "references": [],
                    "fallback": {"error": grobid_error or ""},
                }

            # Persist extraction.json always (spine source of truth)
            extraction_key = (
                f"extract/{work_id}/attempts/{attempt_id}/primary/extraction.json"
            )
            payload_bytes = json.dumps(
                extraction_payload or {},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            object_store_s3.put_bytes(
                extraction_key,
                payload_bytes,
                content_type="application/json",
            )
            create_artifact(
                task.project_id,
                task.user_id,
                attempt_id,
                "extraction.json",
                extraction_key,
                len(payload_bytes),
                "application/json",
            )

            # Index extracted references into the project-scoped graph so work-level
            # links and placeholders appear without manual reindexing.
            try:
                graph = GraphStore(
                    settings=SimpleNamespace(DEFAULT_PROJECT_ID=task.project_id)
                )
                ingest_meta = build_ingested_document_from_spine(
                    work_id=work_id,
                    project_id=task.project_id,
                    include_extraction_data=False,
                ) or {"id": work_id, "project_id": task.project_id}
                graph.index_ingest_upload(ingest_meta)
                graph.index_extraction(
                    ingest_meta=ingest_meta,
                    extraction_data=extraction_payload or {},
                )
            except Exception:
                # Best-effort: extraction should not fail because graph indexing failed.
                pass

            if quality_patch:
                try:
                    merge_attempt_quality_json(attempt_id, quality_patch)
                except Exception:
                    pass

            try:
                if mode == "fulltext":
                    mark_attempt_succeeded(attempt_id)
                else:
                    mark_attempt_partial(attempt_id)
            except Exception:
                pass
            try:
                set_job_state(
                    job_id,
                    "succeeded" if mode == "fulltext" else "partial",
                    progress_json={"stage": "done", "mode": mode},
                )
            except Exception:
                pass
        finally:
            cleanup_temp_path(tmp_pdf_path)
    except Exception as exc:
        try:
            mark_attempt_failed(
                attempt_id, failure_reason=type(exc).__name__, failure_detail=str(exc)
            )
        except Exception:
            pass
        try:
            set_job_state(
                job_id, "failed", progress_json={"stage": "error", "error": str(exc)}
            )
        except Exception:
            pass


class SpineExtractionPool:
    def __init__(self, *, max_workers: int = 1, max_queue: int = 64) -> None:
        """Create an in-process extraction worker pool."""
        self._max_workers = max(1, int(max_workers or 1))
        self._queue: queue.Queue[ExtractionTask] = queue.Queue(
            maxsize=max(1, int(max_queue or 1))
        )
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
                    target=self._worker, daemon=True, name=f"spine-extract-{i+1}"
                )
                t.start()
                self._threads.append(t)
            self._started = True

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                _run_task(task)
            finally:
                self._queue.task_done()

    def enqueue(
        self, *, doc_id: str, project_id: str, user_id: str, force_fallback: bool
    ) -> Optional[str]:
        did = str(doc_id or "").strip()
        pid = str(project_id or "").strip()
        uid = str(user_id or "").strip()
        if not did or not pid or not uid:
            return None

        attempt_id, _state = create_or_get_attempt(
            pid, uid, did, "primary", settings_json={}
        )
        job_id = create_job(
            pid,
            uid,
            attempt_id,
            worker="extract",
            state="queued",
            progress_json={"stage": "queued"},
        )

        self._ensure_started()
        try:
            self._queue.put_nowait(
                ExtractionTask(
                    doc_id=did,
                    project_id=pid,
                    user_id=uid,
                    force_fallback=bool(force_fallback),
                    job_id=job_id,
                    attempt_id=attempt_id,
                )
            )
        except queue.Full:
            try:
                set_job_state(job_id, "failed", progress_json={"stage": "queue_full"})
            except Exception:
                pass
            return None

        return job_id
