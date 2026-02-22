"""Happy-path workflow orchestrator (Phase 10-02).

This module owns background progression for reviewer "chase" runs.

Design constraints:
- Immutable stage artifacts are written once via
  `pipeline_contracts_service.store_stage()`.
- Mutable progress/retries/cancellation are tracked in `pipeline_run_status` tables.
- Global pause is obeyed via `background_state.get_state().get('paused')`.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque

from backend import background_state, attachment_store
from backend.db import connect
from backend.pipeline_contracts import service as pipeline_contracts_service
from backend.settings import settings as app_settings
from backend.span_graph_store import SpanGraphStore
from backend.spine.ids import work_id_from_doc_id
from backend.spine.pipeline_artifacts import StageArtifactAlreadyExists
from backend.spine import pipeline_run_scopes, pipeline_run_status

from .builders import build_citespans_data, build_extract_data


logger = logging.getLogger(__name__)


_SCOPE_TYPE = "claimspan"
_CLAIM_ID_RE = re.compile(r"^cite:(?P<doc>[^:]+):(?P<idx>\d+):")


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_citation_index(claim_id: str) -> int | None:
    cid = str(claim_id or "").strip()
    m = _CLAIM_ID_RE.match(cid)
    if not m:
        parts = cid.split(":")
        if len(parts) >= 3 and parts[0] == "cite" and str(parts[2]).isdigit():
            try:
                return int(parts[2])
            except Exception:
                return None
        return None
    try:
        return int(m.group("idx"))
    except Exception:
        return None


def _attachment_for_target(*, citing_doc_id: str, target_id: str) -> str | None:
    doc_id = str(citing_doc_id or "").strip()
    tid = str(target_id or "").strip()
    if not doc_id or not tid:
        return None

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attachment_id, status
                  FROM attachments
                 WHERE archived=false
                   AND doc_id=%s
                   AND target_id=%s
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (doc_id, tid),
            )
            row = cur.fetchone()
    if not row:
        return None
    attachment_id, status = row
    if str(status or "").strip() != attachment_store.STATUS_MATCHED:
        return None
    return str(attachment_id or "").strip() or None


def _stage_patch(
    *,
    stage: str,
    state: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    message: str | None = None,
    progress: dict | None = None,
) -> dict:
    entry: dict[str, Any] = {"state": str(state)}
    if started_at is not None:
        entry["started_at"] = started_at
    if finished_at is not None:
        entry["finished_at"] = finished_at
    if message is not None:
        entry["message"] = message
    if progress is not None:
        entry["progress"] = progress
    return {"stages": {str(stage): entry}}


class HappyPathOrchestrator:
    def __init__(self, *, workers: int = 1) -> None:
        self._lock = threading.RLock()
        self._queue: Deque[str] = deque()
        self._queued: set[str] = set()
        self._cv = threading.Condition(self._lock)

        n = max(1, int(workers or 1))
        self._workers: list[threading.Thread] = []
        for idx in range(n):
            t = threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"happy-path-worker-{idx+1}",
            )
            t.start()
            self._workers.append(t)

        self._span_graph_store = SpanGraphStore(settings=app_settings)

    def enqueue(self, run_id: str) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        with self._cv:
            if rid in self._queued:
                return
            self._queue.append(rid)
            self._queued.add(rid)
            self._cv.notify()

    def _worker(self) -> None:  # pragma: no cover - background loop
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait(timeout=1.0)
                rid = self._queue.popleft()
                self._queued.discard(rid)

            try:
                self._run_one(rid)
            except Exception:
                logger.exception("Happy-path run failed for run_id=%s", rid)

    def _run_one(self, run_id: str) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return

        run = pipeline_run_status.get_run_status(rid)
        if run is None:
            return

        scope_type = str(run.get("scope_type") or "").strip() or _SCOPE_TYPE
        scope_id = str(run.get("scope_id") or "").strip()
        reviewer_uid = str(run.get("reviewer_uid") or "").strip() or "default"
        citing_doc_id = str(run.get("citing_doc_id") or "").strip()
        if not scope_id or not citing_doc_id:
            pipeline_run_status.upsert_run_status(
                rid,
                scope_type=scope_type,
                scope_id=scope_id or "(missing)",
                reviewer_uid=reviewer_uid,
                citing_doc_id=citing_doc_id or "(missing)",
                state="error",
                error_json={
                    "code": "missing_scope",
                    "message": "Missing scope_id/citing_doc_id",
                },
            )
            return

        if background_state.get_state().get("paused"):
            pipeline_run_status.upsert_run_status(
                rid,
                scope_type=scope_type,
                scope_id=scope_id,
                reviewer_uid=reviewer_uid,
                citing_doc_id=citing_doc_id,
                state="blocked",
            )
            pipeline_run_status.append_event(
                rid, type="run_blocked", payload={"reason": "paused"}
            )
            return

        started_at = _utcnow_z()
        pipeline_run_status.upsert_run_status(
            rid,
            scope_type=scope_type,
            scope_id=scope_id,
            reviewer_uid=reviewer_uid,
            citing_doc_id=citing_doc_id,
            state="running",
            started_at=str(run.get("started_at") or "").strip() or started_at,
        )
        pipeline_run_status.append_event(rid, type="run_started", payload={})

        # Build/write extract and citespans (write-once).
        extract_started = _utcnow_z()
        extract_data = build_extract_data(citing_doc_id=citing_doc_id)
        try:
            pipeline_contracts_service.store_stage(
                run_id=rid,
                stage="extract",
                status="complete",
                data=dict(extract_data),
            )
        except StageArtifactAlreadyExists:
            pass
        extract_finished = _utcnow_z()

        citation_index = _parse_citation_index(scope_id)
        anchors = list(extract_data.get("citation_anchors") or [])
        if citation_index is not None:
            anchors = [
                a
                for a in anchors
                if isinstance(a, dict) and a.get("citation_index") == citation_index
            ]

        citespans_started = _utcnow_z()
        citespans_data = build_citespans_data(
            span_graph_store=self._span_graph_store,
            citing_doc_id=citing_doc_id,
            citation_anchors=anchors,
        )
        try:
            pipeline_contracts_service.store_stage(
                run_id=rid,
                stage="citespans",
                status="complete",
                data=dict(citespans_data),
            )
        except StageArtifactAlreadyExists:
            pass
        citespans_finished = _utcnow_z()

        targets = pipeline_run_status.list_target_status(rid)
        for tgt in targets:
            tid = str(tgt.get("target_id") or "").strip()
            if not tid:
                continue
            if str(tgt.get("state") or "") == "cancelled":
                continue
            pipeline_run_status.upsert_target_status(
                rid,
                tid,
                state=str(tgt.get("state") or "requested"),
                citation_index=tgt.get("citation_index"),
                reference_id=tgt.get("reference_id"),
                attachment_id=tgt.get("attachment_id"),
                stage_state=_stage_patch(
                    stage="extract",
                    state="done",
                    started_at=extract_started,
                    finished_at=extract_finished,
                ),
            )
            pipeline_run_status.upsert_target_status(
                rid,
                tid,
                state=str(tgt.get("state") or "requested"),
                citation_index=tgt.get("citation_index"),
                reference_id=tgt.get("reference_id"),
                attachment_id=tgt.get("attachment_id"),
                stage_state=_stage_patch(
                    stage="citespans",
                    state="done",
                    started_at=citespans_started,
                    finished_at=citespans_finished,
                ),
            )

        # Refresh attachment availability for all targets.
        targets = pipeline_run_status.list_target_status(rid)
        missing = 0
        for tgt in targets:
            tid = str(tgt.get("target_id") or "").strip()
            if not tid:
                continue
            if str(tgt.get("state") or "") == "cancelled":
                continue

            attachment_id = _attachment_for_target(
                citing_doc_id=citing_doc_id, target_id=tid
            )
            if attachment_id:
                next_state = str(tgt.get("state") or "").strip()
                if next_state in {"requested", "blocked"}:
                    next_state = "available"
                pipeline_run_status.upsert_target_status(
                    rid,
                    tid,
                    state=next_state or "available",
                    citation_index=tgt.get("citation_index"),
                    reference_id=tgt.get("reference_id"),
                    attachment_id=attachment_id,
                )
            else:
                missing += 1
                pipeline_run_status.upsert_target_status(
                    rid,
                    tid,
                    state=str(tgt.get("state") or "requested") or "requested",
                    citation_index=tgt.get("citation_index"),
                    reference_id=tgt.get("reference_id"),
                    attachment_id=None,
                    stage_state=_stage_patch(
                        stage="retrieval",
                        state="blocked",
                        message="Awaiting cited PDF attachment",
                    ),
                )

        if missing > 0:
            pipeline_run_status.upsert_run_status(
                rid,
                scope_type=scope_type,
                scope_id=scope_id,
                reviewer_uid=reviewer_uid,
                citing_doc_id=citing_doc_id,
                state="blocked",
            )
            pipeline_run_status.append_event(
                rid,
                type="run_blocked",
                payload={"missing_attachments": int(missing)},
            )
            return

        # TODO(10-02): Run candidate stages when all targets are runnable.
        pipeline_run_status.upsert_run_status(
            rid,
            scope_type=scope_type,
            scope_id=scope_id,
            reviewer_uid=reviewer_uid,
            citing_doc_id=citing_doc_id,
            state="complete",
            finished_at=_utcnow_z(),
        )
        pipeline_run_status.append_event(rid, type="run_finished", payload={})


_ORCH = HappyPathOrchestrator(
    workers=max(1, int(getattr(app_settings, "HAPPY_PATH_WORKERS", 1) or 1))
)


def start_run_for_claimspan(
    *, claim_id: str, reviewer_uid: str, citing_doc_id: str
) -> dict:
    cid = str(claim_id or "").strip()
    ruid = str(reviewer_uid or "").strip() or "default"
    doc_id = str(citing_doc_id or "").strip()
    if not cid:
        raise ValueError("claim_id is required")
    if not doc_id:
        raise ValueError("citing_doc_id is required")

    work_id = work_id_from_doc_id(doc_id)
    run = pipeline_contracts_service.create_run(work_id=work_id, note="happy-path")
    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        raise RuntimeError("create_run did not return run_id")

    pipeline_run_scopes.insert_scope(
        scope_type=_SCOPE_TYPE,
        scope_id=cid,
        reviewer_uid=ruid,
        run_id=run_id,
        citing_doc_id=doc_id,
    )
    pipeline_run_status.upsert_run_status(
        run_id,
        scope_type=_SCOPE_TYPE,
        scope_id=cid,
        reviewer_uid=ruid,
        citing_doc_id=doc_id,
        state="queued" if not background_state.get_state().get("paused") else "blocked",
    )

    citation_index = _parse_citation_index(cid)
    extract_data = build_extract_data(citing_doc_id=doc_id)
    anchors = list(extract_data.get("citation_anchors") or [])
    if citation_index is not None:
        anchors = [
            a
            for a in anchors
            if isinstance(a, dict) and a.get("citation_index") == citation_index
        ]

    if not anchors:
        idx = citation_index if citation_index is not None else 0
        anchors = [
            {"citation_index": int(idx), "reference_id": None, "target_id": None}
        ]

    # Initialize targets in citation order.
    for pos, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            continue
        reference_id = str(anchor.get("reference_id") or "").strip() or None
        target_id = str(anchor.get("target_id") or "").strip() or reference_id
        if not target_id:
            idx = citation_index if citation_index is not None else 0
            target_id = f"ref:{doc_id}:{idx}:{pos}"

        attachment_id = _attachment_for_target(
            citing_doc_id=doc_id, target_id=target_id
        )
        state = "available" if attachment_id else "requested"
        pipeline_run_status.upsert_target_status(
            run_id,
            target_id,
            state=state,
            citation_index=anchor.get("citation_index"),
            reference_id=reference_id,
            attachment_id=attachment_id,
        )

    pipeline_run_status.append_event(run_id, type="run_queued", payload={})
    if background_state.get_state().get("paused"):
        pipeline_run_status.append_event(
            run_id, type="run_blocked", payload={"reason": "paused"}
        )
        return {"run_id": run_id}

    _ORCH.enqueue(run_id)
    return {"run_id": run_id}


def resume_run(run_id: str) -> dict:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    run = pipeline_run_status.get_run_status(rid)
    if run is None:
        raise KeyError("run not found")
    if background_state.get_state().get("paused"):
        pipeline_run_status.upsert_run_status(
            rid,
            scope_type=str(run.get("scope_type") or _SCOPE_TYPE),
            scope_id=str(run.get("scope_id") or ""),
            reviewer_uid=str(run.get("reviewer_uid") or "default"),
            citing_doc_id=str(run.get("citing_doc_id") or ""),
            state="blocked",
        )
        return {"run_id": rid, "blocked": True}
    pipeline_run_status.upsert_run_status(
        rid,
        scope_type=str(run.get("scope_type") or _SCOPE_TYPE),
        scope_id=str(run.get("scope_id") or ""),
        reviewer_uid=str(run.get("reviewer_uid") or "default"),
        citing_doc_id=str(run.get("citing_doc_id") or ""),
        state="queued",
    )
    pipeline_run_status.append_event(rid, type="run_resumed", payload={})
    _ORCH.enqueue(rid)
    return {"run_id": rid}


def cancel_target(run_id: str, target_id: str) -> dict:
    rid = str(run_id or "").strip()
    tid = str(target_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not tid:
        raise ValueError("target_id is required")

    current = pipeline_run_status.get_target_status(rid, tid)
    if current is None:
        raise KeyError("target not found")
    pipeline_run_status.upsert_target_status(
        rid,
        tid,
        state="cancelled",
        citation_index=current.get("citation_index"),
        reference_id=current.get("reference_id"),
        attachment_id=current.get("attachment_id"),
    )
    pipeline_run_status.append_event(
        rid, type="target_cancelled", target_id=tid, payload={}
    )
    return {"run_id": rid, "target_id": tid, "state": "cancelled"}


__all__ = ["start_run_for_claimspan", "resume_run", "cancel_target"]
