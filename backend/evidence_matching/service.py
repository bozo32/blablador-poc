"""High-level orchestration for evidence reruns and FastAPI consumption."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Sequence
from uuid import uuid4

from backend import attachment_store, background_state
from backend.settings import (
    AppSettings,
    apply_execution_profile,
    settings as app_settings,
)

from . import deterministic_matcher, loaders, serializers
from .store import EvidenceRunStore

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import EvidencePipeline

logger = logging.getLogger(__name__)


class EvidenceMatchingService:
    """Coordinate evidence pipeline runs, persistence, and rerun queueing."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        pipeline: "EvidencePipeline" | None = None,
        store: EvidenceRunStore | None = None,
        max_workers: int | None = None,
        load_windows: Callable[[str], Sequence[Any]] | None = None,
        seed_windows: Callable[..., Sequence[Any]] | None = None,
    ) -> None:
        """Initialize the service with optional dependency overrides."""
        self.settings = settings or app_settings
        # Lazily construct the pipeline; importing it pulls in heavy ML deps.
        self.pipeline = pipeline
        self.store = store or EvidenceRunStore(settings=self.settings)
        self.max_workers = max(
            1, int(max_workers or getattr(self.settings, "EVIDENCE_RERUN_WORKERS", 1))
        )
        self._load_claim_windows = load_windows or loaders.load_claim_windows
        self._seed_windows = seed_windows or deterministic_matcher.seed_windows
        self._lock = threading.Lock()
        self._pending_jobs: deque[dict[str, Any]] = deque()
        self._active_claims: dict[str, dict[str, Any]] = {}
        self._claim_text_cache: dict[str, str] = {}

    def _background_paused(self) -> bool:
        try:
            return bool(background_state.get_state().get("paused"))
        except Exception:  # pragma: no cover - pause state failures shouldn't block
            return False

    def _kick_dequeue_locked(self) -> None:
        """Start queued work when not paused and capacity is available.

        Must be called with self._lock held.
        """
        if self._background_paused():
            return
        while len(self._active_claims) < self.max_workers:
            next_job = self._dequeue_job()
            if not next_job:
                return
            self._start_job(next_job)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_current_run(
        self,
        claim_id: str,
        *,
        claim_text: str | None = None,
        cited_attachment_ids: Sequence[str] | None = None,
        force: bool = False,
        execute: bool = True,
    ) -> dict:
        """Run the pipeline immediately when attachment state is stale."""
        if not claim_id:
            raise ValueError("claim_id is required")

        if claim_text and str(claim_text).strip():
            self._claim_text_cache[claim_id] = str(claim_text).strip()

        with self._lock:
            self._kick_dequeue_locked()

        latest = self.store.latest_run(claim_id)
        snapshot = self._attachment_snapshot(claim_id)
        if not force and latest:
            previous_state = (latest.get("metadata") or {}).get("attachments_state")
            if previous_state == snapshot:
                return latest

        if not execute:
            # Caller only wants to register claim_text and inspect current state.
            # Avoid triggering expensive NLI work on read endpoints.
            return latest or {
                "claim_id": claim_id,
                "run_id": None,
                "created_at": None,
                "metadata": {},
                "summary": {"total": 0, "label_counts": {}},
                "candidates": [],
            }

        resolved_claim_text = self._resolve_claim_text(claim_id, claim_text, latest)
        return self._execute_run(
            claim_id,
            claim_text=resolved_claim_text,
            cited_attachment_ids=cited_attachment_ids,
            attachments_state=snapshot,
            note="ensure-current",
        )

    def list_candidates(
        self,
        claim_id: str,
        *,
        label: str | None = None,
        include_neutral: bool = True,
        offset: int = 0,
        limit: int = 10,
    ) -> dict:
        """Return paginated candidates for a claim."""

        def _normalize_label(value: str | None) -> str:
            label_norm = (value or "").strip().lower()
            if label_norm in {"entails", "entail", "entailment"}:
                return "entail"
            if label_norm in {
                "contradicts",
                "contradict",
                "contradiction",
                "refutes",
                "refute",
            }:
                return "contradict"
            if label_norm in {"neutral", "unknown", "none"}:
                return "neutral"
            return label_norm

        with self._lock:
            self._kick_dequeue_locked()

        latest = self.store.latest_run(claim_id)
        if not latest:
            return {
                "candidates": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "run": None,
                "lock_state": self._lock_state(claim_id),
            }

        candidates = list(latest.get("candidates", []))
        if label:
            desired = _normalize_label(label)
            candidates = [
                cand
                for cand in candidates
                if _normalize_label(cand.get("label")) == desired
            ]
        if not include_neutral:
            candidates = [
                cand
                for cand in candidates
                if _normalize_label(cand.get("label")) != "neutral"
            ]

        total = len(candidates)
        window = candidates[offset : offset + limit if limit else None]
        return {
            "candidates": window,
            "total": total,
            "offset": offset,
            "limit": limit,
            "run": {
                "run_id": latest.get("run_id"),
                "created_at": latest.get("created_at"),
            },
            "lock_state": self._lock_state(claim_id),
        }

    def request_rerun(
        self,
        claim_id: str,
        *,
        claim_text: str | None = None,
        note: str | None = None,
        advanced_settings: dict | None = None,
    ) -> dict:
        """Enqueue a rerun, spawning background workers up to the concurrency limit."""
        job = {
            "job_id": uuid4().hex,
            "claim_id": claim_id,
            "claim_text": claim_text,
            "note": note or "manual",
            "advanced_settings": dict(advanced_settings or {}),
            "requested_at": _utcnow(),
        }

        with self._lock:
            if (
                claim_id in self._active_claims
                or len(self._active_claims) >= self.max_workers
            ):
                self._pending_jobs.append(job)
                position = self._queue_position(claim_id, job["job_id"])
                status = "queued"
            else:
                self._start_job(job)
                position = 0
                status = "running"

            self._kick_dequeue_locked()

        return {
            "job_id": job["job_id"],
            "status": status,
            "position": position,
            "locked": True,
        }

    def trigger_auto_rerun(
        self, claim_id: str, *, claim_text: str | None = None
    ) -> dict:
        """Best-effort auto-rerun invoked by attachment lifecycle hooks."""
        if not claim_id or str(claim_id).strip().lower() == "none":
            return {
                "job_id": uuid4().hex,
                "status": "skipped",
                "position": 0,
                "locked": False,
            }
        resolved_claim_text = claim_text or self._claim_text_from_attachments(claim_id)

        if self._background_paused():
            job = {
                "job_id": uuid4().hex,
                "claim_id": claim_id,
                "claim_text": resolved_claim_text,
                "note": "auto-attachment",
                "advanced_settings": {},
                "requested_at": _utcnow(),
            }
            with self._lock:
                self._pending_jobs.append(job)
                position = self._queue_position(claim_id, job["job_id"])
            return {
                "job_id": job["job_id"],
                "status": "queued",
                "position": position,
                "locked": True,
            }

        return self.request_rerun(
            claim_id, claim_text=resolved_claim_text, note="auto-attachment"
        )

    def get_history(self, claim_id: str) -> list[dict]:
        """Expose run history for API responses."""
        with self._lock:
            self._kick_dequeue_locked()
        return self.store.history(claim_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_job(self, job: dict[str, Any]) -> None:
        claim_id = job["claim_id"]
        self._active_claims[claim_id] = {
            "job_id": job["job_id"],
            "started_at": _utcnow(),
            "note": job.get("note"),
        }
        thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            daemon=True,
            name=f"evidence-rerun-{claim_id}",
        )
        thread.start()

    def _run_job(self, job: dict[str, Any]) -> None:
        claim_id = job["claim_id"]
        try:
            latest = self.store.latest_run(claim_id)
            claim_text = self._resolve_claim_text(
                claim_id, job.get("claim_text"), latest
            )
            snapshot = self._attachment_snapshot(claim_id)
            self._execute_run(
                claim_id,
                claim_text=claim_text,
                cited_attachment_ids=None,
                attachments_state=snapshot,
                note=job.get("note"),
                advanced_settings=job.get("advanced_settings"),
            )
        except Exception as exc:  # pragma: no cover - logged for observability
            logger.exception("Evidence rerun failed for claim %s: %s", claim_id, exc)
        finally:
            with self._lock:
                self._active_claims.pop(claim_id, None)
                self._kick_dequeue_locked()

    def _dequeue_job(self) -> dict[str, Any] | None:
        while self._pending_jobs:
            job = self._pending_jobs.popleft()
            if job["claim_id"] in self._active_claims:
                # Claim currently running; push back to queue tail
                self._pending_jobs.append(job)
                continue
            return job
        return None

    def _queue_position(self, claim_id: str, job_id: str) -> int:
        position = 0
        for pending in self._pending_jobs:
            if pending["job_id"] == job_id:
                return position
            if pending["claim_id"] == claim_id:
                position += 1
        return position

    def _execute_run(
        self,
        claim_id: str,
        *,
        claim_text: str,
        cited_attachment_ids: Sequence[str] | None,
        attachments_state: list[dict[str, Any]],
        note: str | None = None,
        advanced_settings: dict | None = None,
    ) -> dict:
        advanced = dict(advanced_settings or {})
        requested_profile = advanced.get("profile")
        (
            effective_settings,
            resolved_profile,
            profile_overrides,
        ) = apply_execution_profile(self.settings, requested_profile)
        if resolved_profile:
            advanced["profile"] = resolved_profile
        if profile_overrides:
            advanced.setdefault("profile_overrides", dict(profile_overrides))

        # Phase 08-08: allow per-rerun HF remote toggle.
        # This sets the per-run settings copy without mutating global settings.
        if "hf_remote" in advanced:
            advanced["hf_remote"] = bool(advanced.get("hf_remote"))
            if advanced["hf_remote"]:
                try:
                    effective_settings = effective_settings.model_copy(
                        update={"HF_REMOTE_INFERENCE": True}
                    )
                except Exception:
                    payload = dict(effective_settings.model_dump())
                    payload["HF_REMOTE_INFERENCE"] = True
                    effective_settings = AppSettings(**payload)

        windows = self._load_claim_windows(claim_id)
        seeds = self._seed_windows(
            claim_text,
            windows,
            cited_attachment_ids=cited_attachment_ids,
            settings_override=effective_settings,
        )

        if effective_settings is self.settings and self.pipeline is not None:
            pipeline = self.pipeline
        else:
            from .pipeline import EvidencePipeline

            pipeline = EvidencePipeline(settings=effective_settings)
            if effective_settings is self.settings and self.pipeline is None:
                self.pipeline = pipeline
        candidates = pipeline.run(
            claim_id=claim_id,
            claim_text=claim_text,
            seeds=seeds,
        )
        serialized = serializers.serialize_candidates(candidates)
        metadata = {
            "claim_text": claim_text,
            "attachments_state": attachments_state,
            "note": note,
            "advanced_settings": advanced,
        }
        return self.store.record_run(claim_id, candidates=serialized, metadata=metadata)

    def _attachment_claim_aliases(self, claim_id: str) -> list[str]:
        """Return claim_id variants that may share attachments.

        Historically, citation-derived claims used ids like:
          cite:{doc_id}:{citation_index}:{segment_id}

        Phase 09 introduced reviewer-scoped claim ids to avoid draft/segmentation
        overwrites:
          cite:{doc_id}:{citation_index}:{reviewer_uid}:{segment_id}

        Attachments placed before this change may still be recorded against the
        legacy (non-reviewer) claim id. To preserve walkthrough usability and
        backward compatibility, evidence runs for reviewer-scoped claim ids
        fall back to the legacy id when no direct attachments exist.
        """
        canonical = str(claim_id or "").strip()
        if not canonical:
            return []
        aliases = [canonical]
        parts = canonical.split(":")
        if (
            len(parts) == 5
            and parts[0] == "cite"
            and parts[1]
            and str(parts[2]).isdigit()
            and parts[4]
        ):
            legacy = ":".join([parts[0], parts[1], parts[2], parts[4]])
            if legacy and legacy not in aliases:
                aliases.append(legacy)
        return aliases

    def _attachment_snapshot(self, claim_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cid in self._attachment_claim_aliases(claim_id):
            for record in attachment_store.list_attachments(claim_id=cid):
                rid = str(record.get("id") or "")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                records.append(record)
        snapshot: list[dict[str, Any]] = []
        for record in records:
            if not attachment_store.is_ready(record):
                continue
            snapshot.append(
                {
                    "id": record.get("id"),
                    "updated_at": record.get("updated_at"),
                    "status": record.get("status"),
                }
            )
        snapshot.sort(key=lambda entry: entry.get("id") or "")
        return snapshot

    def _resolve_claim_text(
        self,
        claim_id: str,
        supplied: str | None,
        latest_run: dict | None,
    ) -> str:
        if supplied:
            return supplied
        if latest_run:
            stored = (latest_run.get("metadata") or {}).get("claim_text")
            if stored:
                return stored
        cached = self._claim_text_cache.get(claim_id)
        if cached:
            return cached
        attachment_text = self._claim_text_from_attachments(claim_id)
        if attachment_text:
            return attachment_text
        raise ValueError(
            (
                "claim_text is required for claim "
                f"{claim_id!r}. Provide it via `/claims/{claim_id}/evidence` "
                "or include it when uploading an attachment before rerunning."
            )
        )

    def _claim_text_from_attachments(self, claim_id: str) -> str | None:
        for cid in self._attachment_claim_aliases(claim_id):
            for record in attachment_store.list_attachments(claim_id=cid):
                text = record.get("claim_text")
                if text:
                    return text
        return None

    def _lock_state(self, claim_id: str) -> dict[str, Any]:
        with self._lock:
            if claim_id in self._active_claims:
                return {"status": "running", "locked": True}
            if any(
                job for job in self._pending_jobs if job.get("claim_id") == claim_id
            ):
                return {"status": "queued", "locked": True}
        return {"status": "idle", "locked": False}


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


evidence_service = EvidenceMatchingService()

__all__ = ["EvidenceMatchingService", "evidence_service"]
