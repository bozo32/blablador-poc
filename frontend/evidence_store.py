"""Session-backed evidence store coordinating UI state and API calls.

Phase 10-04 moves evidence decisions (pin/triage) to durable, server-backed
append-only events. Streamlit session state is no longer the source of truth for
pins/accept/reject.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Dict, List, Optional

import streamlit as st

from frontend import evidence_api

SESSION_KEY = "_evidence_store_state"
STORE_INSTANCE_KEY = "_evidence_store_instance"
DEFAULT_PAGE_SIZE = 5


def _toast(ui: Any, message: str, *, icon: str = "ℹ️", quiet: bool = False) -> None:
    if quiet:
        return
    toast = getattr(ui, "toast", None)
    if callable(toast):  # pragma: no cover - Streamlit runtime only
        toast(message, icon=icon)
        return
    warn = getattr(ui, "warning", None)
    if callable(warn):  # pragma: no cover - Streamlit runtime only
        warn(message)


@dataclass
class EvidenceStore:
    session_state: Any = None
    api: Any = evidence_api
    ui: Any = st
    page_size: int = DEFAULT_PAGE_SIZE
    max_total: int = evidence_api.MAX_TOTAL_CANDIDATES

    def __post_init__(self) -> None:
        """Bind session defaults and precompute paging guardrails."""
        if self.session_state is None:
            self.session_state = st.session_state
        self.session_state.setdefault(
            SESSION_KEY, {"claims": {}, "active_claim_id": None}
        )
        self.max_extra_pages = max(0, (self.max_total // self.page_size) - 1)

    def _normalize_claim_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        normalized = str(text).strip()
        return normalized or None

    def _resolve_claim_text(
        self, claim_state: Dict[str, Any], claim_text: Optional[str]
    ) -> Optional[str]:
        metadata = claim_state.setdefault("metadata", {})
        normalized = self._normalize_claim_text(claim_text)
        if normalized:
            metadata["claim_text"] = normalized
            return normalized
        stored = self._normalize_claim_text(metadata.get("claim_text"))
        if stored:
            return stored
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_claim_state(self, claim_id: str, **metadata: Any) -> Dict[str, Any]:
        root = self._root()
        claims = root.setdefault("claims", {})
        claim_state = claims.setdefault(claim_id, self._default_claim_state(claim_id))
        if metadata:
            claim_state.setdefault("metadata", {}).update(
                {k: v for k, v in metadata.items() if v is not None}
            )
        return claim_state

    def set_active_claim(self, claim_id: Optional[str]) -> Optional[str]:
        root = self._root()
        root["active_claim_id"] = claim_id
        if claim_id:
            self.ensure_claim_state(claim_id)
        return claim_id

    def active_claim_id(self) -> Optional[str]:
        return self._root().get("active_claim_id")

    def sync_for_claim(
        self,
        claim_id: str,
        *,
        claim_text: Optional[str] = None,
        reviewer_uid: str = "default",
        force: bool = False,
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        normalized_provided = self._normalize_claim_text(claim_text)
        stored_metadata_text = self._normalize_claim_text(
            (claim_state.get("metadata") or {}).get("claim_text")
        )
        if normalized_provided and normalized_provided != stored_metadata_text:
            force = True
            claim_state["stale"] = True
        if not force and not claim_state.get("stale") and claim_state.get("candidates"):
            return claim_state
        if claim_state.get("inflight_fetches", 0) >= evidence_api.MAX_LIST_REQUESTS:
            _toast(self.ui, "Evidence fetch already running for this claim.")
            return claim_state
        claim_state["is_loading"] = True
        claim_state["inflight_fetches"] = claim_state.get("inflight_fetches", 0) + 1
        try:
            resolved_claim_text = self._resolve_claim_text(claim_state, claim_text)
            payload = self.api.list_evidence(
                claim_id,
                limit=self._current_limit(claim_state),
                label=claim_state["filters"].get("label"),
                include_neutral=claim_state["filters"].get("include_neutral", True),
                pinned_only=claim_state["filters"].get("pinned_only", False),
                claim_text=resolved_claim_text,
                reviewer_uid=reviewer_uid,
            )
            self.update_from_payload(claim_id, payload)
            claim_state["stale"] = False
            history = self.api.fetch_history(claim_id, limit=5)
            claim_state["history"] = history.get("runs", [])
            claim_state["last_error"] = None

            # Best-effort: hydrate timeline/projection when claim is active.
            if self.active_claim_id() == claim_id and claim_state.get(
                "decisions_stale", True
            ):
                self.sync_decisions(claim_id, reviewer_uid=reviewer_uid)
        except evidence_api.EvidenceApiError as exc:
            claim_state["last_error"] = str(exc)
        finally:
            claim_state["is_loading"] = False
            claim_state["inflight_fetches"] = max(
                0, claim_state.get("inflight_fetches", 1) - 1
            )
        return claim_state

    def sync_decisions(
        self,
        claim_id: str,
        *,
        reviewer_uid: str = "default",
        force: bool = False,
        events_limit: int = 20,
    ) -> Optional[Dict[str, Any]]:
        claim_state = self.ensure_claim_state(claim_id)
        if (
            not force
            and claim_state.get("decisions") is not None
            and not claim_state.get("decisions_stale", True)
        ):
            return claim_state.get("decisions")

        claim_state["decisions_error"] = None
        try:
            decisions = self.api.get_evidence_decisions(
                claim_id,
                reviewer_uid=str(reviewer_uid or "default").strip() or "default",
                events_limit=int(events_limit),
            )
        except evidence_api.EvidenceApiError as exc:
            claim_state["decisions_error"] = str(exc)
            return None
        claim_state["decisions"] = decisions
        claim_state["decisions_stale"] = False
        meta = claim_state.setdefault("decisions_meta", {})
        if isinstance(decisions, dict) and decisions.get("version") is not None:
            meta["version"] = decisions.get("version")
        meta["reviewer_uid"] = str(reviewer_uid or "default").strip() or "default"
        return decisions

    def apply_filter(
        self,
        claim_id: str,
        *,
        label: Optional[str] = None,
        include_neutral: Optional[bool] = None,
        pinned_only: Optional[bool] = None,
        claim_text: Optional[str] = None,
        reviewer_uid: str = "default",
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        filters = claim_state.setdefault("filters", {})
        if label is not None:
            filters["label"] = None if filters.get("label") == label else label
        if include_neutral is not None:
            filters["include_neutral"] = include_neutral
        if pinned_only is not None:
            filters["pinned_only"] = pinned_only
        claim_state["load_more_pages"] = 0
        claim_state["stale"] = True
        return self.sync_for_claim(
            claim_id,
            claim_text=claim_text,
            reviewer_uid=reviewer_uid,
            force=True,
        )

    def load_more(
        self,
        claim_id: str,
        *,
        claim_text: Optional[str] = None,
        reviewer_uid: str = "default",
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        if claim_state.get("load_more_pages", 0) >= self.max_extra_pages:
            _toast(self.ui, "All available evidence candidates are already loaded.")
            return claim_state
        claim_state["load_more_pages"] = claim_state.get("load_more_pages", 0) + 1
        claim_state["stale"] = True
        return self.sync_for_claim(
            claim_id,
            claim_text=claim_text,
            reviewer_uid=reviewer_uid,
            force=True,
        )

    def queue_rerun(
        self,
        claim_id: str,
        *,
        claim_text: Optional[str] = None,
        note: Optional[str] = None,
        advanced_settings: Optional[Dict[str, Any]] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        rerun = claim_state.setdefault("rerun", self._default_rerun_state())
        if rerun.get("inflight"):
            _toast(
                self.ui,
                "Rerun already requested for this claim.",
                icon="⚠️",
                quiet=quiet,
            )
            return rerun
        rerun["inflight"] = True
        try:
            resolved_claim_text = self._resolve_claim_text(claim_state, claim_text)
            job = self.api.request_rerun(
                claim_id,
                claim_text=resolved_claim_text,
                note=note,
                advanced_settings=advanced_settings or {},
            )
            rerun["status"] = job.get("status", "queued")
            rerun["job"] = job
            rerun.setdefault("queue", [])
            if rerun["status"] == "queued":
                rerun["queue"].append(job)
            rerun.pop("error", None)
        except evidence_api.EvidenceApiError as exc:
            rerun["status"] = "error"
            rerun["error"] = str(exc)
            _toast(self.ui, f"Rerun failed: {exc}", icon="⚠️", quiet=quiet)
        finally:
            rerun["inflight"] = False
        return rerun

    def is_selection_locked(self, claim_id: str) -> bool:
        claim_state = self.ensure_claim_state(claim_id)
        lock_state = claim_state.get("lock_state") or {}
        return bool(lock_state.get("locked"))

    def sync_selection(
        self, claim_id: str, *, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fetch persisted evidence selection for a claim and cache it in session."""
        claim_state = self.ensure_claim_state(claim_id)
        if (
            not force
            and claim_state.get("selection") is not None
            and not claim_state.get("selection_stale", True)
        ):
            return claim_state.get("selection")
        claim_state["selection_error"] = None
        try:
            selection = self.api.get_evidence_selection(claim_id)
        except evidence_api.EvidenceApiError as exc:
            claim_state["selection_error"] = str(exc)
            return None
        claim_state["selection"] = selection
        claim_state["selection_stale"] = False
        return selection

    def save_selection(
        self,
        claim_id: str,
        *,
        verdict: str,
        primary_candidate_id: Optional[str] = None,
        secondary: Optional[List[Dict[str, Any]]] = None,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist reviewer verdict and selected candidates to the backend."""
        claim_state = self.ensure_claim_state(claim_id)
        claim_state["selection_error"] = None
        payload: Dict[str, Any] = {
            "verdict": verdict,
            "primary": None,
            "secondary": [],
            "note": note,
        }
        if primary_candidate_id:
            primary = self._find_candidate(claim_state, primary_candidate_id)
            if not primary:
                _toast(self.ui, "Primary selection candidate not found.", icon="⚠️")
                return None
            primary_span = self._candidate_span(primary)
            if not primary_span:
                return None
            payload["primary"] = {
                "candidate_id": primary_candidate_id,
                **primary_span,
            }

        secondary_payload: List[Dict[str, Any]] = []
        for entry in secondary or []:
            candidate_id = (entry or {}).get("candidate_id")
            if not candidate_id:
                continue
            candidate = self._find_candidate(claim_state, candidate_id)
            if not candidate:
                continue
            span = self._candidate_span(candidate)
            if not span:
                continue
            secondary_payload.append(
                {
                    "candidate_id": candidate_id,
                    **span,
                    "rationale": (entry or {}).get("rationale") or "",
                }
            )
        payload["secondary"] = secondary_payload

        try:
            stored = self.api.put_evidence_selection(claim_id, payload)
        except evidence_api.EvidenceApiError as exc:
            claim_state["selection_error"] = str(exc)
            return None

        claim_state["selection"] = stored
        claim_state["selection_stale"] = False
        _toast(self.ui, "Evidence selection saved.", icon="✅")
        return stored

    def preview_excerpt(
        self,
        claim_id: str,
        candidate: Dict[str, Any],
        *,
        before: int = 2,
        after: int = 1,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and cache paragraph-bounded excerpt context for a candidate."""
        claim_state = self.ensure_claim_state(claim_id)
        span = self._candidate_span(candidate)
        if not span:
            return None
        attachment_id = span["attachment_id"]
        span_id = span["span_id"]
        cache_key = f"{attachment_id}:{span_id}:{int(before)}:{int(after)}"
        cache = claim_state.setdefault("excerpt_cache", {})
        if not force and cache_key in cache:
            return cache.get(cache_key)
        try:
            excerpt = self.api.fetch_span_excerpt(
                attachment_id,
                span_id,
                before=before,
                after=after,
            )
        except evidence_api.EvidenceApiError:
            return None
        cache[cache_key] = excerpt
        return excerpt

    def _mark_review_state(
        self, claim_id: str, candidate_id: str, status: str
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        candidate = self._find_candidate(claim_state, candidate_id)
        if not candidate:
            _toast(
                self.ui,
                "Unable to update review state; candidate missing.",
                icon="⚠️",
            )
            return claim_state
        candidate["review_state"] = status
        _toast(self.ui, f"Marked evidence as {status}.")
        return claim_state

    @staticmethod
    def _find_candidate(
        claim_state: Dict[str, Any], candidate_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not candidate_id:
            return None
        for candidate in claim_state.get("candidates", []):
            if candidate.get("id") == candidate_id:
                return candidate
        return None

    def accept_candidate(self, claim_id: str, candidate_id: str) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        candidate = self._find_candidate(claim_state, candidate_id)
        if not candidate:
            _toast(self.ui, "Unable to accept; candidate missing.", icon="⚠️")
            return claim_state
        return self.accept_candidate_target(claim_id, candidate)

    def reject_candidate(self, claim_id: str, candidate_id: str) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        candidate = self._find_candidate(claim_state, candidate_id)
        if not candidate:
            _toast(self.ui, "Unable to reject; candidate missing.", icon="⚠️")
            return claim_state
        return self.reject_candidate_target(claim_id, candidate)

    def toggle_pin(self, claim_id: str, candidate_id: str) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        candidate = self._find_candidate(claim_state, candidate_id)
        if not candidate:
            _toast(self.ui, "Unable to pin; candidate missing.", icon="⚠️")
            return claim_state
        return self.toggle_pin_target(claim_id, candidate)

    def clear_decisions(
        self, claim_id: str, *, reviewer_uid: str = "default"
    ) -> Dict[str, Any]:
        return self._append_decision_event_with_retry(
            claim_id,
            reviewer_uid=reviewer_uid,
            action="clear",
            target=None,
            set_value=None,
            payload={"ui_source": "clear_button"},
        )

    def toggle_pin_target(
        self, claim_id: str, candidate: Dict[str, Any], *, reviewer_uid: str = "default"
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        target = self._decision_target(candidate)
        if not target:
            _toast(self.ui, "Pinned target metadata missing.", icon="⚠️")
            return claim_state
        pinned = bool((candidate.get("decision_state") or {}).get("pinned"))
        action = "unpin" if pinned else "pin"
        return self._append_decision_event_with_retry(
            claim_id,
            reviewer_uid=reviewer_uid,
            action=action,
            target=target,
            set_value=None,
            payload={"ui_source": "evidence_list"},
        )

    def accept_candidate_target(
        self, claim_id: str, candidate: Dict[str, Any], *, reviewer_uid: str = "default"
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        target = self._decision_target(candidate)
        if not target:
            _toast(self.ui, "Decision target metadata missing.", icon="⚠️")
            return claim_state
        triage = str((candidate.get("decision_state") or {}).get("triage") or "none")
        set_value = False if triage == "accepted" else True
        return self._append_decision_event_with_retry(
            claim_id,
            reviewer_uid=reviewer_uid,
            action="accept",
            target=target,
            set_value=set_value,
            payload={"ui_source": "evidence_list"},
        )

    def reject_candidate_target(
        self, claim_id: str, candidate: Dict[str, Any], *, reviewer_uid: str = "default"
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        target = self._decision_target(candidate)
        if not target:
            _toast(self.ui, "Decision target metadata missing.", icon="⚠️")
            return claim_state
        triage = str((candidate.get("decision_state") or {}).get("triage") or "none")
        set_value = False if triage == "rejected" else True
        return self._append_decision_event_with_retry(
            claim_id,
            reviewer_uid=reviewer_uid,
            action="reject",
            target=target,
            set_value=set_value,
            payload={"ui_source": "evidence_list"},
        )

    def _decision_target(self, candidate: Dict[str, Any]) -> Optional[Dict[str, str]]:
        dt = (candidate or {}).get("decision_target")
        if isinstance(dt, dict):
            attachment_id = str(dt.get("attachment_id") or "").strip()
            span_id = str(dt.get("span_id") or "").strip()
            if attachment_id and span_id:
                return {"attachment_id": attachment_id, "span_id": span_id}
        meta = (candidate or {}).get("metadata") or {}
        attachment_id = str((candidate or {}).get("attachment_id") or "").strip()
        span_id = None
        if isinstance(meta, dict):
            span_id = (
                str(meta.get("span_id") or meta.get("anchor_id") or "").strip() or None
            )
        if attachment_id and span_id:
            return {"attachment_id": attachment_id, "span_id": span_id}
        return None

    def _append_decision_event_with_retry(
        self,
        claim_id: str,
        *,
        reviewer_uid: str,
        action: str,
        target: Optional[Dict[str, str]],
        set_value: Optional[bool],
        payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        reviewer = str(reviewer_uid or "default").strip() or "default"

        if claim_state.get("decision_actions_disabled"):
            banner = str(claim_state.get("decision_banner") or "Decisions are locked.")
            _toast(self.ui, banner, icon="⚠️")
            return claim_state

        meta = claim_state.get("decisions_meta") or {}
        expected_version = int(meta.get("version") or 0)
        idempotency_key = uuid4().hex

        def _refresh_and_get_version() -> int:
            decisions = self.sync_decisions(claim_id, reviewer_uid=reviewer, force=True)
            if isinstance(decisions, dict) and decisions.get("version") is not None:
                return int(decisions.get("version") or 0)
            return int((claim_state.get("decisions_meta") or {}).get("version") or 0)

        try:
            resp = self.api.append_evidence_decision_event(
                claim_id,
                reviewer_uid=reviewer,
                idempotency_key=idempotency_key,
                expected_version=expected_version,
                action=action,
                target=target,
                set_value=set_value,
                payload=payload,
            )
        except evidence_api.EvidenceDecisionConflict:
            # Refresh and retry exactly once.
            new_version = _refresh_and_get_version()
            try:
                resp = self.api.append_evidence_decision_event(
                    claim_id,
                    reviewer_uid=reviewer,
                    idempotency_key=idempotency_key,
                    expected_version=new_version,
                    action=action,
                    target=target,
                    set_value=set_value,
                    payload=payload,
                )
            except evidence_api.EvidenceDecisionConflict:
                claim_state["decision_actions_disabled"] = True
                claim_state[
                    "decision_banner"
                ] = "Decision stream changed in another tab. Refresh to continue."
                _toast(self.ui, claim_state["decision_banner"], icon="⚠️")
                return claim_state
            except evidence_api.EvidenceApiError as exc:
                _toast(self.ui, f"Decision write failed: {exc}", icon="⚠️")
                return claim_state
        except evidence_api.EvidenceApiError as exc:
            _toast(self.ui, f"Decision write failed: {exc}", icon="⚠️")
            return claim_state

        if isinstance(resp, dict) and resp.get("version") is not None:
            claim_state.setdefault("decisions_meta", {})["version"] = resp.get(
                "version"
            )
        claim_state["decisions_stale"] = True
        claim_state["stale"] = True
        _toast(self.ui, "Decision saved.", icon="✅", quiet=bool(resp.get("no_op")))
        return claim_state

    def prepare_share_link(
        self, claim_id: str, candidate_id: str
    ) -> Optional[Dict[str, Any]]:
        claim_state = self.ensure_claim_state(claim_id)
        candidate = self._find_candidate(claim_state, candidate_id)
        if not candidate:
            _toast(self.ui, "Unable to find evidence card for sharing.", icon="⚠️")
            return None
        share_payload = json.dumps(
            {
                "claim_id": claim_id,
                "candidate_id": candidate.get("id"),
                "label": candidate.get("label"),
                "title": candidate.get("title"),
                "snippet": (candidate.get("text") or "")[:500],
                "metadata": candidate.get("metadata"),
            },
            ensure_ascii=False,
            indent=2,
        )
        share_state = {
            "candidate": candidate,
            "payload": share_payload,
            "title": candidate.get("title") or candidate.get("id"),
        }
        claim_state["share_target"] = share_state
        _toast(self.ui, "Share details ready below the evidence list.", icon="🔗")
        return share_state

    def open_candidate_pdf(
        self, claim_id: str, candidate_id: str
    ) -> Optional[Dict[str, Any]]:
        claim_state = self.ensure_claim_state(claim_id)
        candidate = self._find_candidate(claim_state, candidate_id)
        if not candidate:
            _toast(self.ui, "Unable to find PDF metadata for this card.", icon="⚠️")
            return None
        metadata = candidate.get("metadata") or {}
        attachment_id = metadata.get("attachment_id") or metadata.get("document_id")
        span_id = metadata.get("span_id") or metadata.get("anchor_id")
        if not attachment_id or not span_id:
            _toast(self.ui, "Card is missing PDF span metadata.", icon="⚠️")
            return None
        try:
            jump = self.api.jump_to_pdf_span(attachment_id, span_id)
        except evidence_api.EvidenceApiError:
            return None
        claim_state["pdf_jump"] = {
            "attachment_id": attachment_id,
            "span_id": span_id,
            "viewer": jump,
        }
        _toast(self.ui, "Open the PDF viewer using the metadata below.", icon="📄")
        return jump

    def update_from_payload(
        self, claim_id: str, payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        if not payload:
            return claim_state
        claim_state["last_payload"] = payload
        candidates = payload.get("candidates", [])
        for candidate in candidates or []:
            label = (candidate.get("label") or "").strip().lower()
            if label in {"entails", "entailment"}:
                candidate["label"] = "entail"
            elif label in {"contradicts", "contradiction", "refutes", "refute"}:
                candidate["label"] = "contradict"
            elif label in {"neutral", "unknown", "none"}:
                candidate["label"] = "neutral"
        claim_state["candidates"] = candidates
        claim_state["total"] = payload.get("total", claim_state.get("total", 0))
        claim_state["offset"] = payload.get("offset", claim_state.get("offset", 0))
        claim_state["run"] = payload.get("run")
        claim_state["lock_state"] = payload.get("lock_state")
        claim_state["pinned"] = payload.get("pinned") or []
        claim_state["decisions_meta"] = payload.get("decisions") or {}

        # Keep UI rerun banner aligned with backend lock state.
        rerun = claim_state.setdefault("rerun", self._default_rerun_state())
        lock_state = claim_state.get("lock_state") or {}
        lock_status = (lock_state.get("status") or "").strip().lower()
        if lock_status in {"queued", "running"}:
            rerun["status"] = lock_status
        else:
            rerun["status"] = "idle"
            rerun["job"] = None
            rerun["queue"] = []
        claim_state["focus_order"] = [
            cand.get("id") for cand in claim_state["candidates"] if cand.get("id")
        ]
        return claim_state

    def _candidate_span(self, candidate: Dict[str, Any]) -> Optional[Dict[str, str]]:
        metadata = (candidate or {}).get("metadata") or {}
        attachment_id = (
            metadata.get("attachment_id")
            or candidate.get("attachment_id")
            or metadata.get("document_id")
        )
        span_id = metadata.get("span_id") or metadata.get("anchor_id")
        if not attachment_id or not span_id:
            _toast(self.ui, "Candidate is missing attachment/span metadata.", icon="⚠️")
            return None
        return {"attachment_id": str(attachment_id), "span_id": str(span_id)}

    def mark_claim_stale(
        self, claim_id: str, *, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        claim_state["stale"] = True
        claim_state["stale_reason"] = reason or claim_state.get("stale_reason")
        return claim_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _root(self) -> Dict[str, Any]:
        return self.session_state.setdefault(
            SESSION_KEY, {"claims": {}, "active_claim_id": None}
        )

    def _default_claim_state(self, claim_id: str) -> Dict[str, Any]:
        return {
            "claim_id": claim_id,
            "pinned": [],
            "filters": {
                "label": None,
                "include_neutral": False,
                "pinned_only": False,
            },
            "load_more_pages": 0,
            "candidates": [],
            "history": [],
            "total": 0,
            "offset": 0,
            "run": None,
            "lock_state": None,
            "focus_order": [],
            "stale": True,
            "stale_reason": None,
            "last_error": None,
            "is_loading": False,
            "inflight_fetches": 0,
            "rerun": self._default_rerun_state(),
            "share_target": None,
            "pdf_jump": None,
            "selection": None,
            "selection_stale": True,
            "selection_error": None,
            "excerpt_cache": {},
            "decisions_meta": {"reviewer_uid": "default", "version": 0},
            "decisions": None,
            "decisions_stale": True,
            "decisions_error": None,
            "decision_actions_disabled": False,
            "decision_banner": None,
        }

    @staticmethod
    def _default_rerun_state() -> Dict[str, Any]:
        return {"status": "idle", "job": None, "queue": [], "inflight": False}

    def _current_limit(self, claim_state: Dict[str, Any]) -> int:
        pages = 1 + claim_state.get("load_more_pages", 0)
        return min(self.max_total, pages * self.page_size)


def _get_store() -> EvidenceStore:
    session_state = st.session_state
    if STORE_INSTANCE_KEY in session_state and isinstance(
        session_state[STORE_INSTANCE_KEY], EvidenceStore
    ):
        return session_state[STORE_INSTANCE_KEY]
    store = EvidenceStore(session_state=session_state)
    session_state[STORE_INSTANCE_KEY] = store
    return store


def ensure_claim_state(claim_id: str, **metadata: Any) -> Dict[str, Any]:
    return _get_store().ensure_claim_state(claim_id, **metadata)


def set_active_claim(claim_id: Optional[str]) -> Optional[str]:
    return _get_store().set_active_claim(claim_id)


def get_active_claim_id() -> Optional[str]:
    return _get_store().active_claim_id()


def sync_for_claim(
    claim_id: str,
    *,
    claim_text: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    return _get_store().sync_for_claim(claim_id, claim_text=claim_text, force=force)


def apply_filter(
    claim_id: str,
    *,
    label: Optional[str] = None,
    include_neutral: Optional[bool] = None,
    pinned_only: Optional[bool] = None,
    claim_text: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_store().apply_filter(
        claim_id,
        label=label,
        include_neutral=include_neutral,
        pinned_only=pinned_only,
        claim_text=claim_text,
    )


def load_more(claim_id: str, *, claim_text: Optional[str] = None) -> Dict[str, Any]:
    return _get_store().load_more(claim_id, claim_text=claim_text)


def queue_rerun(
    claim_id: str,
    *,
    claim_text: Optional[str] = None,
    note: Optional[str] = None,
    advanced_settings: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    return _get_store().queue_rerun(
        claim_id,
        claim_text=claim_text,
        note=note,
        advanced_settings=advanced_settings,
        quiet=quiet,
    )


def update_from_payload(
    claim_id: str, payload: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    return _get_store().update_from_payload(claim_id, payload)


def mark_claim_stale(claim_id: str, *, reason: Optional[str] = None) -> Dict[str, Any]:
    return _get_store().mark_claim_stale(claim_id, reason=reason)


def sync_selection(claim_id: str, *, force: bool = False) -> Optional[Dict[str, Any]]:
    return _get_store().sync_selection(claim_id, force=force)


def save_selection(
    claim_id: str,
    *,
    verdict: str,
    primary_candidate_id: Optional[str] = None,
    secondary: Optional[List[Dict[str, Any]]] = None,
    note: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return _get_store().save_selection(
        claim_id,
        verdict=verdict,
        primary_candidate_id=primary_candidate_id,
        secondary=secondary,
        note=note,
    )


def preview_excerpt(
    claim_id: str,
    candidate: Dict[str, Any],
    *,
    before: int = 2,
    after: int = 1,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    return _get_store().preview_excerpt(
        claim_id,
        candidate,
        before=before,
        after=after,
        force=force,
    )


__all__ = [
    "EvidenceStore",
    "ensure_claim_state",
    "set_active_claim",
    "get_active_claim_id",
    "sync_for_claim",
    "sync_selection",
    "save_selection",
    "preview_excerpt",
    "apply_filter",
    "load_more",
    "queue_rerun",
    "update_from_payload",
    "mark_claim_stale",
]
