"""Session-backed evidence store coordinating UI state and API calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st

from frontend import evidence_api

SESSION_KEY = "_evidence_store_state"
STORE_INSTANCE_KEY = "_evidence_store_instance"
DEFAULT_PAGE_SIZE = 5


def _toast(ui: Any, message: str, *, icon: str = "ℹ️") -> None:
    toast = getattr(ui, "toast", None)
    if callable(toast):  # pragma: no cover - Streamlit runtime only
        toast(message, icon=icon)
        return
    warn = getattr(ui, "warning", None)
    if callable(warn):  # pragma: no cover - Streamlit runtime only
        warn(message)


@dataclass
class EvidenceStore:
    session_state: Optional[Dict[str, Any]] = None
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

    def sync_for_claim(
        self,
        claim_id: str,
        *,
        claim_text: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        if not force and not claim_state.get("stale") and claim_state.get("candidates"):
            return claim_state
        if claim_state.get("inflight_fetches", 0) >= evidence_api.MAX_LIST_REQUESTS:
            _toast(self.ui, "Evidence fetch already running for this claim.")
            return claim_state
        claim_state["is_loading"] = True
        claim_state["inflight_fetches"] = claim_state.get("inflight_fetches", 0) + 1
        try:
            payload = self.api.list_evidence(
                claim_id,
                limit=self._current_limit(claim_state),
                label=claim_state["filters"].get("label"),
                include_neutral=claim_state["filters"].get("include_neutral", True),
                pinned_only=claim_state["filters"].get("pinned_only", False),
                claim_text=claim_text,
            )
            self.update_from_payload(claim_id, payload)
            claim_state["stale"] = False
            history = self.api.fetch_history(claim_id, limit=5)
            claim_state["history"] = history.get("runs", [])
            claim_state["last_error"] = None
        except evidence_api.EvidenceApiError as exc:
            claim_state["last_error"] = str(exc)
        finally:
            claim_state["is_loading"] = False
            claim_state["inflight_fetches"] = max(
                0, claim_state.get("inflight_fetches", 1) - 1
            )
        return claim_state

    def apply_filter(
        self,
        claim_id: str,
        *,
        label: Optional[str] = None,
        include_neutral: Optional[bool] = None,
        pinned_only: Optional[bool] = None,
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
        return self.sync_for_claim(claim_id, force=True)

    def load_more(self, claim_id: str) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        if claim_state.get("load_more_pages", 0) >= self.max_extra_pages:
            _toast(self.ui, "All available evidence candidates are already loaded.")
            return claim_state
        claim_state["load_more_pages"] = claim_state.get("load_more_pages", 0) + 1
        claim_state["stale"] = True
        return self.sync_for_claim(claim_id, force=True)

    def queue_rerun(
        self,
        claim_id: str,
        *,
        claim_text: Optional[str] = None,
        note: Optional[str] = None,
        advanced_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        rerun = claim_state.setdefault("rerun", self._default_rerun_state())
        if rerun.get("inflight"):
            _toast(self.ui, "Rerun already requested for this claim.", icon="⚠️")
            return rerun
        rerun["inflight"] = True
        try:
            job = self.api.request_rerun(
                claim_id,
                claim_text=claim_text,
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
            _toast(self.ui, f"Rerun failed: {exc}", icon="⚠️")
        finally:
            rerun["inflight"] = False
        return rerun

    def update_from_payload(
        self, claim_id: str, payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        if not payload:
            return claim_state
        claim_state["last_payload"] = payload
        claim_state["candidates"] = payload.get("candidates", [])
        claim_state["total"] = payload.get("total", claim_state.get("total", 0))
        claim_state["offset"] = payload.get("offset", claim_state.get("offset", 0))
        claim_state["run"] = payload.get("run")
        claim_state["lock_state"] = payload.get("lock_state")
        claim_state["focus_order"] = [
            cand.get("id") for cand in claim_state["candidates"] if cand.get("id")
        ]
        return claim_state

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
            "pinned_ids": [],
            "filters": {
                "label": None,
                "include_neutral": True,
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
) -> Dict[str, Any]:
    return _get_store().apply_filter(
        claim_id,
        label=label,
        include_neutral=include_neutral,
        pinned_only=pinned_only,
    )


def load_more(claim_id: str) -> Dict[str, Any]:
    return _get_store().load_more(claim_id)


def queue_rerun(
    claim_id: str,
    *,
    claim_text: Optional[str] = None,
    note: Optional[str] = None,
    advanced_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _get_store().queue_rerun(
        claim_id,
        claim_text=claim_text,
        note=note,
        advanced_settings=advanced_settings,
    )


def update_from_payload(
    claim_id: str, payload: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    return _get_store().update_from_payload(claim_id, payload)


def mark_claim_stale(claim_id: str, *, reason: Optional[str] = None) -> Dict[str, Any]:
    return _get_store().mark_claim_stale(claim_id, reason=reason)


__all__ = [
    "EvidenceStore",
    "ensure_claim_state",
    "sync_for_claim",
    "apply_filter",
    "load_more",
    "queue_rerun",
    "update_from_payload",
    "mark_claim_stale",
]
