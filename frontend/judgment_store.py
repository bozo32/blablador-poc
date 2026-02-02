"""Session-backed judgment store coordinating UI state and API calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from frontend import judgment_api

SESSION_KEY = "_judgment_store_state"
STORE_INSTANCE_KEY = "_judgment_store_instance"


def _toast(ui: Any, message: str, *, icon: str = "ℹ️") -> None:
    toast = getattr(ui, "toast", None)
    if callable(toast):  # pragma: no cover - Streamlit runtime only
        toast(message, icon=icon)
        return
    warn = getattr(ui, "warning", None)
    if callable(warn):  # pragma: no cover - Streamlit runtime only
        warn(message)


CalloutKey = Tuple[str, int, Optional[str]]


@dataclass
class JudgmentStore:
    session_state: Any = None
    api: Any = judgment_api
    ui: Any = st

    def __post_init__(self) -> None:
        """Bind session defaults for per-claim and per-doc judgment caching."""
        if self.session_state is None:
            self.session_state = st.session_state
        self.session_state.setdefault(SESSION_KEY, {"claims": {}, "doc_index": {}})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_claim_state(self, claim_id: str) -> Dict[str, Any]:
        root = self._root()
        claims = root.setdefault("claims", {})
        return claims.setdefault(claim_id, self._default_claim_state(claim_id))

    def sync_judgment(self, claim_id: str, *, force: bool = False) -> Dict[str, Any]:
        claim_state = self.ensure_claim_state(claim_id)
        if not force and not claim_state.get("stale") and claim_state.get("fetched"):
            return claim_state

        claim_state["error"] = None
        try:
            judgment = self.api.get_judgment(claim_id)
        except judgment_api.JudgmentApiError as exc:
            claim_state["error"] = str(exc)
            claim_state["stale"] = True
            return claim_state

        claim_state["judgment"] = judgment or None
        claim_state["fetched"] = True
        claim_state["stale"] = False
        self._index_judgment(judgment or {})
        return claim_state

    def save_judgment(
        self,
        claim_id: str,
        *,
        status: str,
        verdict: Optional[str] = None,
        notes: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        claim_state = self.ensure_claim_state(claim_id)
        claim_state["error"] = None

        payload: Dict[str, Any] = {
            "status": status,
            "verdict": verdict,
            "notes": notes,
            "provenance": provenance,
        }
        try:
            stored = self.api.put_judgment(claim_id, payload)
        except judgment_api.JudgmentApiError as exc:
            claim_state["error"] = str(exc)
            _toast(self.ui, f"Unable to save judgment: {exc}", icon="⚠️")
            return None

        claim_state["judgment"] = stored or payload
        claim_state["fetched"] = True
        claim_state["stale"] = False
        claim_state["error"] = None
        self._index_judgment(claim_state["judgment"] or {})
        _toast(self.ui, "Judgment saved.", icon="✅")
        return claim_state["judgment"]

    def sync_doc(
        self,
        doc_id: str,
        *,
        include_drafts: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        root = self._root()
        doc_index = root.setdefault("doc_index", {})
        doc_state = doc_index.setdefault(doc_id, self._default_doc_state(doc_id))

        if (
            not force
            and doc_state.get("fetched")
            and doc_state.get("include_drafts") == include_drafts
            and not doc_state.get("stale")
        ):
            return doc_state

        doc_state["error"] = None
        doc_state["include_drafts"] = include_drafts
        try:
            payload = self.api.list_judgments(
                doc_id=doc_id, include_drafts=include_drafts
            )
        except judgment_api.JudgmentApiError as exc:
            doc_state["error"] = str(exc)
            doc_state["stale"] = True
            return doc_state

        judgments = self._coerce_judgments(payload)
        doc_state["judgments_by_claim_id"] = {}
        doc_state["judgments_by_callout_key"] = {}
        for judgment in judgments:
            claim_id = (judgment or {}).get("claim_id")
            if claim_id:
                doc_state["judgments_by_claim_id"][str(claim_id)] = judgment
                claim_state = self.ensure_claim_state(str(claim_id))
                claim_state["judgment"] = judgment
                claim_state["fetched"] = True
                claim_state["stale"] = False
                self._index_doc_judgment(doc_state, doc_id, judgment)

        doc_state["fetched"] = True
        doc_state["stale"] = False
        return doc_state

    def callout_status(
        self, doc_id: str, citation_index: int, target_id: Optional[str]
    ) -> Dict[str, Any]:
        root = self._root()
        doc_state = (root.get("doc_index") or {}).get(doc_id)
        if not doc_state:
            return {
                "validated": False,
                "outcome": None,
                "claim_ids": [],
            }

        judgments_by_key = doc_state.get("judgments_by_callout_key") or {}
        key: CalloutKey = (doc_id, int(citation_index), target_id)
        matches: List[Dict[str, Any]] = list(judgments_by_key.get(key) or [])
        if not matches and target_id is not None:
            matches = list(
                judgments_by_key.get((doc_id, int(citation_index), None)) or []
            )

        claim_ids = sorted(
            {
                str((entry or {}).get("claim_id"))
                for entry in matches
                if (entry or {}).get("claim_id")
            }
        )
        finals = [
            entry
            for entry in matches
            if (entry or {}).get("status") == "final" and (entry or {}).get("verdict")
        ]
        if not finals:
            return {
                "validated": False,
                "outcome": None,
                "claim_ids": claim_ids,
            }

        verdicts = {
            str((entry or {}).get("verdict"))
            for entry in finals
            if (entry or {}).get("verdict")
        }
        if len(verdicts) == 1:
            outcome = next(iter(verdicts))
        else:
            outcome = "uncertain"

        return {
            "validated": True,
            "outcome": outcome,
            "claim_ids": claim_ids,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _root(self) -> Dict[str, Any]:
        return self.session_state.setdefault(
            SESSION_KEY, {"claims": {}, "doc_index": {}}
        )

    @staticmethod
    def _default_claim_state(claim_id: str) -> Dict[str, Any]:
        return {
            "claim_id": claim_id,
            "judgment": None,
            "stale": True,
            "fetched": False,
            "error": None,
        }

    @staticmethod
    def _default_doc_state(doc_id: str) -> Dict[str, Any]:
        return {
            "doc_id": doc_id,
            "judgments_by_claim_id": {},
            "judgments_by_callout_key": {},
            "include_drafts": False,
            "stale": True,
            "fetched": False,
            "error": None,
        }

    @staticmethod
    def _coerce_judgments(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            items = (
                payload.get("judgments")
                or payload.get("items")
                or payload.get("results")
                or []
            )
            if isinstance(items, list):
                return [entry for entry in items if isinstance(entry, dict)]
        return []

    def _index_judgment(self, judgment: Dict[str, Any]) -> None:
        claim_id = (judgment or {}).get("claim_id")
        if not claim_id:
            return
        provenance = (judgment or {}).get("provenance") or {}
        doc_id = provenance.get("doc_id") or (judgment or {}).get("doc_id")
        citation_index = provenance.get("citation_index") or (judgment or {}).get(
            "citation_index"
        )
        if doc_id is None or citation_index is None:
            return
        root = self._root()
        doc_index = root.setdefault("doc_index", {})
        doc_state = doc_index.setdefault(
            str(doc_id), self._default_doc_state(str(doc_id))
        )
        self._index_doc_judgment(doc_state, str(doc_id), judgment)

    def _index_doc_judgment(
        self, doc_state: Dict[str, Any], doc_id: str, judgment: Dict[str, Any]
    ) -> None:
        provenance = (judgment or {}).get("provenance") or {}
        citation_index = provenance.get("citation_index") or (judgment or {}).get(
            "citation_index"
        )
        if citation_index is None:
            return
        target_id = provenance.get("target_id")
        key: CalloutKey = (doc_id, int(citation_index), target_id)
        by_key = doc_state.setdefault("judgments_by_callout_key", {})
        bucket = by_key.setdefault(key, [])
        bucket.append(judgment)


def _get_store() -> JudgmentStore:
    session_state = st.session_state
    if STORE_INSTANCE_KEY in session_state and isinstance(
        session_state[STORE_INSTANCE_KEY], JudgmentStore
    ):
        return session_state[STORE_INSTANCE_KEY]
    store = JudgmentStore(session_state=session_state)
    session_state[STORE_INSTANCE_KEY] = store
    return store


def sync_judgment(claim_id: str, *, force: bool = False) -> Dict[str, Any]:
    return _get_store().sync_judgment(claim_id, force=force)


def save_judgment(
    claim_id: str,
    *,
    status: str,
    verdict: Optional[str] = None,
    notes: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return _get_store().save_judgment(
        claim_id,
        status=status,
        verdict=verdict,
        notes=notes,
        provenance=provenance,
    )


def sync_doc(
    doc_id: str, *, include_drafts: bool = False, force: bool = False
) -> Dict[str, Any]:
    return _get_store().sync_doc(doc_id, include_drafts=include_drafts, force=force)


def callout_status(
    doc_id: str, citation_index: int, target_id: Optional[str]
) -> Dict[str, Any]:
    return _get_store().callout_status(doc_id, citation_index, target_id)


__all__ = [
    "JudgmentStore",
    "callout_status",
    "save_judgment",
    "sync_doc",
    "sync_judgment",
]
