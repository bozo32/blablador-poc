from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import pytest

from frontend import judgment_api, judgment_store


class StubJudgmentApi:
    def __init__(
        self, *, judgments_by_claim_id: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Capture API interactions for assertions in tests."""
        self.get_calls: List[str] = []
        self.put_calls: List[tuple[str, Dict[str, Any]]] = []
        self.list_calls: List[Dict[str, Any]] = []
        self.judgments_by_claim_id = judgments_by_claim_id or {}
        self.doc_lists: Dict[str, List[Dict[str, Any]]] = {}

    def get_judgment(self, claim_id: str) -> Dict[str, Any]:
        self.get_calls.append(claim_id)
        return self.judgments_by_claim_id.get(claim_id, {})

    def put_judgment(self, claim_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.put_calls.append((claim_id, payload))
        stored = {"claim_id": claim_id, **payload}
        self.judgments_by_claim_id[claim_id] = stored
        return stored

    def list_judgments(
        self, *, doc_id: str | None = None, include_drafts: bool = False
    ):
        self.list_calls.append({"doc_id": doc_id, "include_drafts": include_drafts})
        if not doc_id:
            return {"judgments": []}
        return {"judgments": list(self.doc_lists.get(doc_id, []))}

    def download_export(self, **kwargs):  # pragma: no cover - contract shim
        return {
            "content": "{}",
            "mime": "application/json",
            "filename": "judgments.json",
        }


@pytest.fixture()
def stub_streamlit(monkeypatch):
    state: dict = {}
    toasts: list[tuple[str, str | None]] = []

    def toast(message: str, icon: str | None = None):  # pragma: no cover - UI helper
        toasts.append((message, icon))

    def warning(message: str):  # pragma: no cover - UI helper
        toasts.append((message, "warning"))

    def error(message: str):  # pragma: no cover - UI fallback
        toasts.append((message, None))

    stub = types.SimpleNamespace(
        session_state=state,
        toast=toast,
        warning=warning,
        error=error,
    )
    monkeypatch.setattr(judgment_api, "st", stub)
    monkeypatch.setattr(judgment_store, "st", stub)
    yield stub
    state.clear()
    toasts.clear()
    stub.session_state.pop(judgment_store.STORE_INSTANCE_KEY, None)


def test_sync_judgment_caches_and_skips_refetch(stub_streamlit):
    api = StubJudgmentApi(judgments_by_claim_id={"claim-1": {"claim_id": "claim-1"}})
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )

    state = store.sync_judgment("claim-1")
    assert state["judgment"]["claim_id"] == "claim-1"
    assert api.get_calls == ["claim-1"]

    state = store.sync_judgment("claim-1")
    assert state["judgment"]["claim_id"] == "claim-1"
    assert api.get_calls == ["claim-1"]


def test_save_judgment_updates_cache_and_clears_stale(stub_streamlit):
    api = StubJudgmentApi()
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )

    claim_state = store.ensure_claim_state("claim-1")
    claim_state["stale"] = True

    stored = store.save_judgment(
        "claim-1",
        status="final",
        verdict="support",
        notes={"rationale": "ok"},
        provenance={"doc_id": "doc-1", "citation_index": 1, "target_id": "t1"},
    )
    assert stored
    assert stored["status"] == "final"
    assert stored["verdict"] == "support"
    assert store.ensure_claim_state("claim-1")["stale"] is False


def test_callout_status_no_judgments(stub_streamlit):
    api = StubJudgmentApi()
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )

    status = store.callout_status("doc-1", 1, "t1")
    assert status == {"validated": False, "outcome": None, "claim_ids": []}


def test_callout_status_draft_only_not_validated(stub_streamlit):
    api = StubJudgmentApi()
    api.doc_lists["doc-1"] = [
        {
            "claim_id": "claim-1",
            "status": "draft",
            "verdict": None,
            "provenance": {"doc_id": "doc-1", "citation_index": 1, "target_id": "t1"},
        }
    ]
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.sync_doc("doc-1", include_drafts=True)

    status = store.callout_status("doc-1", 1, "t1")
    assert status["validated"] is False
    assert status["outcome"] is None
    assert status["claim_ids"] == ["claim-1"]


def test_callout_status_single_final_support(stub_streamlit):
    api = StubJudgmentApi()
    api.doc_lists["doc-1"] = [
        {
            "claim_id": "claim-1",
            "status": "final",
            "verdict": "support",
            "provenance": {"doc_id": "doc-1", "citation_index": 1, "target_id": "t1"},
        }
    ]
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.sync_doc("doc-1")

    status = store.callout_status("doc-1", 1, "t1")
    assert status["validated"] is True
    assert status["outcome"] == "support"
    assert status["claim_ids"] == ["claim-1"]


def test_callout_status_disagreement_returns_uncertain(stub_streamlit):
    api = StubJudgmentApi()
    api.doc_lists["doc-1"] = [
        {
            "claim_id": "claim-1",
            "status": "final",
            "verdict": "support",
            "provenance": {"doc_id": "doc-1", "citation_index": 1, "target_id": "t1"},
        },
        {
            "claim_id": "claim-2",
            "status": "final",
            "verdict": "contradict",
            "provenance": {"doc_id": "doc-1", "citation_index": 1, "target_id": "t1"},
        },
    ]
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.sync_doc("doc-1")

    status = store.callout_status("doc-1", 1, "t1")
    assert status["validated"] is True
    assert status["outcome"] == "uncertain"
    assert status["claim_ids"] == ["claim-1", "claim-2"]


def test_callout_status_fallback_when_target_id_missing(stub_streamlit):
    api = StubJudgmentApi()
    api.doc_lists["doc-1"] = [
        {
            "claim_id": "claim-1",
            "status": "final",
            "verdict": "support",
            "provenance": {"doc_id": "doc-1", "citation_index": 1},
        }
    ]
    store = judgment_store.JudgmentStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.sync_doc("doc-1")

    status = store.callout_status("doc-1", 1, "t1")
    assert status["validated"] is True
    assert status["outcome"] == "support"
    assert status["claim_ids"] == ["claim-1"]
