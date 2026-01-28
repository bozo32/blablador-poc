from __future__ import annotations

import copy
import types

import pytest

from frontend import evidence_api, evidence_store


class StubEvidenceApi:
    def __init__(self):
        """Capture API interactions for assertions in tests."""
        self.list_calls: list[tuple[str, dict[str, object]]] = []
        self.history_calls: list[tuple[str, int]] = []
        self.rerun_calls = 0
        self.total = 10

    def list_evidence(self, claim_id: str, **kwargs):
        self.list_calls.append((claim_id, kwargs))
        limit = kwargs.get("limit", 5)
        candidates = [
            {"id": f"{claim_id}-cand-{idx}", "label": kwargs.get("label") or "entails"}
            for idx in range(limit)
        ]
        return {
            "claim_id": claim_id,
            "candidates": candidates,
            "total": self.total,
            "offset": 0,
            "limit": limit,
            "lock_state": {"status": "idle"},
            "run": {"run_id": "run-latest"},
        }

    def fetch_history(self, claim_id: str, limit: int = 5):
        self.history_calls.append((claim_id, limit))
        return {"runs": [{"run_id": f"run-{claim_id}"}]}

    def request_rerun(self, claim_id: str, **_):
        self.rerun_calls += 1
        return {"job_id": f"job-{self.rerun_calls}", "status": "queued"}


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
    monkeypatch.setattr(evidence_api, "st", stub)
    monkeypatch.setattr(evidence_store, "st", stub)
    yield stub
    state.clear()
    toasts.clear()
    stub.session_state.pop(evidence_store.STORE_INSTANCE_KEY, None)


def test_api_stub_request_limit(stub_streamlit, monkeypatch):
    stub_streamlit.session_state.update({"api_url": "http://backend", "api_key": "k"})
    sample_payload = {
        "claim_id": "claim-1",
        "candidates": [{"id": "cand-1"}],
        "total": 5,
        "offset": 0,
        "limit": 5,
    }

    monkeypatch.setattr(
        evidence_api,
        "_request",
        lambda method, path, **kwargs: copy.deepcopy(sample_payload),
    )

    response = evidence_api.list_evidence("claim-1")

    assert (
        response["meta"]["request_slots"]["remaining"] == evidence_api.MAX_LIST_REQUESTS
    )
    assert response["meta"]["remaining_candidates"] == 4

    stub_streamlit.session_state[evidence_api.INFLIGHT_KEY] = {
        "claim-1": evidence_api.MAX_LIST_REQUESTS
    }

    with pytest.raises(evidence_api.EvidenceApiError):
        evidence_api.list_evidence("claim-1")


def test_store_sync_load_more_persistence(stub_streamlit):
    api = StubEvidenceApi()
    store = evidence_store.EvidenceStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
        page_size=2,
        max_total=6,
    )
    state = store.sync_for_claim("claim-sync")
    assert len(state["candidates"]) == 2

    state = store.load_more("claim-sync")
    assert len(state["candidates"]) == 4
    assert api.list_calls[-1][1]["limit"] == 4


def test_store_apply_filter_toggles_label(stub_streamlit):
    api = StubEvidenceApi()
    store = evidence_store.EvidenceStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.sync_for_claim("claim-filter")
    api.list_calls.clear()

    state = store.apply_filter("claim-filter", label="contradicts")
    assert state["filters"]["label"] == "contradicts"
    assert api.list_calls[-1][1]["label"] == "contradicts"

    api.list_calls.clear()
    state = store.apply_filter("claim-filter", label="contradicts")
    assert state["filters"]["label"] is None


def test_request_rerun_queue_limit(stub_streamlit):
    api = StubEvidenceApi()
    store = evidence_store.EvidenceStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.queue_rerun("claim-rerun")
    assert api.rerun_calls == 1

    claim_state = store.ensure_claim_state("claim-rerun")
    claim_state["rerun"]["inflight"] = True
    store.queue_rerun("claim-rerun")
    assert api.rerun_calls == 1


def test_store_sync_marks_stale_from_attachment(stub_streamlit):
    api = StubEvidenceApi()
    store = evidence_store.EvidenceStore(
        session_state=stub_streamlit.session_state,
        api=api,
        ui=stub_streamlit,
    )
    store.sync_for_claim("claim-stale")
    first_call_count = len(api.list_calls)

    store.mark_claim_stale("claim-stale", reason="attachment")
    store.sync_for_claim("claim-stale")

    assert len(api.list_calls) == first_call_count + 1
