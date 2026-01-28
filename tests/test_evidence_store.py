from __future__ import annotations

import copy
import types

import pytest

from frontend import evidence_api


@pytest.fixture()
def stub_streamlit(monkeypatch):
    state: dict = {}
    toasts: list[tuple[str, str | None]] = []

    def toast(message: str, icon: str | None = None):  # pragma: no cover - UI helper
        toasts.append((message, icon))

    def error(message: str):  # pragma: no cover - UI fallback
        toasts.append((message, None))

    stub = types.SimpleNamespace(session_state=state, toast=toast, error=error)
    monkeypatch.setattr(evidence_api, "st", stub)
    yield stub
    state.clear()
    toasts.clear()


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
