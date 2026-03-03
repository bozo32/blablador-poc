import types

import pytest

from frontend import claim_queue


@pytest.fixture(autouse=True)
def session_state(monkeypatch):
    dummy_streamlit = types.SimpleNamespace(session_state={})
    monkeypatch.setattr(claim_queue, "st", dummy_streamlit)
    claim_queue.init_claim_registry()
    yield
    dummy_streamlit.session_state.clear()


def test_auto_match_claim_prefers_reference_hint():
    claim_queue.register_claim(
        "c1",
        claim="Claim referencing Foo",
        reference_id="foo2020",
        callout="(Foo 2020)",
    )
    claim_queue.register_claim(
        "c2",
        claim="Claim referencing Bar",
        reference_id="bar2021",
        callout="(Bar 2021)",
    )

    queue_item = {
        "filename": "foo2020-study.pdf",
        "reference_hint": {"reference_id": "foo2020"},
    }

    matches = claim_queue.auto_match_claim(queue_item)

    assert matches
    assert matches[0]["id"] == "c1"


def test_get_claim_options_truncates_labels():
    claim_queue.register_claim(
        "c3",
        claim=(
            "This is a very long claim that should be truncated for dropdown "
            "readability"
        ),
        callout="(Baz 2022)",
    )

    options = claim_queue.get_claim_options(max_claim_chars=16)

    assert options
    assert options[0]["id"] == "c3"
    assert options[0]["label"].startswith("(Baz 2022)")
    assert options[0]["label"].endswith("…")


def test_cached_dossier_is_isolated_by_project_and_user(monkeypatch):
    calls = []

    def _fake_get_reference_retrieval(
        api_url,
        doc_id,
        reference_id,
        *,
        project_id,
        user_id,
    ):
        calls.append((api_url, doc_id, reference_id, project_id, user_id))
        return {
            "doc_id": doc_id,
            "reference_id": reference_id,
            "project_id": project_id,
            "user_id": user_id,
        }

    monkeypatch.setattr(claim_queue, "get_reference_retrieval", _fake_get_reference_retrieval)
    monkeypatch.setattr(claim_queue.scope_lock, "get_applied_project_id", lambda: "proj-a")
    monkeypatch.setattr(claim_queue.scope_lock, "get_applied_uid", lambda: "user-a")

    first = claim_queue._get_cached_dossier("http://api", "doc-1", "ref-1")
    second = claim_queue._get_cached_dossier("http://api", "doc-1", "ref-1")

    assert first == second
    assert len(calls) == 1

    monkeypatch.setattr(claim_queue.scope_lock, "get_applied_project_id", lambda: "proj-b")
    monkeypatch.setattr(claim_queue.scope_lock, "get_applied_uid", lambda: "user-b")
    third = claim_queue._get_cached_dossier("http://api", "doc-1", "ref-1")

    assert len(calls) == 2
    assert third is not None
    assert third["project_id"] == "proj-b"
    assert third["user_id"] == "user-b"
