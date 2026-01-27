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
