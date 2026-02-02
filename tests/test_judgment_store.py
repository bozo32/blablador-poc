from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.judgment_store import JudgmentStore
from backend.schemas import JudgmentPayload
from backend.settings import settings


@pytest.fixture()
def store(tmp_path: Path, monkeypatch) -> JudgmentStore:
    monkeypatch.setattr(settings, "EVIDENCE_STORE_DIR", tmp_path / "evidence_runs")
    return JudgmentStore(settings=settings)


def test_final_requires_verdict():
    with pytest.raises(ValidationError):
        JudgmentPayload(claim_id="claim-1", status="final", verdict=None)


def test_draft_allows_missing_verdict():
    payload = JudgmentPayload(claim_id="claim-1", status="draft", verdict=None)
    assert payload.status == "draft"
    assert payload.verdict is None


def test_notes_structured_and_optional():
    payload = JudgmentPayload(
        claim_id="claim-1",
        notes={"rationale": "Because."},
    )
    assert payload.notes is not None
    assert payload.notes.rationale == "Because."
    assert payload.notes.caveats is None


def test_filename_collision_guard(store: JudgmentStore):
    claim_a = "a:b"
    claim_b = "a?b"
    assert store._path_for_claim(claim_a).name != store._path_for_claim(claim_b).name


def test_roundtrip_write_and_read(store: JudgmentStore):
    store.upsert(
        "claim-3",
        {
            "status": "final",
            "verdict": "support",
            "notes": {"rationale": "Looks good"},
            "doc_id": "doc-1",
            "citation_index": 2,
            "target_id": "ref-1",
            "sentence_id": "s1",
            "callout": "[2]",
            "doi": "10.0000/example",
            "author": "Smith",
            "year": "2024",
            "claim_text": "Example claim",
        },
    )

    loaded = store.read("claim-3")
    assert loaded is not None
    assert loaded.claim_id == "claim-3"
    assert loaded.status == "final"
    assert loaded.verdict == "support"
    assert loaded.updated_at is not None
    assert loaded.notes is not None
    assert loaded.notes.rationale == "Looks good"
    assert loaded.doc_id == "doc-1"
    assert loaded.citation_index == 2


def test_upsert_updates_timestamp(store: JudgmentStore):
    first = store.upsert("claim-4", {"status": "draft", "verdict": None})
    second = store.upsert("claim-4", {"status": "draft", "verdict": "uncertain"})
    assert first.updated_at is not None
    assert second.updated_at is not None
    assert first.updated_at != second.updated_at
