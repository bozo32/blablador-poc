from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.evidence_selection_store import EvidenceSelectionStore
from backend.settings import settings


@pytest.fixture()
def store(tmp_path: Path, monkeypatch) -> EvidenceSelectionStore:
    monkeypatch.setattr(settings, "EVIDENCE_STORE_DIR", tmp_path / "evidence_runs")
    return EvidenceSelectionStore(settings=settings)


def test_uncertain_requires_note(store: EvidenceSelectionStore):
    with pytest.raises(ValidationError):
        store.upsert(
            "claim-1",
            {
                "verdict": "uncertain",
                "primary": None,
                "secondary": [],
                "note": "",
            },
        )


def test_secondary_requires_rationale(store: EvidenceSelectionStore):
    with pytest.raises(ValidationError):
        store.upsert(
            "claim-2",
            {
                "verdict": "support",
                "primary": {
                    "candidate_id": "cand-1",
                    "attachment_id": "att-1",
                    "span_id": "s1",
                },
                "secondary": [
                    {
                        "candidate_id": "cand-2",
                        "attachment_id": "att-1",
                        "span_id": "s2",
                        "rationale": " ",
                    }
                ],
            },
        )


def test_roundtrip_write_and_read(store: EvidenceSelectionStore):
    stored = store.upsert(
        "claim-3",
        {
            "verdict": "support",
            "primary": {
                "candidate_id": "cand-1",
                "attachment_id": "att-1",
                "span_id": "s1",
            },
            "secondary": [
                {
                    "candidate_id": "cand-2",
                    "attachment_id": "att-2",
                    "span_id": "s9",
                    "rationale": "Extra context",
                }
            ],
            "note": "Looks good",
        },
    )

    loaded = store.read("claim-3")
    assert loaded is not None
    assert loaded.claim_id == "claim-3"
    assert loaded.verdict == "support"
    assert loaded.primary is not None
    assert loaded.primary.span_id == "s1"
    assert loaded.secondary and loaded.secondary[0].rationale == "Extra context"
    assert loaded.note == "Looks good"
    assert stored.updated_at is not None
