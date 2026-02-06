import csv
import json
from pathlib import Path
from io import StringIO

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
            "reviewer_uid": "default",
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


def test_two_reviewers_can_upsert_same_claim_without_overwriting(store: JudgmentStore):
    store.upsert(
        "claim-9",
        {
            "reviewer_uid": "Alice",
            "status": "final",
            "verdict": "support",
            "doc_id": "doc-1",
        },
    )
    store.upsert(
        "claim-9",
        {
            "reviewer_uid": "Bob",
            "status": "final",
            "verdict": "contradict",
            "doc_id": "doc-1",
        },
    )

    alice = store.read("claim-9", reviewer_uid="Alice")
    bob = store.read("claim-9", reviewer_uid="Bob")
    assert alice is not None
    assert bob is not None
    assert alice.verdict == "support"
    assert bob.verdict == "contradict"
    assert alice.reviewer_uid == "Alice"
    assert bob.reviewer_uid == "Bob"

    all_for_claim = store.list_for_claim("claim-9")
    assert {j.reviewer_uid for j in all_for_claim} == {"Alice", "Bob"}


def test_legacy_single_judgment_file_is_still_readable(store: JudgmentStore):
    legacy_path = store._path_for_claim("claim-legacy")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "claim_id": "claim-legacy",
                "status": "draft",
                "verdict": None,
                "notes": None,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    loaded = store.read("claim-legacy", reviewer_uid="default")
    assert loaded is not None
    assert loaded.claim_id == "claim-legacy"
    assert loaded.reviewer_uid == "default"


def test_upsert_updates_timestamp(store: JudgmentStore):
    first = store.upsert("claim-4", {"status": "draft", "verdict": None})
    second = store.upsert("claim-4", {"status": "draft", "verdict": "uncertain"})
    assert first.updated_at is not None
    assert second.updated_at is not None
    assert first.updated_at != second.updated_at


def test_export_defaults_final_only(store: JudgmentStore):
    store.upsert("claim-a", {"status": "draft", "verdict": None, "doc_id": "d1"})
    store.upsert("claim-b", {"status": "final", "verdict": "support", "doc_id": "d1"})
    store.upsert(
        "claim-c",
        {"status": "final", "verdict": "contradict", "doc_id": "d1"},
    )

    exported = store.export_claims(include_drafts=False, mode="core", format="json")
    rows = json.loads(exported.decode("utf-8"))
    assert [row["claim_id"] for row in rows] == ["claim-b", "claim-c"]

    exported_all = store.export_claims(include_drafts=True, mode="core", format="json")
    rows_all = json.loads(exported_all.decode("utf-8"))
    assert [row["claim_id"] for row in rows_all] == ["claim-a", "claim-b", "claim-c"]


def test_export_callouts_grouping_and_ordering(store: JudgmentStore):
    # Group 1: doc_id=None sorts before doc_id="doc-1"
    store.upsert(
        "a",
        {
            "status": "final",
            "verdict": "support",
            "doc_id": None,
            "citation_index": None,
            "target_id": None,
            "callout": None,
            "claim_text": "A",
        },
    )
    # Group 2: deterministic callout chosen from first non-null after claim_id sorting.
    store.upsert(
        "b",
        {
            "status": "final",
            "verdict": "support",
            "doc_id": "doc-1",
            "citation_index": 0,
            "target_id": "ref-1",
            "callout": None,
            "claim_text": "B",
        },
    )
    store.upsert(
        "c",
        {
            "status": "final",
            "verdict": "uncertain",
            "doc_id": "doc-1",
            "citation_index": 0,
            "target_id": "ref-1",
            "callout": "[1]",
            "claim_text": "C",
        },
    )
    store.upsert(
        "d",
        {
            "status": "final",
            "verdict": "contradict",
            "doc_id": "doc-1",
            "citation_index": 0,
            "target_id": "ref-1",
            "callout": "[2]",
            "claim_text": "D",
        },
    )

    exported = store.export_callouts(include_drafts=False, mode="core", format="json")
    groups = json.loads(exported.decode("utf-8"))

    assert len(groups) == 2
    assert groups[0]["doc_id"] is None
    assert groups[0]["citation_index"] is None
    assert groups[0]["target_id"] is None
    assert groups[0]["callout"] is None

    assert groups[1]["doc_id"] == "doc-1"
    assert groups[1]["citation_index"] == 0
    assert groups[1]["target_id"] == "ref-1"
    # claims sorted by claim_id => b,c,d; first non-null callout is from c
    assert groups[1]["callout"] == "[1]"
    assert [c["claim_id"] for c in groups[1]["claims"]] == ["b", "c", "d"]


def test_export_csv_headers_and_row_counts(store: JudgmentStore):
    store.upsert(
        "c1",
        {
            "status": "final",
            "verdict": "support",
            "doc_id": "doc-1",
            "citation_index": 1,
            "target_id": "ref-1",
            "callout": "[1]",
            "claim_text": "T1",
        },
    )
    store.upsert(
        "c2",
        {
            "status": "final",
            "verdict": "contradict",
            "doc_id": "doc-1",
            "citation_index": 1,
            "target_id": "ref-1",
            "callout": "[1]",
            "claim_text": "T2",
        },
    )

    claims_csv = store.export_claims(include_drafts=False, mode="core", format="csv")
    reader = csv.DictReader(StringIO(claims_csv.decode("utf-8")))
    rows = list(reader)
    assert reader.fieldnames == [
        "claim_id",
        "reviewer_uid",
        "status",
        "verdict",
        "claim_text",
    ]
    assert len(rows) == 2

    callouts_csv = store.export_callouts(
        include_drafts=False, mode="core", format="csv"
    )
    reader2 = csv.DictReader(StringIO(callouts_csv.decode("utf-8")))
    rows2 = list(reader2)
    assert reader2.fieldnames == [
        "doc_id",
        "citation_index",
        "target_id",
        "callout",
        "claim_id",
        "reviewer_uid",
        "status",
        "verdict",
        "claim_text",
    ]
    assert len(rows2) == 2


def test_export_claims_verbose_json_includes_notes_object(store: JudgmentStore):
    store.upsert(
        "c1",
        {
            "status": "final",
            "verdict": "support",
            "notes": {"rationale": "Because."},
            "doc_id": "doc-1",
            "citation_index": 1,
            "target_id": "ref-1",
            "callout": "[1]",
            "claim_text": "T1",
        },
    )

    exported = store.export_claims(include_drafts=False, mode="verbose", format="json")
    rows = json.loads(exported.decode("utf-8"))
    assert len(rows) == 1
    assert rows[0]["claim_id"] == "c1"
    assert rows[0]["reviewer_uid"] == "default"
    assert rows[0]["status"] == "final"
    assert rows[0]["verdict"] == "support"
    assert rows[0]["notes"] == {
        "rationale": "Because.",
        "caveats": None,
        "followups": None,
    }
    assert "rationale" not in rows[0]
