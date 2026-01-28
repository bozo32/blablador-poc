from __future__ import annotations

import pytest

from backend.evidence_matching.store import EvidenceRunStore
from backend.settings import settings


@pytest.fixture()
def run_store(tmp_path, monkeypatch) -> EvidenceRunStore:
    monkeypatch.setattr(settings, "EVIDENCE_STORE_DIR", tmp_path / "runs")
    monkeypatch.setattr(settings, "EVIDENCE_HISTORY_DEPTH", 3)
    return EvidenceRunStore(settings=settings)


def _candidate(
    candidate_id: str, *, position: int, score: float, label: str = "entails"
):
    return {
        "id": candidate_id,
        "claim_id": "claim-123",
        "attachment_id": f"att-{candidate_id}",
        "label": label,
        "text": f"Snippet for {candidate_id}",
        "scores": {"position": position, "combined": score},
        "badges": [],
        "metadata": {"page": "1"},
        "spans": [],
    }


def test_store_record_run_persists_latest_snapshot(run_store: EvidenceRunStore):
    payload = run_store.record_run(
        "claim-123",
        candidates=[
            _candidate("cand-1", position=1, score=0.9),
            _candidate("cand-2", position=2, score=0.8, label="contradicts"),
        ],
        metadata={"note": "initial"},
    )

    latest = run_store.latest_run("claim-123")
    assert latest is not None
    assert latest["run_id"] == payload["run_id"]
    assert latest["summary"]["total"] == 2
    assert latest["summary"]["label_counts"]["entails"] == 1
    assert latest["summary"]["label_counts"]["contradicts"] == 1
    assert latest["metadata"]["note"] == "initial"
    assert all("delta" in cand for cand in latest["candidates"])


def test_store_history_is_trimmed_to_configured_depth(run_store: EvidenceRunStore):
    for idx in range(5):
        run_store.record_run(
            "claim-abc",
            candidates=[_candidate(f"c-{idx}", position=1, score=1.0 - idx * 0.1)],
            metadata={"iteration": idx},
        )

    history = run_store.history("claim-abc")
    assert len(history) == 3  # depth enforced via fixture
    # history is newest-first; ensure latest iteration retained
    assert history[0]["metadata"]["iteration"] == 4


def test_store_delta_metadata_reflects_rank_changes(run_store: EvidenceRunStore):
    run_store.record_run(
        "claim-delta",
        candidates=[
            _candidate("cand-1", position=1, score=0.9),
            _candidate("cand-2", position=2, score=0.7, label="neutral"),
        ],
    )

    latest = run_store.record_run(
        "claim-delta",
        candidates=[
            _candidate("cand-2", position=1, score=0.85, label="neutral"),
            _candidate("cand-1", position=2, score=0.75),
            _candidate("cand-3", position=3, score=0.5),
        ],
    )

    c2, c1, c3 = latest["candidates"]
    assert c2["delta"]["status"] == "promoted"
    assert c2["delta"]["rank_change"] == 1
    assert c2["delta"]["diversity_note"] == "neutral-coverage"

    assert c1["delta"]["status"] == "demoted"
    assert c1["delta"]["rank_change"] == -1
    assert c1["delta"]["demotion_reason"] == "rerank-adjustment"
    assert c1["delta"]["score_delta"] < 0

    assert c3["delta"]["status"] == "new"
    assert c3["delta"]["rank_change"] is None
