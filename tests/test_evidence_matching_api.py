from __future__ import annotations

import threading
import time

import pytest

from backend import attachment_pipeline, attachment_store
from backend.evidence_matching.service import EvidenceMatchingService
from backend.evidence_matching.store import EvidenceRunStore
from backend.evidence_matching.types import EvidenceCandidate, EvidenceLabel, RankScores
from backend.settings import settings


# ---------------------------------------------------------------------------
# EvidenceRunStore tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# EvidenceMatchingService tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def service_store(tmp_path, monkeypatch) -> EvidenceRunStore:
    monkeypatch.setattr(settings, "EVIDENCE_STORE_DIR", tmp_path / "service-runs")
    return EvidenceRunStore(settings=settings)


def _build_evidence_candidate(
    candidate_id: str,
    *,
    claim_id: str = "claim-svc",
    label: EvidenceLabel = EvidenceLabel.ENTAILS,
    score: float = 0.9,
    position: int = 1,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        id=candidate_id,
        claim_id=claim_id,
        attachment_id=f"att-{candidate_id}",
        text=f"snippet {candidate_id}",
        label=label,
        scores=RankScores(combined=score, position=position),
    )


def test_service_ensure_current_run_tracks_snapshot(monkeypatch, service_store):
    candidates = [
        _build_evidence_candidate("cand-a", position=1),
        _build_evidence_candidate(
            "cand-b", position=2, label=EvidenceLabel.CONTRADICTS, score=0.7
        ),
    ]

    class SimplePipeline:
        def run(self, **kwargs):
            return candidates

        def summarize_labels(self, cands):  # pragma: no cover - API parity only
            return {}

    monkeypatch.setattr(
        attachment_store,
        "list_attachments",
        lambda claim_id=None: [
            {
                "id": "att-1",
                "status": attachment_store.STATUS_MATCHED,
                "updated_at": "2026-01-28T00:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(attachment_store, "is_ready", lambda record: True)

    service = EvidenceMatchingService(
        settings=settings,
        pipeline=SimplePipeline(),
        store=service_store,
        load_windows=lambda claim_id: ["window"],
        seed_windows=lambda *args, **kwargs: candidates,
    )

    run1 = service.ensure_current_run("claim-svc", claim_text="Alpha claim")
    assert run1["summary"]["total"] == 2

    # Snapshot unchanged -> no rerun
    run2 = service.ensure_current_run("claim-svc")
    assert run2["run_id"] == run1["run_id"]

    # Change attachment timestamp -> rerun
    monkeypatch.setattr(
        attachment_store,
        "list_attachments",
        lambda claim_id=None: [
            {
                "id": "att-1",
                "status": attachment_store.STATUS_MATCHED,
                "updated_at": "2026-01-28T00:05:00Z",
            }
        ],
    )
    run3 = service.ensure_current_run("claim-svc")
    assert run3["run_id"] != run1["run_id"]


def test_service_request_rerun_serializes_queue(monkeypatch, service_store):
    start_event = threading.Event()
    release_event = threading.Event()
    candidates = [_build_evidence_candidate("cand-q")]

    class BlockingPipeline:
        def run(self, **kwargs):
            start_event.set()
            release_event.wait(timeout=2)
            return candidates

        def summarize_labels(self, cands):  # pragma: no cover - API parity
            return {}

    monkeypatch.setattr(
        attachment_store,
        "list_attachments",
        lambda claim_id=None: [
            {
                "id": "att-queue",
                "status": attachment_store.STATUS_MATCHED,
                "updated_at": "2026-01-28T01:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(attachment_store, "is_ready", lambda record: True)

    service = EvidenceMatchingService(
        settings=settings,
        pipeline=BlockingPipeline(),
        store=service_store,
        max_workers=1,
        load_windows=lambda claim_id: ["window"],
        seed_windows=lambda *args, **kwargs: candidates,
    )

    first = service.request_rerun("claim-queue", claim_text="Queued claim")
    assert first["status"] == "running"
    assert start_event.wait(timeout=1), "rerun worker did not start"

    queued = service.request_rerun(
        "claim-queue", claim_text="Queued claim", note="second"
    )
    assert queued["status"] == "queued"

    release_event.set()

    def _history_ready() -> bool:
        return len(service_store.history("claim-queue")) >= 2

    assert _wait_for(_history_ready), "queued rerun did not complete"


def test_service_auto_rerun_triggered_by_attachment_pipeline(tmp_path, monkeypatch):
    settings.ATTACHMENT_DIR = tmp_path / "attachments"
    source = tmp_path / "auto.pdf"
    source.write_bytes(b"%PDF-auto")
    record = attachment_store.create_attachment(
        claim_id="claim-auto",
        doc_id="doc-auto",
        local_path=source,
    )

    monkeypatch.setattr(
        attachment_pipeline.grobid_client,
        "extract_tei",
        lambda *_args, **_kwargs: "<TEI></TEI>",
    )
    monkeypatch.setattr(
        attachment_pipeline.utils, "embed", lambda texts, **_: [[1.0] for _ in texts]
    )
    monkeypatch.setattr(
        attachment_pipeline.extraction,
        "parse_tei",
        lambda *_: {"metadata": {}},
    )
    monkeypatch.setattr(
        attachment_store,
        "save_artifacts",
        lambda *_args, **_kwargs: {
            "sentences": str(tmp_path / "sentences.ndjson"),
            "tei_xml": "tei.xml",
            "tei_json": "tei.json",
        },
    )

    class FakeService:
        def __init__(self):
            self.invocations: list[str] = []

        def trigger_auto_rerun(self, claim_id: str, **_):
            self.invocations.append(claim_id)
            return {"status": "queued"}

    fake_service = FakeService()
    monkeypatch.setattr(attachment_pipeline, "evidence_service", fake_service)

    attachment_pipeline.process_attachment(record["id"])

    assert fake_service.invocations == ["claim-auto"]


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
