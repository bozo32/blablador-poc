from __future__ import annotations

import threading
import time
import types
from typing import cast, Optional

import pytest
from fastapi.testclient import TestClient

# NOTE: Do not stub sys.modules at import time here; it leaks into other tests.


from backend import attachment_pipeline, attachment_store
import backend.main as backend_main
from backend.evidence_matching.pipeline import EvidencePipeline
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
        pipeline=cast(EvidencePipeline, SimplePipeline()),
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
        pipeline=cast(EvidencePipeline, BlockingPipeline()),
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
        claim_text="Attachment rerun text",
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
            self.claim_texts: list[Optional[str]] = []

        def trigger_auto_rerun(
            self, claim_id: str, *, claim_text: Optional[str] = None, **_
        ):
            self.invocations.append(claim_id)
            self.claim_texts.append(claim_text)
            return {"status": "queued"}

    fake_service = FakeService()
    monkeypatch.setattr(attachment_pipeline, "evidence_service", fake_service)

    attachment_pipeline.process_attachment(record["id"])

    assert fake_service.invocations == ["claim-auto"]
    assert fake_service.claim_texts == ["Attachment rerun text"]


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _StubEvidenceService:
    def __init__(self):
        self.store = types.SimpleNamespace(
            latest_run=lambda _claim_id: {"run_id": "r0"}
        )

    def ensure_current_run(
        self, claim_id, claim_text=None, **_
    ):  # pragma: no cover - trivial stub
        self.latest_claim_text = claim_text

    def list_candidates(self, claim_id, **_):
        return {
            "candidates": [
                {
                    "id": "cand-stub",
                    "claim_id": claim_id,
                    "attachment_id": "att-stub",
                    "label": "entails",
                    "text": "snippet",
                    "scores": {"combined": 0.9},
                    "badges": [],
                    "metadata": {},
                    "spans": [],
                    "highlights": [],
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 10,
            "lock_state": {"status": "idle", "locked": False},
            "run": {"run_id": "r1", "created_at": "now"},
        }

    def request_rerun(self, claim_id, **_):
        return {
            "job_id": f"job-{claim_id}",
            "status": "queued",
            "position": 0,
            "locked": True,
        }

    def get_history(self, claim_id):
        return [
            {
                "run_id": "run-hist",
                "created_at": "now",
                "summary": {"total": 1},
                "metadata": {"note": "test"},
            }
        ]


def test_service_api_list_endpoint_returns_payload(monkeypatch):
    stub = _StubEvidenceService()
    monkeypatch.setattr(backend_main, "evidence_service", stub)
    client = TestClient(backend_main.app)

    response = client.get("/claims/claim-api/evidence", params={"label": "entails"})

    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == "claim-api"
    assert data["total"] == 1
    assert data["candidates"][0]["id"] == "cand-stub"


def test_service_api_accepts_claim_text(monkeypatch):
    stub = _StubEvidenceService()
    monkeypatch.setattr(backend_main, "evidence_service", stub)
    client = TestClient(backend_main.app)

    response = client.get(
        "/claims/claim-text/evidence", params={"claim_text": "Fresh claim"}
    )

    assert response.status_code == 200
    assert stub.latest_claim_text == "Fresh claim"


def test_service_api_rerun_endpoint_returns_job_state(monkeypatch):
    stub = _StubEvidenceService()
    monkeypatch.setattr(backend_main, "evidence_service", stub)
    client = TestClient(backend_main.app)

    payload = {"claim_text": "Alpha", "note": "manual"}
    response = client.post("/claims/claim-api/evidence/rerun", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job-claim-api"
    assert data["status"] == "queued"


def test_service_api_history_endpoint_returns_runs(monkeypatch):
    stub = _StubEvidenceService()
    monkeypatch.setattr(backend_main, "evidence_service", stub)
    client = TestClient(backend_main.app)

    response = client.get("/claims/claim-api/evidence/history")

    assert response.status_code == 200
    data = response.json()
    assert data["runs"] and data["runs"][0]["run_id"] == "run-hist"
