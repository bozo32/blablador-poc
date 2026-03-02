from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main as backend_main


def _headers() -> dict[str, str]:
    return {
        "X-Project-Id": "proj-a",
        "X-User-Id": "default",
        "X-Reviewer-Uid": "default",
    }


def test_decisions_read_starts_empty(monkeypatch) -> None:
    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: True)
    client = TestClient(backend_main.app)

    resp = client.get(
        "/claims/claim-decisions/evidence/decisions",
        params={"reviewer_uid": "default"},
        headers=_headers(),
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["claim_id"] == "claim-decisions"
    assert data["reviewer_uid"] == "default"
    assert data["version"] == 0
    assert data["targets"] == {}
    assert data["pinned_targets"] == []


def test_decision_pin_idempotency_occ_and_clear_noop(monkeypatch) -> None:
    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: True)
    client = TestClient(backend_main.app)
    claim_id = "claim-evt"

    payload = {
        "idempotency_key": "k1",
        "expected_version": 0,
        "action": "pin",
        "target": {"attachment_id": "att_dummy", "span_id": "span_dummy"},
        "payload": {"snippet": "Pinned snippet"},
    }
    resp = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json=payload,
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["version"] == 1
    assert data["state"]["pinned"] is True

    # Idempotency replay returns identical payload.
    replay = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json=payload,
        headers=_headers(),
    )
    assert replay.status_code == 200
    assert replay.json() == data

    # OCC conflict with stale expected_version.
    stale = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json={
            **payload,
            "idempotency_key": "k2",
            "expected_version": 0,
        },
        headers=_headers(),
    )
    assert stale.status_code == 409
    body = stale.json()
    assert body.get("detail") == "version conflict"
    assert body.get("current_version") == 1

    # Clear resets state (writes event).
    cleared = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json={
            "idempotency_key": "k3",
            "expected_version": 1,
            "action": "clear",
            "payload": {"snippet": "Cleared"},
        },
        headers=_headers(),
    )
    assert cleared.status_code == 200
    cleared_data = cleared.json()
    assert cleared_data["version"] == 2
    assert cleared_data["no_op"] is False

    # Clear again is a no-op and does not advance version.
    cleared_again = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json={
            "idempotency_key": "k4",
            "expected_version": 2,
            "action": "clear",
        },
        headers=_headers(),
    )
    assert cleared_again.status_code == 200
    again_data = cleared_again.json()
    assert again_data["version"] == 2
    assert again_data["no_op"] is True


class _StubEvidenceService:
    def list_candidates(self, claim_id: str, **_kwargs):
        return {
            "candidates": [],
            "total": 0,
            "offset": 0,
            "limit": 10,
            "lock_state": {"status": "idle", "locked": False},
            "run": {"run_id": "r1", "created_at": "now"},
        }


def test_evidence_overlay_pinned_placeholder_when_missing(monkeypatch) -> None:
    stub = _StubEvidenceService()
    monkeypatch.setattr(backend_main, "evidence_service", stub)
    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: True)
    client = TestClient(backend_main.app)

    claim_id = "claim-overlay"
    pin = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json={
            "idempotency_key": "k1",
            "expected_version": 0,
            "action": "pin",
            "target": {"attachment_id": "att_dummy", "span_id": "span_dummy"},
            "payload": {"snippet": "Pinned snippet"},
        },
        headers=_headers(),
    )
    assert pin.status_code == 200

    resp = client.get(
        f"/claims/{claim_id}/evidence",
        params={"reviewer_uid": "default", "pinned_only": True},
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pinned"], "expected pinned placeholder"
    placeholder = data["pinned"][0]
    assert placeholder["metadata"].get("not_in_current_run") is True
    assert placeholder["decision_state"]["pinned"] is True


def test_evidence_overlay_annotates_candidates(monkeypatch) -> None:
    class Stub:
        def list_candidates(self, claim_id: str, **_kwargs):
            return {
                "candidates": [
                    {
                        "id": "cand-1",
                        "claim_id": claim_id,
                        "attachment_id": "att_dummy",
                        "label": "neutral",
                        "text": "Candidate text",
                        "scores": {},
                        "badges": [],
                        "metadata": {"span_id": "span_dummy"},
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

    stub = Stub()
    monkeypatch.setattr(backend_main, "evidence_service", stub)
    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: True)
    client = TestClient(backend_main.app)

    claim_id = "claim-annotate"
    _ = client.post(
        f"/claims/{claim_id}/evidence/decisions/events",
        params={"reviewer_uid": "default"},
        json={
            "idempotency_key": "k1",
            "expected_version": 0,
            "action": "pin",
            "target": {"attachment_id": "att_dummy", "span_id": "span_dummy"},
        },
        headers=_headers(),
    )

    resp = client.get(
        f"/claims/{claim_id}/evidence",
        params={"reviewer_uid": "default"},
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidates"]
    cand = data["candidates"][0]
    assert cand["decision_target"]["target_key"] == "att_dummy:span_dummy"
    assert cand["decision_state"]["pinned"] is True
    assert "decision:pinned" in (cand.get("badges") or [])
