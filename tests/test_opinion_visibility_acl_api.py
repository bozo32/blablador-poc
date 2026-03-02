from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.spine import opinion_events


def _headers(*, user_id: str = "reviewer-a", reviewer_uid: str = "reviewer-a") -> dict[str, str]:
    return {
        "X-Project-Id": "proj-a",
        "X-User-Id": user_id,
        "X-Reviewer-Uid": reviewer_uid,
    }


def test_visibility_normalization_accepts_shared_alias() -> None:
    assert opinion_events.normalize_visibility("private") == "private"
    assert opinion_events.normalize_visibility("selectable") == "selectable"
    assert opinion_events.normalize_visibility("public") == "public"
    assert opinion_events.normalize_visibility("shared") == "selectable"

    with pytest.raises(ValueError, match="visibility must be one of"):
        opinion_events.normalize_visibility("team-only")


def test_opinion_read_routes_require_project_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: False)
    client = TestClient(backend_main.app)

    resp = client.get(
        "/opinions/events",
        params={"reviewer_uid": "reviewer-a"},
        headers=_headers(),
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "User is not a member of this project"


def test_opinion_read_routes_thread_acl_viewer_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _StubOpinionEvents:
        @staticmethod
        def list_recent_events(**kwargs):
            captured.update(kwargs)
            return [
                {
                    "event_id": 11,
                    "owner_uid": "reviewer-a",
                    "kind": "follow",
                    "target_key": "citespan:1",
                    "visibility": "public",
                    "payload": {"status": "follow"},
                }
            ]

    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: True)
    monkeypatch.setattr(backend_main, "opinion_events", _StubOpinionEvents())

    client = TestClient(backend_main.app)
    resp = client.get(
        "/opinions/events",
        params={"reviewer_uid": "reviewer-a", "limit": 10},
        headers=_headers(user_id="viewer-1", reviewer_uid="reviewer-a"),
    )
    assert resp.status_code == 200
    assert captured["viewer_uid"] == "viewer-1"
    assert captured["viewer_is_project_member"] is True
    assert resp.json()["events"][0]["visibility"] == "public"


def test_opinion_write_requires_actor_and_owner_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_main, "has_project_membership", lambda **_: True)
    client = TestClient(backend_main.app)

    resp = client.post(
        "/opinions/events",
        params={"reviewer_uid": "reviewer-a"},
        headers=_headers(user_id="viewer-1", reviewer_uid="reviewer-a"),
        json={
            "kind": "follow",
            "target_key": "citespan:span-1",
            "visibility": "shared",
            "payload": {"status": "follow"},
        },
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "X-User-Id must match X-Reviewer-Uid for opinion writes"
