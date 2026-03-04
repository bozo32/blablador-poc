from __future__ import annotations

from backend.workflow_happy_path import orchestrator as workflow_orchestrator


def test_run_one_uses_canonical_run_project_for_extract_builder(monkeypatch) -> None:
    run_id = "run-scope-test"
    canonical_project_id = "proj-canonical"
    calls: dict[str, list[str] | str] = {
        "get_run_status": [],
        "list_target_status": [],
        "upsert_run_status": [],
        "append_event": [],
    }

    monkeypatch.setattr(
        workflow_orchestrator.pipeline_runs,
        "get_run",
        lambda rid: {
            "run_id": rid,
            "project_id": canonical_project_id,
            "created_by_user_id": "reviewer-a",
        },
    )
    monkeypatch.setattr(
        workflow_orchestrator.pipeline_run_status,
        "get_run_status",
        lambda rid, *, project_id: (
            calls["get_run_status"].append(project_id),
            {
                "run_id": rid,
                "scope_type": "claimspan",
                "scope_id": "cite:doc-1:0:reviewer-a:1a",
                "reviewer_uid": "reviewer-a",
                "citing_doc_id": "doc-1",
                "project_id": project_id,
                "started_at": None,
            },
        )[1],
    )
    monkeypatch.setattr(
        workflow_orchestrator.pipeline_run_status,
        "list_target_status",
        lambda rid, *, project_id: (
            calls["list_target_status"].append(project_id),
            [],
        )[1],
    )
    monkeypatch.setattr(
        workflow_orchestrator.pipeline_run_status,
        "upsert_run_status",
        lambda rid, *, project_id, **kwargs: calls["upsert_run_status"].append(project_id),
    )
    monkeypatch.setattr(
        workflow_orchestrator.pipeline_run_status,
        "append_event",
        lambda rid, *, project_id, **kwargs: calls["append_event"].append(project_id),
    )
    monkeypatch.setattr(
        workflow_orchestrator,
        "build_extract_data",
        lambda *, citing_doc_id, project_id: (
            calls.__setitem__("build_extract_project", project_id),
            {"structured_doc": {}, "citation_anchors": []},
        )[1],
    )
    monkeypatch.setattr(
        workflow_orchestrator,
        "build_citespans_data",
        lambda **kwargs: {"by_target": {}},
    )
    monkeypatch.setattr(
        workflow_orchestrator.pipeline_contracts_service,
        "store_stage",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        workflow_orchestrator.background_state,
        "get_state",
        lambda: {"paused": False},
    )

    orch = workflow_orchestrator.HappyPathOrchestrator.__new__(
        workflow_orchestrator.HappyPathOrchestrator
    )
    orch._span_graph_store = object()

    orch._run_one(run_id)

    assert calls["build_extract_project"] == canonical_project_id
    assert calls["get_run_status"] == [canonical_project_id]
    assert calls["list_target_status"] == [canonical_project_id, canonical_project_id]
    assert calls["upsert_run_status"] == [canonical_project_id, canonical_project_id]
    assert calls["append_event"] == [canonical_project_id, canonical_project_id]
