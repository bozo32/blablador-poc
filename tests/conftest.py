from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _apply_spine_migrations() -> None:
    """Ensure Postgres spine tables exist for tests.

    Phase 09.3 moves multiple stores (attachments/evidence/judgments/etc.) onto
    Postgres, so unit tests require the DDL to be present.
    """
    from backend.db.migrate import apply_migrations

    apply_migrations()


@pytest.fixture(autouse=True)
def _truncate_spine_backed_tables() -> None:
    """Keep tests isolated from prior runs.

    Phase 09.3 moves multiple stores onto Postgres; graph/spans are now Postgres too.
    """
    from backend.db.pg import connect

    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE
                  evidence_decision_targets,
                  evidence_decision_events,
                  evidence_decision_streams,
                  pipeline_run_events,
                  pipeline_target_status,
                  pipeline_run_status,
                  pipeline_run_scopes,
                  pipeline_stage_artifacts,
                  pipeline_runs,
                  attachment_artifacts,
                  attachment_events,
                  attachments,
                  background_state,
                  evidence_runs,
                  evidence_selections,
                  judgments,
                  user_active_projects,
                  user_project_memberships,
                  confirmed_claims,
                  project_meta,
                  graph_edge_votes,
                  graph_edges,
                  graph_aliases,
                  graph_nodes,
                  span_graph_neighborhood_candidates,
                  span_graph_neighborhood_runs,
                  span_graph_review_marks,
                  span_graph_assertions,
                  span_graph_claim_span_atoms,
                  span_graph_claim_atoms,
                  span_graph_claim_spans,
                  span_graph_span_cite_roles,
                  span_graph_citation_span_index,
                  span_graph_span_cites,
                  span_graph_spans,
                  span_graph_work_cites,
                  span_graph_works
                RESTART IDENTITY CASCADE
                """
            )

    # Note: we intentionally do not delete S3 objects here. Object keys are
    # UUID-scoped in tests, and `delete_all()` can be slow if the bucket has
    # accumulated objects from prior interactive runs.
