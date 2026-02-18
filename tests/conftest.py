from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _apply_spine_migrations() -> None:
    """Ensure Postgres spine tables exist for tests.

    Phase 09.3 moves multiple stores (attachments/evidence/judgments/etc.) onto
    Postgres, so unit tests require the DDL to be present.
    """
    from backend.db.migrate import apply_migrations
    from backend.db.pg import connect

    apply_migrations()

    # Keep tests isolated from prior runs: these tables are now spine-backed.
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE
                  attachment_artifacts,
                  attachment_events,
                  attachments,
                  evidence_runs,
                  evidence_selections,
                  judgments,
                  confirmed_claims,
                  project_meta
                """
            )

    # Note: we intentionally do not delete S3 objects here. Object keys are
    # UUID-scoped in tests, and `delete_all()` can be slow if the bucket has
    # accumulated objects from prior interactive runs.
