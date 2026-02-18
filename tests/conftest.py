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
    from backend.object_store import s3 as object_store_s3

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

    # Clear MinIO/S3 objects from previous runs.
    try:
        object_store_s3.delete_all()
    except Exception:
        # Some unit tests don't require S3; don't block the whole suite.
        pass
