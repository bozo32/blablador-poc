"""Idempotent schema creation for the ingestion spine tables."""

from __future__ import annotations

from backend.db.pg import connect


_DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS works (
      work_id text PRIMARY KEY,
      created_at timestamptz NOT NULL DEFAULT now(),
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      filename text NOT NULL,
      sha256 text NOT NULL,
      size_bytes bigint NOT NULL,
      pdf_object_key text NOT NULL,
      active_attempt_id text NULL,
      tags jsonb NOT NULL DEFAULT '{}'::jsonb
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS attempts (
      attempt_id text PRIMARY KEY,
      work_id text NOT NULL REFERENCES works(work_id) ON DELETE CASCADE,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      kind text NOT NULL,
      state text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      started_at timestamptz NULL,
      finished_at timestamptz NULL,
      schema_version int NOT NULL,
      settings_hash text NOT NULL,
      settings_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      provenance_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      quality_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      failure_reason text NULL,
      failure_detail text NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
      job_id text PRIMARY KEY,
      attempt_id text NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      worker text NOT NULL,
      state text NOT NULL,
      progress_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      heartbeat_at timestamptz NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
      artifact_id text PRIMARY KEY,
      attempt_id text NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      artifact_type text NOT NULL,
      object_key text NOT NULL,
      bytes bigint NULL,
      content_type text NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    # Idempotent alters for existing databases.
    """
    ALTER TABLE works
      ADD COLUMN IF NOT EXISTS project_id text NOT NULL DEFAULT 'default';
    """,
    """
    ALTER TABLE works
      ADD COLUMN IF NOT EXISTS created_by_user_id text NOT NULL DEFAULT 'local';
    """,
    """
    ALTER TABLE attempts
      ADD COLUMN IF NOT EXISTS project_id text NOT NULL DEFAULT 'default';
    """,
    """
    ALTER TABLE attempts
      ADD COLUMN IF NOT EXISTS created_by_user_id text NOT NULL DEFAULT 'local';
    """,
    """
    ALTER TABLE jobs
      ADD COLUMN IF NOT EXISTS project_id text NOT NULL DEFAULT 'default';
    """,
    """
    ALTER TABLE jobs
      ADD COLUMN IF NOT EXISTS created_by_user_id text NOT NULL DEFAULT 'local';
    """,
    """
    ALTER TABLE artifacts
      ADD COLUMN IF NOT EXISTS project_id text NOT NULL DEFAULT 'default';
    """,
    """
    ALTER TABLE artifacts
      ADD COLUMN IF NOT EXISTS created_by_user_id text NOT NULL DEFAULT 'local';
    """,
    """
    CREATE INDEX IF NOT EXISTS attempts_work_id_idx ON attempts(work_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS attempts_project_id_idx ON attempts(project_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_attempt_id_idx ON jobs(attempt_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_project_id_idx ON jobs(project_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS artifacts_attempt_id_idx ON artifacts(attempt_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS artifacts_project_id_idx ON artifacts(project_id);
    """,
]


def apply_migrations() -> None:
    """Apply schema creation DDL.

    Designed to be safe to call multiple times on fresh startup.
    """
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            for stmt in _DDL_STATEMENTS:
                cur.execute(stmt)
