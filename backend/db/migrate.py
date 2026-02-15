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
    # ---------------------------------------------------------------------
    # Identity split scaffolding (Work vs Document vs DocumentVersion)
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS documents (
      document_id text PRIMARY KEY,
      created_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local',
      biblio_work_id text NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS document_versions (
      document_version_id text PRIMARY KEY,
      document_id text NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
      sha256 text NOT NULL,
      size_bytes bigint NOT NULL,
      pdf_object_key text NOT NULL,
      filename text NOT NULL DEFAULT 'document.pdf',
      created_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local'
    );
    """,
    # Backfill/alter for existing databases.
    """
    ALTER TABLE document_versions
      ADD COLUMN IF NOT EXISTS filename text;
    """,
    """
    UPDATE document_versions
       SET filename = 'document.pdf'
     WHERE filename IS NULL;
    """,
    """
    ALTER TABLE document_versions
      ALTER COLUMN filename SET DEFAULT 'document.pdf';
    """,
    """
    ALTER TABLE document_versions
      ALTER COLUMN filename SET NOT NULL;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS document_versions_sha256_uniq
      ON document_versions(sha256);
    """,
    """
    CREATE INDEX IF NOT EXISTS document_versions_document_id_idx
      ON document_versions(document_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS project_documents (
      project_id text NOT NULL,
      document_id text NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
      added_at timestamptz NOT NULL DEFAULT now(),
      added_by_user_id text NOT NULL DEFAULT 'local',
      tags jsonb NOT NULL DEFAULT '{}'::jsonb,
      PRIMARY KEY(project_id, document_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS project_documents_project_id_idx
      ON project_documents(project_id);
    """,
    # ---------------------------------------------------------------------
    # Versioned settings + workflows (project-scoped)
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS settings_bundles (
      bundle_id text PRIMARY KEY,
      project_id text NOT NULL,
      name text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local'
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS settings_bundles_project_name_uniq
      ON settings_bundles(project_id, name);
    """,
    """
    CREATE TABLE IF NOT EXISTS settings_versions (
      version_id text PRIMARY KEY,
      bundle_id text NOT NULL REFERENCES settings_bundles(bundle_id) ON DELETE CASCADE,
      project_id text NOT NULL,
      version int NOT NULL,
      schema_version int NOT NULL,
      config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local',
      locked boolean NOT NULL DEFAULT false
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS settings_versions_bundle_version_uniq
      ON settings_versions(bundle_id, version);
    """,
    """
    CREATE INDEX IF NOT EXISTS settings_versions_project_id_idx
      ON settings_versions(project_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS settings_field_policies (
      policy_id text PRIMARY KEY,
      schema_version int NOT NULL,
      json_path text NOT NULL,
      editable boolean NOT NULL,
      constraints_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS settings_field_policies_schema_idx
      ON settings_field_policies(schema_version);
    """,
    """
    CREATE TABLE IF NOT EXISTS workflow_definitions (
      workflow_id text PRIMARY KEY,
      project_id text NOT NULL,
      name text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local'
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS workflow_definitions_project_name_uniq
      ON workflow_definitions(project_id, name);
    """,
    """
    CREATE TABLE IF NOT EXISTS workflow_versions (
      workflow_version_id text PRIMARY KEY,
      workflow_id text NOT NULL REFERENCES workflow_definitions(workflow_id)
        ON DELETE CASCADE,
      project_id text NOT NULL,
      version int NOT NULL,
      graph_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local',
      locked boolean NOT NULL DEFAULT false
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS workflow_versions_workflow_version_uniq
      ON workflow_versions(workflow_id, version);
    """,
    """
    CREATE INDEX IF NOT EXISTS workflow_versions_project_id_idx
      ON workflow_versions(project_id);
    """,
    # ---------------------------------------------------------------------
    # Stable locators (project-scoped) for evidence/span addressing
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS locators (
      locator_id text PRIMARY KEY,
      project_id text NOT NULL,
      created_by_user_id text NOT NULL DEFAULT 'local',
      document_version_id text NOT NULL REFERENCES document_versions(
        document_version_id
      )
        ON DELETE CASCADE,
      type text NOT NULL,
      payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS locators_project_docver_idx
      ON locators(project_id, document_version_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS locators_project_type_idx
      ON locators(project_id, type);
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
