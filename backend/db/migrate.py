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
    # ---------------------------------------------------------------------
    # Phase 09.3: Spine everywhere (attachments, evidence, judgments, project)
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS project_meta (
      project_id text PRIMARY KEY,
      meta_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      updated_at timestamptz NOT NULL DEFAULT now(),
      updated_by_user_id text NOT NULL DEFAULT 'local'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS background_state (
      project_id text PRIMARY KEY,
      paused boolean NOT NULL DEFAULT false,
      updated_at timestamptz NOT NULL DEFAULT now(),
      reason text NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS attachments (
      attachment_id text PRIMARY KEY,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),

      claim_id text NULL,
      doc_id text NULL,
      citation_index int NULL,
      target_id text NULL,
      source_ingest_id text NULL,

      filename text NOT NULL,
      size_bytes bigint NOT NULL,
      status text NOT NULL,
      error text NULL,
      parsed_at timestamptz NULL,

      archived boolean NOT NULL DEFAULT false,
      archived_at timestamptz NULL,

      attempts int NOT NULL DEFAULT 0,
      max_attempts int NOT NULL DEFAULT 2,

      reference_hint jsonb NOT NULL DEFAULT '{}'::jsonb,
      claim_text text NULL,

      content_sha256 text NULL,

      pdf_object_key text NOT NULL,
      artifacts_json jsonb NOT NULL DEFAULT '{}'::jsonb
    );
    """,
    """
    ALTER TABLE attachments
      ADD COLUMN IF NOT EXISTS content_sha256 text;
    """,
    """
    CREATE INDEX IF NOT EXISTS attachments_project_id_idx
      ON attachments(project_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS attachments_project_content_sha256_idx
      ON attachments(project_id, content_sha256);
    """,
    """
    CREATE INDEX IF NOT EXISTS attachments_claim_id_idx
      ON attachments(claim_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS attachments_doc_id_idx
      ON attachments(doc_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS attachment_events (
      event_id text PRIMARY KEY,
      attachment_id text NOT NULL REFERENCES attachments(attachment_id)
        ON DELETE CASCADE,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      event text NOT NULL,
      detail text NULL,
      at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS attachment_events_attachment_at_idx
      ON attachment_events(attachment_id, at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS attachment_artifacts (
      artifact_id text PRIMARY KEY,
      attachment_id text NOT NULL REFERENCES attachments(attachment_id)
        ON DELETE CASCADE,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      artifact_type text NOT NULL,
      object_key text NOT NULL,
      bytes bigint NULL,
      content_type text NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS attachment_artifacts_attachment_type_idx
      ON attachment_artifacts(attachment_id, artifact_type);
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_runs (
      run_id text PRIMARY KEY,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      claim_id text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      note text NULL,
      summary_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      candidates_object_key text NULL,
      lock_state_json jsonb NOT NULL DEFAULT '{}'::jsonb
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_runs_claim_created_idx
      ON evidence_runs(claim_id, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_selections (
      selection_id text PRIMARY KEY,
      project_id text NOT NULL DEFAULT 'default',
      updated_by_user_id text NOT NULL DEFAULT 'local',
      claim_id text NOT NULL,
      reviewer_uid text NOT NULL DEFAULT 'default',
      updated_at timestamptz NOT NULL DEFAULT now(),
      verdict text NOT NULL,
      primary_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      secondary_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      note text NULL
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS evidence_selections_claim_reviewer_uniq
      ON evidence_selections(claim_id, reviewer_uid);
    """,
    """
    CREATE TABLE IF NOT EXISTS judgments (
      judgment_id text PRIMARY KEY,
      project_id text NOT NULL DEFAULT 'default',
      updated_by_user_id text NOT NULL DEFAULT 'local',
      claim_id text NOT NULL,
      reviewer_uid text NOT NULL DEFAULT 'default',
      updated_at timestamptz NOT NULL DEFAULT now(),
      status text NOT NULL,
      verdict text NULL,
      notes_json jsonb NOT NULL DEFAULT '{}'::jsonb,

      doc_id text NULL,
      citation_index int NULL,
      target_id text NULL,
      sentence_id text NULL,
      callout text NULL,
      reference_id text NULL,
      doi text NULL,
      author text NULL,
      year text NULL,
      claim_text text NULL,
      cited_work_id text NULL,
      citation_anchor jsonb NOT NULL DEFAULT '{}'::jsonb,
      span_selectors jsonb NOT NULL DEFAULT '{}'::jsonb,
      validation_json jsonb NOT NULL DEFAULT '{}'::jsonb
    );
    """,
    # Older DBs may have verdict NOT NULL; drafts allow NULL verdict.
    """
    ALTER TABLE judgments
      ALTER COLUMN verdict DROP NOT NULL;
    """,
    # Older DBs may have year as int; schema expects string (and may be non-numeric).
    """
    ALTER TABLE judgments
      ALTER COLUMN year TYPE text
      USING year::text;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS judgments_claim_reviewer_uniq
      ON judgments(claim_id, reviewer_uid);
    """,
    """
    CREATE INDEX IF NOT EXISTS judgments_doc_id_idx
      ON judgments(doc_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS confirmed_claims (
      project_id text NOT NULL DEFAULT 'default',
      document_id text NOT NULL,
      sentence_id text NOT NULL,
      claim_index int NOT NULL,
      parsed_text text NOT NULL,
      original_text text NULL,
      segmentation_model text NULL,
      reviewer_uid text NOT NULL DEFAULT 'default',
      confirmed_at timestamptz NOT NULL DEFAULT now(),
      confidence double precision NULL,
      citation_index int NULL,
      target_id text NULL,
      sentence_text text NULL,
      PRIMARY KEY(document_id, sentence_id, claim_index)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS confirmed_claims_project_doc_idx
      ON confirmed_claims(project_id, document_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS confirmed_claims_sentence_idx
      ON confirmed_claims(sentence_id);
    """,
    # ---------------------------------------------------------------------
    # Phase 09.3: Durable graph/workboard state (Postgres)
    #
    # These tables replace durable local SQLite graph stores (`data/graph.db`).
    # Table names are prefixed to avoid collisions with ingestion spine tables.
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS graph_nodes (
      project_id text NOT NULL DEFAULT 'default',
      node_id text NOT NULL,
      kind text NOT NULL,
      num int NULL,
      label text NULL,
      properties_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, node_id)
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS graph_nodes_project_num_uniq
      ON graph_nodes(project_id, num)
      WHERE num IS NOT NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS graph_nodes_project_kind_idx
      ON graph_nodes(project_id, kind);
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_aliases (
      project_id text NOT NULL DEFAULT 'default',
      alias text NOT NULL,
      node_id text NOT NULL,
      kind text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, alias)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS graph_aliases_project_node_idx
      ON graph_aliases(project_id, node_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_edges (
      edge_id bigserial PRIMARY KEY,
      project_id text NOT NULL DEFAULT 'default',
      source_id text NOT NULL,
      target_id text NOT NULL,
      kind text NOT NULL,
      ref_id text NOT NULL DEFAULT '',
      enabled boolean NOT NULL DEFAULT true,
      properties_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      UNIQUE(project_id, source_id, target_id, kind, ref_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS graph_edges_project_source_kind_idx
      ON graph_edges(project_id, source_id, kind);
    """,
    """
    CREATE INDEX IF NOT EXISTS graph_edges_project_target_kind_idx
      ON graph_edges(project_id, target_id, kind);
    """,
    """
    CREATE INDEX IF NOT EXISTS graph_edges_project_enabled_kind_idx
      ON graph_edges(project_id, enabled, kind);
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_edge_votes (
      edge_id bigint NOT NULL REFERENCES graph_edges(edge_id) ON DELETE CASCADE,
      reviewer_uid text NOT NULL,
      verdict text NOT NULL,
      confidence double precision NULL,
      comment text NULL,
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(edge_id, reviewer_uid)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS graph_edge_votes_edge_id_idx
      ON graph_edge_votes(edge_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_works (
      project_id text NOT NULL DEFAULT 'default',
      work_id text NOT NULL,
      doi text NULL,
      openalex_id text NULL,
      title text NULL,
      authors_json jsonb NULL,
      year text NULL,
      abstract text NULL,
      abstract_source text NULL,
      abstract_embedding_json jsonb NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, work_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_works_project_doi_idx
      ON span_graph_works(project_id, doi);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_works_project_openalex_idx
      ON span_graph_works(project_id, openalex_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_work_cites (
      project_id text NOT NULL DEFAULT 'default',
      citing_work_id text NOT NULL,
      cited_work_id text NOT NULL,
      source text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, citing_work_id, cited_work_id, source)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_work_cites_project_citing_idx
      ON span_graph_work_cites(project_id, citing_work_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_work_cites_project_cited_idx
      ON span_graph_work_cites(project_id, cited_work_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_spans (
      project_id text NOT NULL DEFAULT 'default',
      span_id text NOT NULL,
      work_id text NULL,
      ingest_id text NULL,
      kind text NOT NULL,
      selector_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      window_fingerprint text NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, span_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_spans_project_work_idx
      ON span_graph_spans(project_id, work_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_spans_project_ingest_idx
      ON span_graph_spans(project_id, ingest_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_span_cites (
      project_id text NOT NULL DEFAULT 'default',
      span_id text NOT NULL,
      cited_work_id text NOT NULL,
      reference_id text NULL,
      citation_index int NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, span_id, cited_work_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_span_cites_project_cited_idx
      ON span_graph_span_cites(project_id, cited_work_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_citation_span_index (
      project_id text NOT NULL DEFAULT 'default',
      ingest_id text NOT NULL,
      citation_index int NOT NULL,
      target_id text NOT NULL,
      span_id text NOT NULL,
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, ingest_id, citation_index, target_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_citation_span_index_project_span_idx
      ON span_graph_citation_span_index(project_id, span_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_span_cite_roles (
      project_id text NOT NULL DEFAULT 'default',
      span_id text NOT NULL,
      cited_work_id text NOT NULL,
      reviewer_uid text NOT NULL,
      role text NOT NULL,
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, span_id, cited_work_id, reviewer_uid)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_span_cite_roles_project_reviewer_idx
      ON span_graph_span_cite_roles(project_id, reviewer_uid);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_claim_spans (
      project_id text NOT NULL DEFAULT 'default',
      claim_span_id text NOT NULL,
      span_id text NOT NULL,
      order_index int NOT NULL,
      selector_json jsonb NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, claim_span_id),
      UNIQUE(project_id, span_id, order_index)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_claim_spans_project_span_idx
      ON span_graph_claim_spans(project_id, span_id, order_index);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_claim_atoms (
      project_id text NOT NULL DEFAULT 'default',
      claim_atom_id text NOT NULL,
      text text NOT NULL,
      created_by text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      supersedes_id text NULL,
      PRIMARY KEY(project_id, claim_atom_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_claim_atoms_project_created_by_idx
      ON span_graph_claim_atoms(project_id, created_by);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_claim_span_atoms (
      project_id text NOT NULL DEFAULT 'default',
      claim_span_id text NOT NULL,
      claim_atom_id text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, claim_span_id, claim_atom_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_assertions (
      project_id text NOT NULL DEFAULT 'default',
      assertion_id text NOT NULL,
      reviewer_uid text NOT NULL,
      verdict text NOT NULL,
      confidence double precision NULL,
      comment text NULL,
      claim_atom_id text NULL,
      claim_span_id text NULL,
      evidence_span_id text NULL,
      evidence_work_id text NULL,
      source text NULL,
      source_key text NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, assertion_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_assertions_project_reviewer_idx
      ON span_graph_assertions(project_id, reviewer_uid);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_assertions_project_claim_span_idx
      ON span_graph_assertions(project_id, claim_span_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_assertions_project_claim_atom_idx
      ON span_graph_assertions(project_id, claim_atom_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_assertions_project_evidence_span_idx
      ON span_graph_assertions(project_id, evidence_span_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_review_marks (
      project_id text NOT NULL DEFAULT 'default',
      claim_span_id text NOT NULL,
      reviewer_uid text NOT NULL,
      mark text NOT NULL,
      updated_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, claim_span_id, reviewer_uid, mark)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_review_marks_project_reviewer_idx
      ON span_graph_review_marks(project_id, reviewer_uid);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_neighborhood_runs (
      project_id text NOT NULL DEFAULT 'default',
      run_id text NOT NULL,
      created_by text NULL,
      context_work_id text NULL,
      context_span_id text NULL,
      context_claim_span_id text NULL,
      context_claim_atom_id text NULL,
      method text NOT NULL,
      params_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY(project_id, run_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_neighborhood_runs_project_work_idx
      ON span_graph_neighborhood_runs(project_id, context_work_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_neighborhood_runs_project_span_idx
      ON span_graph_neighborhood_runs(project_id, context_span_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS span_graph_neighborhood_candidates (
      project_id text NOT NULL DEFAULT 'default',
      run_id text NOT NULL,
      candidate_work_id text NOT NULL,
      bib_intersection int NULL,
      abstract_score double precision NULL,
      rank int NULL,
      detail_json jsonb NULL,
      PRIMARY KEY(project_id, run_id, candidate_work_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS span_graph_neighborhood_candidates_project_work_idx
      ON span_graph_neighborhood_candidates(project_id, candidate_work_id);
    """,
    # ---------------------------------------------------------------------
    # Phase 10-01: Pipeline stage contracts (runs + per-stage artifacts)
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
      run_id text PRIMARY KEY,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      work_id text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      input_fingerprint text NOT NULL,
      caps_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      settings_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      note text NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_runs_work_created_idx
      ON pipeline_runs(work_id, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_stage_artifacts (
      artifact_id text PRIMARY KEY,
      run_id text NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      stage text NOT NULL,
      schema_version int NOT NULL,
      artifact_type text NOT NULL,
      object_key text NOT NULL,
      bytes bigint NULL,
      content_type text NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS pipeline_stage_artifacts_run_stage_uniq
      ON pipeline_stage_artifacts(run_id, stage);
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_stage_artifacts_run_id_idx
      ON pipeline_stage_artifacts(run_id);
    """,
    # ---------------------------------------------------------------------
    # Phase 10-02: Mutable run status (scopes, per-run, per-target, events)
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS pipeline_run_scopes (
      scope_type text NOT NULL,
      scope_id text NOT NULL,
      reviewer_uid text NOT NULL,
      run_id text NOT NULL,
      citing_doc_id text NOT NULL,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_run_scopes_project_scope_idx
      ON pipeline_run_scopes(
        project_id,
        scope_type,
        scope_id,
        reviewer_uid,
        created_at DESC
      );
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_run_scopes_project_run_idx
      ON pipeline_run_scopes(project_id, run_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_run_status (
      run_id text PRIMARY KEY,
      scope_type text NOT NULL,
      scope_id text NOT NULL,
      reviewer_uid text NOT NULL,
      citing_doc_id text NOT NULL,
      project_id text NOT NULL DEFAULT 'default',
      created_by_user_id text NOT NULL DEFAULT 'local',

      state text NOT NULL,
      started_at timestamptz NULL,
      finished_at timestamptz NULL,
      updated_at timestamptz NOT NULL DEFAULT now(),

      error_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      metrics_json jsonb NOT NULL DEFAULT '{}'::jsonb
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_run_status_project_scope_idx
      ON pipeline_run_status(project_id, scope_id, reviewer_uid, updated_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_target_status (
      run_id text NOT NULL,
      target_id text NOT NULL,
      reference_id text NULL,
      citation_index int NULL,
      attachment_id text NULL,

      project_id text NOT NULL DEFAULT 'default',
      updated_at timestamptz NOT NULL DEFAULT now(),

      state text NOT NULL,
      stage_state_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      attempts_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      last_error_json jsonb NOT NULL DEFAULT '{}'::jsonb,

      PRIMARY KEY(run_id, target_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_target_status_project_run_idx
      ON pipeline_target_status(project_id, run_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_target_status_project_target_idx
      ON pipeline_target_status(project_id, target_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_run_events (
      event_id bigserial PRIMARY KEY,
      run_id text NOT NULL,
      target_id text NULL,
      project_id text NOT NULL DEFAULT 'default',
      type text NOT NULL,
      payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS pipeline_run_events_project_run_event_idx
      ON pipeline_run_events(project_id, run_id, event_id);
    """,
    # ---------------------------------------------------------------------
    # Phase 10-04: Evidence decision events (append-only + projection)
    # ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS evidence_decision_streams (
      project_id text NOT NULL DEFAULT 'default',
      claim_id text NOT NULL,
      reviewer_uid text NOT NULL,
      version int NOT NULL DEFAULT 0,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      created_by_user_id text NOT NULL DEFAULT 'local',
      PRIMARY KEY(project_id, claim_id, reviewer_uid)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_decision_events (
      event_id bigserial PRIMARY KEY,
      event_uid text NOT NULL,
      project_id text NOT NULL DEFAULT 'default',
      claim_id text NOT NULL,
      reviewer_uid text NOT NULL,
      created_by_user_id text NOT NULL DEFAULT 'local',
      idempotency_key text NOT NULL,
      action text NOT NULL,
      target_attachment_id text NULL,
      target_span_id text NULL,
      target_key text NULL,
      expected_version int NOT NULL,
      resulting_version int NOT NULL,
      request_fingerprint text NOT NULL,
      set_value boolean NULL,
      payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    """,
    """
    ALTER TABLE evidence_decision_events
      ADD COLUMN IF NOT EXISTS set_value boolean NULL;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS evidence_decision_events_stream_idempotency_uniq
      ON evidence_decision_events(project_id, claim_id, reviewer_uid, idempotency_key);
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS evidence_decision_events_project_event_uid_uniq
      ON evidence_decision_events(project_id, event_uid);
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_decision_events_stream_timeline_idx
      ON evidence_decision_events(project_id, claim_id, reviewer_uid, event_id DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_decision_events_stream_target_idx
      ON evidence_decision_events(project_id, claim_id, reviewer_uid, target_key);
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence_decision_targets (
      project_id text NOT NULL DEFAULT 'default',
      claim_id text NOT NULL,
      reviewer_uid text NOT NULL,
      target_key text NOT NULL,
      attachment_id text NOT NULL,
      span_id text NOT NULL,
      pinned boolean NOT NULL DEFAULT false,
      triage text NOT NULL DEFAULT 'none',
      updated_at timestamptz NOT NULL DEFAULT now(),
      last_event_id bigint NULL,
      last_event_uid text NULL,
      PRIMARY KEY(project_id, claim_id, reviewer_uid, target_key)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS evidence_decision_targets_stream_pinned_idx
      ON evidence_decision_targets(project_id, claim_id, reviewer_uid)
      WHERE pinned=true;
    """,
    # Opinion events table (for follow/ignore/complete status)
    """
    CREATE TABLE IF NOT EXISTS opinion_events (
      project_id text NOT NULL DEFAULT 'default',
      event_id bigserial NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      actor_uid text NOT NULL,
      owner_uid text NOT NULL,
      kind text NOT NULL,
      target_key text NOT NULL,
      doc_id text NULL,
      citation_index int NULL,
      target_id text NULL,
      span_id text NULL,
      visibility text NOT NULL DEFAULT 'private',
      group_id text NULL,
      mode int NOT NULL DEFAULT 0600,
      payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
      idempotency_key text NULL,
      PRIMARY KEY(project_id, event_id)
    );
    """,
    """
    ALTER TABLE opinion_events
      ADD COLUMN IF NOT EXISTS visibility text NOT NULL DEFAULT 'private';
    """,
    """
    ALTER TABLE opinion_events
      ADD COLUMN IF NOT EXISTS group_id text NULL;
    """,
    """
    ALTER TABLE opinion_events
      ADD COLUMN IF NOT EXISTS mode int NOT NULL DEFAULT 0600;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS opinion_events_project_event_id_uniq
      ON opinion_events(project_id, event_id);
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS opinion_events_idempotency_uniq
      ON opinion_events(project_id, owner_uid, idempotency_key)
      WHERE idempotency_key IS NOT NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS opinion_events_owner_kind_idx
      ON opinion_events(project_id, owner_uid, kind);
    """,
    """
    CREATE INDEX IF NOT EXISTS opinion_events_doc_citation_idx
      ON opinion_events(project_id, doc_id, citation_index);
    """,
    """
    CREATE INDEX IF NOT EXISTS opinion_events_target_key_idx
      ON opinion_events(project_id, target_key);
    """,
    """
    CREATE INDEX IF NOT EXISTS opinion_events_owner_span_idx
      ON opinion_events(project_id, owner_uid, span_id)
      WHERE span_id IS NOT NULL;
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
