from __future__ import annotations

from backend.db.pg import connect
from backend.graph_compaction import GraphCompactionService


def _seed_duplicate_doc_key(project_id: str = "default") -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_nodes(
                  project_id, node_id, kind, num, label, properties_json
                )
                VALUES
                  (%s, 'doc:keep', 'document', 1, 'keep', '{"doc_key":"doi:10.1/demo"}'::jsonb),
                  (%s, 'doc:dup', 'document', 2, 'dup', '{"doc_key":"doi:10.1/demo"}'::jsonb),
                  (%s, 'doc:other', 'document', 3, 'other', '{"doc_key":"doi:10.1/other"}'::jsonb)
                """,
                (project_id, project_id, project_id),
            )
            cur.execute(
                """
                INSERT INTO graph_aliases(project_id, alias, node_id, kind)
                VALUES
                  (%s, 'doi:10.1/demo', 'doc:keep', 'doi'),
                  (%s, 'ingest:dup', 'doc:dup', 'ingest')
                """,
                (project_id, project_id),
            )
            cur.execute(
                """
                INSERT INTO graph_edges(project_id, source_id, target_id, kind, ref_id, enabled, properties_json)
                VALUES
                  (%s, 'doc:dup', 'doc:other', 'CITES', 'r1', true, '{"source":"auto"}'::jsonb),
                  (%s, 'doc:keep', 'doc:other', 'CITES', 'r1', false, '{"source":"manual"}'::jsonb)
                """,
                (project_id, project_id),
            )


def _seed_duplicate_ingest_id(project_id: str = "default") -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_nodes(
                  project_id, node_id, kind, num, label, properties_json
                )
                VALUES
                  (
                    %s,
                    'doc:target',
                    'document',
                    90,
                    'target',
                    '{"doc_key":"doi:10.1/target"}'::jsonb
                  ),
                  (
                    %s,
                    'doc:ing-link',
                    'document',
                    30,
                    'linked',
                    '{"ingest_ids":["ing:split"],"title":"Thin"}'::jsonb
                  ),
                  (
                    %s,
                    'doc:ing-rich',
                    'document',
                    1,
                    'rich',
                    '{"ingest_ids":["ing:split"],"title":"Richer","authors":["Ada"],"year":"2024"}'::jsonb
                  ),
                  (
                    %s,
                    'doc:ing-plain',
                    'document',
                    2,
                    'plain',
                    '{"ingest_ids":["ing:split"]}'::jsonb
                  )
                """,
                (project_id, project_id, project_id, project_id),
            )
            cur.execute(
                """
                INSERT INTO graph_aliases(project_id, alias, node_id, kind)
                VALUES
                  (%s, 'ingest:split', 'doc:ing-rich', 'ingest')
                """,
                (project_id,),
            )
            cur.execute(
                """
                INSERT INTO graph_edges(project_id, source_id, target_id, kind, ref_id, enabled, properties_json)
                VALUES
                  (%s, 'doc:ing-link', 'doc:target', 'CITES', 'r2', true, '{"source":"auto"}'::jsonb)
                """,
                (project_id,),
            )


def test_graph_compaction_preview_and_apply() -> None:
    _seed_duplicate_doc_key()
    service = GraphCompactionService()

    preview = service.preview()
    assert preview["duplicate_doc_keys"] == 1
    assert preview["duplicate_nodes"] == 1

    result = service.run_apply()
    assert result["mode"] == "apply"
    assert result["status"] == "completed"
    assert result["report"]["nodes_deleted"] == 1
    assert result["report"]["aliases_repointed"] >= 1
    assert result["report"]["edges_merged"] == 1

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:dup'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 0

            cur.execute(
                "SELECT node_id FROM graph_aliases WHERE project_id=%s AND alias='ingest:dup'",
                ("default",),
            )
            row = cur.fetchone()
            assert row and str(row[0]) == "doc:keep"

            cur.execute(
                """
                SELECT COUNT(1)
                FROM graph_edges
                WHERE project_id=%s
                  AND source_id='doc:keep'
                  AND target_id='doc:other'
                  AND kind='CITES'
                  AND ref_id='r1'
                """,
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 1


def test_graph_compaction_ingest_id_canonical_prefers_links_then_metadata() -> None:
    _seed_duplicate_ingest_id()
    service = GraphCompactionService()

    preview = service.preview()
    assert preview["duplicate_ingest_ids"] == 1
    ingest_groups = preview["ingest_id_dedup"]["groups"]
    assert len(ingest_groups) == 1
    assert ingest_groups[0]["ingest_id"] == "ing:split"
    assert ingest_groups[0]["canonical_node_id"] == "doc:ing-link"

    applied = service.run_apply()
    assert applied["status"] == "completed"
    assert applied["report"]["nodes_deleted"] == 2
    assert applied["report"]["ingest_id_dedup"]["duplicate_ingest_ids"] == 1

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:ing-link'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 1

            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:ing-rich'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 0

            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:ing-plain'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 0

            cur.execute(
                "SELECT node_id FROM graph_aliases WHERE project_id=%s AND alias='ingest:split'",
                ("default",),
            )
            row = cur.fetchone()
            assert row and str(row[0]) == "doc:ing-link"
