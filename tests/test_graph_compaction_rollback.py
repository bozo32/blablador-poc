from __future__ import annotations

from backend.db.pg import connect
from backend.graph_compaction import GraphCompactionService


def _seed_duplicate_doc_key(project_id: str = "default") -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_nodes(project_id, node_id, kind, num, properties_json)
                VALUES
                  (%s, 'doc:keep', 'document', 1, '{"doc_key":"doi:10.3/demo"}'::jsonb),
                  (%s, 'doc:dup', 'document', 2, '{"doc_key":"doi:10.3/demo"}'::jsonb),
                  (%s, 'doc:other', 'document', 3, '{"doc_key":"doi:10.3/other"}'::jsonb)
                """,
                (project_id, project_id, project_id),
            )
            cur.execute(
                """
                INSERT INTO graph_aliases(project_id, alias, node_id, kind)
                VALUES (%s, 'ingest:dup', 'doc:dup', 'ingest')
                """,
                (project_id,),
            )
            cur.execute(
                """
                INSERT INTO graph_edges(project_id, source_id, target_id, kind, ref_id, enabled, properties_json)
                VALUES (%s, 'doc:dup', 'doc:other', 'CITES', 'r1', true, '{}'::jsonb)
                """,
                (project_id,),
            )


def test_graph_compaction_rollback_restores_previous_state() -> None:
    _seed_duplicate_doc_key()
    service = GraphCompactionService()

    apply_result = service.run_apply()
    apply_run_id = str(apply_result["run_id"])

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:dup'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 0

    rollback_result = service.rollback(run_id=apply_run_id)
    assert rollback_result["mode"] == "rollback"
    assert rollback_result["status"] == "completed"
    assert rollback_result["report"]["source_run_id"] == apply_run_id

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:dup'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 1

            cur.execute(
                "SELECT node_id FROM graph_aliases WHERE project_id=%s AND alias='ingest:dup'",
                ("default",),
            )
            row = cur.fetchone()
            assert row and str(row[0]) == "doc:dup"
