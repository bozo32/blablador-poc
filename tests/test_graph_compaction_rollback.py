from __future__ import annotations

import uuid

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
    project_id = f"proj-graph-{uuid.uuid4().hex[:8]}"
    _seed_duplicate_doc_key(project_id=project_id)
    service = GraphCompactionService()

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(1)
                FROM graph_nodes
                WHERE project_id=%s
                  AND kind='document'
                  AND properties_json->>'doc_key'='doi:10.3/demo'
                """,
                (project_id,),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 2

    apply_result = service.run_apply(project_id=project_id)
    apply_run_id = str(apply_result["run_id"])
    deleted_nodes = (
        (apply_result.get("report") or {})
        .get("mutation_journal", {})
        .get("nodes_deleted", [])
    )
    assert isinstance(deleted_nodes, list) and len(deleted_nodes) == 1
    deleted_node_id = str((deleted_nodes[0] or {}).get("node_id") or "").strip()
    assert deleted_node_id

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(1)
                FROM graph_nodes
                WHERE project_id=%s
                  AND kind='document'
                  AND properties_json->>'doc_key'='doi:10.3/demo'
                """,
                (project_id,),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 1

            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id=%s",
                (project_id, deleted_node_id),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 0

    rollback_result = service.rollback(run_id=apply_run_id, project_id=project_id)
    assert rollback_result["mode"] == "rollback"
    assert rollback_result["status"] == "completed"
    assert rollback_result["report"]["source_run_id"] == apply_run_id

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id=%s",
                (project_id, deleted_node_id),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 1

            cur.execute(
                "SELECT node_id FROM graph_aliases WHERE project_id=%s AND alias='ingest:dup'",
                (project_id,),
            )
            row = cur.fetchone()
            assert row and str(row[0]) == "doc:dup"
