from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main as backend_main
from backend.db.pg import connect


def _seed_duplicate_doc_key(project_id: str = "default") -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_nodes(project_id, node_id, kind, num, properties_json)
                VALUES
                  (%s, 'doc:keep', 'document', 1, '{"doc_key":"doi:10.2/demo","ingest_ids":["ing:demo"]}'::jsonb),
                  (%s, 'doc:dup', 'document', 2, '{"doc_key":"doi:10.2/demo","ingest_ids":["ing:demo"]}'::jsonb)
                """,
                (project_id, project_id),
            )


def test_graph_compaction_endpoints_dry_run_then_apply() -> None:
    _seed_duplicate_doc_key()
    client = TestClient(backend_main.app)

    dry = client.post("/maintenance/graph/compact/dry-run", json={})
    assert dry.status_code == 200
    dry_payload = dry.json()
    assert dry_payload["mode"] == "dry-run"
    assert dry_payload["report"]["duplicate_nodes"] == 1
    assert dry_payload["report"]["ingest_id_dedup"]["duplicate_ingest_ids"] == 1

    applied = client.post("/maintenance/graph/compact/apply", json={})
    assert applied.status_code == 200
    payload = applied.json()
    assert payload["mode"] == "apply"
    assert payload["status"] == "completed"
    assert payload["report"]["nodes_deleted"] == 1
    assert payload["report"]["ingest_id_dedup"]["duplicate_ingest_ids"] == 1
    assert isinstance(payload["report"].get("mutation_journal"), dict)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1) FROM graph_nodes WHERE project_id=%s AND node_id='doc:dup'",
                ("default",),
            )
            assert int((cur.fetchone() or [0])[0] or 0) == 0
