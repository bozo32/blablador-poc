from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from backend.db.pg import connect
from backend.settings import AppSettings, settings as app_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass
class _DupGroup:
    doc_key: str
    canonical_node_id: str
    duplicate_node_ids: list[str]


class GraphCompactionService:
    """Durable graph dedup/compaction with journaled rollback support."""

    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        self.settings = settings

    def _project_id(self, project_id: Optional[str] = None) -> str:
        override = str(project_id or "").strip()
        if override:
            return override
        return str(getattr(self.settings, "DEFAULT_PROJECT_ID", "default") or "default")

    def _duplicate_groups(self, *, project_id: str) -> list[_DupGroup]:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COALESCE(properties_json->>'doc_key', '') AS doc_key,
                      ARRAY_AGG(node_id ORDER BY num ASC NULLS LAST, node_id ASC) AS node_ids
                    FROM graph_nodes
                    WHERE project_id=%s
                      AND kind='document'
                      AND COALESCE(properties_json->>'doc_key', '') <> ''
                    GROUP BY COALESCE(properties_json->>'doc_key', '')
                    HAVING COUNT(*) > 1
                    ORDER BY doc_key ASC
                    """,
                    (project_id,),
                )
                rows = cur.fetchall() or []

        groups: list[_DupGroup] = []
        for doc_key, node_ids in rows:
            ids = [str(x) for x in (node_ids or []) if str(x).strip()]
            if len(ids) < 2:
                continue
            groups.append(
                _DupGroup(
                    doc_key=str(doc_key),
                    canonical_node_id=str(ids[0]),
                    duplicate_node_ids=[str(x) for x in ids[1:]],
                )
            )
        return groups

    def preview(self, *, project_id: Optional[str] = None) -> dict[str, Any]:
        pid = self._project_id(project_id)
        groups = self._duplicate_groups(project_id=pid)
        dup_nodes = sum(len(g.duplicate_node_ids) for g in groups)
        return {
            "project_id": pid,
            "strategy": "doc-key-dedup-v1",
            "duplicate_doc_keys": len(groups),
            "duplicate_nodes": int(dup_nodes),
            "groups": [
                {
                    "doc_key": g.doc_key,
                    "canonical_node_id": g.canonical_node_id,
                    "duplicate_node_ids": list(g.duplicate_node_ids),
                }
                for g in groups
            ],
        }

    def run_dry_run(self, *, project_id: Optional[str] = None) -> dict[str, Any]:
        pid = self._project_id(project_id)
        preview = self.preview(project_id=pid)
        run_id = f"gcompact:{uuid.uuid4()}"
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_compaction_runs(
                      run_id, project_id, mode, status, strategy, summary_json, created_at
                    )
                    VALUES (%s, %s, 'dry-run', 'completed', %s, %s::jsonb, now())
                    """,
                    (
                        run_id,
                        pid,
                        str(preview.get("strategy") or "doc-key-dedup-v1"),
                        _json_dumps(preview),
                    ),
                )
        return {
            "run_id": run_id,
            "mode": "dry-run",
            "status": "completed",
            "created_at": _utc_now(),
            "report": preview,
        }

    def _snapshot_project_graph(self, *, project_id: str) -> dict[str, Any]:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      node_id,
                      kind,
                      num,
                      label,
                      properties_json,
                      created_at,
                      updated_at
                    FROM graph_nodes
                    WHERE project_id=%s
                    ORDER BY node_id ASC
                    """,
                    (project_id,),
                )
                node_rows = cur.fetchall() or []

                cur.execute(
                    """
                    SELECT
                      alias,
                      node_id,
                      kind,
                      created_at
                    FROM graph_aliases
                    WHERE project_id=%s
                    ORDER BY alias ASC
                    """,
                    (project_id,),
                )
                alias_rows = cur.fetchall() or []

                cur.execute(
                    """
                    SELECT
                      edge_id,
                      source_id,
                      target_id,
                      kind,
                      ref_id,
                      enabled,
                      properties_json,
                      created_at,
                      updated_at
                    FROM graph_edges
                    WHERE project_id=%s
                    ORDER BY edge_id ASC
                    """,
                    (project_id,),
                )
                edge_rows = cur.fetchall() or []

                cur.execute(
                    """
                    SELECT
                      v.edge_id,
                      v.reviewer_uid,
                      v.verdict,
                      v.confidence,
                      v.comment,
                      v.updated_at
                    FROM graph_edge_votes v
                    JOIN graph_edges e ON e.edge_id=v.edge_id
                    WHERE e.project_id=%s
                    ORDER BY v.edge_id ASC, v.reviewer_uid ASC
                    """,
                    (project_id,),
                )
                vote_rows = cur.fetchall() or []

        return {
            "project_id": project_id,
            "graph_nodes": [
                {
                    "node_id": str(node_id),
                    "kind": str(kind),
                    "num": int(num) if num is not None else None,
                    "label": label,
                    "properties_json": props if isinstance(props, dict) else {},
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                }
                for node_id, kind, num, label, props, created_at, updated_at in node_rows
            ],
            "graph_aliases": [
                {
                    "alias": str(alias),
                    "node_id": str(node_id),
                    "kind": str(kind),
                    "created_at": created_at.isoformat() if created_at else None,
                }
                for alias, node_id, kind, created_at in alias_rows
            ],
            "graph_edges": [
                {
                    "edge_id": int(edge_id),
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "kind": str(kind),
                    "ref_id": str(ref_id or ""),
                    "enabled": bool(enabled),
                    "properties_json": props if isinstance(props, dict) else {},
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                }
                for (
                    edge_id,
                    source_id,
                    target_id,
                    kind,
                    ref_id,
                    enabled,
                    props,
                    created_at,
                    updated_at,
                ) in edge_rows
            ],
            "graph_edge_votes": [
                {
                    "edge_id": int(edge_id),
                    "reviewer_uid": str(reviewer_uid),
                    "verdict": str(verdict),
                    "confidence": confidence,
                    "comment": comment,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                }
                for edge_id, reviewer_uid, verdict, confidence, comment, updated_at in vote_rows
            ],
        }

    def run_apply(self, *, project_id: Optional[str] = None) -> dict[str, Any]:
        pid = self._project_id(project_id)
        run_id = f"gcompact:{uuid.uuid4()}"
        preview = self.preview(project_id=pid)
        groups = self._duplicate_groups(project_id=pid)
        mapping: dict[str, str] = {}
        for g in groups:
            for dup in g.duplicate_node_ids:
                mapping[str(dup)] = str(g.canonical_node_id)

        aliases_repointed = 0
        edges_rewritten = 0
        edges_merged = 0
        nodes_deleted = 0
        snapshot = self._snapshot_project_graph(project_id=pid)

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_compaction_runs(
                      run_id,
                      project_id,
                      mode,
                      status,
                      strategy,
                      summary_json,
                      snapshot_json,
                      created_at
                    )
                    VALUES (%s, %s, 'apply', 'running', %s, %s::jsonb, %s::jsonb, now())
                    """,
                    (
                        run_id,
                        pid,
                        str(preview.get("strategy") or "doc-key-dedup-v1"),
                        _json_dumps(preview),
                        _json_dumps(snapshot),
                    ),
                )

                for dup, canonical in mapping.items():
                    cur.execute(
                        """
                        UPDATE graph_aliases
                        SET node_id=%s
                        WHERE project_id=%s AND node_id=%s
                        """,
                        (canonical, pid, dup),
                    )
                    aliases_repointed += int(cur.rowcount or 0)

                if mapping:
                    cur.execute(
                        """
                        SELECT
                          edge_id,
                          source_id,
                          target_id,
                          kind,
                          ref_id,
                          enabled,
                          properties_json
                        FROM graph_edges
                        WHERE project_id=%s
                          AND (source_id = ANY(%s::text[]) OR target_id = ANY(%s::text[]))
                        ORDER BY edge_id ASC
                        """,
                        (pid, list(mapping.keys()), list(mapping.keys())),
                    )
                    edge_rows = cur.fetchall() or []
                else:
                    edge_rows = []

                for edge_id, source_id, target_id, kind, ref_id, enabled, props in edge_rows:
                    old_id = int(edge_id)
                    new_source = mapping.get(str(source_id), str(source_id))
                    new_target = mapping.get(str(target_id), str(target_id))
                    if new_source == str(source_id) and new_target == str(target_id):
                        continue

                    cur.execute(
                        """
                        SELECT edge_id, enabled
                        FROM graph_edges
                        WHERE project_id=%s
                          AND source_id=%s
                          AND target_id=%s
                          AND kind=%s
                          AND ref_id=%s
                          AND edge_id <> %s
                        LIMIT 1
                        """,
                        (pid, new_source, new_target, str(kind), str(ref_id or ""), old_id),
                    )
                    existing = cur.fetchone()
                    if existing:
                        existing_id, existing_enabled = existing
                        kept_edge_id = int(existing_id)
                        final_enabled = bool(existing_enabled) or bool(enabled)

                        cur.execute(
                            """
                            UPDATE graph_edges
                            SET enabled=%s,
                                properties_json=COALESCE(graph_edges.properties_json, '{}'::jsonb)
                                    || %s::jsonb,
                                updated_at=now()
                            WHERE edge_id=%s
                            """,
                            (
                                final_enabled,
                                _json_dumps(props if isinstance(props, dict) else {}),
                                kept_edge_id,
                            ),
                        )

                        cur.execute(
                            """
                            INSERT INTO graph_edge_votes(
                              edge_id,
                              reviewer_uid,
                              verdict,
                              confidence,
                              comment,
                              updated_at
                            )
                            SELECT %s, reviewer_uid, verdict, confidence, comment, updated_at
                            FROM graph_edge_votes
                            WHERE edge_id=%s
                            ON CONFLICT(edge_id, reviewer_uid) DO UPDATE
                              SET verdict=excluded.verdict,
                                  confidence=excluded.confidence,
                                  comment=excluded.comment,
                                  updated_at=excluded.updated_at
                            """,
                            (kept_edge_id, old_id),
                        )

                        cur.execute("DELETE FROM graph_edges WHERE edge_id=%s", (old_id,))
                        edges_merged += 1
                        continue

                    cur.execute(
                        """
                        UPDATE graph_edges
                        SET source_id=%s,
                            target_id=%s,
                            updated_at=now()
                        WHERE edge_id=%s
                        """,
                        (new_source, new_target, old_id),
                    )
                    edges_rewritten += int(cur.rowcount or 0)

                for dup in mapping.keys():
                    cur.execute(
                        "DELETE FROM graph_nodes WHERE project_id=%s AND node_id=%s",
                        (pid, str(dup)),
                    )
                    nodes_deleted += int(cur.rowcount or 0)

                report = {
                    **preview,
                    "aliases_repointed": int(aliases_repointed),
                    "edges_rewritten": int(edges_rewritten),
                    "edges_merged": int(edges_merged),
                    "nodes_deleted": int(nodes_deleted),
                }
                cur.execute(
                    """
                    UPDATE graph_compaction_runs
                    SET status='completed',
                        summary_json=%s::jsonb,
                        applied_at=now(),
                        updated_at=now()
                    WHERE run_id=%s
                    """,
                    (_json_dumps(report), run_id),
                )
            conn.commit()

        return {
            "run_id": run_id,
            "mode": "apply",
            "status": "completed",
            "created_at": _utc_now(),
            "report": report,
        }

    def rollback(self, *, run_id: str, project_id: Optional[str] = None) -> dict[str, Any]:
        pid = self._project_id(project_id)
        source_run_id = str(run_id or "").strip()
        if not source_run_id:
            raise ValueError("run_id is required")

        rollback_run_id = f"gcompact:{uuid.uuid4()}"
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT project_id, mode, snapshot_json
                    FROM graph_compaction_runs
                    WHERE run_id=%s
                    LIMIT 1
                    """,
                    (source_run_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError("Compaction run not found")

                run_project_id, source_mode, snapshot = row
                if str(run_project_id) != pid:
                    raise ValueError("run_id project does not match current project")
                if str(source_mode) != "apply":
                    raise ValueError("Only apply runs can be rolled back")
                if not isinstance(snapshot, dict):
                    raise ValueError("Compaction run has no rollback snapshot")

                cur.execute(
                    """
                    INSERT INTO graph_compaction_runs(
                      run_id,
                      project_id,
                      mode,
                      status,
                      strategy,
                      source_run_id,
                      summary_json,
                      created_at
                    )
                    VALUES (
                      %s,
                      %s,
                      'rollback',
                      'running',
                      'doc-key-dedup-v1',
                      %s,
                      '{}'::jsonb,
                      now()
                    )
                    """,
                    (rollback_run_id, pid, source_run_id),
                )

                cur.execute(
                    "DELETE FROM graph_edges WHERE project_id=%s",
                    (pid,),
                )
                cur.execute(
                    "DELETE FROM graph_aliases WHERE project_id=%s",
                    (pid,),
                )
                cur.execute(
                    "DELETE FROM graph_nodes WHERE project_id=%s",
                    (pid,),
                )

                for node in snapshot.get("graph_nodes") or []:
                    cur.execute(
                        """
                        INSERT INTO graph_nodes(
                          project_id,
                          node_id,
                          kind,
                          num,
                          label,
                          properties_json,
                          created_at,
                          updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            pid,
                            str(node.get("node_id")),
                            str(node.get("kind")),
                            node.get("num"),
                            node.get("label"),
                            _json_dumps(node.get("properties_json") or {}),
                            node.get("created_at"),
                            node.get("updated_at"),
                        ),
                    )

                for alias in snapshot.get("graph_aliases") or []:
                    cur.execute(
                        """
                        INSERT INTO graph_aliases(
                          project_id,
                          alias,
                          node_id,
                          kind,
                          created_at
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            pid,
                            str(alias.get("alias")),
                            str(alias.get("node_id")),
                            str(alias.get("kind")),
                            alias.get("created_at"),
                        ),
                    )

                for edge in snapshot.get("graph_edges") or []:
                    cur.execute(
                        """
                        INSERT INTO graph_edges(
                          edge_id,
                          project_id,
                          source_id,
                          target_id,
                          kind,
                          ref_id,
                          enabled,
                          properties_json,
                          created_at,
                          updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            int(edge.get("edge_id")),
                            pid,
                            str(edge.get("source_id")),
                            str(edge.get("target_id")),
                            str(edge.get("kind")),
                            str(edge.get("ref_id") or ""),
                            bool(edge.get("enabled")),
                            _json_dumps(edge.get("properties_json") or {}),
                            edge.get("created_at"),
                            edge.get("updated_at"),
                        ),
                    )

                for vote in snapshot.get("graph_edge_votes") or []:
                    cur.execute(
                        """
                        INSERT INTO graph_edge_votes(
                          edge_id,
                          reviewer_uid,
                          verdict,
                          confidence,
                          comment,
                          updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(edge_id, reviewer_uid) DO UPDATE
                          SET verdict=excluded.verdict,
                              confidence=excluded.confidence,
                              comment=excluded.comment,
                              updated_at=excluded.updated_at
                        """,
                        (
                            int(vote.get("edge_id")),
                            str(vote.get("reviewer_uid")),
                            str(vote.get("verdict")),
                            vote.get("confidence"),
                            vote.get("comment"),
                            vote.get("updated_at"),
                        ),
                    )

                cur.execute(
                    """
                    SELECT setval(
                      pg_get_serial_sequence('graph_edges', 'edge_id'),
                      COALESCE((SELECT MAX(edge_id) FROM graph_edges), 1),
                      true
                    )
                    """
                )

                restored = {
                    "project_id": pid,
                    "restored_nodes": len(snapshot.get("graph_nodes") or []),
                    "restored_aliases": len(snapshot.get("graph_aliases") or []),
                    "restored_edges": len(snapshot.get("graph_edges") or []),
                    "restored_votes": len(snapshot.get("graph_edge_votes") or []),
                    "source_run_id": source_run_id,
                }
                cur.execute(
                    """
                    UPDATE graph_compaction_runs
                    SET status='completed',
                        rolled_back_at=now(),
                        updated_at=now(),
                        summary_json=%s::jsonb
                    WHERE run_id=%s
                    """,
                    (_json_dumps(restored), rollback_run_id),
                )
                cur.execute(
                    """
                    UPDATE graph_compaction_runs
                    SET status='rolled_back',
                        rolled_back_at=now(),
                        updated_at=now()
                    WHERE run_id=%s
                    """,
                    (source_run_id,),
                )
            conn.commit()

        return {
            "run_id": rollback_run_id,
            "mode": "rollback",
            "status": "completed",
            "created_at": _utc_now(),
            "report": restored,
        }
