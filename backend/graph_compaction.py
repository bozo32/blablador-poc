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
    group_key: str
    canonical_node_id: str
    duplicate_node_ids: list[str]


@dataclass
class _DocNode:
    node_id: str
    num: Optional[int]
    properties: dict[str, Any]
    has_links: bool
    metadata_score: int


@dataclass
class _CompactionPlan:
    strategy: str
    doc_key_groups: list[_DupGroup]
    ingest_id_groups: list[_DupGroup]
    mapping: dict[str, str]
    report: dict[str, Any]


class GraphCompactionService:
    """Durable graph dedup/compaction with journaled rollback support."""

    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        self.settings = settings

    def _project_id(self, project_id: Optional[str] = None) -> str:
        override = str(project_id or "").strip()
        if override:
            return override
        return str(getattr(self.settings, "DEFAULT_PROJECT_ID", "default") or "default")

    def _node_sort_key(self, node: _DocNode) -> tuple[int, int, int, str]:
        has_links_rank = 0 if bool(node.has_links) else 1
        metadata_rank = -int(node.metadata_score)
        num_rank = int(node.num) if node.num is not None else (10**9)
        return (has_links_rank, metadata_rank, num_rank, str(node.node_id))

    def _metadata_score(self, properties: dict[str, Any]) -> int:
        if not isinstance(properties, dict):
            return 0
        skip_keys = {
            "ingest_ids",
            "ingest_id",
            "resolved",
            "extracted",
            "workflow_assigned",
        }
        score = 0
        for key, value in properties.items():
            if str(key) in skip_keys:
                continue
            if isinstance(value, str) and value.strip():
                score += 1
            elif isinstance(value, bool) and bool(value):
                score += 1
            elif isinstance(value, (int, float)):
                score += 1
            elif isinstance(value, list) and any(str(v).strip() for v in value):
                score += 1
            elif isinstance(value, dict) and value:
                score += 1
        return int(score)

    def _document_nodes(self, *, project_id: str) -> dict[str, _DocNode]:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      n.node_id,
                      n.num,
                      n.properties_json,
                      EXISTS(
                        SELECT 1
                        FROM graph_edges e
                        WHERE e.project_id=n.project_id
                          AND (e.source_id=n.node_id OR e.target_id=n.node_id)
                      ) AS has_links
                    FROM graph_nodes n
                    WHERE n.project_id=%s
                      AND n.kind='document'
                    ORDER BY n.node_id ASC
                    """,
                    (project_id,),
                )
                rows = cur.fetchall() or []

        out: dict[str, _DocNode] = {}
        for node_id, num, properties_json, has_links in rows:
            nid = str(node_id or "").strip()
            if not nid:
                continue
            props = properties_json if isinstance(properties_json, dict) else {}
            out[nid] = _DocNode(
                node_id=nid,
                num=int(num) if num is not None else None,
                properties=props,
                has_links=bool(has_links),
                metadata_score=self._metadata_score(props),
            )
        return out

    def _duplicate_groups_doc_key(
        self, *, nodes_by_id: dict[str, _DocNode]
    ) -> list[_DupGroup]:
        grouped: dict[str, list[_DocNode]] = {}
        for node in nodes_by_id.values():
            doc_key = str((node.properties or {}).get("doc_key") or "").strip()
            if not doc_key:
                continue
            grouped.setdefault(doc_key, []).append(node)

        groups: list[_DupGroup] = []
        for doc_key in sorted(grouped.keys()):
            members = sorted(grouped.get(doc_key) or [], key=self._node_sort_key)
            if len(members) < 2:
                continue
            groups.append(
                _DupGroup(
                    group_key=str(doc_key),
                    canonical_node_id=str(members[0].node_id),
                    duplicate_node_ids=[str(m.node_id) for m in members[1:]],
                )
            )
        return groups

    def _duplicate_groups_ingest_id(
        self, *, nodes_by_id: dict[str, _DocNode]
    ) -> list[_DupGroup]:
        grouped: dict[str, list[_DocNode]] = {}
        for node in nodes_by_id.values():
            props = node.properties or {}
            ingest_ids_raw = props.get("ingest_ids")
            ingest_ids: list[str] = []
            if isinstance(ingest_ids_raw, list):
                ingest_ids.extend(
                    [str(x).strip() for x in ingest_ids_raw if str(x or "").strip()]
                )
            single_ingest = str(props.get("ingest_id") or "").strip()
            if single_ingest:
                ingest_ids.append(single_ingest)
            for ingest_id in sorted(set(ingest_ids)):
                grouped.setdefault(ingest_id, []).append(node)

        groups: list[_DupGroup] = []
        for ingest_id in sorted(grouped.keys()):
            members = sorted(grouped.get(ingest_id) or [], key=self._node_sort_key)
            if len(members) < 2:
                continue
            groups.append(
                _DupGroup(
                    group_key=str(ingest_id),
                    canonical_node_id=str(members[0].node_id),
                    duplicate_node_ids=[str(m.node_id) for m in members[1:]],
                )
            )
        return groups

    def _build_plan(self, *, project_id: str) -> _CompactionPlan:
        strategy = "doc-key+ingest-id-dedup-v2"
        nodes_by_id = self._document_nodes(project_id=project_id)
        doc_groups = self._duplicate_groups_doc_key(nodes_by_id=nodes_by_id)
        ingest_groups = self._duplicate_groups_ingest_id(nodes_by_id=nodes_by_id)

        mapping: dict[str, str] = {}

        def _resolve(node_id: str) -> str:
            current = str(node_id)
            seen: set[str] = set()
            while current in mapping and current not in seen:
                seen.add(current)
                current = str(mapping[current])
            return current

        duplicate_candidates: set[str] = set()
        for group in ingest_groups + doc_groups:
            canonical = _resolve(str(group.canonical_node_id))
            for dup in group.duplicate_node_ids:
                duplicate_candidates.add(str(dup))
                dup_resolved = _resolve(str(dup))
                if dup_resolved == canonical:
                    continue
                mapping[dup_resolved] = canonical

        final_mapping: dict[str, str] = {}
        for dup in sorted(duplicate_candidates):
            resolved = _resolve(str(dup))
            if resolved != str(dup):
                final_mapping[str(dup)] = str(resolved)

        preview_groups = [
            {
                "doc_key": g.group_key,
                "canonical_node_id": g.canonical_node_id,
                "duplicate_node_ids": list(g.duplicate_node_ids),
            }
            for g in doc_groups
        ]
        ingest_report_groups = [
            {
                "ingest_id": g.group_key,
                "canonical_node_id": g.canonical_node_id,
                "duplicate_node_ids": list(g.duplicate_node_ids),
            }
            for g in ingest_groups
        ]
        report = {
            "project_id": project_id,
            "strategy": strategy,
            "duplicate_doc_keys": len(doc_groups),
            "duplicate_ingest_ids": len(ingest_groups),
            "duplicate_nodes": int(len(final_mapping)),
            "groups": preview_groups,
            "ingest_id_dedup": {
                "duplicate_ingest_ids": len(ingest_groups),
                "duplicate_nodes": sum(len(g.duplicate_node_ids) for g in ingest_groups),
                "groups": ingest_report_groups,
            },
        }

        return _CompactionPlan(
            strategy=strategy,
            doc_key_groups=doc_groups,
            ingest_id_groups=ingest_groups,
            mapping=final_mapping,
            report=report,
        )

    def preview(self, *, project_id: Optional[str] = None) -> dict[str, Any]:
        pid = self._project_id(project_id)
        plan = self._build_plan(project_id=pid)
        return dict(plan.report)

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
        plan = self._build_plan(project_id=pid)
        preview = dict(plan.report)
        mapping = dict(plan.mapping)

        aliases_repointed = 0
        edges_rewritten = 0
        edges_merged = 0
        nodes_deleted = 0
        mutation_journal: dict[str, list[dict[str, Any]]] = {
            "aliases_repointed": [],
            "edges_rewritten": [],
            "edges_merged": [],
            "nodes_deleted": [],
        }
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
                        str(preview.get("strategy") or plan.strategy),
                        _json_dumps(preview),
                        _json_dumps(snapshot),
                    ),
                )

                for dup, canonical in mapping.items():
                    cur.execute(
                        """
                        SELECT alias
                        FROM graph_aliases
                        WHERE project_id=%s AND node_id=%s
                        ORDER BY alias ASC
                        """,
                        (pid, dup),
                    )
                    alias_rows = cur.fetchall() or []
                    cur.execute(
                        """
                        UPDATE graph_aliases
                        SET node_id=%s
                        WHERE project_id=%s AND node_id=%s
                        """,
                        (canonical, pid, dup),
                    )
                    aliases_repointed += int(cur.rowcount or 0)
                    for row in alias_rows:
                        alias = str((row or [""])[0] or "").strip()
                        if not alias:
                            continue
                        mutation_journal["aliases_repointed"].append(
                            {
                                "alias": alias,
                                "from_node_id": str(dup),
                                "to_node_id": str(canonical),
                            }
                        )

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
                        mutation_journal["edges_merged"].append(
                            {
                                "deleted_edge_id": int(old_id),
                                "kept_edge_id": int(kept_edge_id),
                                "source_id": str(new_source),
                                "target_id": str(new_target),
                                "kind": str(kind),
                                "ref_id": str(ref_id or ""),
                            }
                        )
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
                    mutation_journal["edges_rewritten"].append(
                        {
                            "edge_id": int(old_id),
                            "old_source_id": str(source_id),
                            "old_target_id": str(target_id),
                            "new_source_id": str(new_source),
                            "new_target_id": str(new_target),
                            "kind": str(kind),
                            "ref_id": str(ref_id or ""),
                        }
                    )

                for dup in mapping.keys():
                    cur.execute(
                        "DELETE FROM graph_nodes WHERE project_id=%s AND node_id=%s",
                        (pid, str(dup)),
                    )
                    nodes_deleted += int(cur.rowcount or 0)
                    mutation_journal["nodes_deleted"].append(
                        {
                            "node_id": str(dup),
                            "canonical_node_id": str(mapping.get(str(dup)) or ""),
                        }
                    )

                report = {
                    **preview,
                    "aliases_repointed": int(aliases_repointed),
                    "edges_rewritten": int(edges_rewritten),
                    "edges_merged": int(edges_merged),
                    "nodes_deleted": int(nodes_deleted),
                    "mutation_journal": mutation_journal,
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
                      'doc-key+ingest-id-dedup-v2',
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
