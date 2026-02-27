from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from backend.db.pg import connect
from backend.settings import AppSettings, settings as app_settings


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _norm_text(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = (value or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    lowered = lowered.replace("https://doi.org/", "")
    lowered = lowered.replace("http://doi.org/", "")
    lowered = lowered.replace("doi:", "")
    lowered = lowered.strip()
    return lowered or None


def _surname(author: Optional[str]) -> Optional[str]:
    if not author:
        return None
    parts = [p for p in re.split(r"\s+", author.strip()) if p]
    if not parts:
        return None
    return parts[-1]


def bib_fingerprint(
    *, title: Optional[str], authors: Optional[Sequence[str]], year: Optional[str]
) -> str:
    first = None
    if authors:
        first = _surname(authors[0])
    key = "|".join(
        [
            _norm_text(first) or "unknown",
            _norm_text(year) or "unknown",
            _norm_text(title) or "untitled",
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"bib:{digest}"


def _safe_bib_key(
    *, title: Optional[str], authors: Optional[Sequence[str]], year: Optional[str]
) -> Optional[str]:
    """Return a bib:... key only when metadata is specific enough.

    bib_fingerprint() is tolerant of missing fields; using it without guards can
    create high-collision aliases. We require a reasonably informative title and
    at least one additional signal (year or authors).
    """
    t = str(title or "").strip()
    y = str(year or "").strip()
    auth = [str(a or "").strip() for a in (authors or [])] if authors else []
    auth = [a for a in auth if a]

    if len(t) < 8:
        return None
    if not y and not auth:
        return None
    if y and not auth and len(t) < 24:
        return None

    return bib_fingerprint(title=t, authors=auth, year=y or None)


def doc_node_id_from_key(doc_key: str) -> str:
    return f"doc:{doc_key}"


def doc_key_for_ingest(ingest_meta: dict) -> Optional[str]:
    sha = (ingest_meta.get("sha256") or "").strip().lower()
    if sha:
        return f"sha256:{sha}"
    return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


class GraphStore:
    """Postgres-backed graph store (nodes + edges + votes).

    Phase 09.3 removes durable local SQLite graph stores (`data/graph.db`).

    Notes:
    - The constructor keeps the legacy `db_path` argument for API compatibility.
    - All rows are scoped by `project_id`.
    """

    def __init__(
        self, db_path: Optional[Path] = None, *, settings: AppSettings = app_settings
    ) -> None:
        """Create a Postgres-backed graph store."""
        self._db_path = db_path
        self.settings = settings

    def _project_id(self, project_id: Optional[str] = None) -> str:
        override = str(project_id or "").strip()
        if override:
            return override
        return str(getattr(self.settings, "DEFAULT_PROJECT_ID", "default") or "default")

    def wipe(self) -> None:
        """Delete all graph rows for the active project (keeps schema)."""
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                # Votes cascade via graph_edges.
                cur.execute("DELETE FROM graph_edges WHERE project_id=%s", (pid,))
                cur.execute("DELETE FROM graph_aliases WHERE project_id=%s", (pid,))
                cur.execute("DELETE FROM graph_nodes WHERE project_id=%s", (pid,))

    def _next_num(self) -> int:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(num), 0) FROM graph_nodes WHERE project_id=%s",
                    (pid,),
                )
                row = cur.fetchone()
                return int((row[0] or 0) + 1)

    def _get_node(self, node_id: str) -> Optional[dict]:
        pid = self._project_id()
        nid = str(node_id or "").strip()
        if not nid:
            return None
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
                    WHERE project_id=%s AND node_id=%s
                    """,
                    (pid, nid),
                )
                row = cur.fetchone()
                if not row:
                    return None

        node_id2, kind, num, label, props_raw, created_at, updated_at = row
        props = props_raw if isinstance(props_raw, dict) else {}
        if props_raw is not None and not isinstance(props_raw, dict):
            try:
                props = json.loads(props_raw)
            except Exception:
                props = {}
        return {
            "node_id": str(node_id2),
            "kind": str(kind),
            "num": int(num) if num is not None else None,
            "label": label,
            "properties": props,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    def _upsert_node(
        self,
        *,
        node_id: str,
        kind: str,
        label: Optional[str] = None,
        merge_properties: Optional[dict] = None,
        project_id: Optional[str] = None,
    ) -> dict:
        pid = self._project_id(project_id)
        existing = self._get_node(node_id)
        now = _now()

        if existing:
            props = dict(existing.get("properties") or {})
            for k, v in (merge_properties or {}).items():
                if v is None:
                    continue
                if k not in props or props.get(k) in (None, "", []):
                    props[k] = v
            new_label = label if label is not None else existing.get("label")
            with connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE graph_nodes
                        SET label=%s, properties_json=%s::jsonb, updated_at=now()
                        WHERE project_id=%s AND node_id=%s
                        """,
                        (new_label, _json_dumps(props), pid, str(node_id)),
                    )
            return self._get_node(node_id) or existing

        num = self._next_num()
        props2 = merge_properties or {}
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
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
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, now(), now())
                    """,
                    (
                        pid,
                        str(node_id),
                        str(kind),
                        int(num),
                        label,
                        _json_dumps(props2),
                    ),
                )
        return self._get_node(node_id) or {
            "node_id": str(node_id),
            "kind": str(kind),
            "num": int(num),
            "label": label,
            "properties": dict(props2),
            "created_at": now,
            "updated_at": now,
        }

    def _set_alias(self, *, alias: str, node_id: str, kind: str) -> None:
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_aliases(
                      project_id,
                      alias,
                      node_id,
                      kind,
                      created_at
                    )
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT(project_id, alias) DO UPDATE
                      SET node_id=excluded.node_id,
                          kind=excluded.kind,
                          created_at=excluded.created_at
                    """,
                    (pid, str(alias), str(node_id), str(kind)),
                )

    def _propagate_ingest_id_to_doc_key(self, *, doc_key: str, ingest_id: str) -> None:
        """Fill missing ingest_ids for existing doc_key nodes.

        Reference nodes created during extraction use doc_key values like
        `doi:...` or `bib:...` but often do not have an ingested PDF yet.
        When a cited PDF is later ingested, we want auto-place to resolve that
        reference without requiring a re-run of resolution on the citing work.

        This is best-effort and only fills when ingest_ids is missing/empty.
        """
        pid = self._project_id()
        key = str(doc_key or "").strip()
        ing = str(ingest_id or "").strip()
        if not key or not ing:
            return
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE graph_nodes
                       SET properties_json = jsonb_set(
                             COALESCE(properties_json, '{}'::jsonb),
                             '{ingest_ids}',
                             to_jsonb(ARRAY[%s]::text[]),
                             true
                           ),
                           updated_at = now()
                     WHERE project_id=%s
                       AND kind='document'
                       AND (COALESCE(properties_json, '{}'::jsonb)->>'doc_key') = %s
                       AND (
                         (COALESCE(properties_json, '{}'::jsonb)->'ingest_ids') IS NULL
                         OR jsonb_array_length(
                              COALESCE(properties_json, '{}'::jsonb)->'ingest_ids'
                            ) = 0
                       )
                    """,
                    (ing, pid, key),
                )

    def resolve_alias(self, alias: str) -> Optional[str]:
        pid = self._project_id()
        a = str(alias or "").strip()
        if not a:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id
                    FROM graph_aliases
                    WHERE project_id=%s AND alias=%s
                    """,
                    (pid, a),
                )
                row = cur.fetchone()
                return str(row[0]) if row and row[0] else None

    def resolve_reference_to_ingest_id(
        self, *, citing_doc_id: str, reference_id: str
    ) -> Optional[str]:
        citing = str(citing_doc_id or "").strip()
        ref = str(reference_id or "").strip()
        if not citing or not ref:
            return None
        node_id = self.resolve_alias(f"ref:{citing}:{ref}")
        if not node_id:
            return None
        node = self._get_node(node_id) or {}
        props = node.get("properties") or {}
        ingest_ids = props.get("ingest_ids") or []
        if isinstance(ingest_ids, list) and ingest_ids:
            value = ingest_ids[0]
            return str(value) if value else None

        doi = normalize_doi(props.get("doi"))
        if doi:
            node_id2 = self.resolve_alias(f"doi:{doi}")
            if node_id2:
                node2 = self._get_node(str(node_id2)) or {}
                props2 = node2.get("properties") or {}
                ingest2 = props2.get("ingest_ids") or []
                if isinstance(ingest2, list) and ingest2:
                    value = ingest2[0]
                    return str(value) if value else None

        doc_key = str(props.get("doc_key") or "").strip()
        if doc_key.startswith("bib:"):
            node_id3 = self.resolve_alias(doc_key)
            if node_id3:
                node3 = self._get_node(str(node_id3)) or {}
                props3 = node3.get("properties") or {}
                ingest3 = props3.get("ingest_ids") or []
                if isinstance(ingest3, list) and ingest3:
                    value = ingest3[0]
                    return str(value) if value else None
        return None

    def _upsert_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        kind: str,
        ref_id: str = "",
        enabled: bool = True,
        merge_properties: Optional[dict] = None,
        project_id: Optional[str] = None,
    ) -> int:
        pid = self._project_id(project_id)
        ref_id_norm = str(ref_id or "")

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, properties_json
                    FROM graph_edges
                    WHERE project_id=%s
                      AND source_id=%s
                      AND target_id=%s
                      AND kind=%s
                      AND ref_id=%s
                    """,
                    (pid, str(source_id), str(target_id), str(kind), ref_id_norm),
                )
                row = cur.fetchone()

        if row:
            edge_id, props_raw = row
            props = props_raw if isinstance(props_raw, dict) else {}
            if props_raw is not None and not isinstance(props_raw, dict):
                try:
                    props = json.loads(props_raw)
                except Exception:
                    props = {}
            for k, v in (merge_properties or {}).items():
                if v is None:
                    continue
                if k not in props or props.get(k) in (None, "", []):
                    props[k] = v
            with connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE graph_edges
                        SET enabled=%s, properties_json=%s::jsonb, updated_at=now()
                        WHERE edge_id=%s
                        """,
                        (bool(enabled), _json_dumps(props), int(edge_id)),
                    )
            return int(edge_id)

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_edges(
                      project_id, source_id, target_id, kind, ref_id, enabled,
                      properties_json, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, now(), now())
                    RETURNING edge_id
                    """,
                    (
                        pid,
                        str(source_id),
                        str(target_id),
                        str(kind),
                        ref_id_norm,
                        bool(enabled),
                        _json_dumps(merge_properties or {}),
                    ),
                )
                row2 = cur.fetchone()
                edge_id_val = row2[0] if row2 else None
                if edge_id_val is None:
                    raise RuntimeError("Failed to create edge")
                new_id = int(edge_id_val)
        return new_id

    def index_ingest_upload(self, ingest_meta: dict) -> Optional[dict]:
        doc_key = doc_key_for_ingest(ingest_meta)
        if not doc_key:
            return None
        node_id = doc_node_id_from_key(doc_key)
        ingest_id = str(ingest_meta.get("id") or "").strip() or None
        props = {
            "doc_key": doc_key,
            "sha256": ingest_meta.get("sha256"),
            "filename": ingest_meta.get("filename"),
            "ingest_ids": [ingest_id] if ingest_id else [],
            "workflow_assigned": False,
        }
        node = self._upsert_node(
            node_id=node_id, kind="document", label=None, merge_properties=props
        )
        if ingest_id:
            self._set_alias(alias=f"ingest:{ingest_id}", node_id=node_id, kind="ingest")
        return node

    def index_extraction(self, *, ingest_meta: dict, extraction_data: dict) -> None:
        doc_key = doc_key_for_ingest(ingest_meta)
        if not doc_key:
            return
        node_id = doc_node_id_from_key(doc_key)
        metadata = (extraction_data or {}).get("metadata") or {}
        title = metadata.get("title")
        authors = metadata.get("authors") or []
        year = metadata.get("year")
        doi_meta = normalize_doi(metadata.get("doi"))
        bib_meta = _safe_bib_key(title=title, authors=authors, year=year)
        ingest_id = str(ingest_meta.get("id") or "").strip() or None

        self._upsert_node(
            node_id=node_id,
            kind="document",
            label=None,
            merge_properties={
                "title": title,
                "authors": authors,
                "year": year,
                "filename": ingest_meta.get("filename"),
                "ingest_ids": [ingest_id] if ingest_id else [],
                "sha256": ingest_meta.get("sha256"),
                "doi": doi_meta,
                "extracted": True,
            },
        )
        if ingest_id:
            self._set_alias(alias=f"ingest:{ingest_id}", node_id=node_id, kind="ingest")

        if doi_meta:
            self._set_alias(alias=f"doi:{doi_meta}", node_id=node_id, kind="doi")
        if bib_meta:
            self._set_alias(alias=str(bib_meta), node_id=node_id, kind="bib")

        # If this work corresponds to an existing reference node (doi/bib),
        # backfill that node with this ingest_id so auto-place can resolve
        # without requiring re-resolution of citing works.
        try:
            if ingest_id:
                if doi_meta:
                    self._propagate_ingest_id_to_doc_key(
                        doc_key=f"doi:{doi_meta}",
                        ingest_id=str(ingest_id),
                    )
                if bib_meta:
                    self._propagate_ingest_id_to_doc_key(
                        doc_key=str(bib_meta),
                        ingest_id=str(ingest_id),
                    )
        except Exception:
            pass

        for ref in (extraction_data or {}).get("references") or []:
            if not isinstance(ref, dict):
                continue
            grobid = (
                (ref.get("grobid") or {}) if isinstance(ref.get("grobid"), dict) else {}
            )
            doi = normalize_doi(ref.get("doi") or grobid.get("doi"))
            ref_title = grobid.get("title")
            ref_authors = grobid.get("authors") or []
            ref_year = grobid.get("year")
            bib_ref = _safe_bib_key(title=ref_title, authors=ref_authors, year=ref_year)
            ref_key = (
                f"doi:{doi}"
                if doi
                else (
                    bib_ref
                    or bib_fingerprint(
                        title=ref_title, authors=ref_authors, year=ref_year
                    )
                )
            )
            ref_node_id = self.resolve_alias(str(ref_key)) or doc_node_id_from_key(
                ref_key
            )
            self._upsert_node(
                node_id=ref_node_id,
                kind="document",
                label=None,
                merge_properties={
                    "doc_key": ref_key,
                    "doi": doi,
                    "title": ref_title,
                    "authors": ref_authors,
                    "year": ref_year,
                    "raw_reference": ref.get("raw_reference"),
                },
            )
            ref_id = str(ref.get("id") or "")
            if ingest_id and ref_id:
                self._set_alias(
                    alias=f"ref:{ingest_id}:{ref_id}",
                    node_id=ref_node_id,
                    kind="reference",
                )
            self._upsert_edge(
                source_id=node_id,
                target_id=ref_node_id,
                kind="CITES",
                ref_id=ref_id,
                enabled=True,
                merge_properties={
                    "source": "auto",
                    "ingest_id": ingest_id,
                    "reference_id": ref_id or None,
                },
            )

    def index_resolution(
        self, *, ingest_meta: dict, resolution_data: Sequence[dict]
    ) -> None:
        ingest_id = str(ingest_meta.get("id") or "").strip() or None
        if not ingest_id:
            return
        try:
            doc_key = doc_key_for_ingest(ingest_meta)
            if doc_key:
                self._upsert_node(
                    node_id=doc_node_id_from_key(doc_key),
                    kind="document",
                    label=None,
                    merge_properties={"resolved": True},
                )
        except Exception:
            pass

        for entry in resolution_data or []:
            if not isinstance(entry, dict):
                continue
            ref_id = str(entry.get("reference_id") or "").strip()
            if not ref_id:
                continue
            node_id = self.resolve_alias(f"ref:{ingest_id}:{ref_id}")
            if not node_id:
                continue
            doi = normalize_doi(entry.get("doi"))
            bib = _safe_bib_key(
                title=entry.get("title"),
                authors=entry.get("authors") or [],
                year=entry.get("year"),
            )

            ingest_ids: list[str] = []
            for alias_key in [f"doi:{doi}" if doi else None, bib]:
                if not alias_key:
                    continue
                target_node_id = self.resolve_alias(str(alias_key))
                if not target_node_id:
                    continue
                target_node = self._get_node(str(target_node_id)) or {}
                props = target_node.get("properties") or {}
                raw_ids = props.get("ingest_ids")
                if isinstance(raw_ids, list):
                    ingest_ids = [str(x) for x in raw_ids if str(x).strip()]
                if ingest_ids:
                    break

            self._upsert_node(
                node_id=node_id,
                kind="document",
                label=None,
                merge_properties={
                    "doi": doi,
                    "title": entry.get("title"),
                    "year": entry.get("year"),
                    "ingest_ids": ingest_ids or None,
                    "resolved": True,
                },
            )

    def index_confirmed_claims(self, payload: dict) -> None:
        ingest_id = str(payload.get("document_id") or "").strip()
        if not ingest_id:
            return
        doc_node_id = self.resolve_alias(f"ingest:{ingest_id}")
        if not doc_node_id:
            return
        sentence_id = str(payload.get("sentence_id") or "").strip() or None
        citation_index = payload.get("citation_index")
        target_id = payload.get("target_id")
        confirmed = payload.get("confirmed_claims") or []
        if not isinstance(confirmed, list) or not confirmed:
            return

        self._upsert_node(
            node_id=doc_node_id,
            kind="document",
            label=None,
            merge_properties={"workflow_assigned": True},
        )

        for claim in confirmed:
            if not isinstance(claim, dict):
                continue
            claim_index = claim.get("claim_index")
            claim_id = f"claim:{ingest_id}:{sentence_id or 'unknown'}:{claim_index}"
            self._upsert_node(
                node_id=claim_id,
                kind="claim",
                label=None,
                merge_properties={
                    "document_id": ingest_id,
                    "sentence_id": sentence_id,
                    "citation_index": citation_index,
                    "target_id": target_id,
                    "claim_index": claim_index,
                    "parsed_text": claim.get("parsed_text"),
                },
            )
            self._upsert_edge(
                source_id=doc_node_id,
                target_id=claim_id,
                kind="HAS_CLAIM",
                ref_id=str(claim_index or ""),
                enabled=True,
                merge_properties={"source": "auto"},
            )
            if ingest_id and target_id:
                ref_node = self.resolve_alias(f"ref:{ingest_id}:{target_id}")
                if ref_node:
                    self._upsert_edge(
                        source_id=claim_id,
                        target_id=ref_node,
                        kind="CLAIM_ABOUT",
                        ref_id=str(target_id),
                        enabled=True,
                        merge_properties={"source": "auto"},
                    )

    def mark_doc_assigned_for_attachment(
        self, *, doc_id: Optional[str], claim_id: Optional[str]
    ) -> None:
        if not doc_id or not claim_id:
            return
        doc_node_id = self.resolve_alias(f"ingest:{doc_id}")
        if not doc_node_id:
            return
        self._upsert_node(
            node_id=doc_node_id,
            kind="document",
            label=None,
            merge_properties={"workflow_assigned": True},
        )

    def link_reference_to_ingest(
        self, *, citing_doc_id: str, reference_id: str, cited_ingest_id: str
    ) -> None:
        """Attach an uploaded cited work to a reference node.

        Auto-place requires `ref:{citing_doc_id}:{reference_id}` to resolve to a
        document node with `properties.ingest_ids[0] == cited_ingest_id`.

        This method is invoked when a reviewer assigns a Source Bin PDF to a
        specific reference target (via attachment placement).
        """
        citing = str(citing_doc_id or "").strip()
        ref = str(reference_id or "").strip()
        cited = str(cited_ingest_id or "").strip()
        if not citing or not ref or not cited:
            return
        ref_node = self.resolve_alias(f"ref:{citing}:{ref}")
        if not ref_node:
            return
        self._upsert_node(
            node_id=str(ref_node),
            kind="document",
            label=None,
            merge_properties={"ingest_ids": [cited]},
        )

    def ledger_rows(self, *, project_id: Optional[str] = None) -> List[dict]:
        pid = self._project_id(project_id)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id, num, label, properties_json
                    FROM graph_nodes
                    WHERE project_id=%s AND kind='document'
                    ORDER BY num ASC
                    """,
                    (pid,),
                )
                nodes = cur.fetchall() or []

        if not nodes:
            return []

        docs: Dict[str, dict] = {}
        for node_id, num, label, props_raw in nodes:
            props = props_raw if isinstance(props_raw, dict) else {}
            if props_raw is not None and not isinstance(props_raw, dict):
                try:
                    props = json.loads(props_raw)
                except Exception:
                    props = {}
            docs[str(node_id)] = {
                "node_id": str(node_id),
                "num": int(num or 0),
                "label": label,
                "properties": props,
            }

        anchored_nodes = {
            node_id
            for node_id, doc in docs.items()
            if bool((doc.get("properties") or {}).get("ingest_ids"))
        }

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_id, target_id
                    FROM graph_edges
                    WHERE project_id=%s AND kind='CITES' AND enabled=true
                    """,
                    (pid,),
                )
                cite_rows = cur.fetchall() or []

        outgoing: Dict[str, List[str]] = {k: [] for k in docs.keys()}
        incoming: Dict[str, List[str]] = {k: [] for k in docs.keys()}
        for src, tgt in cite_rows:
            src2 = str(src)
            tgt2 = str(tgt)
            if src2 in outgoing and tgt2 in docs:
                outgoing[src2].append(tgt2)
            if tgt2 in incoming and src2 in docs:
                incoming[tgt2].append(src2)

        rows: List[dict] = []
        for node_id, doc in docs.items():
            props = doc.get("properties") or {}
            authors = props.get("authors") or []
            year = props.get("year")
            title = props.get("title")
            filename = props.get("filename")
            first = _surname(authors[0]) if authors else None
            short = "Unknown"
            if first and year:
                short = f"{first} ({year})"
            elif first:
                short = first
            elif title:
                short = str(title)[:28]
            elif filename:
                name = str(filename)
                if name.lower().endswith(".pdf"):
                    name = name[:-4]
                short = name[:28]

            apa = props.get("raw_reference")
            if not apa:
                bits = []
                if authors:
                    bits.append(", ".join(str(a) for a in authors[:6]))
                if year:
                    bits.append(str(year))
                if title:
                    bits.append(str(title))
                if not title and filename:
                    bits.append(str(filename))
                doi = props.get("doi")
                if doi:
                    bits.append(f"doi:{doi}")
                apa = ". ".join(bit for bit in bits if bit) or short

            assigned = bool(props.get("workflow_assigned"))
            anchored = bool(props.get("ingest_ids"))
            extracted = bool(props.get("extracted"))
            resolved = bool(props.get("resolved"))
            degree = len(outgoing.get(node_id) or []) + len(incoming.get(node_id) or [])
            status = "green" if (assigned and anchored and degree > 0) else "orange"

            incoming_live = [
                docs[s]["num"]
                for s in (incoming.get(node_id) or [])
                if s in anchored_nodes
            ]
            outgoing_live = [
                docs[t]["num"]
                for t in (outgoing.get(node_id) or [])
                if t in anchored_nodes
            ]

            rows.append(
                {
                    "num": int(doc.get("num") or 0),
                    "node_id": node_id,
                    "short": short,
                    "title": title or filename,
                    "apa": apa,
                    "status": status,
                    "assigned": assigned,
                    "anchored": anchored,
                    "extracted": extracted,
                    "resolved": resolved,
                    "incoming": sorted(
                        [docs[s]["num"] for s in incoming.get(node_id) or []]
                    ),
                    "outgoing": sorted(
                        [docs[t]["num"] for t in outgoing.get(node_id) or []]
                    ),
                    "incoming_live": sorted([int(n) for n in incoming_live if n]),
                    "outgoing_live": sorted([int(n) for n in outgoing_live if n]),
                    "ingest_id": (props.get("ingest_ids") or [None])[0],
                }
            )

        rows.sort(key=lambda r: r.get("num", 0))
        return rows

    def ledger_options(self, *, project_id: Optional[str] = None) -> List[dict]:
        rows = self.ledger_rows(project_id=project_id)
        return [
            {
                "num": r.get("num"),
                "short": r.get("short"),
                "apa": r.get("apa"),
                "status": r.get("status"),
            }
            for r in rows
        ]

    def _node_id_by_num(
        self, num: int, *, project_id: Optional[str] = None
    ) -> Optional[str]:
        pid = self._project_id(project_id)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id
                    FROM graph_nodes
                    WHERE project_id=%s AND kind='document' AND num=%s
                    """,
                    (pid, int(num)),
                )
                row = cur.fetchone()
                return str(row[0]) if row and row[0] else None

    def set_outgoing(
        self,
        *,
        source_num: int,
        target_nums: Sequence[int],
        project_id: Optional[str] = None,
    ) -> None:
        pid = self._project_id(project_id)
        source_id = self._node_id_by_num(int(source_num), project_id=pid)
        if not source_id:
            raise KeyError(f"Unknown source document number: {source_num}")
        desired: set[str] = set()
        for n in target_nums or []:
            tgt = self._node_id_by_num(int(n), project_id=pid)
            if tgt and tgt != source_id:
                desired.add(tgt)

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, target_id
                    FROM graph_edges
                    WHERE project_id=%s AND kind='CITES' AND source_id=%s
                    """,
                    (pid, str(source_id)),
                )
                existing = cur.fetchall() or []
        existing_targets = {str(tgt): int(eid) for eid, tgt in existing}

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                for tgt, edge_id in existing_targets.items():
                    cur.execute(
                        """
                        UPDATE graph_edges
                        SET enabled=%s, updated_at=now()
                        WHERE edge_id=%s
                        """,
                        (bool(tgt in desired), int(edge_id)),
                    )

        for tgt in desired:
            if tgt in existing_targets:
                continue
            self._upsert_edge(
                source_id=source_id,
                target_id=tgt,
                kind="CITES",
                ref_id="manual",
                enabled=True,
                merge_properties={"source": "manual"},
                project_id=pid,
            )

    def set_incoming(
        self,
        *,
        target_num: int,
        source_nums: Sequence[int],
        project_id: Optional[str] = None,
    ) -> None:
        pid = self._project_id(project_id)
        target_id = self._node_id_by_num(int(target_num), project_id=pid)
        if not target_id:
            raise KeyError(f"Unknown target document number: {target_num}")
        desired_sources: set[str] = set()
        for n in source_nums or []:
            src = self._node_id_by_num(int(n), project_id=pid)
            if src and src != target_id:
                desired_sources.add(src)

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, source_id
                    FROM graph_edges
                    WHERE project_id=%s AND kind='CITES' AND target_id=%s
                    """,
                    (pid, str(target_id)),
                )
                existing = cur.fetchall() or []
        existing_sources = {str(src): int(eid) for eid, src in existing}

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                for src, edge_id in existing_sources.items():
                    cur.execute(
                        """
                        UPDATE graph_edges
                        SET enabled=%s, updated_at=now()
                        WHERE edge_id=%s
                        """,
                        (bool(src in desired_sources), int(edge_id)),
                    )

        for src in desired_sources:
            if src in existing_sources:
                continue
            self._upsert_edge(
                source_id=src,
                target_id=target_id,
                kind="CITES",
                ref_id="manual",
                enabled=True,
                merge_properties={"source": "manual"},
                project_id=pid,
            )

    def set_assigned(
        self,
        *,
        doc_num: int,
        assigned: bool,
        project_id: Optional[str] = None,
    ) -> None:
        pid = self._project_id(project_id)
        node_id = self._node_id_by_num(int(doc_num), project_id=pid)
        if not node_id:
            raise KeyError(f"Unknown document number: {doc_num}")
        self._upsert_node(
            node_id=node_id,
            kind="document",
            label=None,
            merge_properties={"workflow_assigned": bool(assigned)},
            project_id=pid,
        )

    def _get_edge(self, edge_id: int) -> Optional[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, source_id, target_id, kind, ref_id, enabled,
                           properties_json, created_at, updated_at
                    FROM graph_edges
                    WHERE edge_id=%s AND project_id=%s
                    """,
                    (int(edge_id), pid),
                )
                row = cur.fetchone()
                if not row:
                    return None

        (
            edge_id2,
            source_id,
            target_id,
            kind,
            ref_id,
            enabled,
            props_raw,
            created_at,
            updated_at,
        ) = row
        props = props_raw if isinstance(props_raw, dict) else {}
        if props_raw is not None and not isinstance(props_raw, dict):
            try:
                props = json.loads(props_raw)
            except Exception:
                props = {}

        return {
            "edge_id": int(edge_id2),
            "source_id": str(source_id),
            "target_id": str(target_id),
            "kind": str(kind),
            "ref_id": str(ref_id or ""),
            "enabled": bool(enabled),
            "properties": props,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    def get_edge(self, edge_id: int) -> Optional[dict]:
        return self._get_edge(int(edge_id))

    def get_claim_node(self, claim_id: str) -> Optional[dict]:
        node = self._get_node(str(claim_id or "").strip())
        if not node:
            return None
        if str(node.get("kind")) != "claim":
            return None
        return node

    def list_claim_nodes(self) -> List[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT node_id
                    FROM graph_nodes
                    WHERE project_id=%s AND kind='claim'
                    ORDER BY num ASC
                    """,
                    (pid,),
                )
                rows = cur.fetchall() or []
        nodes: List[dict] = []
        for (node_id,) in rows:
            node = self._get_node(str(node_id))
            if node:
                nodes.append(node)
        return nodes

    def upsert_claim_link(
        self,
        *,
        source_claim_id: str,
        target_claim_id: str,
        source: str,
        creator_uid: Optional[str] = None,
        explored_by: Optional[Sequence[str]] = None,
    ) -> int:
        src = str(source_claim_id or "").strip()
        tgt = str(target_claim_id or "").strip()
        if not src or not tgt:
            raise ValueError("source_claim_id and target_claim_id are required")
        if src == tgt:
            raise ValueError("Cannot link a claim to itself")

        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, properties_json
                    FROM graph_edges
                    WHERE project_id=%s AND source_id=%s AND target_id=%s
                      AND kind='CLAIM_LINK' AND ref_id=''
                    """,
                    (pid, src, tgt),
                )
                row = cur.fetchone()

        props: dict = {}
        if row:
            props_raw = row[1]
            props = props_raw if isinstance(props_raw, dict) else {}
            if props_raw is not None and not isinstance(props_raw, dict):
                try:
                    props = json.loads(props_raw)
                except Exception:
                    props = {}

        props["source"] = str(source or "").strip() or "auto"
        if creator_uid:
            text = str(creator_uid or "").strip()
            if (
                text
                and props.get("source") == "manual"
                and not props.get("creator_uid")
            ):
                props["creator_uid"] = text

        if explored_by:
            merged: List[str] = []
            existing = props.get("explored_by")
            if isinstance(existing, list):
                merged.extend(str(x) for x in existing if str(x).strip())
            for uid in explored_by:
                text = str(uid or "").strip()
                if text and text not in merged:
                    merged.append(text)
            if merged:
                props["explored_by"] = merged

        if row:
            edge_id_val = row[0]
            if edge_id_val is None:
                raise RuntimeError("Corrupt graph edge row")
            edge_id = int(edge_id_val)
            with connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE graph_edges
                        SET enabled=true, properties_json=%s::jsonb, updated_at=now()
                        WHERE edge_id=%s
                        """,
                        (_json_dumps(props), edge_id),
                    )
            return edge_id

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_edges(
                      project_id, source_id, target_id, kind, ref_id, enabled,
                      properties_json, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, 'CLAIM_LINK', '', true, %s::jsonb, now(), now())
                    RETURNING edge_id
                    """,
                    (pid, src, tgt, _json_dumps(props)),
                )
                row2 = cur.fetchone()
                edge_id_val = row2[0] if row2 else None
                if edge_id_val is None:
                    raise RuntimeError("Failed to create claim link edge")
                new_id = int(edge_id_val)
        return new_id

    def upsert_topology_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        kind: str,
        source: str,
        creator_uid: Optional[str] = None,
        explored_by: Optional[Sequence[str]] = None,
        enabled: bool = True,
    ) -> int:
        src = str(source_id or "").strip()
        tgt = str(target_id or "").strip()
        kind_norm = str(kind or "").strip()
        if not src or not tgt or not kind_norm:
            raise ValueError("source_id, target_id, and kind are required")
        if src == tgt:
            raise ValueError("Cannot link a node to itself")

        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT edge_id, properties_json
                    FROM graph_edges
                    WHERE project_id=%s
                      AND source_id=%s
                      AND target_id=%s
                      AND kind=%s
                      AND ref_id=''
                    """,
                    (pid, src, tgt, kind_norm),
                )
                row = cur.fetchone()

        props: dict = {}
        if row:
            props_raw = row[1]
            props = props_raw if isinstance(props_raw, dict) else {}
            if props_raw is not None and not isinstance(props_raw, dict):
                try:
                    props = json.loads(props_raw)
                except Exception:
                    props = {}

        props["source"] = str(source or "").strip() or "auto"
        if creator_uid:
            text = str(creator_uid or "").strip()
            if (
                text
                and props.get("source") == "manual"
                and not props.get("creator_uid")
            ):
                props["creator_uid"] = text

        if explored_by:
            merged: List[str] = []
            existing = props.get("explored_by")
            if isinstance(existing, list):
                merged.extend(str(x) for x in existing if str(x).strip())
            for uid in explored_by:
                text = str(uid or "").strip()
                if text and text not in merged:
                    merged.append(text)
            if merged:
                props["explored_by"] = merged

        if row:
            edge_id_val = row[0]
            if edge_id_val is None:
                raise RuntimeError("Corrupt graph edge row")
            edge_id = int(edge_id_val)
            with connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE graph_edges
                        SET enabled=%s, properties_json=%s::jsonb, updated_at=now()
                        WHERE edge_id=%s
                        """,
                        (bool(enabled), _json_dumps(props), edge_id),
                    )
            return edge_id

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_edges(
                      project_id, source_id, target_id, kind, ref_id, enabled,
                      properties_json, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, '', %s, %s::jsonb, now(), now())
                    RETURNING edge_id
                    """,
                    (pid, src, tgt, kind_norm, bool(enabled), _json_dumps(props)),
                )
                row2 = cur.fetchone()
                edge_id_val = row2[0] if row2 else None
                if edge_id_val is None:
                    raise RuntimeError("Failed to create topology edge")
                new_id = int(edge_id_val)
        return new_id

    def upsert_manual_work_cites_work(
        self,
        *,
        citing_ingest_id: str,
        cited_ingest_id: str,
        reviewer_uid: str,
        enabled: bool = True,
    ) -> int:
        citing = str(citing_ingest_id or "").strip()
        cited = str(cited_ingest_id or "").strip()
        if not citing or not cited:
            raise ValueError("citing_ingest_id and cited_ingest_id are required")
        return self.upsert_topology_edge(
            source_id=citing,
            target_id=cited,
            kind="WORK_CITES_WORK",
            source="manual",
            creator_uid=str(reviewer_uid or "default"),
            enabled=bool(enabled),
        )

    def list_edges_by_kind(
        self, *, kind: str, include_disabled: bool = True
    ) -> List[dict]:
        pid = self._project_id()
        kind_norm = str(kind or "").strip()
        if not kind_norm:
            return []
        clause = "" if include_disabled else "AND enabled=true"
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT edge_id
                    FROM graph_edges
                    WHERE project_id=%s AND kind=%s {clause}
                    ORDER BY edge_id ASC
                    """,
                    (pid, kind_norm),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (edge_id,) in rows:
            edge = self._get_edge(int(edge_id))
            if edge:
                out.append(edge)
        return out

    def list_edges(
        self,
        *,
        kind: str,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        include_disabled: bool = True,
    ) -> List[dict]:
        pid = self._project_id()
        kind_norm = str(kind or "").strip()
        if not kind_norm:
            return []
        clauses = ["project_id=%s", "kind=%s"]
        params: List[Any] = [pid, kind_norm]
        if source_id is not None:
            clauses.append("source_id=%s")
            params.append(str(source_id))
        if target_id is not None:
            clauses.append("target_id=%s")
            params.append(str(target_id))
        if not include_disabled:
            clauses.append("enabled=true")
        where = " AND ".join(clauses)
        with connect() as conn:
            with conn.cursor() as cur:
                sql = (
                    f"SELECT edge_id FROM graph_edges WHERE {where} "
                    "ORDER BY edge_id ASC"
                )
                cur.execute(sql, tuple(params))
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (edge_id,) in rows:
            edge = self._get_edge(int(edge_id))
            if edge:
                out.append(edge)
        return out

    def set_edge_enabled(
        self, *, edge_id: int, enabled: bool, merge_properties: Optional[dict] = None
    ) -> None:
        edge = self._get_edge(int(edge_id))
        if not edge:
            raise KeyError("Edge not found")
        props = edge.get("properties") or {}
        for k, v in (merge_properties or {}).items():
            if v is None:
                continue
            props[k] = v
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE graph_edges
                    SET enabled=%s, properties_json=%s::jsonb, updated_at=now()
                    WHERE edge_id=%s
                    """,
                    (bool(enabled), _json_dumps(props), int(edge_id)),
                )

    def delete_claim_link(self, *, edge_id: int, reviewer_uid: str) -> None:
        edge = self._get_edge(int(edge_id))
        if not edge or edge.get("kind") != "CLAIM_LINK":
            raise KeyError("Claim link edge not found")

        props = edge.get("properties") or {}
        src = str(props.get("source") or "")
        if src != "manual":
            raise PermissionError("Only manual edges can be deleted")
        creator = str(props.get("creator_uid") or "").strip()
        reviewer = str(reviewer_uid or "").strip() or "default"
        if not creator or creator != reviewer:
            raise PermissionError("Only the creator can delete this manual edge")

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE graph_edges
                    SET enabled=false, updated_at=now()
                    WHERE edge_id=%s
                    """,
                    (int(edge_id),),
                )

    def upsert_edge_vote(
        self,
        *,
        edge_id: int,
        reviewer_uid: str,
        verdict: str,
        confidence: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> dict:
        edge = self._get_edge(int(edge_id))
        if not edge:
            raise KeyError("Edge not found")
        if not edge.get("enabled"):
            props = edge.get("properties") or {}
            if (
                str(edge.get("kind") or "") == "CLAIM_LINK"
                and str(props.get("source") or "") == "manual"
            ):
                raise KeyError("Edge not found")

        reviewer = str(reviewer_uid or "").strip() or "default"
        ver = str(verdict or "").strip()
        if ver not in {"support", "contradict", "neutral", "uncertain"}:
            raise ValueError(
                "verdict must be one of: support, contradict, neutral, uncertain"
            )

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_edge_votes(
                      edge_id, reviewer_uid, verdict, confidence, comment, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT(edge_id, reviewer_uid) DO UPDATE
                      SET verdict=excluded.verdict,
                          confidence=excluded.confidence,
                          comment=excluded.comment,
                          updated_at=excluded.updated_at
                    RETURNING
                      edge_id,
                      reviewer_uid,
                      verdict,
                      confidence,
                      comment,
                      updated_at
                    """,
                    (int(edge_id), reviewer, ver, confidence, comment),
                )
                row = cur.fetchone()

        edge_id2, reviewer_uid2, verdict2, conf2, comment2, updated_at = row
        return {
            "edge_id": int(edge_id2),
            "reviewer_uid": str(reviewer_uid2),
            "verdict": str(verdict2),
            "confidence": conf2,
            "comment": comment2,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    def list_edge_votes(self, edge_id: int) -> List[dict]:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      edge_id,
                      reviewer_uid,
                      verdict,
                      confidence,
                      comment,
                      updated_at
                    FROM graph_edge_votes
                    WHERE edge_id=%s
                    ORDER BY updated_at DESC, reviewer_uid ASC
                    """,
                    (int(edge_id),),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for edge_id2, reviewer_uid, verdict, conf, comment, updated_at in rows:
            out.append(
                {
                    "edge_id": int(edge_id2),
                    "reviewer_uid": str(reviewer_uid),
                    "verdict": str(verdict),
                    "confidence": conf,
                    "comment": comment,
                    "updated_at": updated_at.isoformat().replace("+00:00", "Z")
                    if updated_at is not None
                    else None,
                }
            )
        return out

    def edge_vote_aggregates(self, edge_id: int) -> dict:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT verdict, COUNT(*)
                    FROM graph_edge_votes
                    WHERE edge_id=%s
                    GROUP BY verdict
                    """,
                    (int(edge_id),),
                )
                rows = cur.fetchall() or []
        counts = {str(verdict): int(n or 0) for verdict, n in rows}
        n_support = int(counts.get("support") or 0)
        n_contradict = int(counts.get("contradict") or 0)
        n_neutral = int(counts.get("neutral") or 0)
        n_uncertain = int(counts.get("uncertain") or 0)
        n_total = n_support + n_contradict + n_neutral + n_uncertain
        return {
            "n_support": n_support,
            "n_contradict": n_contradict,
            "n_neutral": n_neutral,
            "n_uncertain": n_uncertain,
            "n_total": n_total,
        }

    def list_claim_links_for_claim(
        self,
        *,
        claim_id: str,
        sources: Optional[Sequence[str]] = None,
        enabled_only: bool = True,
    ) -> List[dict]:
        pid = self._project_id()
        node_id = str(claim_id or "").strip()
        if not node_id:
            return []
        clauses = [
            "project_id=%s",
            "kind='CLAIM_LINK'",
            "(source_id=%s OR target_id=%s)",
        ]
        params: List[Any] = [pid, node_id, node_id]
        if enabled_only:
            clauses.append("enabled=true")
        where = " AND ".join(clauses)
        with connect() as conn:
            with conn.cursor() as cur:
                sql = (
                    f"SELECT edge_id FROM graph_edges WHERE {where} "
                    "ORDER BY edge_id ASC"
                )
                cur.execute(sql, tuple(params))
                rows = cur.fetchall() or []

        want = {str(s).strip() for s in (sources or []) if str(s).strip()}
        edges: List[dict] = []
        for (edge_id,) in rows:
            edge = self._get_edge(int(edge_id))
            if not edge:
                continue
            if want:
                src = str((edge.get("properties") or {}).get("source") or "")
                if src not in want:
                    continue
            edges.append(edge)
        return edges
