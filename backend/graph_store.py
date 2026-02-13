from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    *,
    title: Optional[str],
    authors: Optional[Sequence[str]],
    year: Optional[str],
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


SCHEMA: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS nodes (
        node_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        num INTEGER UNIQUE,
        label TEXT,
        properties_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS aliases (
        alias TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        ref_id TEXT NOT NULL DEFAULT '',
        enabled INTEGER NOT NULL DEFAULT 1,
        properties_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(source_id, target_id, kind, ref_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_source_kind ON edges(source_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target_kind ON edges(target_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edges_enabled_kind ON edges(enabled, kind)",
    # Phase 09: per-edge multi-user votes (consensus aggregates computed from rows).
    """
    CREATE TABLE IF NOT EXISTS edge_votes (
        edge_id INTEGER NOT NULL,
        reviewer_uid TEXT NOT NULL,
        verdict TEXT NOT NULL,
        confidence REAL,
        comment TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(edge_id, reviewer_uid),
        FOREIGN KEY(edge_id) REFERENCES edges(edge_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edge_votes_edge_id ON edge_votes(edge_id)",
]


class GraphStore:
    def __init__(self, db_path: Path):
        """SQLite-backed graph store (nodes + edges)."""
        self._path = _ensure_parent(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            for ddl in SCHEMA:
                self._conn.execute(ddl)

    def _next_num(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(num), 0) AS max_num FROM nodes"
        ).fetchone()
        return int(row["max_num"] or 0) + 1

    def _get_node(self, node_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT node_id, kind, num, label, properties_json, created_at, updated_at
            FROM nodes
            WHERE node_id=?
            """,
            (node_id,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["properties"] = json.loads(data.pop("properties_json") or "{}")
        except Exception:
            data["properties"] = {}
        return data

    def _upsert_node(
        self,
        *,
        node_id: str,
        kind: str,
        label: Optional[str] = None,
        merge_properties: Optional[dict] = None,
    ) -> dict:
        existing = self._get_node(node_id)
        now = _now()
        if existing:
            props = dict(existing.get("properties") or {})
            for k, v in (merge_properties or {}).items():
                if v is None:
                    continue
                if k not in props or props.get(k) in (None, "", []):
                    props[k] = v
                else:
                    # special-case: lists should extend.
                    if isinstance(props.get(k), list) and isinstance(v, list):
                        merged = list(props.get(k) or [])
                        for item in v:
                            if item not in merged:
                                merged.append(item)
                        props[k] = merged
            new_label = label if label is not None else existing.get("label")
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE nodes
                    SET label=?, properties_json=?, updated_at=?
                    WHERE node_id=?
                    """,
                    (
                        new_label,
                        json.dumps(props, ensure_ascii=True),
                        now,
                        node_id,
                    ),
                )
            return self._get_node(node_id) or existing

        num = self._next_num()
        props = merge_properties or {}
        with self._conn:
            self._conn.execute(
                """
                    INSERT INTO nodes(
                        node_id,
                        kind,
                        num,
                        label,
                        properties_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    node_id,
                    kind,
                    num,
                    label,
                    json.dumps(props, ensure_ascii=True),
                    now,
                    now,
                ),
            )
        return self._get_node(node_id) or {
            "node_id": node_id,
            "kind": kind,
            "num": num,
            "label": label,
            "properties": dict(props),
        }

    def _set_alias(self, *, alias: str, node_id: str, kind: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO aliases(alias, node_id, kind, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (alias, node_id, kind, _now()),
            )

    def resolve_alias(self, alias: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT node_id FROM aliases WHERE alias=?",
            (alias,),
        ).fetchone()
        return str(row["node_id"]) if row else None

    def resolve_reference_to_ingest_id(
        self, *, citing_doc_id: str, reference_id: str
    ) -> Optional[str]:
        """Return an anchored ingest_id for a citing doc's reference_id (target_id)."""
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

        # Fallback: attempt to resolve the reference's doc_key/doi to an ingested
        # document node via aliases.
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
    ) -> None:
        now = _now()
        ref_id_norm = str(ref_id or "")
        row = self._conn.execute(
            """
            SELECT edge_id, properties_json, enabled
            FROM edges
            WHERE source_id=? AND target_id=? AND kind=? AND ref_id=?
            """,
            (source_id, target_id, kind, ref_id_norm),
        ).fetchone()
        if row:
            try:
                props = json.loads(row["properties_json"] or "{}")
            except Exception:
                props = {}
            for k, v in (merge_properties or {}).items():
                if v is None:
                    continue
                if k not in props or props.get(k) in (None, "", []):
                    props[k] = v
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE edges
                    SET enabled=?, properties_json=?, updated_at=?
                    WHERE edge_id=?
                    """,
                    (
                        1 if enabled else 0,
                        json.dumps(props, ensure_ascii=True),
                        now,
                        int(row["edge_id"]),
                    ),
                )
            return

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO edges(
                    source_id,
                    target_id,
                    kind,
                    ref_id,
                    enabled,
                    properties_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    target_id,
                    kind,
                    ref_id_norm,
                    1 if enabled else 0,
                    json.dumps(merge_properties or {}, ensure_ascii=True),
                    now,
                    now,
                ),
            )

    # --- Public indexing -------------------------------------------------

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
            },
        )
        if ingest_id:
            self._set_alias(alias=f"ingest:{ingest_id}", node_id=node_id, kind="ingest")

        # If this ingested doc has a DOI, alias the DOI key to the ingest node.
        # This prevents "duplicate document nodes" while allowing reference
        # resolution (doi -> ingest_id) via resolve_alias.
        if doi_meta:
            self._set_alias(alias=f"doi:{doi_meta}", node_id=node_id, kind="doi")
        if bib_meta:
            self._set_alias(alias=str(bib_meta), node_id=node_id, kind="bib")

        # References -> document nodes + CITES edges.
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
            # Prefer any existing alias for this reference (eg. doi -> ingested doc).
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
            # If the resolution corresponds to an ingested document, anchor the
            # reference node to that ingest id so downstream views (Surfing) can
            # resolve reference_id -> ingest_id.
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

        # Mark document as assigned in workflow.
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

    # --- Ledger view + editing -------------------------------------------

    def ledger_rows(self) -> List[dict]:
        nodes = self._conn.execute(
            """
            SELECT node_id, kind, num, label, properties_json
            FROM nodes
            WHERE kind='document'
            ORDER BY num ASC
            """
        ).fetchall()
        if not nodes:
            return []
        docs: Dict[str, dict] = {}
        for row in nodes:
            props = {}
            try:
                props = json.loads(row["properties_json"] or "{}")
            except Exception:
                props = {}
            docs[str(row["node_id"])] = {
                "node_id": str(row["node_id"]),
                "num": int(row["num"] or 0),
                "label": row["label"],
                "properties": props,
            }

        anchored_nodes = {
            node_id
            for node_id, doc in docs.items()
            if bool((doc.get("properties") or {}).get("ingest_ids"))
        }

        # Edges: only enabled CITES.
        cite_rows = self._conn.execute(
            "SELECT source_id, target_id FROM edges WHERE kind='CITES' AND enabled=1"
        ).fetchall()
        outgoing: Dict[str, List[str]] = {k: [] for k in docs.keys()}
        incoming: Dict[str, List[str]] = {k: [] for k in docs.keys()}
        for edge in cite_rows:
            src = str(edge["source_id"])
            tgt = str(edge["target_id"])
            if src in outgoing and tgt in docs:
                outgoing[src].append(tgt)
            if tgt in incoming and src in docs:
                incoming[tgt].append(src)

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

    def ledger_options(self) -> List[dict]:
        rows = self.ledger_rows()
        return [
            {
                "num": r.get("num"),
                "short": r.get("short"),
                "apa": r.get("apa"),
                "status": r.get("status"),
            }
            for r in rows
        ]

    def _node_id_by_num(self, num: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT node_id FROM nodes WHERE kind='document' AND num=?",
            (int(num),),
        ).fetchone()
        return str(row["node_id"]) if row else None

    def set_outgoing(self, *, source_num: int, target_nums: Sequence[int]) -> None:
        source_id = self._node_id_by_num(int(source_num))
        if not source_id:
            raise KeyError(f"Unknown source document number: {source_num}")
        desired: set[str] = set()
        for n in target_nums or []:
            tgt = self._node_id_by_num(int(n))
            if tgt and tgt != source_id:
                desired.add(tgt)

        existing = self._conn.execute(
            "SELECT edge_id, target_id FROM edges WHERE kind='CITES' AND source_id=?",
            (source_id,),
        ).fetchall()
        existing_targets = {
            str(row["target_id"]): int(row["edge_id"]) for row in existing
        }
        with self._conn:
            for tgt, edge_id in existing_targets.items():
                self._conn.execute(
                    "UPDATE edges SET enabled=?, updated_at=? WHERE edge_id=?",
                    (1 if tgt in desired else 0, _now(), edge_id),
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
            )

    def set_incoming(self, *, target_num: int, source_nums: Sequence[int]) -> None:
        target_id = self._node_id_by_num(int(target_num))
        if not target_id:
            raise KeyError(f"Unknown target document number: {target_num}")
        desired_sources: set[str] = set()
        for n in source_nums or []:
            src = self._node_id_by_num(int(n))
            if src and src != target_id:
                desired_sources.add(src)

        existing = self._conn.execute(
            "SELECT edge_id, source_id FROM edges WHERE kind='CITES' AND target_id=?",
            (target_id,),
        ).fetchall()
        existing_sources = {
            str(row["source_id"]): int(row["edge_id"]) for row in existing
        }
        with self._conn:
            for src, edge_id in existing_sources.items():
                self._conn.execute(
                    "UPDATE edges SET enabled=?, updated_at=? WHERE edge_id=?",
                    (1 if src in desired_sources else 0, _now(), edge_id),
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
            )

    def set_assigned(self, *, doc_num: int, assigned: bool) -> None:
        node_id = self._node_id_by_num(int(doc_num))
        if not node_id:
            raise KeyError(f"Unknown document number: {doc_num}")
        self._upsert_node(
            node_id=node_id,
            kind="document",
            label=None,
            merge_properties={"workflow_assigned": bool(assigned)},
        )

    # --- Claim graph (Phase 09) -------------------------------------------

    def _get_edge(self, edge_id: int) -> Optional[dict]:
        row = self._conn.execute(
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
            FROM edges
            WHERE edge_id=?
            """,
            (int(edge_id),),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["properties"] = json.loads(data.pop("properties_json") or "{}")
        except Exception:
            data["properties"] = {}
        data["edge_id"] = int(data.get("edge_id") or 0)
        data["enabled"] = bool(int(data.get("enabled") or 0))
        return data

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
        rows = self._conn.execute(
            "SELECT node_id FROM nodes WHERE kind='claim' ORDER BY num ASC"
        ).fetchall()
        nodes: List[dict] = []
        for row in rows:
            node = self._get_node(str(row["node_id"]))
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
        """Create or re-enable a materialized claim->claim edge.

        Uses edges(kind='CLAIM_LINK') and stores provenance in properties_json.
        """
        src = str(source_claim_id or "").strip()
        tgt = str(target_claim_id or "").strip()
        if not src or not tgt:
            raise ValueError("source_claim_id and target_claim_id are required")
        if src == tgt:
            raise ValueError("Cannot link a claim to itself")

        now = _now()
        row = self._conn.execute(
            """
            SELECT edge_id, properties_json
            FROM edges
            WHERE source_id=? AND target_id=? AND kind='CLAIM_LINK' AND ref_id=''
            """,
            (src, tgt),
        ).fetchone()

        props: dict = {}
        if row:
            try:
                props = json.loads(row["properties_json"] or "{}")
            except Exception:
                props = {}

        props["source"] = str(source or "").strip() or "auto"
        if creator_uid:
            text = str(creator_uid or "").strip()
            if text:
                # Only set creator for manual edges; keep existing creator if present.
                if props.get("source") == "manual" and not props.get("creator_uid"):
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
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE edges
                    SET enabled=1, properties_json=?, updated_at=?
                    WHERE edge_id=?
                    """,
                    (
                        json.dumps(props, ensure_ascii=True),
                        now,
                        int(row["edge_id"]),
                    ),
                )
            return int(row["edge_id"])

        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO edges(
                    source_id,
                    target_id,
                    kind,
                    ref_id,
                    enabled,
                    properties_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 'CLAIM_LINK', '', 1, ?, ?, ?)
                """,
                (src, tgt, json.dumps(props, ensure_ascii=True), now, now),
            )
        return int(cur.lastrowid)

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
        """Create or re-enable a generic topology edge.

        This is the same settled-topology pattern used for claim links: an edge
        row (with enabled flag) plus per-reviewer votes stored in edge_votes.
        """
        src = str(source_id or "").strip()
        tgt = str(target_id or "").strip()
        kind_norm = str(kind or "").strip()
        if not src or not tgt or not kind_norm:
            raise ValueError("source_id, target_id, and kind are required")
        if src == tgt:
            raise ValueError("Cannot link a node to itself")

        now = _now()
        row = self._conn.execute(
            """
            SELECT edge_id, properties_json
            FROM edges
            WHERE source_id=? AND target_id=? AND kind=? AND ref_id=''
            """,
            (src, tgt, kind_norm),
        ).fetchone()

        props: dict = {}
        if row:
            try:
                props = json.loads(row["properties_json"] or "{}")
            except Exception:
                props = {}

        props["source"] = str(source or "").strip() or "auto"
        if creator_uid:
            text = str(creator_uid or "").strip()
            if text:
                if props.get("source") == "manual" and not props.get("creator_uid"):
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
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE edges
                    SET enabled=?, properties_json=?, updated_at=?
                    WHERE edge_id=?
                    """,
                    (
                        1 if enabled else 0,
                        json.dumps(props, ensure_ascii=True),
                        now,
                        int(row["edge_id"]),
                    ),
                )
            return int(row["edge_id"])

        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO edges(
                    source_id,
                    target_id,
                    kind,
                    ref_id,
                    enabled,
                    properties_json,
                    created_at,
                    updated_at
                ) VALUES(?, ?, ?, '', ?, ?, ?, ?)
                """,
                (
                    src,
                    tgt,
                    kind_norm,
                    1 if enabled else 0,
                    json.dumps(props, ensure_ascii=True),
                    now,
                    now,
                ),
            )
        return int(cur.lastrowid)

    def upsert_manual_work_cites_work(
        self,
        *,
        citing_ingest_id: str,
        cited_ingest_id: str,
        reviewer_uid: str,
        enabled: bool = True,
    ) -> int:
        """Create/enable a manual work->work citation edge.

        This is a work-level override for cases where extraction/resolution is
        wrong or incomplete. It uses the generic topology edge mechanism.
        """
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
        kind_norm = str(kind or "").strip()
        if not kind_norm:
            return []
        where_enabled = "" if include_disabled else "AND enabled=1"
        rows = self._conn.execute(
            f"""
            SELECT edge_id
            FROM edges
            WHERE kind=?
              {where_enabled}
            ORDER BY edge_id ASC
            """,
            (kind_norm,),
        ).fetchall()
        out: List[dict] = []
        for r in rows:
            edge = self._get_edge(int(r["edge_id"]))
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
        kind_norm = str(kind or "").strip()
        if not kind_norm:
            return []
        clauses = ["kind=?"]
        params: List[Any] = [kind_norm]
        if source_id is not None:
            clauses.append("source_id=?")
            params.append(str(source_id))
        if target_id is not None:
            clauses.append("target_id=?")
            params.append(str(target_id))
        if not include_disabled:
            clauses.append("enabled=1")
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT edge_id FROM edges WHERE {where} ORDER BY edge_id ASC",
            tuple(params),
        ).fetchall()
        out: List[dict] = []
        for r in rows:
            edge = self._get_edge(int(r["edge_id"]))
            if edge:
                out.append(edge)
        return out

    def set_edge_enabled(
        self,
        *,
        edge_id: int,
        enabled: bool,
        merge_properties: Optional[dict] = None,
    ) -> None:
        edge = self._get_edge(int(edge_id))
        if not edge:
            raise KeyError("Edge not found")
        props = edge.get("properties") or {}
        for k, v in (merge_properties or {}).items():
            if v is None:
                continue
            props[k] = v
        with self._conn:
            self._conn.execute(
                (
                    "UPDATE edges SET enabled=?, properties_json=?, updated_at=? "
                    "WHERE edge_id=?"
                ),
                (
                    1 if enabled else 0,
                    json.dumps(props, ensure_ascii=True),
                    _now(),
                    int(edge_id),
                ),
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

        with self._conn:
            self._conn.execute(
                "UPDATE edges SET enabled=0, updated_at=? WHERE edge_id=?",
                (_now(), int(edge_id)),
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
            # Allow votes on disabled edges so consensus can re-enable topology.
            # Exception: manually deleted claim links should stay inert.
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

        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO edge_votes(
                    edge_id,
                    reviewer_uid,
                    verdict,
                    confidence,
                    comment,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id, reviewer_uid) DO UPDATE SET
                    verdict=excluded.verdict,
                    confidence=excluded.confidence,
                    comment=excluded.comment,
                    updated_at=excluded.updated_at
                """,
                (int(edge_id), reviewer, ver, confidence, comment, now),
            )

        row = self._conn.execute(
            """
            SELECT edge_id, reviewer_uid, verdict, confidence, comment, updated_at
            FROM edge_votes
            WHERE edge_id=? AND reviewer_uid=?
            """,
            (int(edge_id), reviewer),
        ).fetchone()
        return (
            dict(row)
            if row
            else {
                "edge_id": int(edge_id),
                "reviewer_uid": reviewer,
                "verdict": ver,
                "confidence": confidence,
                "comment": comment,
                "updated_at": now,
            }
        )

    def list_edge_votes(self, edge_id: int) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT edge_id, reviewer_uid, verdict, confidence, comment, updated_at
            FROM edge_votes
            WHERE edge_id=?
            ORDER BY updated_at DESC, reviewer_uid ASC
            """,
            (int(edge_id),),
        ).fetchall()
        return [dict(r) for r in rows]

    def edge_vote_aggregates(self, edge_id: int) -> dict:
        rows = self._conn.execute(
            """
            SELECT verdict, COUNT(*) AS n
            FROM edge_votes
            WHERE edge_id=?
            GROUP BY verdict
            """,
            (int(edge_id),),
        ).fetchall()
        counts = {str(r["verdict"]): int(r["n"] or 0) for r in rows}
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
        node_id = str(claim_id or "").strip()
        if not node_id:
            return []
        where_enabled = "AND enabled=1" if enabled_only else ""
        rows = self._conn.execute(
            f"""
            SELECT edge_id
            FROM edges
            WHERE kind='CLAIM_LINK'
              {where_enabled}
              AND (source_id=? OR target_id=?)
            ORDER BY edge_id ASC
            """,
            (node_id, node_id),
        ).fetchall()
        edges: List[dict] = []
        want = {str(s).strip() for s in (sources or []) if str(s).strip()}
        for row in rows:
            edge = self._get_edge(int(row["edge_id"]))
            if not edge:
                continue
            if want:
                src = str((edge.get("properties") or {}).get("source") or "")
                if src not in want:
                    continue
            edges.append(edge)
        return edges
