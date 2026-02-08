from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _norm_ws(value: Optional[str]) -> str:
    return " ".join(str(value or "").split()).strip()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _span_id(
    *,
    anchor_id: str,
    kind: str,
    selector: dict,
    window_fingerprint: Optional[str],
) -> str:
    sel = selector or {}
    exact = _norm_ws(sel.get("exact"))
    prefix = _norm_ws(sel.get("prefix"))
    suffix = _norm_ws(sel.get("suffix"))
    fp = _norm_ws(window_fingerprint)
    raw = "|".join([anchor_id, kind, exact, prefix, suffix, fp])
    return f"span:{_sha256(raw)}"


def _claim_span_id(*, span_id: str, order_index: int, selector: Optional[dict]) -> str:
    sel = selector or {}
    exact = _norm_ws(sel.get("exact"))
    prefix = _norm_ws(sel.get("prefix"))
    suffix = _norm_ws(sel.get("suffix"))
    raw = "|".join([span_id, str(int(order_index)), exact, prefix, suffix])
    return f"claimspan:{_sha256(raw)}"


SCHEMA: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS works (
        work_id TEXT PRIMARY KEY,
        doi TEXT,
        openalex_id TEXT,
        title TEXT,
        authors_json TEXT,
        year TEXT,
        abstract TEXT,
        abstract_source TEXT,
        abstract_embedding_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi)",
    "CREATE INDEX IF NOT EXISTS idx_works_openalex ON works(openalex_id)",
    """
    CREATE TABLE IF NOT EXISTS work_cites (
        citing_work_id TEXT NOT NULL,
        cited_work_id TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(citing_work_id, cited_work_id, source)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_work_cites_citing "
        "ON work_cites(citing_work_id)"
    ),
    ("CREATE INDEX IF NOT EXISTS idx_work_cites_cited " "ON work_cites(cited_work_id)"),
    """
    CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT PRIMARY KEY,
        work_id TEXT,
        ingest_id TEXT,
        kind TEXT NOT NULL,
        selector_json TEXT NOT NULL,
        window_fingerprint TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_spans_work_id ON spans(work_id)",
    "CREATE INDEX IF NOT EXISTS idx_spans_ingest_id ON spans(ingest_id)",
    """
    CREATE TABLE IF NOT EXISTS span_cites (
        span_id TEXT NOT NULL,
        cited_work_id TEXT NOT NULL,
        reference_id TEXT,
        citation_index INTEGER,
        created_at TEXT NOT NULL,
        PRIMARY KEY(span_id, cited_work_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_span_cites_cited ON span_cites(cited_work_id)",
    """
    CREATE TABLE IF NOT EXISTS span_cite_roles (
        span_id TEXT NOT NULL,
        cited_work_id TEXT NOT NULL,
        reviewer_uid TEXT NOT NULL,
        role TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(span_id, cited_work_id, reviewer_uid)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_span_cite_roles_reviewer "
        "ON span_cite_roles(reviewer_uid)"
    ),
    """
    CREATE TABLE IF NOT EXISTS claim_spans (
        claim_span_id TEXT PRIMARY KEY,
        span_id TEXT NOT NULL,
        order_index INTEGER NOT NULL,
        selector_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(span_id, order_index)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_claim_spans_span "
        "ON claim_spans(span_id, order_index)"
    ),
    """
    CREATE TABLE IF NOT EXISTS claim_atoms (
        claim_atom_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        supersedes_id TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_claim_atoms_created_by ON claim_atoms(created_by)",
    """
    CREATE TABLE IF NOT EXISTS claim_span_atoms (
        claim_span_id TEXT NOT NULL,
        claim_atom_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(claim_span_id, claim_atom_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS assertions (
        assertion_id TEXT PRIMARY KEY,
        reviewer_uid TEXT NOT NULL,
        verdict TEXT NOT NULL,
        confidence REAL,
        comment TEXT,
        claim_atom_id TEXT,
        claim_span_id TEXT,
        evidence_span_id TEXT,
        evidence_work_id TEXT,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_assertions_reviewer ON assertions(reviewer_uid)",
    "CREATE INDEX IF NOT EXISTS idx_assertions_claim_span ON assertions(claim_span_id)",
    "CREATE INDEX IF NOT EXISTS idx_assertions_claim_atom ON assertions(claim_atom_id)",
    (
        "CREATE INDEX IF NOT EXISTS idx_assertions_evidence_span "
        "ON assertions(evidence_span_id)"
    ),
    """
    CREATE TABLE IF NOT EXISTS neighborhood_runs (
        run_id TEXT PRIMARY KEY,
        created_by TEXT,
        context_work_id TEXT,
        context_span_id TEXT,
        context_claim_span_id TEXT,
        context_claim_atom_id TEXT,
        method TEXT NOT NULL,
        params_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_neighborhood_runs_work "
        "ON neighborhood_runs(context_work_id)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_neighborhood_runs_span "
        "ON neighborhood_runs(context_span_id)"
    ),
    """
    CREATE TABLE IF NOT EXISTS neighborhood_candidates (
        run_id TEXT NOT NULL,
        candidate_work_id TEXT NOT NULL,
        bib_intersection INTEGER,
        abstract_score REAL,
        rank INTEGER,
        detail_json TEXT,
        PRIMARY KEY(run_id, candidate_work_id)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_neighborhood_candidates_work "
        "ON neighborhood_candidates(candidate_work_id)"
    ),
]


class SpanGraphStore:
    def __init__(self, db_path: Path):
        """SQLite-backed store for span-first graph structures."""
        self._path = _ensure_parent(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            for ddl in SCHEMA:
                self._conn.execute(ddl)

    # --- Work -------------------------------------------------------------

    def upsert_work(
        self,
        *,
        work_id: str,
        doi: Optional[str] = None,
        openalex_id: Optional[str] = None,
        title: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        year: Optional[str] = None,
        abstract: Optional[str] = None,
        abstract_source: Optional[str] = None,
    ) -> dict:
        now = _now()
        authors_json = _json_dumps(list(authors or []))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO works(
                    work_id, doi, openalex_id, title, authors_json, year,
                    abstract, abstract_source, abstract_embedding_json,
                    created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(work_id) DO UPDATE SET
                    doi=COALESCE(excluded.doi, doi),
                    openalex_id=COALESCE(excluded.openalex_id, openalex_id),
                    title=COALESCE(excluded.title, title),
                    authors_json=CASE
                        WHEN excluded.authors_json IS NOT NULL
                        THEN excluded.authors_json
                        ELSE authors_json
                    END,
                    year=COALESCE(excluded.year, year),
                    abstract=COALESCE(excluded.abstract, abstract),
                    abstract_source=COALESCE(excluded.abstract_source, abstract_source),
                    updated_at=excluded.updated_at
                """,
                (
                    str(work_id),
                    doi,
                    openalex_id,
                    title,
                    authors_json,
                    year,
                    abstract,
                    abstract_source,
                    None,
                    now,
                    now,
                ),
            )
        return self.get_work(str(work_id)) or {"work_id": str(work_id)}

    def get_work(self, work_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT work_id, doi, openalex_id, title, authors_json, year,
                   abstract, abstract_source, abstract_embedding_json,
                   created_at, updated_at
            FROM works WHERE work_id=?
            """,
            (str(work_id),),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["authors"] = json.loads(data.pop("authors_json") or "[]")
        except Exception:
            data["authors"] = []
        data.pop("abstract_embedding_json", None)
        return data

    # --- Spans ------------------------------------------------------------

    def upsert_span(
        self,
        *,
        kind: str,
        selector: dict,
        window_fingerprint: Optional[str],
        ingest_id: Optional[str] = None,
        work_id: Optional[str] = None,
    ) -> dict:
        ingest_id = str(ingest_id or "").strip() or None
        work_id = str(work_id or "").strip() or None
        if bool(ingest_id) == bool(work_id):
            raise ValueError("Exactly one of ingest_id or work_id is required")
        anchor_id = f"ingest:{ingest_id}" if ingest_id else f"work:{work_id}"
        span_id = _span_id(
            anchor_id=anchor_id,
            kind=str(kind or "").strip(),
            selector=selector or {},
            window_fingerprint=window_fingerprint,
        )
        now = _now()
        selector_json = _json_dumps(selector or {})
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO spans(
                    span_id, work_id, ingest_id, kind, selector_json,
                    window_fingerprint, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(span_id) DO UPDATE SET
                    work_id=COALESCE(excluded.work_id, work_id),
                    ingest_id=COALESCE(excluded.ingest_id, ingest_id),
                    kind=excluded.kind,
                    selector_json=excluded.selector_json,
                    window_fingerprint=COALESCE(
                        excluded.window_fingerprint,
                        window_fingerprint
                    ),
                    updated_at=excluded.updated_at
                """,
                (
                    span_id,
                    work_id,
                    ingest_id,
                    str(kind),
                    selector_json,
                    _norm_ws(window_fingerprint) or None,
                    now,
                    now,
                ),
            )
        return self.get_span(span_id) or {"span_id": span_id}

    def get_span(self, span_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT span_id, work_id, ingest_id, kind, selector_json,
                   window_fingerprint, created_at, updated_at
            FROM spans WHERE span_id=?
            """,
            (str(span_id),),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["selector"] = json.loads(data.pop("selector_json") or "{}")
        except Exception:
            data["selector"] = {}
        return data

    def add_span_cites(
        self,
        *,
        span_id: str,
        cites: Iterable[dict],
    ) -> int:
        now = _now()
        inserted = 0
        with self._conn:
            for entry in cites or []:
                cited_work_id = str((entry or {}).get("cited_work_id") or "").strip()
                if not cited_work_id:
                    continue
                ref_id = (entry or {}).get("reference_id")
                cite_idx = (entry or {}).get("citation_index")
                try:
                    cite_idx_val = int(cite_idx) if cite_idx is not None else None
                except Exception:
                    cite_idx_val = None
                cur = self._conn.execute(
                    """
                    INSERT OR REPLACE INTO span_cites(
                        span_id, cited_work_id, reference_id, citation_index, created_at
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (str(span_id), cited_work_id, ref_id, cite_idx_val, now),
                )
                inserted += int(cur.rowcount or 0)
        return inserted

    def set_span_cite_role(
        self,
        *,
        span_id: str,
        cited_work_id: str,
        reviewer_uid: str,
        role: str,
    ) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO span_cite_roles(
                    span_id, cited_work_id, reviewer_uid, role, updated_at
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(span_id, cited_work_id, reviewer_uid)
                DO UPDATE SET role=excluded.role, updated_at=excluded.updated_at
                """,
                (
                    str(span_id),
                    str(cited_work_id),
                    str(reviewer_uid),
                    str(role),
                    now,
                ),
            )

    # --- ClaimSpans -------------------------------------------------------

    def upsert_claim_spans(self, *, span_id: str, items: Sequence[dict]) -> List[dict]:
        now = _now()
        out: List[dict] = []
        with self._conn:
            for item in items or []:
                try:
                    order_index = int((item or {}).get("order_index"))
                except Exception:
                    continue
                selector = (item or {}).get("selector")
                selector_json = _json_dumps(selector) if selector is not None else None
                claim_span_id = _claim_span_id(
                    span_id=str(span_id),
                    order_index=order_index,
                    selector=selector if isinstance(selector, dict) else None,
                )
                self._conn.execute(
                    """
                    INSERT INTO claim_spans(
                        claim_span_id, span_id, order_index, selector_json,
                        created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(span_id, order_index) DO UPDATE SET
                        selector_json=COALESCE(excluded.selector_json, selector_json),
                        updated_at=excluded.updated_at
                    """,
                    (
                        claim_span_id,
                        str(span_id),
                        order_index,
                        selector_json,
                        now,
                        now,
                    ),
                )
                out.append(
                    {
                        "claim_span_id": claim_span_id,
                        "span_id": str(span_id),
                        "order_index": order_index,
                        "selector": selector if isinstance(selector, dict) else None,
                    }
                )
        return out

    # --- Assertions -------------------------------------------------------

    def create_assertion(self, *, payload: dict) -> dict:
        assertion_id = str(payload.get("assertion_id") or "").strip() or None
        if not assertion_id:
            assertion_id = f"assert:{_sha256(_json_dumps(payload) + '|' + _now())}"
        reviewer_uid = str(payload.get("reviewer_uid") or "").strip()
        verdict = str(payload.get("verdict") or "").strip()
        if not reviewer_uid:
            raise ValueError("reviewer_uid is required")
        if not verdict:
            raise ValueError("verdict is required")

        claim_atom_id = payload.get("claim_atom_id")
        claim_span_id = payload.get("claim_span_id")
        if bool(claim_atom_id) == bool(claim_span_id):
            raise ValueError(
                "Exactly one of claim_atom_id or claim_span_id is required"
            )

        evidence_span_id = payload.get("evidence_span_id")
        evidence_work_id = payload.get("evidence_work_id")
        if not evidence_span_id and not evidence_work_id:
            raise ValueError("evidence_span_id or evidence_work_id is required")

        confidence = payload.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else None
        except Exception:
            confidence_val = None

        comment = payload.get("comment")
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO assertions(
                    assertion_id, reviewer_uid, verdict, confidence, comment,
                    claim_atom_id, claim_span_id,
                    evidence_span_id, evidence_work_id,
                    created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(assertion_id),
                    reviewer_uid,
                    verdict,
                    confidence_val,
                    comment,
                    claim_atom_id,
                    claim_span_id,
                    evidence_span_id,
                    evidence_work_id,
                    now,
                ),
            )
        return self.get_assertion(str(assertion_id)) or {"assertion_id": assertion_id}

    def get_assertion(self, assertion_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT assertion_id, reviewer_uid, verdict, confidence, comment,
                   claim_atom_id, claim_span_id, evidence_span_id, evidence_work_id,
                   created_at
            FROM assertions WHERE assertion_id=?
            """,
            (str(assertion_id),),
        ).fetchone()
        return dict(row) if row else None

    def list_assertions_for_claim_span(
        self, *, claim_span_id: str, reviewer_uid: Optional[str] = None
    ) -> List[dict]:
        if reviewer_uid:
            rows = self._conn.execute(
                """
                SELECT assertion_id, reviewer_uid, verdict, confidence, comment,
                       claim_atom_id, claim_span_id, evidence_span_id, evidence_work_id,
                       created_at
                FROM assertions
                WHERE claim_span_id=? AND reviewer_uid=?
                ORDER BY created_at DESC
                """,
                (str(claim_span_id), str(reviewer_uid)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT assertion_id, reviewer_uid, verdict, confidence, comment,
                       claim_atom_id, claim_span_id, evidence_span_id, evidence_work_id,
                       created_at
                FROM assertions
                WHERE claim_span_id=?
                ORDER BY created_at DESC
                """,
                (str(claim_span_id),),
            ).fetchall()
        return [dict(r) for r in rows]
