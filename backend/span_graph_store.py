from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import uuid

from backend import text_selectors


_CITE_CLAIM_RE = re.compile(
    r"^cite:(?P<doc>[^:]+):(?P<idx>\d+):(?:(?P<reviewer>[^:]+):)?(?P<seg>.+)$"
)
_SEG_LETTER_RE = re.compile(r"^\d+([a-z])$", re.IGNORECASE)
_SEG_NUM_RE = re.compile(r"^(?:seg-)?(\d+)$", re.IGNORECASE)


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
    CREATE TABLE IF NOT EXISTS citation_span_index (
        ingest_id TEXT NOT NULL,
        citation_index INTEGER NOT NULL,
        target_id TEXT NOT NULL,
        span_id TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(ingest_id, citation_index, target_id)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_citation_span_index_span "
        "ON citation_span_index(span_id)"
    ),
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
        source TEXT,
        source_key TEXT,
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
    CREATE TABLE IF NOT EXISTS review_marks (
        claim_span_id TEXT NOT NULL,
        reviewer_uid TEXT NOT NULL,
        mark TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(claim_span_id, reviewer_uid, mark)
    )
    """,
    (
        "CREATE INDEX IF NOT EXISTS idx_review_marks_reviewer "
        "ON review_marks(reviewer_uid)"
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
        self._ensure_schema_migrations()

    def _ensure_schema_migrations(self) -> None:
        """Apply additive migrations for existing DB files."""

        def _table_exists(name: str) -> bool:
            row = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (str(name),),
            ).fetchone()
            return bool(row)

        def _columns(name: str) -> set[str]:
            try:
                rows = self._conn.execute(f"PRAGMA table_info({name})").fetchall()
            except Exception:
                return set()
            return {str(r[1]) for r in rows}

        # citation_span_index may be missing in older DBs.
        if not _table_exists("citation_span_index"):
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS citation_span_index (
                        ingest_id TEXT NOT NULL,
                        citation_index INTEGER NOT NULL,
                        target_id TEXT NOT NULL,
                        span_id TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(ingest_id, citation_index, target_id)
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_citation_span_index_span
                    ON citation_span_index(span_id)
                    """
                )

        # Ensure selection assertion provenance columns exist.
        cols = _columns("assertions")
        for col in ("source", "source_key"):
            if col not in cols:
                with self._conn:
                    self._conn.execute(f"ALTER TABLE assertions ADD COLUMN {col} TEXT")

    def compact_assertions(
        self, *, dry_run: bool = False, aggressive: bool = False
    ) -> dict:
        """Compact the assertions table to reduce duplicate noise.

        - De-duplicates identical assertions.
        - Optionally removes legacy mirrored selection assertions when a
          deterministic `sel:` assertion exists for the same claim_span+reviewer.
        """
        self._ensure_schema_migrations()
        before = int(
            (
                self._conn.execute("SELECT COUNT(1) AS n FROM assertions").fetchone()
                or {}
            )["n"]
        )

        dedupe_count = int(
            (
                self._conn.execute(
                    """
                    SELECT COUNT(1) AS n
                    FROM assertions
                    WHERE rowid NOT IN (
                        SELECT MAX(rowid)
                        FROM assertions
                        GROUP BY reviewer_uid,
                                 claim_span_id,
                                 verdict,
                                 COALESCE(evidence_span_id,''),
                                 COALESCE(evidence_work_id,''),
                                 COALESCE(source,''),
                                 COALESCE(source_key,'')
                    )
                    """
                ).fetchone()
                or {}
            )["n"]
        )

        legacy_count = 0
        if aggressive:
            legacy_count = int(
                (
                    self._conn.execute(
                        """
                        SELECT COUNT(1) AS n
                        FROM assertions a
                        WHERE a.assertion_id NOT LIKE 'sel:%'
                          AND (a.source IS NULL OR a.source='' OR a.source='selection')
                          AND a.claim_span_id IS NOT NULL
                          AND EXISTS (
                            SELECT 1 FROM assertions s
                            WHERE s.assertion_id LIKE 'sel:%'
                              AND s.reviewer_uid = a.reviewer_uid
                              AND s.claim_span_id = a.claim_span_id
                          )
                        """
                    ).fetchone()
                    or {}
                )["n"]
            )

        selection_dupes = 0
        if aggressive:
            selection_dupes = int(
                (
                    self._conn.execute(
                        """
                        SELECT COUNT(1) AS n
                        FROM assertions
                        WHERE assertion_id LIKE 'sel:%'
                          AND rowid NOT IN (
                            SELECT MAX(rowid)
                            FROM assertions
                            WHERE assertion_id LIKE 'sel:%'
                            GROUP BY reviewer_uid, claim_span_id
                          )
                        """
                    ).fetchone()
                    or {}
                )["n"]
            )

        if not dry_run:
            with self._conn:
                self._conn.execute(
                    """
                    DELETE FROM assertions
                    WHERE rowid NOT IN (
                        SELECT MAX(rowid)
                        FROM assertions
                        GROUP BY reviewer_uid,
                                 claim_span_id,
                                 verdict,
                                 COALESCE(evidence_span_id,''),
                                 COALESCE(evidence_work_id,''),
                                 COALESCE(source,''),
                                 COALESCE(source_key,'')
                    )
                    """
                )
                if aggressive:
                    self._conn.execute(
                        """
                        DELETE FROM assertions
                        WHERE assertion_id LIKE 'sel:%'
                          AND rowid NOT IN (
                            SELECT MAX(rowid)
                            FROM assertions
                            WHERE assertion_id LIKE 'sel:%'
                            GROUP BY reviewer_uid, claim_span_id
                          )
                        """
                    )
                    self._conn.execute(
                        """
                        DELETE FROM assertions
                        WHERE assertion_id NOT LIKE 'sel:%'
                          AND (source IS NULL OR source='' OR source='selection')
                          AND claim_span_id IS NOT NULL
                          AND EXISTS (
                            SELECT 1 FROM assertions s
                            WHERE s.assertion_id LIKE 'sel:%'
                              AND s.reviewer_uid = assertions.reviewer_uid
                              AND s.claim_span_id = assertions.claim_span_id
                          )
                        """
                    )

        after = int(
            (
                self._conn.execute("SELECT COUNT(1) AS n FROM assertions").fetchone()
                or {}
            )["n"]
        )

        return {
            "dry_run": bool(dry_run),
            "aggressive": bool(aggressive),
            "before": before,
            "after": after,
            "dedupe_candidates": dedupe_count,
            "legacy_candidates": legacy_count,
            "selection_dupe_candidates": selection_dupes,
            "deleted": max(0, before - after)
            if not dry_run
            else (dedupe_count + legacy_count + selection_dupes),
        }

    # --- Neighborhood runs -------------------------------------------------

    def create_neighborhood_run(
        self,
        *,
        created_by: str,
        context_span_id: Optional[str],
        method: str,
        params: dict,
    ) -> str:
        run_id = f"nbr:{uuid.uuid4()}"
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO neighborhood_runs(
                    run_id, created_by, context_work_id, context_span_id,
                    context_claim_span_id, context_claim_atom_id,
                    method, params_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(created_by or "system"),
                    None,
                    str(context_span_id) if context_span_id else None,
                    None,
                    None,
                    str(method or "unknown"),
                    _json_dumps(params or {}),
                    now,
                ),
            )
        return run_id

    def add_neighborhood_candidates(
        self, *, run_id: str, candidates: List[dict]
    ) -> int:
        now = _now()
        inserted = 0
        with self._conn:
            for idx, cand in enumerate(candidates or []):
                work_id = str((cand or {}).get("work_id") or "").strip()
                if not work_id:
                    continue
                bib = (cand or {}).get("bib_intersection")
                score = (cand or {}).get("abstract_score")
                try:
                    bib_val = int(bib) if bib is not None else None
                except Exception:
                    bib_val = None
                try:
                    score_val = float(score) if score is not None else None
                except Exception:
                    score_val = None
                rank = (cand or {}).get("rank")
                try:
                    rank_val = int(rank) if rank is not None else (idx + 1)
                except Exception:
                    rank_val = idx + 1

                detail = (cand or {}).get("detail")
                detail_json = _json_dumps(detail) if isinstance(detail, dict) else None

                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO neighborhood_candidates(
                        run_id, candidate_work_id, bib_intersection, abstract_score,
                        rank, detail_json
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(run_id),
                        work_id,
                        bib_val,
                        score_val,
                        rank_val,
                        detail_json,
                    ),
                )
                inserted += 1

        # Touch run timestamp for traceability.
        self._conn.execute(
            "UPDATE neighborhood_runs SET created_at=? WHERE run_id=?",
            (now, str(run_id)),
        )
        return inserted

    def get_neighborhood_run(self, *, run_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT run_id, created_by, context_work_id, context_span_id,
                   context_claim_span_id, context_claim_atom_id,
                   method, params_json, created_at
            FROM neighborhood_runs
            WHERE run_id=?
            """,
            (str(run_id),),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["params"] = json.loads(data.pop("params_json") or "{}")
        except Exception:
            data["params"] = {}
        return data

    def list_neighborhood_candidates(
        self, *, run_id: str, limit: int = 50
    ) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT run_id, candidate_work_id, bib_intersection, abstract_score,
                   rank, detail_json
            FROM neighborhood_candidates
            WHERE run_id=?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (str(run_id), int(limit)),
        ).fetchall()
        out: List[dict] = []
        for row in rows:
            data = dict(row)
            raw = data.pop("detail_json", None)
            if raw:
                try:
                    data["detail"] = json.loads(raw)
                except Exception:
                    data["detail"] = None
            else:
                data["detail"] = None
            out.append(data)
        return out

    # --- Indexing adapters ------------------------------------------------

    def index_claim_confirmation(self, payload: dict) -> Optional[dict]:
        """Index a ClaimConfirmationRequest payload into spans + claim spans.

        This is a migration bridge: the current app already produces confirmed
        claim segments keyed by (document_id, sentence_id, citation_index,
        target_id). We create:

        - one `Span(kind=citation_window)` anchored to the ingest (document_id)
        - `ClaimSpan`s for each confirmed claim index
        - a placeholder cited `Work` id derived from (document_id, target_id)

        Anchoring note:
        v1 stores a composite `window_fingerprint` that embeds citation_index and
        target_id so we can resolve spans later even when the raw sentence text
        isn't available at the call site.
        """
        doc_id = str(payload.get("document_id") or "").strip()
        if not doc_id:
            return None
        citation_index = payload.get("citation_index")
        try:
            cite_idx = int(citation_index)
        except Exception:
            return None

        target_id = str(payload.get("target_id") or "").strip() or None
        sentence_text = str(payload.get("sentence_text") or "").strip()
        if not sentence_text:
            return None

        citation_anchor = payload.get("citation_anchor")
        cited_work_id = str(payload.get("cited_work_id") or "").strip() or None
        anchor_quote = None
        anchor_fp = None
        if isinstance(citation_anchor, dict):
            anchor_quote = citation_anchor.get("anchor_quote")
            anchor_fp = citation_anchor.get("window_fingerprint")
            cited_work_id = (
                cited_work_id
                or str(citation_anchor.get("cited_work_id") or "").strip()
                or None
            )

        # Prefer citation anchor selectors/fingerprint when available.
        selector = (
            anchor_quote
            if isinstance(anchor_quote, dict)
            else text_selectors.build_anchor_quote(sentence_text)
        )
        # Stable window fingerprint (do not bake citation_index/target_id).
        window_fingerprint = (
            str(anchor_fp).strip()
            if isinstance(anchor_fp, str) and str(anchor_fp).strip()
            else text_selectors.fingerprint(sentence_text)
        )

        span = self.upsert_span(
            kind="citation_window",
            selector=selector,
            window_fingerprint=window_fingerprint,
            ingest_id=doc_id,
        )

        if target_id:
            self.upsert_citation_span_index(
                ingest_id=doc_id,
                citation_index=cite_idx,
                target_id=target_id,
                span_id=str(span.get("span_id")),
            )

        if not cited_work_id and target_id:
            cited_work_id = f"ref:{doc_id}:{target_id}"
        if cited_work_id:
            self.upsert_work(work_id=cited_work_id)
            self.add_span_cites(
                span_id=str(span.get("span_id")),
                cites=[
                    {
                        "cited_work_id": cited_work_id,
                        "reference_id": target_id,
                        "citation_index": cite_idx,
                    }
                ],
            )

        confirmed = payload.get("confirmed_claims") or []
        raw_indexes: list[int] = []
        for entry in confirmed:
            if not isinstance(entry, dict):
                continue
            try:
                raw_indexes.append(int(entry.get("claim_index")))
            except Exception:
                continue

        # Normalize claim indexes to a stable 1-based order_index for claim spans.
        # The legacy system has mixed 0-based and 1-based claim_index usage.
        shift = 1 if raw_indexes and min(raw_indexes) == 0 else 0
        items = [
            {"order_index": int(idx) + shift, "selector": None} for idx in raw_indexes
        ]
        claim_spans = self.upsert_claim_spans(
            span_id=str(span.get("span_id")), items=items
        )
        return {
            "span_id": str(span.get("span_id")),
            "claim_spans": claim_spans,
            "cited_work_id": cited_work_id,
        }

    def find_citation_span(
        self, *, ingest_id: str, citation_index: int, target_id: Optional[str]
    ) -> Optional[dict]:
        """Best-effort find a citation_window span for a citation context."""
        ingest_id = str(ingest_id or "").strip()
        if not ingest_id:
            return None
        ci = int(citation_index)

        if target_id:
            mapped = self.get_citation_span_id(
                ingest_id=ingest_id,
                citation_index=ci,
                target_id=str(target_id).strip(),
            )
            if mapped:
                return self.get_span(mapped)

        like = f"%ci:{ci}%"
        args: list[str] = [ingest_id, like]
        sql = (
            "SELECT span_id, work_id, ingest_id, kind, selector_json, "
            "window_fingerprint, created_at, updated_at "
            "FROM spans WHERE ingest_id=? AND kind='citation_window' "
            "AND window_fingerprint LIKE ?"
        )
        if target_id:
            sql += " AND window_fingerprint LIKE ?"
            args.append(f"%t:{str(target_id).strip()}%")
        sql += " ORDER BY updated_at DESC LIMIT 1"
        row = self._conn.execute(sql, tuple(args)).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["selector"] = json.loads(data.pop("selector_json") or "{}")
        except Exception:
            data["selector"] = {}
        return data

    # --- Citation span index ------------------------------------------------

    def upsert_citation_span_index(
        self,
        *,
        ingest_id: str,
        citation_index: int,
        target_id: str,
        span_id: str,
    ) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO citation_span_index(
                    ingest_id, citation_index, target_id, span_id, updated_at
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(ingest_id, citation_index, target_id)
                DO UPDATE SET span_id=excluded.span_id, updated_at=excluded.updated_at
                """,
                (
                    str(ingest_id),
                    int(citation_index),
                    str(target_id),
                    str(span_id),
                    now,
                ),
            )

    def get_citation_span_id(
        self, *, ingest_id: str, citation_index: int, target_id: str
    ) -> Optional[str]:
        row = self._conn.execute(
            """
            SELECT span_id FROM citation_span_index
            WHERE ingest_id=? AND citation_index=? AND target_id=?
            LIMIT 1
            """,
            (str(ingest_id), int(citation_index), str(target_id)),
        ).fetchone()
        if not row:
            return None
        return str(row["span_id"] or "").strip() or None

    def get_claim_span(self, *, span_id: str, order_index: int) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT claim_span_id, span_id, order_index, selector_json,
                   created_at, updated_at
            FROM claim_spans
            WHERE span_id=? AND order_index=?
            """,
            (str(span_id), int(order_index)),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        raw = data.pop("selector_json", None)
        if raw:
            try:
                data["selector"] = json.loads(raw)
            except Exception:
                data["selector"] = None
        else:
            data["selector"] = None
        return data

    def list_claim_spans_for_span(self, *, span_id: str) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT claim_span_id, span_id, order_index, selector_json,
                   created_at, updated_at
            FROM claim_spans
            WHERE span_id=?
            ORDER BY order_index ASC
            """,
            (str(span_id),),
        ).fetchall()
        out: List[dict] = []
        for row in rows:
            data = dict(row)
            raw = data.pop("selector_json", None)
            if raw:
                try:
                    data["selector"] = json.loads(raw)
                except Exception:
                    data["selector"] = None
            else:
                data["selector"] = None
            out.append(data)
        return out

    # --- Review marks (unknown vs not_supported) --------------------------

    def mark_checked(self, *, claim_span_id: str, reviewer_uid: str) -> None:
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO review_marks(claim_span_id, reviewer_uid, mark, updated_at)
                VALUES(?, ?, 'checked', ?)
                ON CONFLICT(claim_span_id, reviewer_uid, mark)
                DO UPDATE SET updated_at=excluded.updated_at
                """,
                (str(claim_span_id), str(reviewer_uid), now),
            )

    def has_checked(self, *, claim_span_id: str, reviewer_uid: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM review_marks
            WHERE claim_span_id=? AND reviewer_uid=? AND mark='checked'
            LIMIT 1
            """,
            (str(claim_span_id), str(reviewer_uid)),
        ).fetchone()
        return bool(row)

    # --- Legacy claim_id helpers -----------------------------------------

    def parse_cite_claim_id(self, claim_id: str) -> Optional[dict]:
        match = _CITE_CLAIM_RE.match(str(claim_id or "").strip())
        if not match:
            return None
        doc_id = (match.group("doc") or "").strip()
        idx_raw = (match.group("idx") or "").strip()
        seg_id = (match.group("seg") or "").strip()
        reviewer = (match.group("reviewer") or "").strip() or "default"
        try:
            cite_idx = int(idx_raw)
        except Exception:
            return None

        order_index = None
        m = _SEG_LETTER_RE.match(seg_id)
        if m:
            order_index = 1 + (ord(m.group(1).lower()) - ord("a"))
        else:
            m2 = _SEG_NUM_RE.match(seg_id)
            if m2:
                try:
                    order_index = int(m2.group(1))
                except Exception:
                    order_index = None
        return {
            "document_id": doc_id,
            "citation_index": cite_idx,
            "reviewer_uid": reviewer,
            "segment_id": seg_id,
            "order_index": order_index,
        }

    # --- Status computation ------------------------------------------------

    def claim_span_status(self, *, claim_span_id: str, reviewer_uid: str) -> dict:
        """Compute the v1 status lattice for a claim span and reviewer.

        The current evidence selection is represented as a deterministic `sel:*`
        assertion. Non-selection assertions (manual) also contribute.
        """
        reviewer_uid = str(reviewer_uid or "").strip() or "default"
        rows = self.list_assertions_for_claim_span(
            claim_span_id=str(claim_span_id),
            reviewer_uid=reviewer_uid,
        )
        current = self.get_current_selection_assertion(
            claim_span_id=str(claim_span_id),
            reviewer_uid=reviewer_uid,
        )

        non_selection = [
            r for r in rows if not str(r.get("assertion_id") or "").startswith("sel:")
        ]

        support_pairs = {
            (
                str(r.get("evidence_span_id") or ""),
                str(r.get("evidence_work_id") or ""),
            )
            for r in non_selection
            if (r.get("verdict") == "support")
        }
        contra_pairs = {
            (
                str(r.get("evidence_span_id") or ""),
                str(r.get("evidence_work_id") or ""),
            )
            for r in non_selection
            if (r.get("verdict") == "contradict")
        }
        support_pairs.discard(("", ""))
        contra_pairs.discard(("", ""))

        n_support = len(support_pairs) + (
            1 if (current or {}).get("verdict") == "support" else 0
        )
        n_contra = len(contra_pairs) + (
            1 if (current or {}).get("verdict") == "contradict" else 0
        )

        checked = self.has_checked(
            claim_span_id=str(claim_span_id),
            reviewer_uid=reviewer_uid,
        )
        checked = checked or bool(rows) or bool(current)

        if n_support and n_contra:
            status = "contested"
        elif n_support:
            status = "supported"
        elif n_contra:
            status = "contradicted"
        else:
            status = "not_supported" if checked else "unknown"

        return {
            "claim_span_id": str(claim_span_id),
            "reviewer_uid": reviewer_uid,
            "status": status,
            "checked": bool(checked),
            "n_support": int(n_support),
            "n_contradict": int(n_contra),
        }

    def span_status(self, *, span_id: str, reviewer_uid: str) -> dict:
        reviewer_uid = str(reviewer_uid or "").strip() or "default"
        claim_spans = self.list_claim_spans_for_span(span_id=str(span_id))
        child = [
            self.claim_span_status(
                claim_span_id=str(cs.get("claim_span_id") or ""),
                reviewer_uid=reviewer_uid,
            )
            for cs in claim_spans
            if str(cs.get("claim_span_id") or "").strip()
        ]

        statuses = [c.get("status") for c in child]
        if "contradicted" in statuses:
            status = "contradicted"
        elif "not_supported" in statuses:
            status = "not_supported"
        elif "contested" in statuses:
            status = "contested"
        elif "unknown" in statuses:
            status = "unknown"
        else:
            status = "supported" if statuses else "unknown"

        counts = {
            "supported": 0,
            "contradicted": 0,
            "contested": 0,
            "not_supported": 0,
            "unknown": 0,
        }
        for s in statuses:
            if s in counts:
                counts[str(s)] += 1

        return {
            "span_id": str(span_id),
            "reviewer_uid": reviewer_uid,
            "status": status,
            "n_claim_spans": int(len(statuses)),
            "n_supported": int(counts["supported"]),
            "n_contradicted": int(counts["contradicted"]),
            "n_contested": int(counts["contested"]),
            "n_not_supported": int(counts["not_supported"]),
            "n_unknown": int(counts["unknown"]),
        }

    def span_bundle(
        self,
        *,
        span_id: str,
        reviewer_uid: str,
        include_history: bool = False,
    ) -> Optional[dict]:
        span = self.get_span(str(span_id))
        if not span:
            return None
        reviewer_uid = str(reviewer_uid or "").strip() or "default"

        cites = self.list_span_cites(span_id=str(span_id))
        for entry in cites:
            cited_work_id = str(entry.get("cited_work_id") or "").strip()
            if not cited_work_id:
                continue
            role = self.get_span_cite_role(
                span_id=str(span_id),
                cited_work_id=cited_work_id,
                reviewer_uid=reviewer_uid,
            )
            entry["role"] = role or "unknown"

        claim_spans = self.list_claim_spans_for_span(span_id=str(span_id))
        claim_spans_payload: List[dict] = []
        for cs in claim_spans:
            cs_id = str(cs.get("claim_span_id") or "").strip()
            if not cs_id:
                continue
            status = self.claim_span_status(
                claim_span_id=cs_id,
                reviewer_uid=reviewer_uid,
            )
            current = self.get_current_selection_assertion(
                claim_span_id=cs_id,
                reviewer_uid=reviewer_uid,
            )
            history_n_total = 0
            if include_history:
                history_n_total = len(
                    self.list_assertions_for_claim_span(
                        claim_span_id=cs_id,
                        reviewer_uid=reviewer_uid,
                    )
                )
            claim_spans_payload.append(
                {
                    "claim_span_id": cs_id,
                    "span_id": str(span_id),
                    "order_index": int(cs.get("order_index") or 0),
                    "status": status.get("status"),
                    "checked": bool(status.get("checked")),
                    "n_support": int(status.get("n_support") or 0),
                    "n_contradict": int(status.get("n_contradict") or 0),
                    "current": current,
                    "history_n_total": int(history_n_total),
                }
            )
        claim_spans_payload.sort(key=lambda item: int(item.get("order_index") or 0))

        return {
            "span": span,
            "reviewer_uid": reviewer_uid,
            "span_status": self.span_status(
                span_id=str(span_id),
                reviewer_uid=reviewer_uid,
            ),
            "cites": cites,
            "claim_spans": claim_spans_payload,
            "include_history": bool(include_history),
        }

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

    def list_span_cites(self, *, span_id: str) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT span_id, cited_work_id, reference_id, citation_index, created_at
            FROM span_cites
            WHERE span_id=?
            ORDER BY created_at ASC
            """,
            (str(span_id),),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_span_cite_role(
        self, *, span_id: str, cited_work_id: str, reviewer_uid: str
    ) -> Optional[str]:
        row = self._conn.execute(
            """
            SELECT role FROM span_cite_roles
            WHERE span_id=? AND cited_work_id=? AND reviewer_uid=?
            LIMIT 1
            """,
            (str(span_id), str(cited_work_id), str(reviewer_uid)),
        ).fetchone()
        if not row:
            return None
        return str(row["role"] or "").strip() or None

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
        source = payload.get("source")
        source_key = payload.get("source_key")
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO assertions(
                    assertion_id, reviewer_uid, verdict, confidence, comment,
                    claim_atom_id, claim_span_id,
                    evidence_span_id, evidence_work_id,
                    source, source_key,
                    created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(source) if source is not None else None,
                    str(source_key) if source_key is not None else None,
                    now,
                ),
            )
        return self.get_assertion(str(assertion_id)) or {"assertion_id": assertion_id}

    def upsert_selection_assertion(
        self,
        *,
        claim_id: str,
        reviewer_uid: str,
        verdict: str,
        claim_span_id: str,
        evidence_span_id: Optional[str],
        evidence_work_id: Optional[str],
        comment: Optional[str],
    ) -> dict:
        """Idempotently persist the current selection verdict as an assertion.

        Evidence selections are "current state" (one per claim). We write them
        as a single upserted assertion keyed by (reviewer_uid, claim_span_id)
        to avoid unbounded duplicate rows.
        """
        key = "|".join(["selection", str(reviewer_uid), str(claim_span_id)])
        assertion_id = f"sel:{_sha256(key)}"
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO assertions(
                    assertion_id, reviewer_uid, verdict, confidence, comment,
                    claim_atom_id, claim_span_id,
                    evidence_span_id, evidence_work_id,
                    source, source_key,
                    created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assertion_id,
                    str(reviewer_uid),
                    str(verdict),
                    None,
                    comment,
                    None,
                    str(claim_span_id),
                    str(evidence_span_id) if evidence_span_id else None,
                    str(evidence_work_id) if evidence_work_id else None,
                    "selection",
                    str(claim_id),
                    now,
                ),
            )
        return self.get_assertion(assertion_id) or {"assertion_id": assertion_id}

    def get_assertion(self, assertion_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT assertion_id, reviewer_uid, verdict, confidence, comment,
                   claim_atom_id, claim_span_id, evidence_span_id, evidence_work_id,
                   source, source_key,
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
                       source, source_key,
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
                       source, source_key,
                       created_at
                FROM assertions
                WHERE claim_span_id=?
                ORDER BY created_at DESC
                """,
                (str(claim_span_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_current_selection_assertion(
        self, *, claim_span_id: str, reviewer_uid: str
    ) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT assertion_id, reviewer_uid, verdict, confidence, comment,
                   claim_atom_id, claim_span_id, evidence_span_id, evidence_work_id,
                   source, source_key,
                   created_at
            FROM assertions
            WHERE claim_span_id=?
              AND reviewer_uid=?
              AND assertion_id LIKE 'sel:%'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(claim_span_id), str(reviewer_uid)),
        ).fetchone()
        return dict(row) if row else None
