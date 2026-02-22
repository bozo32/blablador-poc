from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from backend import text_selectors
from backend.db.pg import connect
from backend.settings import AppSettings, settings as app_settings


_CITE_CLAIM_RE = re.compile(
    r"^cite:(?P<doc>[^:]+):(?P<idx>\d+):(?:(?P<reviewer>[^:]+):)?(?P<seg>.+)$"
)
_SEG_LETTER_RE = re.compile(r"^\d+([a-z])$", re.IGNORECASE)
_SEG_NUM_RE = re.compile(r"^(?:seg-)?(\d+)$", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


class SpanGraphStore:
    """Postgres-backed span graph store.

    Phase 09.3 removes durable local SQLite span-graph state (`data/graph.db`).
    The constructor retains the legacy `db_path` parameter for API compatibility.
    """

    def __init__(
        self, db_path: Optional[Path] = None, *, settings: AppSettings = app_settings
    ) -> None:
        """Create a Postgres-backed span graph store."""
        self._db_path = db_path
        self.settings = settings

    def _project_id(self) -> str:
        return str(getattr(self.settings, "DEFAULT_PROJECT_ID", "default") or "default")

    def wipe(self) -> None:
        """Delete all span-graph rows for the active project (keeps schema)."""
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM span_graph_neighborhood_candidates "
                    "WHERE project_id=%s",
                    (pid,),
                )
                cur.execute(
                    "DELETE FROM span_graph_neighborhood_runs WHERE project_id=%s",
                    (pid,),
                )
                cur.execute(
                    "DELETE FROM span_graph_review_marks WHERE project_id=%s", (pid,)
                )
                cur.execute(
                    "DELETE FROM span_graph_assertions WHERE project_id=%s", (pid,)
                )
                cur.execute(
                    "DELETE FROM span_graph_claim_span_atoms WHERE project_id=%s",
                    (pid,),
                )
                cur.execute(
                    "DELETE FROM span_graph_claim_atoms WHERE project_id=%s", (pid,)
                )
                cur.execute(
                    "DELETE FROM span_graph_claim_spans WHERE project_id=%s", (pid,)
                )
                cur.execute(
                    "DELETE FROM span_graph_span_cite_roles WHERE project_id=%s", (pid,)
                )
                cur.execute(
                    "DELETE FROM span_graph_citation_span_index WHERE project_id=%s",
                    (pid,),
                )
                cur.execute(
                    "DELETE FROM span_graph_span_cites WHERE project_id=%s", (pid,)
                )
                cur.execute("DELETE FROM span_graph_spans WHERE project_id=%s", (pid,))
                cur.execute(
                    "DELETE FROM span_graph_work_cites WHERE project_id=%s", (pid,)
                )
                cur.execute("DELETE FROM span_graph_works WHERE project_id=%s", (pid,))

    def compact_assertions(
        self, *, dry_run: bool = False, aggressive: bool = False
    ) -> dict:
        """Compact the assertions table to reduce duplicate noise.

        Mirrors the legacy SQLite behavior:
        - De-duplicate identical assertions (keep most recent per key).
        - Optionally remove legacy mirrored selection assertions when a deterministic
          `sel:` assertion exists for the same claim_span+reviewer.
        """
        pid = self._project_id()
        sel_pat = "sel:%"
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(1) FROM span_graph_assertions WHERE project_id=%s",
                    (pid,),
                )
                before = int((cur.fetchone() or [0])[0] or 0)

                cur.execute(
                    """
                    SELECT COUNT(1)
                    FROM span_graph_assertions a
                    WHERE project_id=%s
                      AND EXISTS (
                        SELECT 1
                        FROM span_graph_assertions b
                        WHERE b.project_id=a.project_id
                          AND b.reviewer_uid=a.reviewer_uid
                          AND COALESCE(b.claim_span_id,'') =
                              COALESCE(a.claim_span_id,'')
                          AND b.verdict=a.verdict
                          AND COALESCE(b.evidence_span_id,'') =
                              COALESCE(a.evidence_span_id,'')
                          AND COALESCE(b.evidence_work_id,'') =
                              COALESCE(a.evidence_work_id,'')
                          AND COALESCE(b.source,'') = COALESCE(a.source,'')
                          AND COALESCE(b.source_key,'') = COALESCE(a.source_key,'')
                          AND (
                            b.created_at > a.created_at
                            OR (
                              b.created_at=a.created_at
                              AND b.assertion_id > a.assertion_id
                            )
                          )
                      )
                    """,
                    (pid,),
                )
                dedupe_candidates = int((cur.fetchone() or [0])[0] or 0)

                legacy_candidates = 0
                selection_dupe_candidates = 0
                if aggressive:
                    cur.execute(
                        """
                        SELECT COUNT(1)
                        FROM span_graph_assertions a
                        WHERE a.project_id=%s
                          AND a.assertion_id NOT LIKE %s
                          AND (
                            a.source IS NULL
                            OR a.source=''
                            OR a.source='selection'
                          )
                          AND a.claim_span_id IS NOT NULL
                          AND EXISTS (
                            SELECT 1 FROM span_graph_assertions s
                            WHERE s.project_id=a.project_id
                              AND s.assertion_id LIKE %s
                              AND s.reviewer_uid=a.reviewer_uid
                              AND s.claim_span_id=a.claim_span_id
                          )
                        """,
                        (pid, sel_pat, sel_pat),
                    )
                    legacy_candidates = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute(
                        """
                        SELECT COUNT(1)
                        FROM span_graph_assertions a
                        WHERE a.project_id=%s
                          AND a.assertion_id LIKE %s
                          AND EXISTS (
                            SELECT 1 FROM span_graph_assertions b
                            WHERE b.project_id=a.project_id
                              AND b.assertion_id LIKE %s
                              AND b.reviewer_uid=a.reviewer_uid
                              AND b.claim_span_id=a.claim_span_id
                              AND (
                                b.created_at > a.created_at
                                OR (
                                  b.created_at=a.created_at
                                  AND b.assertion_id > a.assertion_id
                                )
                              )
                          )
                        """,
                        (pid, sel_pat, sel_pat),
                    )
                    selection_dupe_candidates = int((cur.fetchone() or [0])[0] or 0)

        if not dry_run:
            with connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    # De-dupe identical assertions; keep newest row.
                    cur.execute(
                        """
                        DELETE FROM span_graph_assertions a
                        WHERE a.project_id=%s
                          AND EXISTS (
                            SELECT 1
                            FROM span_graph_assertions b
                            WHERE b.project_id=a.project_id
                              AND b.reviewer_uid=a.reviewer_uid
                              AND COALESCE(b.claim_span_id,'') =
                                  COALESCE(a.claim_span_id,'')
                              AND b.verdict=a.verdict
                              AND COALESCE(b.evidence_span_id,'') =
                                  COALESCE(a.evidence_span_id,'')
                              AND COALESCE(b.evidence_work_id,'') =
                                  COALESCE(a.evidence_work_id,'')
                              AND COALESCE(b.source,'') = COALESCE(a.source,'')
                              AND COALESCE(b.source_key,'') = COALESCE(a.source_key,'')
                              AND (
                                b.created_at > a.created_at
                                OR (
                                  b.created_at=a.created_at
                                  AND b.assertion_id > a.assertion_id
                                )
                              )
                          )
                        """,
                        (pid,),
                    )

                    if aggressive:
                        # Keep newest sel:* per (reviewer_uid, claim_span_id).
                        cur.execute(
                            """
                            DELETE FROM span_graph_assertions a
                            WHERE a.project_id=%s
                              AND a.assertion_id LIKE %s
                              AND EXISTS (
                                SELECT 1 FROM span_graph_assertions b
                                WHERE b.project_id=a.project_id
                                  AND b.assertion_id LIKE %s
                                  AND b.reviewer_uid=a.reviewer_uid
                                  AND b.claim_span_id=a.claim_span_id
                                  AND (
                                    b.created_at > a.created_at
                                    OR (
                                      b.created_at=a.created_at
                                      AND b.assertion_id > a.assertion_id
                                    )
                                  )
                              )
                            """,
                            (pid, sel_pat, sel_pat),
                        )
                        # Remove legacy mirrored selection assertions.
                        cur.execute(
                            """
                            DELETE FROM span_graph_assertions a
                            WHERE a.project_id=%s
                              AND a.assertion_id NOT LIKE %s
                              AND (
                                a.source IS NULL
                                OR a.source=''
                                OR a.source='selection'
                              )
                              AND a.claim_span_id IS NOT NULL
                              AND EXISTS (
                                SELECT 1 FROM span_graph_assertions s
                                WHERE s.project_id=a.project_id
                                  AND s.assertion_id LIKE %s
                                  AND s.reviewer_uid=a.reviewer_uid
                                  AND s.claim_span_id=a.claim_span_id
                              )
                            """,
                            (pid, sel_pat, sel_pat),
                        )

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(1) FROM span_graph_assertions WHERE project_id=%s",
                    (pid,),
                )
                after = int((cur.fetchone() or [0])[0] or 0)

        return {
            "dry_run": bool(dry_run),
            "aggressive": bool(aggressive),
            "before": int(before),
            "after": int(after),
            "dedupe_candidates": int(dedupe_candidates),
            "legacy_candidates": int(legacy_candidates),
            "selection_dupe_candidates": int(selection_dupe_candidates),
            "deleted": max(0, int(before) - int(after))
            if not dry_run
            else int(dedupe_candidates + legacy_candidates + selection_dupe_candidates),
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
        pid = self._project_id()
        run_id = f"nbr:{uuid.uuid4()}"
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_neighborhood_runs(
                      project_id, run_id, created_by,
                      context_work_id, context_span_id,
                      context_claim_span_id, context_claim_atom_id,
                      method, params_json, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                    """,
                    (
                        pid,
                        str(run_id),
                        str(created_by or "system"),
                        None,
                        str(context_span_id) if context_span_id else None,
                        None,
                        None,
                        str(method or "unknown"),
                        _json_dumps(params or {}),
                    ),
                )
        return run_id

    def add_neighborhood_candidates(
        self, *, run_id: str, candidates: List[dict]
    ) -> int:
        pid = self._project_id()
        inserted = 0
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                for cand in candidates or []:
                    if not isinstance(cand, dict):
                        continue
                    work_id = str(
                        cand.get("candidate_work_id") or cand.get("work_id") or ""
                    ).strip()
                    if not work_id:
                        continue
                    cur.execute(
                        """
                        INSERT INTO span_graph_neighborhood_candidates(
                          project_id, run_id, candidate_work_id,
                          bib_intersection, abstract_score, rank, detail_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT(project_id, run_id, candidate_work_id) DO UPDATE
                          SET bib_intersection=excluded.bib_intersection,
                              abstract_score=excluded.abstract_score,
                              rank=excluded.rank,
                              detail_json=excluded.detail_json
                        """,
                        (
                            pid,
                            str(run_id),
                            work_id,
                            cand.get("bib_intersection"),
                            cand.get("abstract_score"),
                            cand.get("rank"),
                            _json_dumps(cand.get("detail"))
                            if cand.get("detail") is not None
                            else "null",
                        ),
                    )
                    inserted += 1
        return inserted

    def get_neighborhood_run(self, *, run_id: str) -> Optional[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, created_by, context_work_id, context_span_id,
                           context_claim_span_id, context_claim_atom_id,
                           method, params_json, created_at
                    FROM span_graph_neighborhood_runs
                    WHERE project_id=%s AND run_id=%s
                    """,
                    (pid, str(run_id)),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            run_id2,
            created_by,
            context_work_id,
            context_span_id,
            context_claim_span_id,
            context_claim_atom_id,
            method,
            params_raw,
            created_at,
        ) = row
        params = params_raw if isinstance(params_raw, dict) else {}
        if params_raw is not None and not isinstance(params_raw, dict):
            try:
                params = json.loads(params_raw)
            except Exception:
                params = {}
        return {
            "run_id": str(run_id2),
            "created_by": created_by,
            "context_work_id": context_work_id,
            "context_span_id": context_span_id,
            "context_claim_span_id": context_claim_span_id,
            "context_claim_atom_id": context_claim_atom_id,
            "method": method,
            "params": params,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
        }

    def list_neighborhood_candidates(
        self, *, run_id: str, limit: int = 200
    ) -> List[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      run_id,
                      candidate_work_id,
                      bib_intersection,
                      abstract_score,
                      rank,
                      detail_json
                    FROM span_graph_neighborhood_candidates
                    WHERE project_id=%s AND run_id=%s
                    ORDER BY rank ASC NULLS LAST, candidate_work_id ASC
                    LIMIT %s
                    """,
                    (pid, str(run_id), int(limit)),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (
            run_id2,
            candidate_work_id,
            bib_intersection,
            abstract_score,
            rank,
            detail_raw,
        ) in rows:
            detail = detail_raw if isinstance(detail_raw, dict) else None
            if detail_raw is not None and not isinstance(detail_raw, dict):
                try:
                    detail = json.loads(detail_raw)
                except Exception:
                    detail = None
            out.append(
                {
                    "run_id": str(run_id2),
                    "candidate_work_id": str(candidate_work_id),
                    "bib_intersection": bib_intersection,
                    "abstract_score": abstract_score,
                    "rank": rank,
                    "detail": detail,
                }
            )
        return out

    # --- Indexing adapters ------------------------------------------------

    def index_claim_confirmation(self, payload: dict) -> Optional[dict]:
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

        selector = (
            anchor_quote
            if isinstance(anchor_quote, dict)
            else text_selectors.build_anchor_quote(sentence_text)
        )
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
        pid = self._project_id()
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

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      span_id,
                      work_id,
                      ingest_id,
                      kind,
                      selector_json,
                      window_fingerprint,
                      created_at,
                      updated_at
                    FROM span_graph_spans
                    WHERE project_id=%s AND ingest_id=%s AND kind='citation_window'
                    ORDER BY updated_at DESC
                    LIMIT 25
                    """,
                    (pid, ingest_id),
                )
                rows = cur.fetchall() or []

        # Fallback: prefer any row with matching ci/t in the fingerprint.
        want_ci = f"ci:{ci}"
        want_t = f"t:{str(target_id).strip()}" if target_id else None
        best = None
        for row in rows:
            wf = str(row[5] or "")
            if want_ci in wf and (want_t is None or want_t in wf):
                best = row
                break
        if best is None and rows:
            best = rows[0]
        if not best:
            return None

        (
            span_id,
            work_id,
            ingest_id2,
            kind,
            selector_raw,
            window_fingerprint,
            created_at,
            updated_at,
        ) = best
        selector = selector_raw if isinstance(selector_raw, dict) else {}
        if selector_raw is not None and not isinstance(selector_raw, dict):
            try:
                selector = json.loads(selector_raw)
            except Exception:
                selector = {}
        return {
            "span_id": str(span_id),
            "work_id": work_id,
            "ingest_id": ingest_id2,
            "kind": str(kind),
            "selector": selector,
            "window_fingerprint": window_fingerprint,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    # --- Citation span index ---------------------------------------------

    def upsert_citation_span_index(
        self, *, ingest_id: str, citation_index: int, target_id: str, span_id: str
    ) -> None:
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_citation_span_index(
                      project_id,
                      ingest_id,
                      citation_index,
                      target_id,
                      span_id,
                      updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT(project_id, ingest_id, citation_index, target_id)
                    DO UPDATE
                      SET span_id=excluded.span_id,
                          updated_at=excluded.updated_at
                    """,
                    (
                        pid,
                        str(ingest_id),
                        int(citation_index),
                        str(target_id),
                        str(span_id),
                    ),
                )

    def get_citation_span_id(
        self, *, ingest_id: str, citation_index: int, target_id: str
    ) -> Optional[str]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT span_id
                    FROM span_graph_citation_span_index
                    WHERE project_id=%s
                      AND ingest_id=%s
                      AND citation_index=%s
                      AND target_id=%s
                    LIMIT 1
                    """,
                    (pid, str(ingest_id), int(citation_index), str(target_id)),
                )
                row = cur.fetchone()
        return str(row[0]).strip() if row and row[0] else None

    def get_claim_span(self, *, span_id: str, order_index: int) -> Optional[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      claim_span_id,
                      span_id,
                      order_index,
                      selector_json,
                      created_at,
                      updated_at
                    FROM span_graph_claim_spans
                    WHERE project_id=%s AND span_id=%s AND order_index=%s
                    """,
                    (pid, str(span_id), int(order_index)),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            claim_span_id,
            span_id2,
            order_index2,
            selector_raw,
            created_at,
            updated_at,
        ) = row
        selector = selector_raw if isinstance(selector_raw, dict) else None
        if selector_raw is not None and not isinstance(selector_raw, dict):
            try:
                selector = json.loads(selector_raw)
            except Exception:
                selector = None
        return {
            "claim_span_id": str(claim_span_id),
            "span_id": str(span_id2),
            "order_index": int(order_index2 or 0),
            "selector": selector,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    def list_claim_spans_for_span(self, *, span_id: str) -> List[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      claim_span_id,
                      span_id,
                      order_index,
                      selector_json,
                      created_at,
                      updated_at
                    FROM span_graph_claim_spans
                    WHERE project_id=%s AND span_id=%s
                    ORDER BY order_index ASC
                    """,
                    (pid, str(span_id)),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (
            claim_span_id,
            span_id2,
            order_index2,
            selector_raw,
            created_at,
            updated_at,
        ) in rows:
            selector = selector_raw if isinstance(selector_raw, dict) else None
            if selector_raw is not None and not isinstance(selector_raw, dict):
                try:
                    selector = json.loads(selector_raw)
                except Exception:
                    selector = None
            out.append(
                {
                    "claim_span_id": str(claim_span_id),
                    "span_id": str(span_id2),
                    "order_index": int(order_index2 or 0),
                    "selector": selector,
                    "created_at": created_at.isoformat().replace("+00:00", "Z")
                    if created_at is not None
                    else None,
                    "updated_at": updated_at.isoformat().replace("+00:00", "Z")
                    if updated_at is not None
                    else None,
                }
            )
        return out

    # --- Review marks -----------------------------------------------------

    def mark_checked(self, *, claim_span_id: str, reviewer_uid: str) -> None:
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_review_marks(
                      project_id, claim_span_id, reviewer_uid, mark, updated_at
                    )
                    VALUES (%s, %s, %s, 'checked', now())
                    ON CONFLICT(project_id, claim_span_id, reviewer_uid, mark)
                    DO UPDATE SET updated_at=excluded.updated_at
                    """,
                    (pid, str(claim_span_id), str(reviewer_uid)),
                )

    def has_checked(self, *, claim_span_id: str, reviewer_uid: str) -> bool:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM span_graph_review_marks
                    WHERE project_id=%s
                      AND claim_span_id=%s
                      AND reviewer_uid=%s
                      AND mark='checked'
                    LIMIT 1
                    """,
                    (pid, str(claim_span_id), str(reviewer_uid)),
                )
                row = cur.fetchone()
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

    # --- Works ------------------------------------------------------------

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
        abstract_embedding: Optional[dict] = None,
        merge: Optional[dict] = None,
    ) -> dict:
        pid = self._project_id()
        wid = str(work_id or "").strip()
        if not wid:
            raise ValueError("work_id is required")
        merged: dict = {}
        if isinstance(merge, dict):
            merged.update(merge)
        # Back-compat with older call sites passing explicit kwargs.
        for k, v in {
            "doi": doi,
            "openalex_id": openalex_id,
            "title": title,
            "authors": list(authors) if authors is not None else None,
            "year": year,
            "abstract": abstract,
            "abstract_source": abstract_source,
            "abstract_embedding": abstract_embedding,
        }.items():
            if v is None:
                continue
            merged[k] = v
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_works(
                      project_id,
                      work_id,
                      doi,
                      openalex_id,
                      title,
                      authors_json,
                      year,
                      abstract,
                      abstract_source,
                      abstract_embedding_json,
                      created_at,
                      updated_at
                    )
                    VALUES (
                      %s,
                      %s,
                      %s,
                      %s,
                      %s,
                      %s::jsonb,
                      %s,
                      %s,
                      %s,
                      %s::jsonb,
                      now(),
                      now()
                    )
                    ON CONFLICT(project_id, work_id) DO UPDATE
                      SET doi=COALESCE(span_graph_works.doi, excluded.doi),
                          openalex_id=COALESCE(
                            span_graph_works.openalex_id,
                            excluded.openalex_id
                          ),
                          title=COALESCE(span_graph_works.title, excluded.title),
                          authors_json=COALESCE(
                            span_graph_works.authors_json,
                            excluded.authors_json
                          ),
                          year=COALESCE(span_graph_works.year, excluded.year),
                          abstract=COALESCE(
                            span_graph_works.abstract,
                            excluded.abstract
                          ),
                          abstract_source=COALESCE(
                            span_graph_works.abstract_source,
                            excluded.abstract_source
                          ),
                          abstract_embedding_json=COALESCE(
                            span_graph_works.abstract_embedding_json,
                            excluded.abstract_embedding_json
                          ),
                          updated_at=now()
                    """,
                    (
                        pid,
                        wid,
                        merged.get("doi"),
                        merged.get("openalex_id"),
                        merged.get("title"),
                        _json_dumps(merged.get("authors"))
                        if merged.get("authors") is not None
                        else "null",
                        merged.get("year"),
                        merged.get("abstract"),
                        merged.get("abstract_source"),
                        _json_dumps(merged.get("abstract_embedding"))
                        if merged.get("abstract_embedding") is not None
                        else "null",
                    ),
                )
        return self.get_work(wid) or {"work_id": wid}

    def get_work(self, work_id: str) -> Optional[dict]:
        pid = self._project_id()
        wid = str(work_id or "").strip()
        if not wid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT work_id, doi, openalex_id, title, authors_json, year,
                           abstract, abstract_source, abstract_embedding_json,
                           created_at, updated_at
                    FROM span_graph_works
                    WHERE project_id=%s AND work_id=%s
                    """,
                    (pid, wid),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            work_id2,
            doi,
            openalex_id,
            title,
            authors_raw,
            year,
            abstract,
            abstract_source,
            abstract_embedding_raw,
            created_at,
            updated_at,
        ) = row
        authors = authors_raw if isinstance(authors_raw, list) else None
        if authors_raw is not None and not isinstance(authors_raw, list):
            try:
                authors = json.loads(authors_raw)
            except Exception:
                authors = None
        abstract_embedding = (
            abstract_embedding_raw if isinstance(abstract_embedding_raw, dict) else None
        )
        if abstract_embedding_raw is not None and not isinstance(
            abstract_embedding_raw, dict
        ):
            try:
                abstract_embedding = json.loads(abstract_embedding_raw)
            except Exception:
                abstract_embedding = None
        return {
            "work_id": str(work_id2),
            "doi": doi,
            "openalex_id": openalex_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "abstract_source": abstract_source,
            "abstract_embedding": abstract_embedding,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    # --- Spans ------------------------------------------------------------

    def upsert_span(
        self,
        *,
        kind: str,
        selector: dict,
        window_fingerprint: Optional[str],
        work_id: Optional[str] = None,
        ingest_id: Optional[str] = None,
    ) -> dict:
        pid = self._project_id()
        kind_norm = str(kind or "").strip()
        if not kind_norm:
            raise ValueError("kind is required")
        anchor_id = str(work_id or ingest_id or "").strip()
        if not anchor_id:
            raise ValueError("work_id or ingest_id is required")
        sid = _span_id(
            anchor_id=anchor_id,
            kind=kind_norm,
            selector=selector or {},
            window_fingerprint=window_fingerprint,
        )
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_spans(
                      project_id, span_id, work_id, ingest_id, kind,
                      selector_json, window_fingerprint, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, now(), now())
                    ON CONFLICT(project_id, span_id) DO UPDATE
                      SET work_id=COALESCE(span_graph_spans.work_id, excluded.work_id),
                          ingest_id=COALESCE(
                            span_graph_spans.ingest_id,
                            excluded.ingest_id
                          ),
                          kind=excluded.kind,
                          selector_json=excluded.selector_json,
                          window_fingerprint=excluded.window_fingerprint,
                          updated_at=now()
                    """,
                    (
                        pid,
                        sid,
                        str(work_id) if work_id else None,
                        str(ingest_id) if ingest_id else None,
                        kind_norm,
                        _json_dumps(selector or {}),
                        str(window_fingerprint) if window_fingerprint else None,
                    ),
                )
        return self.get_span(sid) or {"span_id": sid}

    def get_span(self, span_id: str) -> Optional[dict]:
        pid = self._project_id()
        sid = str(span_id or "").strip()
        if not sid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      span_id,
                      work_id,
                      ingest_id,
                      kind,
                      selector_json,
                      window_fingerprint,
                      created_at,
                      updated_at
                    FROM span_graph_spans
                    WHERE project_id=%s AND span_id=%s
                    """,
                    (pid, sid),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            span_id2,
            work_id,
            ingest_id,
            kind,
            selector_raw,
            window_fingerprint,
            created_at,
            updated_at,
        ) = row
        selector = selector_raw if isinstance(selector_raw, dict) else {}
        if selector_raw is not None and not isinstance(selector_raw, dict):
            try:
                selector = json.loads(selector_raw)
            except Exception:
                selector = {}
        return {
            "span_id": str(span_id2),
            "work_id": work_id,
            "ingest_id": ingest_id,
            "kind": str(kind),
            "selector": selector,
            "window_fingerprint": window_fingerprint,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }

    def add_span_cites(self, *, span_id: str, cites: List[dict]) -> int:
        pid = self._project_id()
        inserted = 0
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                for cite in cites or []:
                    if not isinstance(cite, dict):
                        continue
                    cited_work_id = str(cite.get("cited_work_id") or "").strip()
                    if not cited_work_id:
                        continue
                    cur.execute(
                        """
                        INSERT INTO span_graph_span_cites(
                          project_id,
                          span_id,
                          cited_work_id,
                          reference_id,
                          citation_index,
                          created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, now())
                        ON CONFLICT(project_id, span_id, cited_work_id) DO UPDATE
                          SET reference_id=excluded.reference_id,
                              citation_index=excluded.citation_index
                        """,
                        (
                            pid,
                            str(span_id),
                            cited_work_id,
                            str(cite.get("reference_id"))
                            if cite.get("reference_id") is not None
                            else None,
                            int(cite.get("citation_index"))
                            if cite.get("citation_index") is not None
                            else None,
                        ),
                    )
                    inserted += 1
        return inserted

    def list_span_cites(self, *, span_id: str) -> List[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      span_id,
                      cited_work_id,
                      reference_id,
                      citation_index,
                      created_at
                    FROM span_graph_span_cites
                    WHERE project_id=%s AND span_id=%s
                    ORDER BY cited_work_id ASC
                    """,
                    (pid, str(span_id)),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for span_id2, cited_work_id, reference_id, citation_index, created_at in rows:
            out.append(
                {
                    "span_id": str(span_id2),
                    "cited_work_id": str(cited_work_id),
                    "reference_id": reference_id,
                    "citation_index": citation_index,
                    "created_at": created_at.isoformat().replace("+00:00", "Z")
                    if created_at is not None
                    else None,
                }
            )
        return out

    def get_span_cite_role(
        self, *, span_id: str, cited_work_id: str, reviewer_uid: str
    ) -> Optional[str]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role
                    FROM span_graph_span_cite_roles
                    WHERE project_id=%s
                      AND span_id=%s
                      AND cited_work_id=%s
                      AND reviewer_uid=%s
                    """,
                    (pid, str(span_id), str(cited_work_id), str(reviewer_uid)),
                )
                row = cur.fetchone()
        return str(row[0]) if row and row[0] else None

    def set_span_cite_role(
        self, *, span_id: str, cited_work_id: str, reviewer_uid: str, role: str
    ) -> None:
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_span_cite_roles(
                      project_id, span_id, cited_work_id, reviewer_uid, role, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT(project_id, span_id, cited_work_id, reviewer_uid)
                    DO UPDATE SET role=excluded.role, updated_at=excluded.updated_at
                    """,
                    (
                        pid,
                        str(span_id),
                        str(cited_work_id),
                        str(reviewer_uid),
                        str(role),
                    ),
                )

    def upsert_claim_spans(self, *, span_id: str, items: Sequence[dict]) -> List[dict]:
        pid = self._project_id()
        out: List[dict] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            try:
                order_index = int(item.get("order_index"))
            except Exception:
                continue
            selector = (
                item.get("selector") if isinstance(item.get("selector"), dict) else None
            )
            csid = _claim_span_id(
                span_id=str(span_id), order_index=order_index, selector=selector
            )
            with connect(autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO span_graph_claim_spans(
                          project_id,
                          claim_span_id,
                          span_id,
                          order_index,
                          selector_json,
                          created_at,
                          updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s::jsonb, now(), now())
                        ON CONFLICT(project_id, span_id, order_index) DO UPDATE
                          SET claim_span_id=excluded.claim_span_id,
                              selector_json=excluded.selector_json,
                              updated_at=excluded.updated_at
                        """,
                        (
                            pid,
                            csid,
                            str(span_id),
                            int(order_index),
                            _json_dumps(selector) if selector is not None else "null",
                        ),
                    )
            got = self.get_claim_span(
                span_id=str(span_id), order_index=int(order_index)
            )
            if got:
                out.append(got)
        out.sort(key=lambda r: int(r.get("order_index") or 0))
        return out

    # --- Claim atoms ------------------------------------------------------

    def create_claim_atom(
        self,
        *,
        payload: Optional[dict] = None,
        text: Optional[str] = None,
        created_by: Optional[str] = None,
        reviewer_uid: Optional[str] = None,
        supersedes_id: Optional[str] = None,
        claim_atom_id: Optional[str] = None,
    ) -> dict:
        payload = payload or {
            "text": text,
            "created_by": created_by,
            "reviewer_uid": reviewer_uid,
            "supersedes_id": supersedes_id,
            "claim_atom_id": claim_atom_id,
        }

        reviewer_uid = (
            str(
                payload.get("reviewer_uid") or payload.get("created_by") or "default"
            ).strip()
            or "default"
        )
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text is required")
        pid = self._project_id()
        claim_atom_id = (
            str(payload.get("claim_atom_id") or "").strip() or f"atom:{uuid.uuid4()}"
        )
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_claim_atoms(
                      project_id,
                      claim_atom_id,
                      text,
                      created_by,
                      created_at,
                      updated_at,
                      supersedes_id
                    )
                    VALUES (%s, %s, %s, %s, now(), now(), %s)
                    """,
                    (
                        pid,
                        claim_atom_id,
                        text,
                        reviewer_uid,
                        str(payload.get("supersedes_id"))
                        if payload.get("supersedes_id") is not None
                        else None,
                    ),
                )
        return self.get_claim_atom(claim_atom_id) or {"claim_atom_id": claim_atom_id}

    def get_claim_atom(self, claim_atom_id: str) -> Optional[dict]:
        pid = self._project_id()
        aid = str(claim_atom_id or "").strip()
        if not aid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      claim_atom_id,
                      text,
                      created_by,
                      created_at,
                      updated_at,
                      supersedes_id
                    FROM span_graph_claim_atoms
                    WHERE project_id=%s AND claim_atom_id=%s
                    """,
                    (pid, aid),
                )
                row = cur.fetchone()
        if not row:
            return None
        claim_atom_id2, text, created_by, created_at, updated_at, supersedes_id = row
        return {
            "claim_atom_id": str(claim_atom_id2),
            "text": text,
            "created_by": created_by,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
            "supersedes_id": supersedes_id,
        }

    def list_claim_atoms(self, *, created_by: Optional[str] = None) -> List[dict]:
        pid = self._project_id()
        clauses = ["project_id=%s"]
        params: list[Any] = [pid]
        if created_by:
            clauses.append("created_by=%s")
            params.append(str(created_by))
        where = " AND ".join(clauses)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT claim_atom_id
                    FROM span_graph_claim_atoms
                    WHERE {where}
                    ORDER BY updated_at DESC, claim_atom_id ASC
                    """,
                    tuple(params),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (aid,) in rows:
            atom = self.get_claim_atom(str(aid))
            if atom:
                out.append(atom)
        return out

    def link_claim_span_atom(self, *, claim_span_id: str, claim_atom_id: str) -> None:
        pid = self._project_id()
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_claim_span_atoms(
                      project_id, claim_span_id, claim_atom_id, created_at
                    )
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT(project_id, claim_span_id, claim_atom_id) DO NOTHING
                    """,
                    (pid, str(claim_span_id), str(claim_atom_id)),
                )

    def list_claim_span_atoms(self, *, claim_span_id: str) -> List[dict]:
        pid = self._project_id()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT claim_atom_id
                    FROM span_graph_claim_span_atoms
                    WHERE project_id=%s AND claim_span_id=%s
                    ORDER BY claim_atom_id ASC
                    """,
                    (pid, str(claim_span_id)),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (aid,) in rows:
            atom = self.get_claim_atom(str(aid))
            if atom:
                out.append(atom)
        return out

    # --- Assertions -------------------------------------------------------

    def create_assertion(self, *, payload: dict) -> dict:
        pid = self._project_id()
        assertion_id = (
            str(payload.get("assertion_id") or "").strip() or f"ast:{uuid.uuid4()}"
        )
        reviewer_uid = (
            str(payload.get("reviewer_uid") or "default").strip() or "default"
        )
        verdict = str(payload.get("verdict") or "").strip()
        if verdict not in {"support", "contradict", "neutral", "uncertain"}:
            raise ValueError("Invalid verdict")

        claim_atom_id = str(payload.get("claim_atom_id") or "").strip() or None
        claim_span_id = str(payload.get("claim_span_id") or "").strip() or None
        evidence_span_id = str(payload.get("evidence_span_id") or "").strip() or None
        evidence_work_id = str(payload.get("evidence_work_id") or "").strip() or None
        if not evidence_span_id and not evidence_work_id:
            raise ValueError("evidence_span_id or evidence_work_id is required")

        confidence_val = None
        confidence = payload.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else None
        except Exception:
            confidence_val = None

        comment = payload.get("comment")
        source = payload.get("source")
        source_key = payload.get("source_key")

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO span_graph_assertions(
                      project_id,
                      assertion_id,
                      reviewer_uid,
                      verdict,
                      confidence,
                      comment,
                      claim_atom_id,
                      claim_span_id,
                      evidence_span_id,
                      evidence_work_id,
                      source,
                      source_key,
                      created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT(project_id, assertion_id) DO UPDATE
                      SET reviewer_uid=excluded.reviewer_uid,
                          verdict=excluded.verdict,
                          confidence=excluded.confidence,
                          comment=excluded.comment,
                          claim_atom_id=excluded.claim_atom_id,
                          claim_span_id=excluded.claim_span_id,
                          evidence_span_id=excluded.evidence_span_id,
                          evidence_work_id=excluded.evidence_work_id,
                          source=excluded.source,
                          source_key=excluded.source_key,
                          created_at=excluded.created_at
                    """,
                    (
                        pid,
                        assertion_id,
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
                    ),
                )
        return self.get_assertion(assertion_id) or {"assertion_id": assertion_id}

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
        key = "|".join(["selection", str(reviewer_uid), str(claim_span_id)])
        assertion_id = f"sel:{_sha256(key)}"
        return self.create_assertion(
            payload={
                "assertion_id": assertion_id,
                "reviewer_uid": str(reviewer_uid),
                "verdict": str(verdict),
                "confidence": None,
                "comment": comment,
                "claim_atom_id": None,
                "claim_span_id": str(claim_span_id),
                "evidence_span_id": str(evidence_span_id) if evidence_span_id else None,
                "evidence_work_id": str(evidence_work_id) if evidence_work_id else None,
                "source": "selection",
                "source_key": str(claim_id),
            }
        )

    def get_assertion(self, assertion_id: str) -> Optional[dict]:
        pid = self._project_id()
        aid = str(assertion_id or "").strip()
        if not aid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      assertion_id,
                      reviewer_uid,
                      verdict,
                      confidence,
                      comment,
                      claim_atom_id,
                      claim_span_id,
                      evidence_span_id,
                      evidence_work_id,
                      source,
                      source_key,
                      created_at
                    FROM span_graph_assertions
                    WHERE project_id=%s AND assertion_id=%s
                    """,
                    (pid, aid),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            assertion_id2,
            reviewer_uid,
            verdict,
            confidence,
            comment,
            claim_atom_id,
            claim_span_id,
            evidence_span_id,
            evidence_work_id,
            source,
            source_key,
            created_at,
        ) = row
        return {
            "assertion_id": str(assertion_id2),
            "reviewer_uid": str(reviewer_uid),
            "verdict": str(verdict),
            "confidence": confidence,
            "comment": comment,
            "claim_atom_id": claim_atom_id,
            "claim_span_id": claim_span_id,
            "evidence_span_id": evidence_span_id,
            "evidence_work_id": evidence_work_id,
            "source": source,
            "source_key": source_key,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
        }

    def list_assertions_for_claim_span(
        self, *, claim_span_id: str, reviewer_uid: Optional[str] = None
    ) -> List[dict]:
        pid = self._project_id()
        clauses = ["project_id=%s", "claim_span_id=%s"]
        params: list[Any] = [pid, str(claim_span_id)]
        if reviewer_uid:
            clauses.append("reviewer_uid=%s")
            params.append(str(reviewer_uid))
        where = " AND ".join(clauses)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT assertion_id
                    FROM span_graph_assertions
                    WHERE {where}
                    ORDER BY created_at DESC
                    """,
                    tuple(params),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (aid,) in rows:
            ast = self.get_assertion(str(aid))
            if ast:
                out.append(ast)
        return out

    def get_current_selection_assertion(
        self, *, claim_span_id: str, reviewer_uid: str
    ) -> Optional[dict]:
        pid = self._project_id()
        sel_pat = "sel:%"
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT assertion_id
                    FROM span_graph_assertions
                    WHERE project_id=%s AND claim_span_id=%s AND reviewer_uid=%s
                      AND assertion_id LIKE %s
                    ORDER BY created_at DESC, assertion_id DESC
                    LIMIT 1
                    """,
                    (pid, str(claim_span_id), str(reviewer_uid), sel_pat),
                )
                row = cur.fetchone()
        return self.get_assertion(str(row[0])) if row and row[0] else None

    # --- Status computation ----------------------------------------------

    def claim_span_status(self, *, claim_span_id: str, reviewer_uid: str) -> dict:
        reviewer_uid = str(reviewer_uid or "").strip() or "default"
        rows = self.list_assertions_for_claim_span(
            claim_span_id=str(claim_span_id), reviewer_uid=reviewer_uid
        )
        current = self.get_current_selection_assertion(
            claim_span_id=str(claim_span_id), reviewer_uid=reviewer_uid
        )

        non_selection = [
            r for r in rows if not str(r.get("assertion_id") or "").startswith("sel:")
        ]

        support_pairs = {
            (str(r.get("evidence_span_id") or ""), str(r.get("evidence_work_id") or ""))
            for r in non_selection
            if (r.get("verdict") == "support")
        }
        contra_pairs = {
            (str(r.get("evidence_span_id") or ""), str(r.get("evidence_work_id") or ""))
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
            claim_span_id=str(claim_span_id), reviewer_uid=reviewer_uid
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

        span_status = self.span_status(span_id=str(span_id), reviewer_uid=reviewer_uid)

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
                claim_span_id=cs_id, reviewer_uid=reviewer_uid
            )
            current = self.get_current_selection_assertion(
                claim_span_id=cs_id, reviewer_uid=reviewer_uid
            )
            history_n_total = 0
            if include_history:
                history_n_total = len(
                    self.list_assertions_for_claim_span(
                        claim_span_id=cs_id, reviewer_uid=reviewer_uid
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

        claim_spans_payload.sort(key=lambda r: int(r.get("order_index") or 0))
        return {
            "span": span,
            "span_status": span_status,
            "reviewer_uid": reviewer_uid,
            "cites": cites,
            "claim_spans": claim_spans_payload,
            "include_history": bool(include_history),
        }

    # --- Nav helpers (Phase 10-03) ----------------------------------------

    def list_citing_contexts_for_work(
        self,
        *,
        work_id: str,
        reviewer_uid: str,
        limit: int = 500,
    ) -> List[dict]:
        """List citing contexts for a cited work.

        Contexts are derived from span-graph cite rows and enriched (best-effort)
        with confirmed_claims snippet/sentence_id for the given reviewer.
        """
        pid = self._project_id()
        wid = str(work_id or "").strip()
        reviewer = str(reviewer_uid or "").strip() or "default"
        if not wid:
            return []

        out: List[dict] = []
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (
                      s.ingest_id,
                      COALESCE(c.citation_index, 1000000),
                      COALESCE(c.reference_id, '')
                    )
                      s.ingest_id AS citing_doc_id,
                      c.citation_index,
                      c.reference_id,
                      cc.sentence_id,
                      cc.claim_index,
                      cc.parsed_text
                    FROM span_graph_span_cites c
                    JOIN span_graph_spans s
                      ON s.project_id=c.project_id
                     AND s.span_id=c.span_id
                    LEFT JOIN confirmed_claims cc
                      ON cc.project_id=c.project_id
                     AND cc.document_id=s.ingest_id
                     AND cc.citation_index=c.citation_index
                     AND cc.target_id=c.reference_id
                     AND cc.reviewer_uid=%s
                    WHERE c.project_id=%s
                      AND c.cited_work_id=%s
                      AND s.kind='citation_window'
                    ORDER BY
                      s.ingest_id ASC,
                      COALESCE(c.citation_index, 1000000) ASC,
                      COALESCE(c.reference_id, '') ASC,
                      COALESCE(cc.claim_index, 1000000) ASC
                    LIMIT %s
                    """,
                    (reviewer, pid, wid, int(limit)),
                )
                rows = cur.fetchall() or []

        for (
            citing_doc_id,
            citation_index,
            reference_id,
            sentence_id,
            claim_index,
            parsed_text,
        ) in rows:
            snippet = None
            try:
                snippet = str(parsed_text or "").strip() or None
            except Exception:
                snippet = None
            if snippet and len(snippet) > 260:
                snippet = snippet[:259].rstrip() + "..."
            out.append(
                {
                    "citing_doc_id": str(citing_doc_id or "").strip() or None,
                    "citation_index": int(citation_index)
                    if citation_index is not None
                    else None,
                    "reference_id": str(reference_id or "").strip() or None,
                    "sentence_id": str(sentence_id or "").strip() or None,
                    "claim_id": None,
                    "snippet": snippet,
                }
            )

        # Stable ordering for clients.
        out.sort(
            key=lambda r: (
                str(r.get("citing_doc_id") or ""),
                int(r.get("citation_index") or 1000000),
                str(r.get("reference_id") or ""),
                str(r.get("sentence_id") or ""),
            )
        )
        return out

    def list_spans_for_ingest(
        self,
        *,
        ingest_id: str,
        kind: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        pid = self._project_id()
        iid = str(ingest_id or "").strip()
        if not iid:
            return []
        clauses = ["project_id=%s", "ingest_id=%s"]
        params: list[Any] = [pid, iid]
        if kind:
            clauses.append("kind=%s")
            params.append(str(kind))
        where = " AND ".join(clauses)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT span_id
                    FROM span_graph_spans
                    WHERE {where}
                    ORDER BY updated_at DESC, span_id ASC
                    LIMIT %s
                    """,
                    tuple(params + [int(limit)]),
                )
                rows = cur.fetchall() or []
        out: List[dict] = []
        for (sid,) in rows:
            span = self.get_span(str(sid))
            if span:
                out.append(span)
        return out

    def get_claim_span_by_id(self, *, claim_span_id: str) -> Optional[dict]:
        pid = self._project_id()
        csid = str(claim_span_id or "").strip()
        if not csid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      claim_span_id,
                      span_id,
                      order_index,
                      selector_json,
                      created_at,
                      updated_at
                    FROM span_graph_claim_spans
                    WHERE project_id=%s AND claim_span_id=%s
                    """,
                    (pid, csid),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            claim_span_id2,
            span_id,
            order_index,
            selector_raw,
            created_at,
            updated_at,
        ) = row
        selector = selector_raw if isinstance(selector_raw, dict) else None
        if selector_raw is not None and not isinstance(selector_raw, dict):
            try:
                selector = json.loads(selector_raw)
            except Exception:
                selector = None
        return {
            "claim_span_id": str(claim_span_id2),
            "span_id": str(span_id),
            "order_index": int(order_index or 0),
            "selector": selector,
            "created_at": created_at.isoformat().replace("+00:00", "Z")
            if created_at is not None
            else None,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
        }
