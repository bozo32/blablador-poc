import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.settings import settings
from backend.schemas import ClaimConfirmationRequest


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        filename TEXT,
        uploaded_at TEXT,
        source_path TEXT,
        tei_path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentences (
        sentence_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        citation_index INTEGER,
        target_id TEXT,
        sentence_text TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS claims (
        claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id TEXT NOT NULL,
        claim_index INTEGER NOT NULL,
        parsed_text TEXT NOT NULL,
        original_text TEXT,
        segmentation_model TEXT,
        reviewer_uid TEXT NOT NULL,
        confirmed_at TEXT NOT NULL,
        confidence REAL,
        UNIQUE(sentence_id, claim_index),
        FOREIGN KEY(sentence_id) REFERENCES sentences(sentence_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_sentence_id TEXT NOT NULL,
        target_sentence_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(source_sentence_id)
            REFERENCES sentences(sentence_id)
            ON DELETE CASCADE,
        FOREIGN KEY(target_sentence_id)
            REFERENCES sentences(sentence_id)
            ON DELETE CASCADE
    )
    """,
]


def _ensure_db_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class ClaimStore:
    def __init__(self, db_path: Path):
        """Initialize the on-disk SQLite store and ensure schema exists."""
        self._path = _ensure_db_path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        for ddl in SCHEMA:
            self._conn.execute(ddl)
        self._conn.commit()

    def wipe(self) -> None:
        """Delete all claim-store rows (keeps schema)."""
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        with self._conn:
            for row in rows or []:
                name = str(row[0])
                if not name:
                    continue
                self._conn.execute(f'DELETE FROM "{name}"')

    def persist_confirmed_claims(self, payload: ClaimConfirmationRequest) -> int:
        if not payload.confirmed_claims:
            return 0
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO documents (doc_id) VALUES (?)
                """,
                (payload.document_id,),
            )
            self._conn.execute(
                """
                INSERT INTO sentences (
                    sentence_id,
                    doc_id,
                    citation_index,
                    target_id,
                    sentence_text,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sentence_id) DO UPDATE SET
                    sentence_text = excluded.sentence_text,
                    updated_at = excluded.updated_at
                """,
                (
                    payload.sentence_id,
                    payload.document_id,
                    payload.citation_index,
                    payload.target_id,
                    payload.sentence_text,
                    now,
                ),
            )
            rows = 0
            for claim in payload.confirmed_claims:
                self._conn.execute(
                    """
                    INSERT INTO claims (
                        sentence_id,
                        claim_index,
                        parsed_text,
                        original_text,
                        segmentation_model,
                        reviewer_uid,
                        confirmed_at,
                        confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sentence_id, claim_index) DO UPDATE SET
                        parsed_text = excluded.parsed_text,
                        original_text = excluded.original_text,
                        segmentation_model = excluded.segmentation_model,
                        reviewer_uid = excluded.reviewer_uid,
                        confirmed_at = excluded.confirmed_at,
                        confidence = excluded.confidence
                    """,
                    (
                        payload.sentence_id,
                        claim.claim_index,
                        claim.parsed_text,
                        claim.original_text,
                        payload.segmentation_model,
                        payload.reviewer_uid,
                        now,
                        claim.confidence,
                    ),
                )
                rows += 1
        return rows


claim_store = ClaimStore(settings.CLAIM_DB_PATH)
