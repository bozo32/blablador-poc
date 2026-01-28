import json
import sqlite3
from pathlib import Path

from backend.settings import settings


EXPORT_PATH = settings.CLAIM_DB_PATH.parent / "claims.ndjson"


def export_claims(out_path: Path | str = EXPORT_PATH) -> int:
    """Export confirmed claims to NDJSON for downstream ingestion."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(settings.CLAIM_DB_PATH))
    conn.row_factory = sqlite3.Row

    query = """
    SELECT
        c.claim_id,
        c.sentence_id,
        c.claim_index,
        c.parsed_text,
        c.original_text,
        c.segmentation_model,
        c.reviewer_uid,
        c.confirmed_at,
        c.confidence,
        s.doc_id,
        s.citation_index,
        s.target_id,
        s.sentence_text,
        d.filename,
        d.uploaded_at
    FROM claims c
    JOIN sentences s ON c.sentence_id = s.sentence_id
    LEFT JOIN documents d ON s.doc_id = d.doc_id
    ORDER BY c.confirmed_at DESC
    """

    exported = 0
    with conn, out_path.open("w", encoding="utf-8") as fh:
        for row in conn.execute(query):
            record = {key: row[key] for key in row.keys()}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
    return exported


if __name__ == "__main__":
    count = export_claims()
    print(f"Exported {count} confirmed claims to {EXPORT_PATH}")
