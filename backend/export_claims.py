import json
from pathlib import Path

from backend.db.pg import connect


EXPORT_PATH = Path(__file__).parent.parent / "data" / "claims.ndjson"


def export_claims(out_path: Path | str = EXPORT_PATH) -> int:
    """Export confirmed claims to NDJSON for downstream ingestion."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0

    query = """
    SELECT
      project_id,
      document_id,
      sentence_id,
      claim_index,
      parsed_text,
      original_text,
      segmentation_model,
      reviewer_uid,
      confirmed_at,
      confidence,
      citation_index,
      target_id,
      sentence_text
    FROM confirmed_claims
    ORDER BY confirmed_at DESC
    """

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall() or []

    with out_path.open("w", encoding="utf-8") as fh:
        for (
            project_id,
            document_id,
            sentence_id,
            claim_index,
            parsed_text,
            original_text,
            segmentation_model,
            reviewer_uid,
            confirmed_at,
            confidence,
            citation_index,
            target_id,
            sentence_text,
        ) in rows:
            record = {
                "project_id": project_id,
                "document_id": document_id,
                # Back-compat with older exporters.
                "doc_id": document_id,
                "sentence_id": sentence_id,
                "claim_index": claim_index,
                "parsed_text": parsed_text,
                "original_text": original_text,
                "segmentation_model": segmentation_model,
                "reviewer_uid": reviewer_uid,
                "confirmed_at": confirmed_at.isoformat().replace("+00:00", "Z")
                if confirmed_at is not None
                else None,
                "confidence": confidence,
                "citation_index": citation_index,
                "target_id": target_id,
                "sentence_text": sentence_text,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
    return exported


if __name__ == "__main__":
    count = export_claims()
    print(f"Exported {count} confirmed claims to {EXPORT_PATH}")
