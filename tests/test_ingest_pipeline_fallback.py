from __future__ import annotations

from pathlib import Path

import backend.ingest_pipeline as ingest_pipeline
from backend.graph_store import GraphStore
from backend.ingestion_store import create_ingested_document, get_ingested_document


def test_ingest_pipeline_falls_back_to_header_and_references(tmp_path, monkeypatch):
    # This test targets the legacy ingestion store behavior.
    monkeypatch.setenv("SPINE_MODE", "legacy")
    monkeypatch.setenv("SPINE_PDF_SOURCE", "legacy")

    ingestion_dir = tmp_path / "ingestion"
    graph_db = tmp_path / "graph.db"
    store = GraphStore(graph_db)

    meta = create_ingested_document(
        b"not-a-real-pdf",
        "x.pdf",
        ingestion_dir=ingestion_dir,
    )
    doc_id = str(meta["id"])

    def _fulltext(_pdf_path: Path) -> str:
        raise ingest_pipeline.grobid_client.GrobidError("boom")

    def _header(_pdf_path: Path) -> str:
        return (
            "<TEI xmlns='http://www.tei-c.org/ns/1.0'>"
            "<teiHeader><fileDesc><titleStmt><title>Doc</title></titleStmt>"
            "<sourceDesc><biblStruct><idno type='DOI'>10.1/abc</idno>"
            "</biblStruct></sourceDesc></fileDesc></teiHeader>"
            "<text><body/></text></TEI>"
        )

    def _refs(_pdf_path: Path) -> str:
        return (
            "<TEI xmlns='http://www.tei-c.org/ns/1.0'>"
            "<teiHeader/><text><body>"
            "<listBibl><biblStruct xml:id='b0'>"
            "<idno type='DOI'>10.2/xyz</idno>"
            "<analytic><title level='a'>Ref</title></analytic>"
            "</biblStruct></listBibl>"
            "</body></text></TEI>"
        )

    monkeypatch.setattr(
        ingest_pipeline.grobid_client, "extract_tei_fulltext", _fulltext
    )
    monkeypatch.setattr(ingest_pipeline.grobid_client, "extract_tei_header", _header)
    monkeypatch.setattr(ingest_pipeline.grobid_client, "extract_tei_references", _refs)

    def _resolve(entries):
        ref_id = str((entries or [{}])[0].get("id") or "b0")
        return [
            {
                "reference_id": ref_id,
                "doi": "10.2/xyz",
                "title": "Ref",
                "year": "2024",
                "authors": ["Doe"],
                "status": "match",
                "selected_source": "grobid",
            }
        ]

    monkeypatch.setattr(ingest_pipeline, "resolve_references", _resolve)

    ingest_pipeline.run_full_ingest_pipeline(
        doc_id,
        graph_store=store,
        ingestion_dir=ingestion_dir,
    )

    updated = get_ingested_document(doc_id, ingestion_dir)
    assert updated is not None
    assert updated["extraction"]["status"] == "complete"
    assert updated["body_extraction"]["status"] == "error"
    assert updated["resolution"]["status"] == "complete"
    assert updated["resolution"]["data"]
