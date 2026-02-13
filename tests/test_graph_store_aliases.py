from __future__ import annotations

from backend.graph_store import GraphStore


def test_bib_alias_collapses_reference_to_ingested_doc(tmp_path):
    store = GraphStore(tmp_path / "graph.db")

    # Ingest a doc with no DOI but strong bib metadata.
    ingest_meta = {"id": "doc-a", "sha256": "a" * 64, "filename": "a.pdf"}
    store.index_ingest_upload(ingest_meta)
    store.index_extraction(
        ingest_meta=ingest_meta,
        extraction_data={
            "metadata": {
                "title": (
                    "Scientific Autonomy, Public Accountability, and the Rise of "
                    "Peer Review"
                ),
                "authors": ["Baldwin"],
                "year": "2018",
                "doi": None,
            },
            "references": [],
        },
    )

    # Ingest a second doc that cites the first using only bib metadata.
    ingest_b = {"id": "doc-b", "sha256": "b" * 64, "filename": "b.pdf"}
    store.index_ingest_upload(ingest_b)
    store.index_extraction(
        ingest_meta=ingest_b,
        extraction_data={
            "metadata": {"title": "Muller", "authors": ["Muller"], "year": "2025"},
            "references": [
                {
                    "id": "b4",
                    "doi": None,
                    "grobid": {
                        "title": (
                            "Scientific Autonomy, Public Accountability, and the Rise "
                            "of Peer Review"
                        ),
                        "authors": ["Baldwin"],
                        "year": "2018",
                    },
                }
            ],
        },
    )

    # Ensure the reference alias resolves to the ingested doc-a node_id, not a
    # separate doc:bib node.
    ref_node = store.resolve_alias("ref:doc-b:b4")
    assert ref_node is not None
    assert ref_node == store.resolve_alias("ingest:doc-a")
