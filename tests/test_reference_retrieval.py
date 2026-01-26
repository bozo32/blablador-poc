import pytest

from backend import reference_retrieval


DOC_ID = "doc-1"
REFERENCE_ID = "ref-1"


def _base_document():
    return {
        "id": DOC_ID,
        "extraction": {
            "data": {
                "references": [
                    {
                        "id": REFERENCE_ID,
                        "raw_reference": "Doe, J. 2020. Sample Work.",
                        "doi": "10.1000/sample",
                        "grobid": {
                            "authors": ["Jane Doe"],
                            "year": "2020",
                            "title": "Sample Work",
                            "url": "https://publisher.example/pdf",
                        },
                        "url": None,
                    }
                ]
            }
        },
        "resolution": {
            "data": [
                {
                    "reference_id": REFERENCE_ID,
                    "doi": "10.1000/sample",
                    "status": "match",
                    "confidence": 0.92,
                    "grobid": {
                        "title": "Sample Work",
                        "doi": "10.1000/sample",
                        "url": "https://grobid.example/sample",
                    },
                    "crossref": {
                        "title": "Sample Work",
                        "doi": "10.1000/sample",
                        "url": "https://crossref.example/sample",
                        "publisher_url": "https://publisher.example/pdf",
                        "confidence": 0.88,
                    },
                    "openalex": {
                        "title": "Sample Work",
                        "doi": "10.1000/sample",
                        "url": "https://openalex.example/sample",
                        "confidence": 0.75,
                    },
                }
            ]
        },
    }


@pytest.fixture(autouse=True)
def mock_ingestion(monkeypatch):
    doc = _base_document()

    def _get_ingested_document(doc_id):
        assert doc_id == DOC_ID
        return doc

    monkeypatch.setattr(
        reference_retrieval, "get_ingested_document", _get_ingested_document
    )
    yield doc


def test_builds_happy_path_dossier(mock_ingestion):
    dossier = reference_retrieval.build_retrieval_dossier(DOC_ID, REFERENCE_ID)

    assert dossier.document_id == DOC_ID
    assert dossier.reference_id == REFERENCE_ID
    assert "Jane Doe" in dossier.canonical_citation
    assert dossier.doi == "10.1000/sample"
    assert dossier.primary_url == "https://publisher.example/pdf"
    assert dossier.manual_instructions is None
    assert dossier.sources, "sources should include grobid/crossref/openalex"


def test_missing_links_emit_manual_instructions(mock_ingestion):
    mock_ingestion["resolution"]["data"][0].update(
        {"doi": None, "selected_source": None, "publisher_url": None, "url": None}
    )
    mock_ingestion["resolution"]["data"][0]["grobid"]["url"] = None
    mock_ingestion["resolution"]["data"][0]["crossref"]["url"] = None
    mock_ingestion["resolution"]["data"][0]["crossref"]["publisher_url"] = None
    mock_ingestion["resolution"]["data"][0]["openalex"]["url"] = None
    mock_ingestion["extraction"]["data"]["references"][0]["grobid"]["url"] = None
    mock_ingestion["extraction"]["data"]["references"][0]["doi"] = None

    dossier = reference_retrieval.build_retrieval_dossier(DOC_ID, REFERENCE_ID)

    assert dossier.primary_url is None
    assert dossier.manual_instructions.startswith("No direct link available")


def test_resolver_confidence_serialized(mock_ingestion):
    dossier = reference_retrieval.build_retrieval_dossier(DOC_ID, REFERENCE_ID)

    assert dossier.resolver_confidence == pytest.approx(0.92)
    # Ensure individual source confidences are preserved
    confidence_map = {src.label: src.confidence for src in dossier.sources}
    assert confidence_map["Grobid"] is None
    assert confidence_map["Crossref"] == pytest.approx(0.88)
    assert confidence_map["Openalex"] == pytest.approx(0.75)
