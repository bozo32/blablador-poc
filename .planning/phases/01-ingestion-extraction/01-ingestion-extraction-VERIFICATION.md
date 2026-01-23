---
phase: 01-ingestion-extraction
verified: 2026-01-23T19:40:00Z
status: passed
score: 12/12 must-haves verified
human_verification:
  - test: "Upload a PDF from the Streamlit sidebar"
    expected: "Document appears in the ingested list with filename, size, and timestamp"
    why_human: "Requires running UI + backend to confirm live upload flow"
  - test: "Run extraction for an uploaded PDF"
    expected: "Metadata, citations, and bibliography populate in the UI"
    why_human: "Depends on live GROBID service and real PDF parsing"
  - test: "Run reference resolution after extraction"
    expected: "Resolution table includes DOI/confidence when available"
    why_human: "Depends on Crossref API and configured mailto"
---

# Phase 1: Ingestion + Extraction Verification Report

**Phase Goal:** Users can ingest PDFs and see structured metadata, citations, and normalized references.
**Verified:** 2026-01-23T19:40:00Z
**Status:** passed
**Re-verification:** Yes — user-approved

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | User can upload a PDF and receive an ingestion record ID | ✓ VERIFIED | `backend/main.py` exposes `POST /ingest` and returns stored metadata via `create_ingested_document`. |
| 2 | User can list uploaded documents and see upload metadata | ✓ VERIFIED | `backend/main.py` `GET /ingest` returns `list_ingested_documents` metadata. |
| 3 | Stored ingestion record persists on disk for later extraction | ✓ VERIFIED | `backend/ingestion_store.py` writes `metadata.json` and `source.pdf` per doc ID. |
| 4 | User can trigger GROBID extraction for an uploaded PDF | ✓ VERIFIED | `backend/main.py` `POST /ingest/{doc_id}/extract` calls `grobid_client.extract_tei`. |
| 5 | Extracted metadata and citations are stored with the ingestion record | ✓ VERIFIED | `backend/ingestion_store.py` `store_extraction` persists `extraction` payload. |
| 6 | TEI/XML output is persisted for reprocessing | ✓ VERIFIED | `backend/ingestion_store.py` writes `extraction/tei.xml` in `store_extraction`. |
| 7 | User can resolve bibliography entries to canonical identifiers | ✓ VERIFIED | `backend/main.py` `POST /ingest/{doc_id}/resolve` calls `resolve_references`. |
| 8 | Resolved references include DOI and confidence scores when available | ✓ VERIFIED | `backend/reference_resolver.py` returns `doi` and `confidence` fields. |
| 9 | Resolution results are stored with the ingestion record | ✓ VERIFIED | `backend/ingestion_store.py` `store_resolution` persists `resolution` payload. |
| 10 | User can upload a PDF from the UI and see it listed | ✓ VERIFIED | `frontend/ui.py` sidebar uploader calls `upload_pdf` and refreshes `ingested_docs`. |
| 11 | User can view extracted metadata, citations, and resolution status | ✓ VERIFIED | `frontend/ui.py` `draw_ingestion_panel` renders metadata/citations/references/resolution tables. |
| 12 | User can trigger extraction and resolution from the UI | ✓ VERIFIED | `frontend/ui.py` buttons call `trigger_extraction` and `trigger_resolution`. |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `backend/ingestion_store.py` | Local ingestion storage utilities | ✓ VERIFIED | 177 lines; writes metadata + TEI + resolution payloads; used by `backend/main.py`. |
| `backend/settings.py` | Ingestion/GROBID/Crossref settings | ✓ VERIFIED | 241 lines; defines `INGESTION_DIR`, `GROBID_URL`, `CROSSREF_*`. |
| `backend/schemas.py` | Ingestion/extraction/resolution schemas | ✓ VERIFIED | 195 lines; includes `IngestedDocument`, `ExtractionResponse`, `ResolutionResponse`. |
| `backend/main.py` | Ingestion, extraction, resolution endpoints | ✓ VERIFIED | 470 lines; `/ingest`, `/extract`, `/resolve` routes wired to storage and parsers. |
| `backend/grobid_client.py` | GROBID TEI client | ✓ VERIFIED | 36 lines; `extract_tei` posts PDF to GROBID. |
| `backend/extraction.py` | TEI parsing for metadata/citations | ✓ VERIFIED | 152 lines; parses metadata, citations, references. |
| `backend/reference_resolver.py` | Crossref resolver | ✓ VERIFIED | 111 lines; resolves DOI/query with confidence. |
| `frontend/ingestion_api.py` | UI HTTP helpers | ✓ VERIFIED | 49 lines; wraps `/ingest`, `/extract`, `/resolve`. |
| `frontend/ui.py` | Streamlit ingestion UI | ✓ VERIFIED | 995 lines; ingestion panel and actions wired to API helpers. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `backend/main.py` | `backend/ingestion_store.py` | `create_ingested_document`, `list_ingested_documents`, `store_extraction`, `store_resolution` | WIRED | Imports and calls ingestion store helpers for upload/list/extract/resolve. |
| `backend/settings.py` | `backend/ingestion_store.py` | `INGESTION_DIR` | WIRED | `ingestion_store.DEFAULT_INGESTION_DIR` uses settings. |
| `backend/main.py` | `backend/grobid_client.py` | `extract_tei` | WIRED | Extraction endpoint posts PDF to GROBID client. |
| `backend/main.py` | `backend/extraction.py` | `parse_tei` | WIRED | Extraction endpoint parses TEI to metadata/citations. |
| `backend/ingestion_store.py` | `extraction payload` | `store_extraction` | WIRED | TEI + parsed payload stored under `extraction` in metadata. |
| `backend/main.py` | `backend/reference_resolver.py` | `resolve_references` | WIRED | Resolution endpoint normalizes bibliography entries. |
| `backend/reference_resolver.py` | `https://api.crossref.org/works` | HTTP GET | WIRED | Crossref API requests with `mailto` parameter. |
| `backend/ingestion_store.py` | `resolution payload` | `store_resolution` | WIRED | Resolution results persisted in metadata. |
| `frontend/ui.py` | `/ingest` | `upload_pdf` | WIRED | Sidebar uploader calls helper which POSTs to API. |
| `frontend/ui.py` | `/ingest/{doc_id}/extract` | `trigger_extraction` | WIRED | UI button triggers extraction. |
| `frontend/ui.py` | `/ingest/{doc_id}/resolve` | `trigger_resolution` | WIRED | UI button triggers resolution. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
| --- | --- | --- |
| ING-01 | ✓ SATISFIED | None found in code. |
| ING-02 | ✓ SATISFIED | None found in code. |
| ING-03 | ✓ SATISFIED | None found in code. |
| ING-04 | ✓ SATISFIED | None found in code. |
| ING-05 | ✓ SATISFIED | None found in code. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| (none) | - | - | - | - |

### Human Verification Required

### 1. Upload PDF in UI

**Test:** Upload a PDF from the sidebar uploader.
**Expected:** Document appears in the ingested list with filename, size, and timestamp.
**Why human:** Requires running the Streamlit UI and backend to validate the upload flow.

### 2. Run Extraction

**Test:** Click "Run Extraction" for an uploaded PDF.
**Expected:** Metadata, citations, and bibliography tables populate in the UI.
**Why human:** Depends on a live GROBID service and parsing real TEI output.

### 3. Run Resolution

**Test:** Click "Resolve References" after extraction.
**Expected:** Resolution table includes DOI/confidence when Crossref returns data.
**Why human:** Depends on Crossref API availability and configured `CROSSREF_MAILTO`.

### Gaps Summary

All must-haves are present and wired. Live service validation is required to confirm the end-to-end ingestion, extraction, and resolution flows.

---

_Verified: 2026-01-23T18:12:02Z_
_Verifier: Claude (gsd-verifier)_
