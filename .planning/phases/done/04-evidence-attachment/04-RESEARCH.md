# Phase 4 Research — Evidence Attachment

**Date:** 2026-01-26  
**Scope:** Retrieval instructions, attachment UX, backend persistence, parsing pipeline readiness for downstream evidence matching (EVD-01 — EVD-03)

## 1. Problem Framing

Reviewers need to move from curated claims to evidence sourcing. Before matching spans we must: (1) help them obtain the cited PDF, (2) let them attach it to a claim with clear feedback, and (3) parse + persist the attachment so later phases can index sentences. Reliability matters more than fancy UX; attachments become ground truth for the rest of the workflow.

## 2. Retrieval Instructions (EVD-01)

- **Inputs:** extracted reference metadata (GROBID), resolver outputs (Crossref/OpenAlex), and stored ingestion info.  
- **Best practice:** build a small "retrieval dossier" that surfaces: canonical citation (APA/shortened), DOI/URL, resolution confidence, and quick actions (copy DOI, open OpenAlex).  
- **Backend support:** expose `GET /references/{doc_id}/{reference_id}/retrieval` returning normalized identifiers, recommended download sources (Crossref URL, publisher PDF if available), and fallback instructions ("request via library" when no direct link).  
- **UI pattern:** inline expandable card inside the claim queue with "Copy instructions" and "Open source" buttons. Provide callouts when metadata is incomplete; fall back to manual instructions referencing the bibliographic string.

## 3. Attachment UX (EVD-02)

- Use Streamlit's file-uploader but wrap it in a custom drop zone to match the roadmap requirement (hover highlight, badge, skeleton loader).  
- Persist attachment state in `st.session_state` keyed by `queue_id`. Track: `file_name`, `local_path`, `status`, timestamps, errors.  
- Provide explicit detach/reset flow; default to single attachment per claim.  
- Accessibility hints: show file size + type; disable drop area while parsing to avoid duplicate uploads.

## 4. Backend Attachment Lifecycle (EVD-02/03)

**Storage:**

- Add `attachments` table (`id`, `claim_id`, `doc_id`, `file_path`, `filename`, `status`, `error`, `uploaded_at`, `parsed_at`).  
- Link to `sentences` table via `attachment_id`.  
- Persist attachments on disk under a deterministic folder (e.g., `data/attachments/{claim_id}/`).

**API surface:**

- `POST /claims/{claim_id}/attachment`: accepts metadata and (optionally) file via multipart/form-data. For MVP we can accept a local path (UI already saved file) and move/copy server-side.  
- `GET /claims/{claim_id}/attachment`: returns attachment status for polling.  
- Emit events/logging for monitoring since parsing can be slow.

**Parsing pipeline:**

- Reuse existing extraction helpers (`extraction.parse_tei`) to turn PDFs into TEI JSON.  
- After parsing, extract sentence windows with unique IDs and persist them for the claim.  
- Convert parsed sentences to NDJSON rows so later phases (matching/ranking) can rehydrate quickly.  
- Keep parsing synchronous initially; if average file sizes slow down UI we can dispatch to background tasks in Phase 5.

## 5. Verification Hooks

- **Must-haves:** highlight drop target, badge + loader lifecycle, detach confirmation, durable backend metadata, NDJSON export includes attachments.  
- **Testing:** add CLI smoke command to upload sample PDF + confirm `attachments.status` transitions `pending → parsing → ready`.  
- **Telemetry:** log parsing errors by claim/attachment ID; surface in UI as retry button.

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Large PDFs cause slow parsing | Cap upload size (e.g., 25 MB) and show explicit warning; consider async parsing in later phases |
| Reviewers attach wrong document | Show attachment metadata (title/author if extractable) after parsing; allow detach & re-upload |
| File path handling between Streamlit (frontend) and FastAPI (backend) | Upload to a shared workspace directory that both layers can access (e.g., `data/tmp_uploads`), then move server-side |
| Schema drift when attachments added | Write Alembic-style migration helper or `claim_store.ensure_schema()` that creates new tables/columns idempotently |

## 7. Implementation Order Recommendation

1. Retrieval dossier endpoint/UI (unblocks drop target by giving reviewers clarity).  
2. Frontend attachment drop target + local state persistence.  
3. Backend attachment route + parser + NDJSON export.  
4. Polish: loaders/detach UX, error retries, telemetry.

---

This research file should be referenced by planners/executors for Phase 4.
