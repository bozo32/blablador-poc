# Workflow Stages: Backend Notes (WIP)

## Governance Metadata

- Doc role: implementation notes
- Authority tier: reference-only (non-normative)
- Status: provisional/WIP
- Owner: repo maintainers
- Last reviewed: 2026-03-03
- Canonical for: historical backend-change notes only
- Superseded by: `docs/WORKFLOW_PROTOCOL.md` and phase verification artifacts when conflicts exist

This document captures backend-only structural changes found in a recent demo-driven working tree, and reframes them as stage-by-stage workflow outcomes.

Baseline reference:
- Branch: `feature/09-1-ingestion-replumbing-solid-spine`
- Last commit before the demo rush: `1efc9e9` (docs-only)

## What Changed (Backend Only)

### Attachments: reuse cited PDFs across many claims

Outcome:
- Allow one cited PDF upload to be reused without mutating the original attachment row.

Changes:
- `backend/attachment_store.py`: add `clone_attachment(source_attachment_id, ...)` which duplicates an attachment row while reusing the same `pdf_object_key` + artifacts. If the source is already `matched`, the clone is created as `matched` immediately.
- `backend/schemas.py`: add `AttachmentCloneRequest` payload.
- `backend/main.py`: add `POST /attachments/{attachment_id}/clone` endpoint returning a new attachment.

Why this matters:
- Without cloning, assigning a “global” Source Bin attachment to a specific claim/doc/target mutates it, making it unavailable for other citing documents and confusing assignment UI.

### Graph: persist doc-level reference -> ingest mapping

Outcome:
- Once a reviewer associates a cited PDF with a bibliography/reference target, future auto-place can resolve that reference to the same ingested cited work.

Changes:
- `backend/graph_store.py`: add `link_reference_to_ingest(citing_doc_id, reference_id, cited_ingest_id)` which sets `ingest_ids=[cited_ingest_id]` on the `ref:{citing_doc_id}:{reference_id}` node.
- `backend/main.py`: on attachment placement (`PATCH /attachments/{id}`), if `source_ingest_id` is present, call `link_reference_to_ingest(...)`.
- `backend/main.py`: on clone (`POST /attachments/{id}/clone`), best-effort call `link_reference_to_ingest(...)`.
- `backend/graph_store.py`: add best-effort backfill `_propagate_ingest_id_to_doc_key(doc_key, ingest_id)` invoked when a new work is indexed, to fill missing `ingest_ids` on existing `doc_key` document nodes.

Why this matters:
- Graph “cited work” nodes are only visible/resolvable once targets can map to an ingest id. This is the missing link when the graph shows placeholders like `https://` but not an actual cited work.

### Evidence reruns: automatic sweep after assignment

Outcome:
- When a cited source is assigned (placed or cloned), evidence reruns are triggered so the reviewer sees candidates without manual rerun.

Changes:
- `backend/main.py`: after successful placement and after clone, call `evidence_service.trigger_auto_rerun(claim_id, claim_text=...)` (best-effort).

### Ingestion: allow forced re-extraction

Outcome:
- Make stepwise diagnosis easier by letting you force a new extraction job even when extraction is already complete.

Changes:
- `backend/main.py`: `POST /ingest/{doc_id}/extract?force=true` bypasses early-return for `running/complete` states.

### Judgment store: robust JSON decoding

Outcome:
- Avoid runtime errors when Postgres JSON/JSONB columns are returned decoded vs stringified depending on driver/cursor.

Changes:
- `backend/judgment_store.py`: `_decode_json_cell()` used for `notes`, `validation`, `citation_anchor`, `span_selectors`.

### Attachments: sentence fallback from TEI

Outcome:
- If `sentences.ndjson` is missing but `tei_xml` exists, evidence review can still produce sentence rows.

Changes:
- `backend/attachment_store.py`: `load_sentences_for_attachment()` now synthesizes sentence rows from TEI XML using `lxml` (best-effort).

### NLI: dev-time disable

Outcome:
- Allow local workflows to run without downloading/initializing large NLI models.

Changes:
- `backend/nli.py`: if env `NLI_DISABLE` is truthy, `assess()` returns `[]`.

### Attachments listing: tolerant serialization

Outcome:
- Avoid one bad record breaking `/attachments` response.

Changes:
- `backend/main.py`: `GET /attachments` uses `_serialize_attachment()` in a try/except per record.

## Stage-by-Stage Outcomes (Current vs Desired)

### Stage: Ingestion
Current (baseline):
- Extract/resolve run, but re-running for diagnosis requires manual resets or code changes.

Desired:
- Deterministic “force re-extract” for a doc id to reproduce extraction/resolution issues.

Backend changes supporting desired:
- `/ingest/{doc_id}/extract?force=true`.

### Stage: Filtering / Matching (attachments)
Current (baseline):
- Source Bin attachments can become "stuck" as claim/doc-scoped rows, preventing reuse.

Desired:
- Global cited PDFs remain global; claim-scoped placements are per-claim clones.

Backend changes supporting desired:
- `clone_attachment()` + `/attachments/{id}/clone`.

### Stage: Reading
Current (baseline):
- Evidence review depends on sentence artifacts; seed data may lack `sentences.ndjson`.

Desired:
- Reading/evidence stays usable with TEI-only artifacts.

Backend changes supporting desired:
- TEI->sentences synthesis fallback.

### Stage: Chasing
Current (baseline):
- Manual association may not create durable doc-level mapping; graph still shows unresolved cited nodes.

Desired:
- When a cited PDF is associated once, future auto-place resolves the same reference target to the ingested cited work.

Backend changes supporting desired:
- `link_reference_to_ingest()` invoked on placement/clone.
- Best-effort propagation to `doc_key` nodes.

### Stage: Surfing (Graph navigation)
Current (baseline):
- Graph can show placeholders for cited targets without showing the cited work node.

Desired:
- Cited work nodes appear once mapping exists; graph uses `ingest_ids` to render “available” work nodes.

Backend changes supporting desired:
- Reference->ingest linking and ingest id propagation.

## Recommendation: What To Retain

High value / low risk:
- Clone attachments (`clone_attachment`, `AttachmentCloneRequest`, `/attachments/{id}/clone`).
- Link reference->ingest mapping in graph (`link_reference_to_ingest`).
- Trigger evidence rerun after placement/clone.
- Forced extraction (`force=true`).
- Judgment JSON decoding helper.

Useful but optional (confirm before keeping):
- TEI->sentences synthesis fallback (adds `lxml` dependency usage).
- NLI_DISABLE shortcut (changes behavior based on env).
