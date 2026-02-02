# Architecture

This repo is a local-first citation checking tool with a Streamlit UI and a FastAPI backend.

## High-level

- `frontend/`: Streamlit UI + client-side state helpers.
- `backend/`: FastAPI app + retrieval/matching pipeline.
- `data/`: local artifacts (attachments, evidence runs, selections, caches). Not intended for git.
- `.planning/`: planning docs and project state.

## Core workflow

1) Ingest a citing document
- UI uploads a PDF and triggers extraction + reference resolution.

2) Select citations and chase claims
- `Document text` shows citation callouts.
- User follows citations into a `Chase queue` and segments them into claims.

3) Attach cited sources
- User uploads cited PDFs.
- Backend processes PDFs into TEI, extracts sentences/paragraphs, and matches candidate spans.

4) Evidence review
- UI requests an evidence rerun for a claim.
- Backend seeds candidate windows (BM25/FAISS), runs NLI, and returns ranked candidates.

5) Human decision capture
- Reviewer labels candidates and records an overall source assessment.

## Backend

- App: `backend/main.py` (FastAPI).
- Evidence matching:
  - `backend/evidence_matching/service.py`: orchestrates runs, history, attachment snapshots.
  - `backend/evidence_matching/deterministic_matcher.py`: deterministic seeding + dedupe.
  - `backend/nli.py`: NLI scoring (truncation, batch control, MPS fallback).
- Attachments:
  - `backend/attachment_pipeline.py`: TEI parsing + sentence/paragraph extraction.

Design rules:
- Read endpoints must not trigger expensive compute.
- Expensive model calls are behind explicit rerun/processing triggers.

## Frontend

- Main UI: `frontend/ui.py`.
- Reusable UI blocks: `frontend/components/*`.
  - `frontend/components/chase_queue.py`: VSCode-like collapsible chase queue.

State rules:
- Keep one canonical session-state key per data item.
- Any duplicated widgets must use unique widget keys and sync to the canonical value.

## Verification

- Automated: `pytest -q`.
- Manual smoke:
  - chase queue grows and supports open/close
  - switching citations prompts on unsaved edits
  - evidence rerun works without triggering compute on read-only GET
  - attachments UI groups by source (UI-level), even if backend is claim-scoped
