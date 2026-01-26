# Roadmap: OS-ERIN

## Overview

OS-ERIN delivers an end-to-end citation validation workflow, starting with reliable PDF ingestion and citation extraction, then moving through claim review, evidence matching, and human judgment capture. The phases build progressively from structured extraction to evidence-backed validation and exportable judgments, with lightweight workspace organization utilities at the end.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Ingestion + Extraction** - PDFs ingested, parsed, and references normalized.
- [x] **Phase 2: Citation Context Navigation** - Citation callouts and trees are navigable.
- [ ] **Phase 3: Claim Selection + Editing** - Claims are extracted and editable for review.
- [ ] **Phase 4: Evidence Attachment** - Cited PDFs are attached and parsed for evidence search.
- [ ] **Phase 5: Evidence Matching + Ranking** - Evidence candidates are matched and ranked.
- [ ] **Phase 6: Evidence Review + Selection** - Evidence is highlighted and selectable.
- [ ] **Phase 7: Validation + Export** - Judgments are recorded and exportable.
- [ ] **Phase 8: Workspace Organization** - Folder browsing and retrieval utilities round out the workflow.

## Phase Details

### Phase 1: Ingestion + Extraction
**Goal**: Users can ingest PDFs and see structured metadata, citations, and normalized references.
**Depends on**: Nothing (first phase)
**Requirements**: ING-01, ING-02, ING-03, ING-04, ING-05
**Success Criteria** (what must be TRUE):
  1. User can upload a PDF and see it available for analysis.
  2. User can view extracted metadata (title, authors, year) for the uploaded PDF.
  3. User can view in-text citations and bibliography entries extracted from the PDF.
  4. User can see resolved reference identifiers with confidence when available.
**Plans**: 4 plans
Plans:
- [x] 01-01-PLAN.md — Local ingestion storage and upload API
- [x] 01-02-PLAN.md — GROBID extraction and metadata/citation parsing
- [x] 01-03-PLAN.md — Reference resolution via Crossref
- [x] 01-04-PLAN.md — Streamlit ingestion UI

### Phase 2: Citation Context Navigation
**Goal**: Users can navigate citation context and follow cited works for validation.
**Depends on**: Phase 1
**Requirements**: NAV-01, NAV-02, NAV-03
**Success Criteria** (what must be TRUE):
  1. User can open citation context for a selected callout.
  2. User can follow a citation to load cited context into the validation pane.
  3. User can browse a citation tree of referenced and cited-by works when available.
**Plans**: 6 plans
Plans:
- [x] 02-01-PLAN.md — Backend citation context and graph endpoints
- [x] 02-02-PLAN.md — Streamlit callout navigation and citation tree UI
- [x] 02-03-PLAN.md — OpenAlex citation graph expansion wiring
- [x] 02-04-PLAN.md — Callout context selection fixes
- [x] 02-05-PLAN.md — GROBID consolidation + Crossref/OpenAlex mismatch resolution
- [x] 02-06-PLAN.md — Mismatch UI bibliography context + consolidation flag

### Phase 3: Claim Selection + Editing
**Goal**: Users can review and refine extracted claims from citing text.
**Depends on**: Phase 2
**Requirements**: CLM-01, CLM-02
**Success Criteria** (what must be TRUE):
  1. User can see candidate claims segmented from citing text.
  2. User can select and edit an extracted claim before validation.
**Plans**: 3 plans
Plans:
- [ ] 03-01-PLAN.md — Claim queue state and segmentation helpers
- [ ] 03-02-PLAN.md — Queue rendering and parsing confirmation UI
- [ ] 03-03-PLAN.md — Modal claim editor with autosave and reset

### Phase 4: Evidence Attachment
**Goal**: Users can attach cited PDFs and prepare them for evidence retrieval.
**Depends on**: Phase 3
**Requirements**: EVD-01, EVD-02, EVD-03
**Success Criteria** (what must be TRUE):
  1. User can request retrieval instructions for a cited work from the claim context.
  2. User can drop a cited PDF onto a claim and see it attached.
  3. User can see the attached cited PDF parsed and ready for evidence search.
**Plans:** 3 plans
Plans:
- [ ] 04-01-PLAN.md — Retrieval dossier API + claim UI action
- [ ] 04-02-PLAN.md — Attachment queue UX with drop surfaces + accessibility fallback
- [ ] 04-03-PLAN.md — Attachment persistence + parsing pipeline integration

### Phase 5: Evidence Matching + Ranking
**Goal**: Users receive ranked evidence candidates that link claims to cited text.
**Depends on**: Phase 4
**Requirements**: EVD-04, EVD-05, EVD-06
**Success Criteria** (what must be TRUE):
  1. User can view evidence candidates matched to the selected claim.
  2. User sees a reranked list of top evidence options for the claim.
  3. User sees top-N evidence candidates with entail/contradict labels per ranking rules.
**Plans**: TBD

### Phase 6: Evidence Review + Selection
**Goal**: Users can inspect evidence in-context and choose the best match.
**Depends on**: Phase 5
**Requirements**: EVD-07, EVD-08
**Success Criteria** (what must be TRUE):
  1. User sees evidence spans highlighted in the cited PDF.
  2. User can select an entailing span or choose "none of the above".
**Plans**: TBD

### Phase 7: Validation + Export
**Goal**: Users can record verdicts, annotate them, and export results.
**Depends on**: Phase 6
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04
**Success Criteria** (what must be TRUE):
  1. User can record a verdict (support/contradict/uncertain) for a claim.
  2. User can add notes to a judgment.
  3. Citation callouts show validated vs unvalidated status.
  4. User can export judgments and metadata as CSV or JSON.
**Plans**: TBD

### Phase 8: Workspace Organization
**Goal**: Users can navigate their PDF workspace and copy retrieval details.
**Depends on**: Phase 7
**Requirements**: WS-01, WS-02
**Success Criteria** (what must be TRUE):
  1. User can browse a project folder structure for PDFs.
  2. User can copy citation retrieval info (author/year/DOI) to clipboard.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 2 → 2.1 → 2.2 → 3 → 3.1 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Ingestion + Extraction | 4/4 | Complete | 2026-01-23 |
| 2. Citation Context Navigation | 6/6 | Complete | 2026-01-24 |
| 3. Claim Selection + Editing | 0/TBD | Not started | - |
| 4. Evidence Attachment | 0/TBD | Not started | - |
| 5. Evidence Matching + Ranking | 0/TBD | Not started | - |
| 6. Evidence Review + Selection | 0/TBD | Not started | - |
| 7. Validation + Export | 0/TBD | Not started | - |
| 8. Workspace Organization | 0/TBD | Not started | - |
