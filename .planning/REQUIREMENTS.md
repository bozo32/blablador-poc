# Requirements: OS-ERIN

**Defined:** 2026-01-23
**Core Value:** Readers can validate a citation by linking a claim to supporting (or contradicting) evidence from the cited source, with the judgment captured and reusable.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Ingestion + Extraction

- [ ] **ING-01**: User can upload a PDF for analysis
- [ ] **ING-02**: System parses uploaded PDFs with GROBID into TEI/XML
- [ ] **ING-03**: System extracts metadata (title, authors, year) from PDFs
- [ ] **ING-04**: System extracts in-text citations and bibliography entries
- [ ] **ING-05**: System resolves references to canonical identifiers (DOI/OpenAlex/Crossref) with confidence

### Citation Navigation + Context

- [x] **NAV-01**: User can view citation context (sentence/section) for a selected callout
- [x] **NAV-02**: User can follow a citation to load cited context into the validation pane
- [x] **NAV-03**: User can browse a citation tree of referenced and cited-by works when available

### Claim + Evidence

- [ ] **CLM-01**: System segments citing text into candidate claims
- [ ] **CLM-02**: User can select and edit extracted claim parses
- [x] **EVD-01**: User can request retrieval instructions for a cited work from the claim context
- [x] **EVD-02**: User can drop a retrieved cited PDF onto the claim context to attach it
- [x] **EVD-03**: System parses dropped cited PDFs with GROBID into TEI/XML
- [ ] **EVD-04**: System runs deterministic matching to find closest cited spans to the claim
- [ ] **EVD-05**: System reranks candidate spans and shows top evidence options
- [ ] **EVD-06**: System presents top-N evidence candidates (default 3 entailing + 3 contradicting when strongest contradicting outranks weakest entailing)
- [ ] **EVD-07**: Evidence spans are highlighted/anchored in the cited PDF
- [ ] **EVD-08**: User can select an entailing span or choose "none of the above"

### Validation + Judgment

- [x] **VAL-01**: User can record a verdict for a claim vs cited source (support/contradict/uncertain)
- [x] **VAL-02**: User can add annotations/notes to a judgment
- [x] **VAL-03**: Citation shows validation status (validated vs unvalidated)
- [x] **VAL-04**: User can export judgments and metadata (CSV/JSON)

### Workspace + Organization

- [ ] **WS-01**: User can browse project folder structure for PDFs
- [ ] **WS-02**: User can copy citation retrieval info (author/year/DOI) to clipboard

### UX + Workflow Acceleration

- [ ] **UX-01**: Citation interactions are inline (no end-of-block citation rows)
  - In-document citations are underlined/inline clickable (chips are optional, not required).
  - End-of-block citation button rows are removed.

- [ ] **UX-02**: Workspace layout is vertically dense and settings are tucked away
  - The large header is removed from the top of the page (still accessible, not top-chrome).
  - Settings move behind a gear/drawer (popover/slide-out).
  - A “dense” mode reduces vertical padding/spacing without hiding core actions.

### Robust Segmentation

- [ ] **SEG-01**: Sentence segmentation fallback when TEI/GROBID segmentation looks wrong
  - Auto-detect heuristics trigger fallback segmentation.
  - Fallback segmenter produces stable sentence boundaries for UI navigation.
  - Citation selection remains reliable even when fallback is used.

### Source Ingestion + Background Processing

- [ ] **ATT-01**: Global “Source bin” upload (no per-claim/source-specific drop targets)
  - User can drop/upload source PDFs into a single shared bin.
  - Uploaded PDFs are stored and tracked as “unassigned sources” until matched.

- [ ] **ATT-02**: Background processing and prefetch
  - Immediately after upload: conversion + GROBID parse runs in the background.
  - After claims are saved: evidence runs start automatically for relevant sources so results are ready by arrival.

### PDF Review Ergonomics

- [ ] **PDF-01**: “View PDF” action copies a short excerpt snippet for Cmd-F
  - In chasing claims view, a button opens the source PDF and copies a deterministic 3–4 word snippet from the excerpt to the clipboard.

### Recursive Retrieval + Graph Navigation

- [ ] **NAV-04**: Recursive retrieval across a citation tree
  - User can follow citations from cited sources to additional sources and keep expanding the tree.

- [ ] **GPH-01**: Node graph is validated and integrated into navigation
  - Graph is not a dead-end tab; selecting nodes routes into the chase/workspace view.

### Demo / Tech-Showcase

- [ ] **DEMO-01**: Synthetic demo corpus + replayable trace for an end-to-end tech demo
  - Dataset includes 5–15 PDFs with a known citation graph.
  - Includes seeded artifacts (attachments parsed, evidence selections, judgments) to avoid waiting during demos.
  - Includes a replayable “trace” (event log) of a realistic walk + annotate session.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Workspace + Organization

- **WS-03**: User can manage a codebook of applied tags for text segments

### Collaboration + Analytics

- **COL-01**: Multiple reviewers can collaborate and adjudicate judgments
- **COL-02**: Disagreement heatmap highlights conflicting evidence across claims

### Validation + Judgment (v2)

- [ ] **VAL-05**: Claim validation supports a 3-level assessment
  - **Level 1 (Equivalence)**: Reviewer selects the equivalent cited-source statement from top-N candidate matches; no justification required.
  - **Level 2 (Valid In Source)**: Reviewer records whether the selected source statement is valid within the cited source (support/contradict/uncertain) and MUST provide a justification annotation describing what validation was performed (e.g., study quality assessment, primary-source retrieval for secondary citation).
  - **Level 3 (Transferable To Citing Context)**: Reviewer records whether the source-valid statement transfers to the citing context (support/contradict/uncertain) and MUST provide a justification annotation listing the variables/moderators considered for transportability.

- [ ] **VAL-06**: Required justification annotations are enforced for finalization
  - **Rule:** Level 2 and Level 3 cannot be marked Final unless their annotation contains at least one non-empty field.
  - **Rule:** Level 1 never requires annotation.
  - **Rule:** Callout "validated" status is driven by Level 3 being Final (not Level 2).

- [ ] **VAL-07**: Shared annotation schema with context-specific prompts
  - **Model:** Level 2 and Level 3 reuse the same structured annotation fields but show different prompt/help text.
  - **Minimum fields (structured):** `method` (what validation was done), `variables` (list), `rationale`, `caveats`, `followups`, `evidence_type`.
  - **UI:** Level 2 shows "valid-in-source" placeholder text; Level 3 shows "transportability" placeholder text.

- [ ] **VAL-08**: Export includes tri-level assessment explicitly
  - **Claims export:** includes Level 1 selection (selected span metadata), Level 2 verdict + annotation, Level 3 verdict + annotation.
  - **Callouts export:** groups claims and includes Level 3 outcome used for validation state.
  - **Defaults:** export remains final-only by default, with include-drafts toggle.

### Intelligence + Expansion

- **INT-01**: Smart citation classification (support/contradict/mention) at scale
- **INT-02**: Multi-hop citation trail exploration for indirect evidence chains
- **INT-03**: Retraction/errata alerts linked to cited works

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| EOSC node governance and coalition planning | Tracked in CIT-EOSC proposal, not part of this app |
| Large-corpus indexing and batch validation | v1 focuses on single-paper workflow |
| Fully automated validation without human judgment | Requires human adjudication to ensure trust |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ING-01 | Phase 1 | Complete |
| ING-02 | Phase 1 | Complete |
| ING-03 | Phase 1 | Complete |
| ING-04 | Phase 1 | Complete |
| ING-05 | Phase 1 | Complete |
| NAV-01 | Phase 2 | Complete |
| NAV-02 | Phase 2 | Complete |
| NAV-03 | Phase 2 | Complete |
| CLM-01 | Phase 3 | Pending |
| CLM-02 | Phase 3 | Pending |
| EVD-01 | Phase 4 | Complete |
| EVD-02 | Phase 4 | Complete |
| EVD-03 | Phase 4 | Complete |
| EVD-04 | Phase 5 | Pending |
| EVD-05 | Phase 5 | Pending |
| EVD-06 | Phase 5 | Pending |
| EVD-07 | Phase 6 | Pending |
| EVD-08 | Phase 6 | Pending |
| VAL-01 | Phase 7 | Complete |
| VAL-02 | Phase 7 | Complete |
| VAL-03 | Phase 7 | Complete |
| VAL-04 | Phase 7 | Complete |
| WS-01 | Phase 8 | Pending |
| WS-02 | Phase 8 | Pending |
| UX-01 | Phase 8 | Pending |
| UX-02 | Phase 8 | Pending |
| SEG-01 | Phase 8 | Pending |
| ATT-01 | Phase 8 | Pending |
| ATT-02 | Phase 8 | Pending |
| PDF-01 | Phase 8 | Pending |
| NAV-04 | Phase 8.1 | Pending |
| GPH-01 | Phase 8.1 | Pending |
| DEMO-01 | Phase 8.2 | Pending |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 33
- Unmapped: 0

---
*Requirements defined: 2026-01-23*
*Last updated: 2026-02-02 after Phase 8 planning expansion*
