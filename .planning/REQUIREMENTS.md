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

- [ ] **NAV-01**: User can view citation context (sentence/section) for a selected callout
- [ ] **NAV-02**: User can follow a citation to load cited context into the validation pane
- [ ] **NAV-03**: User can browse a citation tree of referenced and cited-by works when available

### Claim + Evidence

- [ ] **CLM-01**: System segments citing text into candidate claims
- [ ] **CLM-02**: User can select and edit extracted claim parses
- [ ] **EVD-01**: User can request retrieval instructions for a cited work from the claim context
- [ ] **EVD-02**: User can drop a retrieved cited PDF onto the claim context to attach it
- [ ] **EVD-03**: System parses dropped cited PDFs with GROBID into TEI/XML
- [ ] **EVD-04**: System runs deterministic matching to find closest cited spans to the claim
- [ ] **EVD-05**: System reranks candidate spans and shows top evidence options
- [ ] **EVD-06**: System presents top-N evidence candidates (default 3 entailing + 3 contradicting when strongest contradicting outranks weakest entailing)
- [ ] **EVD-07**: Evidence spans are highlighted/anchored in the cited PDF
- [ ] **EVD-08**: User can select an entailing span or choose "none of the above"

### Validation + Judgment

- [ ] **VAL-01**: User can record a verdict for a claim vs cited source (support/contradict/uncertain)
- [ ] **VAL-02**: User can add annotations/notes to a judgment
- [ ] **VAL-03**: Citation shows validation status (validated vs unvalidated)
- [ ] **VAL-04**: User can export judgments and metadata (CSV/JSON)

### Workspace + Organization

- [ ] **WS-01**: User can browse project folder structure for PDFs
- [ ] **WS-02**: User can copy citation retrieval info (author/year/DOI) to clipboard

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Workspace + Organization

- **WS-03**: User can manage a codebook of applied tags for text segments

### Collaboration + Analytics

- **COL-01**: Multiple reviewers can collaborate and adjudicate judgments
- **COL-02**: Disagreement heatmap highlights conflicting evidence across claims

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
| NAV-01 | Phase 2 | Pending |
| NAV-02 | Phase 2 | Pending |
| NAV-03 | Phase 2 | Pending |
| CLM-01 | Phase 3 | Pending |
| CLM-02 | Phase 3 | Pending |
| EVD-01 | Phase 4 | Pending |
| EVD-02 | Phase 4 | Pending |
| EVD-03 | Phase 4 | Pending |
| EVD-04 | Phase 5 | Pending |
| EVD-05 | Phase 5 | Pending |
| EVD-06 | Phase 5 | Pending |
| EVD-07 | Phase 6 | Pending |
| EVD-08 | Phase 6 | Pending |
| VAL-01 | Phase 7 | Pending |
| VAL-02 | Phase 7 | Pending |
| VAL-03 | Phase 7 | Pending |
| VAL-04 | Phase 7 | Pending |
| WS-01 | Phase 8 | Pending |
| WS-02 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Requirements defined: 2026-01-23*
*Last updated: 2026-01-23 after Phase 1 completion*
