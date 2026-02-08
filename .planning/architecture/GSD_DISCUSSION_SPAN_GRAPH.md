# GSD Discussion Input: Span-First Graph + Neighborhood Contradiction Search

This document is the discussion-oriented version of:

- `.planning/architecture/SPAN_GRAPH_MODEL.md` (architecture + conditions)
- `.planning/architecture/SPAN_GRAPH_SPECS.md` (implementable MVP spec)
- `.planning/architecture/SPAN_GRAPH_FEASIBILITY.md` (adequacy + feasibility)

Goal: make it easy to step through the design, surface disagreements early, and lock decisions before execution.

## 1) What We Are Building (One-liner)

A span-first knowledge graph where:

- `Work` nodes connect via `Span` provenance (in-text citation windows)
- `Span` decomposes into `ClaimSpan`s that map to user `ClaimAtom`s
- reviewers record `Assertion`s (support/contradict/neutral/uncertain) referencing evidence as `Span`/`Work`
- the system can also propose *non-cited* nearby works via a bibliographic neighborhood search to detect cherry-picking

## 2) Why We’re Doing This (Problems the current system can’t represent cleanly)

- Per-claim source placement makes multi-claim citing spans brittle (10a has sources, 10b doesn’t).
- Secondary citation breaks simple “evidence span” types (evidence can itself be a citing span).
- Multi-reviewer work needs persistent, attributable disagreement rather than overwriting state.
- Credibility assessment requires seeing neutral/reputational citations and searching nearby contradictions.

## 3) Requirements / Conditions (Must Capture)

Structural/provenance:
- C1: Work -> Span -> ClaimSpan exploration
- C2: Span cites multiple works
- C5: ClaimSpans compose a Span (AND across segments)
- C8: Secondary citation with no type collision
- C10: Stable anchoring across reprocessing

Epistemic:
- C3: OR satisfaction (one of several cited works may support)
- C4: Neutral/reputational citation is first-class
- C7: Contradiction yields contested states
- C9: Multi-reviewer disagreement is attributable and persistent
- C11: Auditability / replay

Discovery:
- C13: Neighborhood contradiction search (bib intersection + abstract filter) to detect cherry-picking

## 4) Proposed Data Model (Key separation: structure vs assertion)

Canonical-ish structure nodes:
- `Work`: paper/work (DOI/OpenAlex/fingerprint)
- `Ingest`: local pdf instance (sha256, filename)
- `Span`: anchored text span within a work/ingest (quote selector + fingerprint)
- `ClaimSpan`: anchored segments composing a span

User semantics:
- `ClaimAtom`: user proposition (not anchored)

Epistemic events:
- `Assertion`: reviewer-attributed verdict relating a claim to evidence
  - verdict: `support|contradict|neutral|uncertain`
  - evidence endpoint: `Span` (preferred) and/or `Work` shorthand

Discovery/audit:
- `NeighborhoodRun`: a recorded run that proposes nearby candidate works (not necessarily cited)
- `work_cites`: work-level citation edges from external metadata (OpenAlex), used to generate neighborhoods

Computed views (not stored as ground truth):
- per-reviewer claim/claimspan/span status: supported/contested/unknown/background-only
- span adequacy: AND across claimspans; contradiction propagates to contested
- cherry-pick signals: compare cited/used evidence set vs discovered neighborhood candidates

Status lattice (v1):
- `unknown`: not yet assessed
- `supported`: support exists, no contradict
- `contradicted`: contradict exists, no support
- `contested`: support and contradict both exist OR reviewers disagree in consensus view
- `not_supported`: assessed but no supporting evidence exists

AND aggregation across claimspans (v1):
- if any child contradicted => parent contradicted
- else if any child not_supported => parent not_supported
- else if any child contested => parent contested
- else if any child unknown => parent unknown
- else => parent supported

v2 note (status hygiene):
- Add an explicit reviewer "checked" marker so `unknown` vs `not_supported` is not inferred from the presence/absence of neutral/uncertain evidence.

## 5) Implementation Plan (Incremental Cutover)

Phase 1: Persist structure in parallel
- On segmentation save/confirm: create `Span` + `ClaimSpan` rows
- Persist `Span -> Work` citations when known (from existing citation context)

Phase 2: Record assertions in parallel
- When user judges evidence for a claim: write an `Assertion` linked to claimspan/atom and evidence span/work

Phase 3: Neighborhood search
- Add `work_cites` + `works.abstract` fields
- Implement `POST /neighborhood/search` that:
  - builds candidate list via bibliographic intersection
  - ranks/filters via abstract similarity
  - persists `NeighborhoodRun` + `NeighborhoodCandidates`

Phase 4: UI cutover
- Graph tab becomes span/work-based explorer (Work graph -> expand to spans -> expand to claimspans)
- Evidence view becomes span-scoped (per-claim view becomes a filter over shared span evidence)

## 6) Decision Points (Need to Decide Before Execution)

These materially affect schema, migration, and correctness.

D1. Span anchoring source-of-truth
- Option A: anchor `Span` to `ingest_id` first, map to `work_id` later (recommended)
- Option B: require `work_id` early (cleaner conceptually, harder operationally)

D2. ClaimSpan scope
- Option A: canonical `ClaimSpan`s, reviewer-specific `ClaimAtom` + `Assertion` (recommended)
- Option B: reviewer-scoped `ClaimSpan`s (simpler, but consensus becomes messy)

D3. Evidence endpoint
- Option A: evidence is `Span` (recommended; supports secondary citation)
- Option B: evidence is `Work` only (naive; loses provenance)

D4. Neighborhood method v1
- Option A: bibliographic coupling (shared references) from OpenAlex
- Option B: co-citation (cited-by overlap)
- Option C: references-of-cited (two-hop)
Recommended v1: start with whichever OpenAlex endpoint is easiest/fastest + cached locally.

D5. Abstract similarity
- Option A: simple lexical TF-IDF-ish scoring (fast, no model deps)
- Option B: embeddings (better ranking, more infra)
Recommended v1: lexical, add embeddings later.

D6. Persistence of neighborhood runs
- Option A: persist only when user explicitly requests a search (recommended)
- Option B: always persist automatically (lots of noise)

### Decisions Locked (2026-02-08)

- D1: A (anchor `Span` to `ingest_id` first; map to `work_id` later)
- D2: A (canonical `ClaimSpan`s; reviewer-specific `ClaimAtom` + `Assertion`)
- D3: A (evidence endpoint is `Span`, with optional `Work` shorthand)
- D4: v1 neighborhood uses bibliographic intersection (user-configurable)
- D5: v1 similarity uses title/abstract/keyword similarity; see v2 note for user-curated statements
- D6: A (persist only when explicitly requested)

## 7) Open Questions / Likely Discussion Hotspots

Q1. What is the default “context” for neighborhood search?
- span-level (the specific citing span)
- claimspan/atom-level (the specific claim)
- work-level (whole paper)

Q2. How should neutral citations impact “adequacy”?
- neutral never counts as support
- neutral reduces suspicion / affects credibility narrative

Q3. How do we prevent “any cited work can satisfy any claimspan” confusion?
- Introduce `Requirement` objects (claimspan -> candidate works)
- or add assertion role metadata (`evidentiary|reputational|background`)

Q4. How do we represent cherry-picking signals in UI?
- show “nearby contradictory works exist” badge
- show candidate list grouped by relation (contradict/support/unknown)
- show “not cited but nearby” contrasts

## 8) Minimal Example (Sanity Check)

One citing span S in Work A cites Works B and C.

- `Span(S) CITES Work(B)`
- `Span(S) CITES Work(C)`
- `Span(S) COMPOSED_OF ClaimSpan(1), ClaimSpan(2)`
- Reviewer R asserts:
  - ClaimSpan(1) supported by Span in Work(B)
  - ClaimSpan(2) supported by Span in Work(C)
- Reviewer R2 asserts:
  - ClaimSpan(1) contradicted by Span in Work(C)

Computed:
- ClaimSpan(1) = contested (support + contradict)
- Span(S) adequacy = contested (AND over claimspans)

Neighborhood run:
- Starting from Work(B) and Work(C), find nearby works via bibliographic coupling
- Filter by abstract match to ClaimSpan(1) text
- Present candidates to reviewers as “possible contradiction checks”

## 9) Exit Criteria for the Discussion Round

- D1-D6 resolved with defaults written down
- UI representation for “discovered but not cited” agreed
- Confirmed that the incremental cutover won’t strand current workflows

## v2 Notes (Not Implemented in v1)

- Neighborhood search query curation: derive candidate "statements" from a work's title/abstract/keywords, present them to the user for editing/selection, and use the selected statements as the similarity filter input.
