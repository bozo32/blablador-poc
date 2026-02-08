# Span-First Graph Model (Work -> Span -> Claim)

This document defines a span-first knowledge graph model that supports exploration and analysis at three levels:

- `Work` (paper/work)
- `Span` (anchored citing span within a work; may cite multiple works)
- `ClaimSpan` + `ClaimAtom` (segmented claims composed from a span)

The intent is to make the project structurally correct for:

- multi-citation spans
- secondary citation chains
- joint conditionality across claim segments
- neutral / reputational citation behavior
- multi-reviewer disagreement with attributable rationales

It is written as core architecture documentation: (1) what must be representable and why it matters, and (2) the data structure that is adequate to support it.

## Terms (Non-Negotiable Distinctions)

- `Work`: a scholarly work (paper/book/etc). Identity is best-effort canonical (DOI/OpenAlex), but duplicates may exist early.
- `Ingest`: a local PDF instance we processed. An ingest may map to a `Work` (or remain unmapped).
- `Span`: an anchored text span within a `Work` (or `Ingest`) located by robust selectors (quote selector + fingerprint), not by offsets.
- `CiteSpan`: a `Span` with one or more citations (`Span -> Work` edges). This is the provenance origin of work-to-work citation.
- `ClaimSpan`: an anchored sub-span (or segment) that composes a `Span` and serves as the infrastructural anchor for user-created claims.
- `ClaimAtom`: a user-generated proposition (semantic unit). Not necessarily anchored to text; versionable; merge/split friendly.
- `Assertion`: a reviewer-attributed act of relating a `ClaimAtom` (or `ClaimSpan`) to evidence (`Span` / `Work`) with a verdict.

Key principle: *provenance is structural; epistemology is asserted.* Do not store subjective judgments as if they are structural truth.

## 1) Conditions We Must Capture (and Why They Matter)

### C1. Three-level exploration: Work, citing span, claim segments
Must capture:
- `Work` -> `Span` -> `ClaimSpan` composition
Why it matters:
- Users need to see where a citation link comes from (which in-text span) and then how that span decomposes into evaluable claims.

### C2. A citing span may cite multiple works
Must capture:
- `Span` has 0..N outgoing `CITES` edges to `Work` nodes
Why it matters:
- Many citations are bundled (e.g., “see A; B; C”), and the span’s rhetorical function may be satisfied by any subset.

### C3. A citing span may be “validated” by one of several cited works (OR satisfaction)
Must capture:
- Mapping from claim-segment requirements to candidate cited works is not necessarily 1:1
- Some claims are satisfied if *any* cited work supports them
Why it matters:
- Otherwise the system forces false precision (“all cited works support this”) or blocks progress.

### C4. Neutral / reputational citing is first-class (not a bug)
Must capture:
- Assertions can be `neutral` or “background/reputational” without being treated as evidence for a claim.
Why it matters:
- Credibility assessment depends on whether the author is citing for authority vs evidence; neutral must remain visible.

Note:
- This can be represented either as assertion metadata and/or as a reviewer-attributed role annotation on `Span -> Work` citation edges (e.g., `evidentiary|background|reputational|unknown`).

### C5. Joint conditionality across claim segments (AND composition)
Must capture:
- A span may contain multiple `ClaimSpan`s where all must be satisfied for the span to be adequately backed
Why it matters:
- “A and B” is not supported if only A is supported.

### C6. Mixed satisfaction: multiple claim segments satisfied by different cited works
Must capture:
- Claim segment 1 supported by cited work X; claim segment 2 supported by cited work Y
Why it matters:
- Prevents forcing a single “one paper backs the whole thing” story.

### C7. Contradiction across cited works can undermine adequacy
Must capture:
- Contradicting evidence may coexist with supporting evidence for a claim segment
- “Contested” is a stable state distinct from “supported”
Why it matters:
- Real-world discourse includes disputes; surfacing contestation is critical for trust and reviewer coordination.

### C8. Secondary citation (A cites B, B cites C) without type collisions
Must capture:
- The same anchored text in Work B can be used as evidence for A’s claim (as a `Span`) while also being a `CiteSpan` that cites Work C
Why it matters:
- If “EvidenceSpan” and “ClaimSpan” are disjoint types, secondary citation forces duplication and breaks provenance.

### C9. Multi-reviewer disagreement is attributable and persists
Must capture:
- Different reviewers can record different assertions (support/contradict/neutral) for the same claim segment and evidence
- Disagreement is a first-class object, not overwritten state
Why it matters:
- Collaboration requires seeing *who* disagrees, *why*, and where disagreements cluster.

### C10. Stable anchoring across reprocessing and segmentation drift
Must capture:
- Anchors should survive PDF re-extraction and reviewer segmentation differences
Why it matters:
- Offsets and local reference IDs drift; without stable anchors, the graph becomes non-reproducible.

### C11. Auditability: we can reconstruct “what did we believe then?”
Must capture:
- Assertions are time/version-stamped and can be replayed
Why it matters:
- Review is iterative; we need a timeline for debugging and demo trace replay.

### C12. Incremental adoption with backward compatibility
Must capture:
- Existing claim workflow continues to function during migration
Why it matters:
- A full rewrite that breaks the app is not acceptable; we need an adapter period.

### C13. Neighborhood contradiction search to detect cherry-picking
Must capture:
- The system can surface *non-cited* nearby works that are likely relevant and potentially contradictory
- Neighborhood is generated from bibliographic intersection (e.g., bibliographic coupling / co-citation) and filtered by abstract relevance
Why it matters:
- Cherry-picking is empirically common; credibility assessment requires actively looking for nearby disagreement, not only evaluating the cited set.

## 2) Data Structure Adequate to Support These Conditions

### 2.1 Node Types

#### `Work`
Represents a paper/work.

Minimal properties:
- `work_id` (string; canonical preferred)
- `doi` (optional)
- `title`, `authors`, `year` (optional)
- `abstract` / `abstract_embedding` (optional; enables neighborhood filtering)

#### `Ingest`
Represents a processed local PDF.

Minimal properties:
- `ingest_id` (uuid)
- `sha256`
- `filename` (+ aliases)
- `work_id` (optional link to `Work`)

#### `Span`
Anchored text span within a citing work/ingest.

Minimal properties:
- `span_id` (deterministic hash)
- `work_id` (or `ingest_id` until mapped)
- `selector` (quote selector: exact/prefix/suffix)
- `window_fingerprint` (sha256 of normalized window text)
- `kind` (e.g., `citation_window`, `evidence_excerpt`, `other`)

Notes:
- `Span` is the unifying infrastructure object; a `Span` can be used as evidence *and* can itself cite other works.

#### `ClaimSpan`
Anchored claim segment derived from a `Span`.

Minimal properties:
- `claim_span_id` (deterministic hash of `span_id` + claim selector)
- `span_id` (parent)
- `selector` (quote selector or segment selector)
- `order_index` (for UI ordering)

#### `ClaimAtom`
User-generated proposition.

Minimal properties:
- `claim_atom_id` (uuid)
- `text` (normalized string)
- `created_by` (reviewer uid)
- `revision_of` / `supersedes` (optional)

#### `Assertion`
Reviewer-attributed assessment linking claim to evidence.

Minimal properties:
- `assertion_id` (uuid)
- `reviewer_uid`
- `verdict` (`support|contradict|neutral|uncertain`)
- `confidence` (0..1 optional)
- `comment` (optional)
- `created_at`
- `policy_version` (optional)

#### `NeighborhoodRun` (optional, recommended)
Represents a search run that proposes nearby works for contradiction checking.

Minimal properties:
- `run_id` (uuid)
- `seed_work_ids` (the works/spans we started from)
- `method` (e.g., `bib_coupling`, `co_citation`)
- `filters` (e.g., abstract similarity threshold)
- `created_at`

### 2.2 Edge Types

Structural edges (canonical-ish):
- `IN_WORK`: `Span -> Work` (where the span lives)
- `CITES`: `Span -> Work` (span cites 0..N works)  (C2)
- `COMPOSED_OF`: `Span -> ClaimSpan` (span is composed of claim segments) (C1/C5)
- `REALIZES`: `ClaimSpan -> ClaimAtom` (claim segment maps to one or more claim atoms) (bridge)

Reviewer-attributed citation role annotations (interpretive, not canonical structure):
- `CITE_ROLE`: `(Span, Work, reviewer_uid) -> role` where role is `evidentiary|background|reputational|unknown`.

Structural edges (bibliographic network, optional but important for C13):
- `WORK_CITES`: `Work -> Work` (work-level citation edges from external metadata; not derived from a local span)

Epistemic edges (subjective, attributed):
- `ABOUT`: `Assertion -> ClaimAtom` (or `Assertion -> ClaimSpan` in an MVP)
- `EVIDENCE`: `Assertion -> Span` (evidence is a span; supports secondary citation cleanly) (C8)
- `EVIDENCE_WORK`: `Assertion -> Work` (optional shorthand when you only know the cited work, not the excerpt)

Discovery edges (derived, provenance-heavy; keep separate from canonical structure):
- `DISCOVERED`: `NeighborhoodRun -> Work` with scores/metrics (`bib_intersection`, `abstract_score`)

Optional disambiguation edges (recommended as complexity rises):
- `REQUIRES`: `ClaimSpan -> Requirement` and `Requirement -> Work` candidates
  - This allows “these are candidate cited works for this claim segment” without assuming all citations are evidentiary.

### 2.3 Computed Views (Not Stored as Primary Truth)

To avoid conflating assertion and structure, compute these on demand:

- `ClaimAtomStatus(reviewer)`: derived from that reviewer’s assertions (support/contradict/neutral)
- `ClaimSpanStatus(reviewer)`: aggregation of attached claim atoms
- `SpanAdequacy(reviewer)`: AND over `ClaimSpanStatus`, with contested propagation
- `CherryPickSignals`: derived by comparing what is cited/used vs what the neighborhood suggests (C13)

#### Status lattice (v1)

We use a small set of computed statuses at multiple levels (ClaimAtom, ClaimSpan, Span, and across reviewers):

- `unknown`: not yet assessed (no support/contradict evidence recorded; no explicit "checked" mark)
- `supported`: at least one support assertion exists and no contradict assertion exists
- `contradicted`: at least one contradict assertion exists and no support assertion exists
- `contested`: both support and contradict assertions exist (conflict)
- `not_supported`: assessed but no support evidence exists (see Notes)

Notes:
- `neutral` and `uncertain` assertions are preserved and shown.
- `unknown` is the safe default. We intentionally keep this state so later revisions can map previously-unrepresented cases into a new explicit status without losing information.
- `not_supported` is reserved for cases where a reviewer has actively checked and still found no supporting evidence.
  - v1 approximation: if a reviewer recorded any `neutral`/`uncertain` evidence assertions for a claim but no `support`/`contradict`, treat it as `not_supported`.
  - v2 improvement: add an explicit reviewer "checked" marker (e.g., `review_marks`) so `unknown` vs `not_supported` is not inferred.
- `contested` is reserved for actual conflict (support vs contradict), or for disagreement when aggregating across reviewers.

#### Aggregation rules (v1)

Within a ClaimSpan / ClaimAtom (OR across evidence):
- Compute status from the set of assertions using the lattice above.

Span adequacy across composed ClaimSpans (AND):
- If any child is `contradicted` => span is `contradicted`
- Else if any child is `not_supported` => span is `not_supported`
- Else if any child is `contested` => span is `contested`
- Else if any child is `unknown` => span is `unknown`
- Else => span is `supported`

Across reviewers (consensus view):
- If all reviewers' computed statuses match => that status
- Else => `contested`

Defaults:
- OR within a claim segment (any evidence span can support)
- AND across claim segments (span is adequately backed only if all segments are supported)
- Contradiction yields `contested` unless policy explicitly overrides

### 2.4 Identity and Anchoring

Stable IDs:
- `span_id = hash(work_id, selector, window_fingerprint, policy_version)`
- `claim_span_id = hash(span_id, claim_selector, order_index)`

Anchors must be robust to reprocessing:
- Prefer quote selector + fingerprint; do not rely on raw offsets as the stored anchor.

### 2.5 How This Captures the Hard Cases

- Multi-citation span: one `Span` with multiple `CITES` edges. Assertions choose which cited work/spans matter. (C2/C3)
- Reputational citing: assertion verdict `neutral` + optional role metadata; still attached to `Span` for audit. (C4)
- Joint conditionality: multiple `ClaimSpan`s under one `Span`; span adequacy is AND across segments. (C5)
- Mixed satisfaction: claim segment 1 assertions point to evidence in Work X; claim segment 2 to Work Y. (C6)
- Contradiction undermines: contradictory assertions make `contested`; span adequacy propagates. (C7)
- Secondary citation: `Assertion(EVIDENCE -> Span_B)` while `Span_B CITES Work_C`; no type collision required. (C8)
- Multi-reviewer disagreement: assertions are per reviewer; consensus is a computed aggregate. (C9)
- Neighborhood contradiction search: `NeighborhoodRun` proposes works via bibliographic network + abstract filter; reviewers can then attach assertions (contradict/support/neutral) referencing those works/spans. (C13)

## Non-Goals (For First Implementation Pass)

- Perfect canonical work identity resolution (duplicates acceptable; provide merge tooling later).
- Perfect span re-resolution in all PDFs (start with best-effort + fallbacks).
- Fully automated claim-atom clustering across reviewers (start with per-reviewer atoms; add clustering later).
