# Feasibility Review: Span-First Graph Rebuild

This document evaluates whether `.planning/architecture/SPAN_GRAPH_SPECS.md` is:

1) adequate with respect to the architectural requirements/conditions, and
2) implementable in a sane structure within this repo.

## A) Adequacy vs Requirements

### Covers the core conditions

- C1 (3-level exploration): explicit `Work` + `Span` + `ClaimSpan` hierarchy.
- C2 (multi-citation): `span_cites` supports 0..N cited works per span.
- C3 (OR satisfaction): assertions attach to evidence spans/works; status is computed.
- C4 (neutral): `verdict=neutral` supported; role metadata can be added without schema change.
- C5 (AND composition): `Span -> ClaimSpan` composition supports AND adequacy.
- C6 (mixed satisfaction): each claim span/atom can reference different evidence spans.
- C7 (contradiction): contradictory assertions produce `contested` computed state.
- C8 (secondary citation): evidence endpoint is `Span`, so a span can be both evidence and cite origin.
- C9 (multi-reviewer): assertions are reviewer-attributed rows.
- C10 (stable anchoring): selectors + fingerprints stored on `Span`.
- C11 (auditability): assertions are append-only events with timestamps.
- C12 (incremental adoption): adapter plan included.
- C13 (neighborhood contradiction search): discovery runs + work-level citation edges allow surfacing non-cited nearby works.

### Known gaps / likely naive points

- Work identity resolution remains best-effort; duplicates will occur.
- Span re-resolution can fail if extracted text changes too much; needs fallbacks.
- Mapping from multi-cited spans to claim-specific evidence is underspecified until `Requirement`/role metadata is introduced.

Low-risk improvement already compatible with v1:
- Add reviewer-attributed citation roles for `Span -> Work` edges so users can mark `evidentiary|background|reputational|unknown` without treating role as canonical structure.

Status hygiene note:
- Keeping `unknown` separate from `not_supported` reduces schema thrash later when we discover additional meaningful states.
- Neighborhood quality depends on external metadata completeness (OpenAlex citation graph + abstracts).

The model is still adequate because these gaps do not block correctness; they affect ergonomics and precision.

## B) Implementability in this Repo

### Current repo realities (risk areas)

- There is already a `backend/graph_store.py` implementing a work/claim graph and a separate evidence/attachment pipeline.
- Current UI has claim-centric ids (`cite:...`, `claim:...`) and per-claim attachments.
- There is already early work on stable anchoring (`backend/text_selectors.py` and `cited_work_id` + `citation_anchor` fields in judgments).

Main risks:
- messy/dead code paths around legacy claim id formats
- attachment and evidence-matching assumptions keyed by claim_id
- multiple sources of truth (judgment_store vs graph_store vs evidence_store)

### Why SQLite is sane here

- Existing graph store is SQLite; extending it is consistent.
- We can keep tables small and indexed; exploration queries are local and bounded.

### Incremental cutover strategy (minimize breakage)

1) Add `Span` persistence on segmentation save/confirm without changing any UI.
2) Add assertion persistence alongside current judgment writes.
3) Add span-scoped attachment placement behind the existing auto-place behavior.
4) Move Graph tab to span/work-based endpoints once the adapter is stable.
5) Remove claim-id attachment duplication once span-scoped evidence is reliable.

This reduces the chance that a mid-flight rebuild bricks walkthroughs.

### Testing strategy (must be added early)

- Golden trace test: load one doc, create one cite span with two claim spans, attach one cited pdf, record two reviewers with disagreement; assert computed statuses.
- Secondary citation test: Work A assertion uses evidence span in Work B; ensure graph traversal works even if Work C absent.
- Migration test: legacy `cite:...` claim id resolves to the same `span_id`/`claim_span_id` after reprocessing.

## C) Decisions Needed Before Execution

These are choices that materially affect implementation details:

1) Span identity source-of-truth:
   - `work_id` anchored spans (requires work mapping), or
   - `ingest_id` anchored spans (works immediately, later map to work_id)

2) ClaimSpan segmentation scope:
   - canonical shared claim spans, with reviewer-specific claim atoms, or
   - reviewer-scoped claim spans (simpler, but makes consensus harder)

3) Evidence endpoint:
   - store evidence as `Span` only (recommended), with optional `Work` shorthand

4) Neighborhood search persistence:
   - store neighborhood runs + candidate lists for audit/replay (recommended), or
   - compute on-demand only (less storage, weaker auditability)

Default recommendation for sanity:
- anchor spans to `ingest_id` first (always available), map to `work_id` when resolved
- keep `ClaimSpan` canonical, store reviewer-specific `ClaimAtom` + `Assertion`
- persist neighborhood runs when the user explicitly requests a search (so traces are replayable)
