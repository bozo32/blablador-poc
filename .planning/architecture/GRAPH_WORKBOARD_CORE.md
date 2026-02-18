# Graph Workboard Core (First-Principles Spec)

Defined: 2026-02-18

This document specifies the durable "graph"/"workboard" core needed for OS-ERIN.
It is written from first principles, then mapped onto the current implementation.

## Problem Statement

As soon as citation validation exceeds trivial working memory (eg. 2 clauses x 3 sources),
reviewers need a durable, navigable workboard that answers:

- What is left to do?
- What is done, by whom, and with what confidence?
- What is disputed/contested?
- How does work propagate across a citation tree?

Phase 09.3 requires the system to operate with no durable local `data/**` state.
Therefore any user-authored graph/workboard state must be persisted in Postgres (Spine).

## Hard Requirements

- Durability: authored state survives restarts, container rebuilds, and `rm -rf ./data`.
- Source of truth: authored state lives in Postgres; local SQLite may exist only as a disposable cache.
- Multi-reviewer correctness: per-reviewer actions do not overwrite each other; aggregation is reproducible.
- Stable identity: user actions attach to stable IDs and remain valid across reindex/re-extract runs.
- Rebuildability: derived/indexed graph state can be regenerated from Spine artifacts.
- Adequate today: current UI endpoints must continue to work while the storage backend changes.

## Entity Model (Stable Identity)

All durable authored state MUST target one of these stable IDs.

- Project: `project_id`
- Reviewer: `reviewer_uid` (scoped to project)
- Document (ingested work): `document_id` (Spine ingest/work id)
- Reference anchor: `(document_id, citation_index, target_id)`
- Span: `span_id` referencing a selector window in a document
- ClaimSpan: `claim_span_id` (or `(span_id, order_index)`)
- Work: `work_id` (eg. OpenAlex/DOI/bib-key), independent of whether a PDF is ingested

Selectors must be stored as JSON (anchor-quote / offsets) to support future markup.

## Data Classes: Authored vs Derived

Authored (durable; never silently regenerated)

- Ledger edits: manual citation links, assigned flags, explicit enable/disable of links
- Claim graph edits: manual claim links, per-edge votes, reviewer-specific notes
- Workboard state: checked marks, per-cite roles, assertions (including selections)
- (Optional) layout hints: pins/grouping/ordering used for navigation

Derived (rebuildable; may be cached)

- Document/work aliases inferred from extraction/resolution (doi/bib)
- Auto citation links inferred from extraction
- Claim nodes inferred from confirmed-claims
- Convenience indexes/materializations for fast querying

## Operations (Write Model)

- Upsert derived indexes idempotently from Spine artifacts.
- Record reviewer actions as append-only or upserted-by-key rows:
  - mark checked / un-checked
  - set cite role
  - create/update/delete manual edges
  - vote on edges
  - create/update assertions and selection decisions

Every authored write includes: `project_id`, `reviewer_uid`, `source`, `created_at/updated_at`.

## Queries (Read Model)

- Ledger view: documents + incoming/outgoing + anchored/extracted/resolved + assigned.
- Workboard status: per span/claim span/reviewer: unknown/not_supported/supported/contradicted/contested.
- "What is left": fast query listing unreviewed items for a reviewer, with grouping by citation tree.
- Subgraph slices for navigation: claim subgraph with votes; span bundle with per-claim-span status.

## Future Capacities (Must Be Supported)

These are not necessarily v1 UI features, but the durable core MUST not block them.

1) User annotation/markup of documents (comments on spans)
- Store annotations keyed by selector + `document_id` (optionally `span_id`).
- Fields: author/reviewer, body, tags, resolved state, timestamps.

2) User-created entailment links between spans within a document (argument mapping)
- Model as typed edges between span-like endpoints: `ENTAILS`, `SUPPORTS`, `CONTRADICTS`, etc.
- Keep per-reviewer assertions/votes separate from existence of an edge.

3) Differential epistemic weight by reviewer
- Store reviewer weights in project config or a `reviewer_weights` table.
- Aggregations become weighted sums instead of counts; per-reviewer rows remain the source data.

4) Propagation of edge conditions to descendants
- Treat propagation as derived state computed from authored edges + authored judgments.
- Optional materialization for performance; must always be rebuildable.

## Current Implementation (Contrast)

Today, `backend/graph_store.py` and `backend/span_graph_store.py` are SQLite-backed and mix:

- Derived indexing (doc nodes, aliases, auto edges, claim-node indexing)
- Authored user state (manual edges, votes, checked marks, assertions)

This violates Phase 09.3 durability: `data/graph.db` is currently a system-of-record for workboard state.

## Recommended Path (Optimized Clean x Extensible x Adequate)

- Move all AUTHORED graph/workboard tables to Postgres first.
  - Keep the API surface stable; swap storage behind the existing endpoints.
- Treat DERIVED graph indexes as rebuildable; optionally cache locally.
- Separate authored vs derived at the schema level to avoid accidental regeneration.

Phase 09.3 consequence:

- It is acceptable to keep a local cache DB only if deleting it does not lose user-authored state.
