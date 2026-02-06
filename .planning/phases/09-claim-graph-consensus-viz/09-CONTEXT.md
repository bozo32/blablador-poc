# Phase 9: Claim Graph + Multi-User Consensus - Context

**Gathered:** 2026-02-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Add local multi-user review (no real auth) and an interactive claim-level graph that visualizes per-edge consensus vs disagreement. The graph is optimized for local exploration of moderate project sizes (roughly ~20 papers), with provenance layers for edges (auto/manual/external_search), reviewer-attributed votes/comments, and inspection/compare surfaces that make disagreement easy to spot.

</domain>

<decisions>
## Implementation Decisions

### Edge meaning + vote vocabulary
- Primary edge reviewers vote on: claim -> claim.
- Edge votes: support / contradict / neutral / uncertain.
- Per-vote confidence: optional.
- Consensus display uses raw counts/ratios (not confidence-weighted), even if confidence is present.
- Edges with zero votes can still be shown, but are visually marked as unvoted.

### Reviewer identity + disagreement UX
- Reviewer identity is a free-text name (reviewer_uid) via a single global "Current user" dropdown.
- Reviewer list is per-project (stored in project metadata), not global.
- If a project has no active reviewer yet, prompt the user to pick a name.
- Adding a reviewer switches to that reviewer immediately.
- Names are unique with trim + collapsed-whitespace normalization and case-insensitive comparison.
- No reviewer rename/delete management in MVP.
- Switching Current user scopes ALL saves (claim judgments, edge votes, manual graph edits) to that reviewer.
- Votes/comments are attributable; names are always shown (no anonymous mode).
- Edge vote comments: optional.

### Compare mode (MVP)
- Compare mode exists in the Graph tab as a drawer/panel.
- Compare is independent of Current user; you can pick any two reviewers.
- The last-selected compare pair is persisted per project.
- Compare supports both claim judgments and edge votes.
- Compare is side-by-side and highlights disagreements.
- Compare includes a disagreements list (both claims + edges):
  - includes missing-vs-present as a difference
  - sorts support-vs-contradict conflicts first
  - clicking an item navigates + focuses the relevant claim/edge and opens the inspection/compare view

### Graph renderer + exploration UX
- Renderer: Cytoscape.js embedded as a Streamlit component.
- Core navigation: smooth pan/zoom; stable selection.
- Graph tab default: center on the currently selected claim (if any); otherwise show instructions.
- Expansion is explicit (buttons), not automatic on selection.
- Controls (hops 1-3, edge cap per node, min_votes) live in a sidebar.
- Edge click opens a right-side inspection panel (not popover/modal).
- Node click shows claim details + provenance and offers a jump link back to the document view.
- Selection is single-select.
- Find claim control exists:
  - fuzzy match
  - dropdown results; selecting a result focuses/centers the node

### Paper grouping + collapse behavior
- Claims are visually grouped by paper.
- Paper groups can be collapsed.
- Collapsed state is represented as a single paper node; clicking expands in place.
- Paper grouping labels are short (doc_id/short title), not full titles.

### Edge styling + provenance layers
- Consensus support vs contradict is visible at a glance.
- MVP encoding: neutral edge line plus a small mid-edge ratio badge showing support vs contradict balance.
- Ratio badges are always visible (kept small/low-noise).
- Edge thickness scales with vote volume (n_total) and may also reflect confidence when present.
- Provenance styles:
  - auto: solid
  - manual: solid + emphasized
  - external_search: dotted
- Provenance layers visible by default: auto + manual + external_search.

### Inspection UI (edge + node)
- Edge inspection panel shows two columns:
  - left (green): supporters + comments
  - right (red): contradictors + comments
  - neutrals/uncertain collapsed
- Node inspection shows claim text and provenance (document, callout, sentence_id).

### Candidate edges + manual edges
- Candidate edge list lives in the node inspection panel.
- Candidates are ordered best-match first.
- Adding a candidate edge immediately adds it to the graph and selects the new edge.
- Manual edge creation is only via candidate list (no free-form drawing/connect).
- Manual edge removal/undo exists.

### Data model separation (entities vs assertions vs opinions)
- Separate canonical entities (document, claim) from assertions (claim links) and opinions (per-user votes on assertions).
- Edges do not change kind/type based on consensus; consensus is computed/aggregated.
- Nodes and edges store provenance fields: source (auto/manual/external_search), timestamps, and creator_uid for manual creation.

### Stable claim identity (disambiguation)
- Goal: claim node IDs remain stable across re-extract/re-segment.
- Tiered identity:
  - primary: doc_id + tei_sentence_id when available
  - fallback: doc_id + text_fingerprint
- Claim nodes store aliases (e.g., TEI sentence id + fingerprint) and reprocessing attempts alias-based relinking before creating new nodes.

### Consensus encoding (aggregates on edges)
- For each claim-edge, maintain aggregated counts: n_support, n_contradict, n_neutral, n_total.
- Derived ratios drive the UI badge.

### Materialization + visibility rules (avoid hedgehog)
- Do not dump all possible claim-to-claim edges into the graph.
- Expanding a neighborhood persists the newly surfaced edges as explored.
- Explored edges are attributed to the active reviewer (who explored/surfaced them).
- Default visibility shows all reviewers' explored edges; users can filter to:
  - My explored edges
  - Explored by others
- If edge-cap hides edges, affected nodes show per-node "+N more" stubs to expand.
- Unexplored candidates are computed on demand and are never persisted unless explicitly added as edges.
- External_search provenance edges persist once explored.
- Graph filter settings (hops, edge-cap, min_votes, provenance toggles) persist per project.
- Manual edge deletion is creator-scoped: only the creator can delete it, and deletion removes it globally.

### Exports
- Exports always include reviewer_uid, even for single-reviewer projects.

### Non-goals (MVP)
- No real authentication/permissions.
- No web-scale whole-corpus atlas.
- No full ontology export (RDF/OWL); only leave export hooks for later.

### Claude's Discretion
- Exact layout algorithm/tuning inside Cytoscape (as long as readability-first intent holds).
- Exact ratio badge rendering and edge thickness scaling (as long as it conveys consensus + vote volume).
- Exact text fingerprint details (as long as deterministic and robust to minor text shifts).
- Judgment storage layout on disk / in files, as long as it is backward compatible and exports include reviewer_uid.
- External search ingestion mechanics/caching implementation details, as long as provenance is external_search and edges behave per the rules above.

</decisions>

<specifics>
## Specific Ideas

- Graph is used to inspect consensus and disagreement.
- "Cherry-picking" should be derived (computed overlay such as icons/halos), not authored as new edge types.

</specifics>

<deferred>
## Deferred Ideas

- Cherry-picking signals/overlays beyond basic consensus visualization.
- Broader cross-project/atlas-scale graph exploration.
- Ontology export (RDF/OWL).

</deferred>

---

*Phase: 09-claim-graph-consensus-viz*
*Context gathered: 2026-02-06*
