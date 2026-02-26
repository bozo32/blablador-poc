# Phase 10-03: Graph As Navigation (Cytoscape Routing) - Context

**Gathered:** 2026-02-22
**Status:** Ready for research/planning

<domain>
## Phase Boundary

Turn the graph into a navigation surface: selecting nodes/edges routes into the main workspace (document/span/claim). Add recursive retrieval hooks so reviewers can follow citations from cited sources and keep building the chase.

In scope:
- Selection contract and routing behavior from graph -> workspace context.
- Recursive expansion that queues requested works (no auto-download).

Out of scope:
- New annotation/comment systems for edges/nodes (durable decisions/events live in 10-04).

</domain>

<decisions>
## Implementation Decisions

### Workspace model + routing targets
- Center panel uses tabs: `Reading`, `Chasing`, `Surfing`.
- Graph lives in `Surfing`; the reader lives in `Reading`.
- Node/edge click is **soft focus**: show a compact popover; do not route until user clicks an explicit action.
- Popover actions include `Go Read` (default) and `Go Chase`.
- `Go Read` switches to `Reading` and jumps to citation context.
- `Open PDF` is always present when a PDF exists; it switches to `Reading` (not a separate PDF-only jump).

### Chasing routing semantics (what gets focused)
- If cited-side spans are not resolved yet, routing to a cited work goes to `Chasing` with that work selected; the user performs normal review to select candidate cited citespans.
- `Chasing` top shows the citing citespan and its segmented citespans/claimspans.
- `Chasing` includes a dropdown for candidate/selected cited citespans.
- When routing to `Chasing` and multiple citing claimspans exist, default selection is **last active**.
- If user clicks:
  - citing citespan/claimspan node -> route to `Chasing` for that citing selection.
  - cited work node -> route to `Chasing` with that work selected.
  - cited citespan node -> route to `Chasing` with that work + cited citespan selected.
  - cited claimspan node -> route to `Chasing` with that work + cited citespan + claimspan selected.
- If the clicked cited citespan/claimspan does not exist yet: route to `Chasing` with a clear "Create/Select cited citespan" prompt (no auto-creation).

### Selection contract
- First-class selectable types: `work`, `citespan`, `claimspan`, `edge`.
- Selection ids are **structured** identifiers (not opaque) to keep routing debuggable.
- Edges are entities: selection is `{type:'edge', id:<edge_id>}`.
- Edge click emphasizes the relationship; edge popover offers `Go source` and `Go target`.
- Non-routable/unavailable works are still selectable as `type='work'` and render as `state='missing'` (or `requested/blocked`); popover shows Retrieval (primary) and `Go Chase`.

### Ambiguity + determinism
- When a selection maps to multiple possible routing targets:
  - Default is "pick best + show chooser" (best default = **last active target**).
  - For a cited work referenced in multiple citing contexts: **always ask** (do not remember last choice).
- Chooser appears inside the graph popover.
- Chooser ordering: grouped by citing document, then citation order.
- Chooser row content: citing doc title + excerpt snippet derived from the claimspan text.

### Graph model (semantics)
- Graph is a **span-centric chase map**.
- Default layout: left-to-right flow.
- Default view scope: current focus only (active citing citespan/claimspan + immediate cited neighbors).
- Edge kinds that are visually distinct: chase + segmentation + assessment.
- No arrowheads by default.

### Labels, shapes, density
- Labels are short:
  - Works: short bib-style label.
  - Spans: short text snippet.
- Show cited-side citespan/claimspan nodes on selection (expand-in-place for the selected work).
- On citing side, show both citespan parent and claimspans children.
- Add a simple density toggle: `Show claimspans` on/off.
- Shapes:
  - work = rounded rectangle
  - citespan = circle
  - claimspan = small pill
- Selection styling: strong outline for selection + dim non-neighbors.

### State + assessment encoding
- Graph state is live/polling-derived.
- Graph uses color/icon only for processing state (no text chips on nodes); use small corner badges.
- Processing states are distinct (at least `requested` vs `blocked` are separate).
- Minimum state vocabulary the UI must handle (nodes/edges; rollups allowed): `missing`, `requested`, `blocked`, `available`, `processing`, `done`, `error`, `cancelled`, `mixed`.
- "Done" for cited work means **assessment recorded**.
- Aggregated/mixed state:
  - Show as a ring/pie indicator.
  - Popover shows a list of items (clickable) rather than counts.
- Edges inherit target state.
- Assessment encoding uses traffic-light palette:
  - supports = green
  - contradicts = red
  - inconsistent = orange with pattern
  - silent = grey (and visually distinct from unassessed)
- Assessment encoded on both nodes and edges.
- Precedence is hybrid: processing states (missing/blocked/processing/error) override; otherwise assessed nodes use assessment colors.
- Cancelled vs error: cancelled is grey stop; error is red.

### Popover content + actions
- Popover is compact/action-first.
- Show a one-line assessment badge when available.
- For missing/unavailable cited works: show both Retrieval and Go Chase, with Retrieval primary.
- Retrieval info presentation: copy action plus external links when available.
- Expand is available on all cited-side nodes.

</decisions>

<specifics>
## Specific Ideas

- Expand is a user-driven action that also queues requested works for the next hop (no auto-download).
- Expansion is one hop per click; append nodes without forcing a relayout (manual reset/center controls exist).
- Expansions are persisted per project.
- Recursively queued requests appear in the same requested-works queue, grouped by parent source, and show the parent label.
- Matching/assigning an uploaded PDF should auto-link the request and update the graph.
- Low-quality citations at expand time: show counts (confident vs low-confidence) and let the user choose what to include.
- Expand can be queued pending prerequisites; it should be cancelable from the popover.
- Expand completion uses a subtle inline notice; failures show inline errors in popover + queue.

</specifics>

<deferred>
## Deferred Ideas

- Edge/node annotations (“what other users added to comment their decision”) as a drilldown from the graph popover. This is a durable decisions/events capability and belongs in 10-04.

</deferred>

---

*Phase: 10-contracts-core-workflow-simplification*
*Context gathered: 2026-02-22*
