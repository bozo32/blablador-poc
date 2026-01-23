# Phase 2: Citation Context Navigation - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Enable users to navigate citation context and follow cited works for validation (callout interactions, context display, and citation tree view). This phase does not add claim segmentation or evidence workflows.

</domain>

<decisions>
## Implementation Decisions

### Callout interactions
- Single click on a citation callout opens context.
- Opening a callout replaces the right-pane content (no stack or tabs).
- Each citation is its own clickable badge when multiple appear in a sentence.
- No keyboard shortcuts for callout navigation in v1.

### Context content
- Show the citation sentence plus the adjacent sentences (before/after).
- Default cited context uses the preceding span up to the sentence boundary.
- Display cleaned text (no raw TEI), with background highlight on callouts.

### Citation tree behavior
- Show both references and cited-by relationships by default.
- Tree depth is user-controlled.
- Missing data appears as stub nodes labeled “data unavailable.”
- Tree is presented as a node graph view.

### Empty/error states
- If context is unavailable, show an inline message in the context pane.
- If graph data is missing, show an empty graph with a “no data” message.
- API errors surface as toast notifications with a retry action.
- Loading uses skeleton placeholders.

### Claude's Discretion
- Exact styling of callout highlights and skeleton placeholders.
- Graph layout algorithm and node label formatting.

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-citation-context-navigation*
*Context gathered: 2026-01-23*
