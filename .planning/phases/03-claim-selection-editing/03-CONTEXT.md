# Phase 3: Claim Selection + Editing - Context

**Gathered:** 2026-01-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Users review extracted claims from citing text, edit them, and confirm a final claim for validation. The phase covers claim segmentation, selection, and editing within the citation context workflow.

</domain>

<decisions>
## Implementation Decisions

### Claim segmentation rules
- Default segmentation is clause-level splitting (compound sentences become multiple candidates).
- When a sentence has multiple claims, show multiple candidate parses.
- Keep attribution phrases ("we find", "suggests") in the default claim text.
- Keep parentheticals/hedges in the claim text (do not strip by default).

### Editing workflow
- Claim edits happen in a modal editor launched from the right pane candidate list.
- Edits auto-save per field; no global save button required.
- Show original text on hover for edited claims.
- Provide a "Reset to original" action per claim.

### Selection behavior + queue
- Clicking a claim callout auto-queues it for parsing in the right pane; reading can continue.
- Right pane shows a queue ordered oldest-first.
- Each queued claim is a collapsible panel with cited source author/year in the header.
- Panel body shows the full citing sentence and candidate claim parses (editable).
- "Confirm parsing" collapses the panel and marks status as processed.
- Status colors: gray = not processed, green = processed.
- Panels can be manually collapsed at any time.

### Claude's Discretion
- Exact modal layout, spacing, and microcopy.
- Visual styling of the queue and status dot.

</decisions>

<specifics>
## Specific Ideas

- Right pane bottom should hold the claim queue so readers can keep scanning the main text.
- Each claim panel header shows cited source author (year) for quick recognition.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-claim-selection-editing*
*Context gathered: 2026-01-24*
