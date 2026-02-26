# Phase 7: Validation + Export - Context

**Gathered:** 2026-02-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Add final per-claim judgment capture and export:
- Record a verdict for a claim (support/contradict/uncertain), with a draft/unreviewed state allowed before finalizing.
- Add optional structured notes to the judgment.
- Reflect validation status (including outcome) in citation callouts.
- Export judgments + metadata as JSON or CSV.

</domain>

<decisions>
## Implementation Decisions

### Verdict model
- Verdict labels: Support / Contradict / Uncertain.
- Verdict applies to the claim (one verdict per claim), not per callout.
- Draft/Unreviewed is allowed (claim can be saved without a final verdict).
- “Validated” status is only true once a final verdict is set (draft does not count).
- Mixed/inconsistent evidence collapses into Uncertain (no separate “mixed” verdict).

### Notes + annotations
- Notes are structured fields (not a single freeform blob).
- Notes are always optional (no verdict requires notes).
- Notes editing happens inline on the claim card.
- After saving, notes are collapsed by default (preview + expand).

### Validated status in callouts
- Callouts show status via icon + color.
- Callouts reflect verdict outcome (support vs contradict vs uncertain) with outcome-specific styling.
- Clicking the callout’s status indicator opens judgment details (navigates to the claim/judgment view).

### Export format + scope
- Default export format: JSON. CSV is also supported.
- Provide both export shapes as separate files: per-claim and per-callout.
- Default export includes only final verdicts (exclude Draft/Unreviewed unless explicitly chosen).
- Export supports two modes: core fields vs verbose metadata.

### Claude's Discretion
- Exact structured note field names/count (e.g., rationale/caveats/follow-ups).
- Exact icons/colors used for callout outcome styling.
- Exact field lists for core vs verbose export modes.

</decisions>

<specifics>
## Specific Ideas

- Keep callouts minimal: status via icon/color, but clickable to jump into judgment details.
- Notes should not visually dominate the claim view (collapsed by default after save).

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 07-validation-export*
*Context gathered: 2026-02-02*
