# Phase 06: Evidence Review + Selection - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Enable reviewers to inspect evidence in-context and select the best supporting/contradicting span (or choose none-of-the-above) for each parsed claim segment.

This phase focuses on review + selection UX; workflow customization UI (patch-board) and new retrieval strategies are out of scope.

</domain>

<decisions>
## Implementation Decisions

### In-Source Highlight View
- **Source structure:** Use cited PDF GROBID TEI as the canonical structure for section/paragraph context.
- **Grouping:** Group candidates by TEI section path; if no headings exist, group everything under a single `Body` section (do not synthesize headings).
- **Top hits:** Show top-N ranked hits at the top, then the same hits grouped by TEI section.
- **Context window:** Default excerpt is `two sentences before + highlighted span sentence + one sentence after`, without crossing paragraph boundaries. Bullets and tables count as paragraphs.
- **Collapsed preview:** Collapsed excerpt shows a 1-line preview + highlight badge (verdict + score/confidence).

### Selection + Verdict Semantics
- **Granularity:** Selection is per claim-segment (parsed clause), not per whole claim sentence.
- **Outcomes:** `support` / `contradict` / `uncertain` / `none`.
- **Uncertain requires note:** Choosing `uncertain` requires a short comment.
- **Selection cardinality:** One primary span, with optional secondary span(s).
- **Secondary rationale:** Selecting a secondary span prompts for a short rationale (why it is added).
- **None-of-the-above:** Persist `none` but keep candidates visible and changeable later.

### Navigation + Anchoring Model
- **Right rail:** Right rail stays stable as the user anchor.
- **Center pane:** Center pane can switch between `Citing` and `Source` views.
- **Candidate interaction:** Clicking a candidate expands it inline (accordion) rather than changing the overall layout.
- **Followed citations:** Followed citations remain pinned in the right rail until explicitly unfollowed.

### Empty/Weak Evidence States
- **Not parsed yet:** If cited PDF is attached but not parsed/section-indexed, show status and wait; selection disabled until ready.
- **No candidates:** Show "None found" and clear next actions (within existing capabilities).
- **Low confidence:** Show low-confidence candidates but visually de-emphasize them (still accessible).

### Model Execution
- **Local-first:** Keep local model execution.
- **Future hooks:** Add clean provider hooks so API-backed model execution can be added later (no UI exposed in this phase).

### Claude's Discretion
- Exact highlight styling (colors, emphasis) and preview formatting.
- Top-N count in the “top hits” section.
- Exact confidence thresholds/heuristics for “de-emphasize”.

</decisions>

<specifics>
## Specific Ideas

- Candidates should not be decontextualized sentences; they must be shown as highlighted spans within surrounding context.
- Use accordions/collapsers between relevant bits of text to support fast scanning.

</specifics>

<deferred>
## Deferred Ideas

- Workflow patch-board (ComfyUI-like) and workflow benchmarking UI in settings.
- API-hosted compute providers (e.g., Hugging Face inference endpoints) as an alternative to local execution.

</deferred>

---

*Phase: 06-evidence-review-selection*
*Context gathered: 2026-01-31*
