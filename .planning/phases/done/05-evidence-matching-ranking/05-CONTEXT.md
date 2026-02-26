# Phase 5: Evidence Matching + Ranking - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Surface ranked evidence candidates for the active claim, showing entail/contradict labels, snippets, and enough metadata for reviewers to act. No new evidence-generation or review features beyond delivering and managing the ranked list.

</domain>

<decisions>
## Implementation Decisions

### Candidate layout & metadata
- Full-width cards, arranged in a tight vertical stack, each led by the snippet (with inline highlight on the matched span) followed by metadata separated with horizontal rules.
- Cards include: compact entail/contradict chip plus subtle rank chip near the header, inline highlight for multi-span counts (badge), badges for match provenance (cited vs heuristic), heuristic origin text chip, machine vs human curation chip, OCR quality badge when applicable, and initials chip whenever a reviewer already acted on the candidate.
- Required metadata shows page + section (use placeholder text when missing) and citation reference inline in the footer; hover reveals additional metadata rather than extra buttons.
- Snippets default to ~600 characters, collapsed to 3–4 lines with an expand/collapse control; duplication cues are skipped.
- Claim reminder shows a one-line claim excerpt that links back to the editor and carries a badge if edits are unsaved.
- Inline actions: primary accept and secondary reject buttons live left-aligned in the footer, always visible, with keyboard navigation (arrow keys move focus, Enter triggers primary action) and no hover-only control reliance.
- Additional affordances per card: inline “Open in PDF” button (glow hover) jumping the viewer to the span, star icon to pin favorites, share-link icon for deep link copy, inline notes block (collapsed summary), conflict icon for contradictions with prior accepted evidence, warning banner skipped, and no snippet copy control.
- Background tint differentiates entail vs contradict, while a global progress bar summarises entail/contrad counts across the stack.
- Cards show numeric confidence scores plus a mini sparkline visualizing token-level confidence, rule dividers between sections, and a “Why matched” bullet list beneath metadata.

### Ranking feedback
- Display raw ranking scores per candidate, along with subtle rank chips and arrow-only delta badges when reruns change ordering; high-confidence demotions due to diversity are explained in the rationale sidebar.
- Dedicated rationale sidebar auto-syncs with card hover, supports multi-pin comparison, and houses: ranking rationales, diversity/demotion notes, entail/contrad label explanations, delta-vs-current accepted evidence lines, confidence waterfall chart, detailed section describing the ranking model configuration, and links out to documentation for entail/contrad scoring formulas.
- Sidebar provides downloadable rank-change history, JSON export control, and advanced-mode toggle (accessed via the advanced menu) with subtle toast feedback when mode switches.
- Details modal per candidate breaks down heuristic contributions; badges mark reviewer overrides and manual rejects that influenced ordering.
- Neutral/unknown candidates are grouped in their own section within the same list; a global inline chip warns when overall ranking confidence drops below threshold.
- Rerank completion fires a toast with a direct link to run logs/JSON export; stale rankings show a banner prompting rerun, and lack of entail/contrad labels results in a separate neutral section rather than warnings.

### Interaction flow
- Matching/ranking auto-runs when attachments change for the focused claim; manual rerun sits in the overflow menu.
- Manual reruns queue (multiple allowed) and can proceed even while claim text is being edited; reviewers may add optional notes describing rerun intent.
- Rerun confirmation only appears when unsaved claim edits exist; otherwise reruns fire immediately with inline card skeletons replacing content until completion.
- During a rerun, accept/reject actions on the existing list are locked, but the UI allows other navigation; queued reruns are disclosed via tooltip-only status, and there is no rate limit.
- Advanced parameter adjustments live in a modal sheet reached from overflow; settings apply globally across claims, with available undo to restore the previous ordering if the new run is worse.
- Completion toast includes a link to logs/JSON export; rerun mode changes trigger subtle toasts, and rerun parameters do not display custom badges.
- Manual reruns can be launched alongside ongoing runs, with queued status managed internally; overflow-launched reruns may open the parameter sheet (planner decides exact anchoring) and we skip parameter-deviation indicators.

### Result coverage
- Default view shows the top 5 candidates; “Load more” fetches fixed batches of 5 with the button indicating remaining count and allowing up to two concurrent requests.
- Load-more state persists per claim, so returning reviewers see the same expanded count; cap total retrieved candidates at 50.
- Filter chips allow toggling entail/contrad/neutral within a single combined list; chips also provide a shortcut to jump directly to neutral candidates, auto-loading more if needed, and displaying inline empty-state copy when no neutral evidence exists even after loading.
- Guarantee at least one contradicting candidate slot when available; if fewer than five total candidates exist, show whatever is available without backfilling.
- Virtual scrolling handles long lists; reviewers can collapse sections via accordion controls, and pinned candidates stay at the top even when sections collapse or filters change.
- Shortcut chip to neutral, per-label warnings are omitted, and “Load more” auto-loads for neutral jump while respecting the two-request cap; concurrent requests beyond two are blocked.
- Download/export for more-than-visible candidates is not provided here (JSON export already covered in ranking feedback section).

### Claude's Discretion
- Accessibility implementations (ARIA live regions, contrast adjustments) follow planner judgment.
- Presentation/anchoring of the advanced parameter sheet (slide-out vs centered modal) is up to planner discretion.
- Whether “Load more” should support label-specific controls beyond the global button is left to planner judgment.

</decisions>

<specifics>
## Specific Ideas

No external references cited beyond the decisions above — open to standard approaches that satisfy them.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-evidence-matching-ranking*
*Context gathered: 2026-01-27*
