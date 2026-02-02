# Phase 8: Workspace + Fast-Path Review UX - Context

**Gathered:** 2026-02-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Make review fast and interruption-free by moving citation interactions inline in the document text, introducing a single global source-bin upload surface with background conversion/parsing, and auto-triggering evidence runs so evidence is ready when the reviewer opens it. The workspace becomes vertically denser (no big header; settings behind a gear/drawer) without losing usability.

Out of scope for this phase: full recursive retrieval and a full node-graph navigation surface (Phase 8.1).

</domain>

<decisions>
## Implementation Decisions

### Inline citation interactions -> citing-span collection
- In-text citations are link-style underlined.
- Clicking an inline citation adds/activates the relevant citing span (snippet) into the right-pane citing-span list (this list is the collector of spans-of-interest).
- Multi-citation clusters do not require source selection here; this phase treats the citing span as the unit of interest. Divergent sources become relevant later at retrieval/chasing time.
- Selected/active inline-citation feedback: Claude decides the exact treatment.

### Global source bin (left pane)
- Location: left pane, always visible, as a collapsible/stacked section alongside navigation.
- Default organization: simple recency list.
- Upload: drag/drop + upload button; supports multi-file.
- Duplicate uploads: detect and dedupe.
- Per-item actions: View PDF, Assign/Re-place, Retry.
- Removal: archive (reversible); archived items hidden by default behind a "Show archived" affordance.
- Bulk actions: bulk retry.
- Match/placement flow:
  - System auto-suggests matches.
  - Auto-place silently only when confidence is strict/high.
  - If low/ambiguous confidence: leave unplaced and prompt (needs placement).
  - Users can correct placement after the fact (re-place to another record).
  - Placement destination is visible in an item details view/hover (not always inline).
  - Show a short placement audit trail (1-2 lines: why + timestamp).
- Status model:
  - Empty marker: not started.
  - Orange: processing.
  - Yellow: processed but not placed.
  - Green: processed + placed (appears in citation tree at the record level).
  - Red: failed.
  - Processing presentation: progress bar (not a spinner).
- Notifications: no popups/toasts; rely on bin status changes.

### Auto background processing (conversion/parsing/evidence)
- Source conversion + parsing starts immediately on upload (in the background).
- Auto evidence runs:
  - Trigger immediately after claim save.
  - Trigger again on claim re-save/edit (claim_text changes).
  - Trigger when new relevant sources become placed (green) after a claim was saved (backfill affected claims).
  - "Relevant" sources = placed (green) sources tied to cited works (not any processed source in the bin).
- Evidence-view behavior while runs are active: show partial results + live updating, but prevent evidence selection until the run completes.
- Manual controls remain: users can Run/Rerun explicitly.
- Pause/resume:
  - One global pause toggle.
  - Pause lets current work finish but prevents starting new work.
  - Pause state persists across app restarts; pending work auto-resumes on startup when not paused.
  - Manual Run/Rerun overrides pause for that action.
  - Paused state is shown via a global banner with queued/processing/failed counts.
- Failures:
  - Auto-retry a few times before marking failed.
  - A non-blocking global failed badge exists; clicking it opens the Source bin with failures highlighted.
- Prioritization:
  - Evidence runs prioritize the claim the user is currently viewing.
  - Source processing/placement prioritizes sources relevant to the current document/claim.
  - Favor UI responsiveness even if background queues grow.
- Batching/queue semantics:
  - If multiple sources become placed in quick succession: batch with a short quiet period then run once.
  - If an evidence run is in progress and user clicks Rerun: queue rerun after current completes.
  - If new sources become placed during an evidence run: schedule a follow-up run.
  - Cancel controls: allow cancel of queued evidence runs only.
  - Evidence run history: keep last 5 runs visible.
  - Per-claim indicator states: queued / running / ready / failed.
- Saved reviewer decisions:
  - Auto reruns should not silently change saved selections/judgments; if a source PDF is replaced/changes, existing selections may be cleared or explicitly flagged.
  - Re-placing a source or replacing a source PDF is treated as a reason to re-parse and auto-rerun evidence for affected claims.

### Dense workspace layout + pane roles
- Overall layout is 3 panes:
  - Left: source bin + compact navigation.
  - Center: main work area with tabs.
  - Right: ordered list of citing spans (snippets) with completion status.
- Center tabs for Phase 8: Document / Review.
  - Review contains chasing/evidence/judgment flows; chasing is treated as a first-class review mode (not an overlay).
  - The active citing span selected on the right drives what appears in the center Review view.
  - Multiple segmented claims can exist within a citing span; those appear stacked in the center.
- Right pane (citing spans list): each row shows snippet + completion status.
- Settings:
  - Settings are behind a gear that opens a right-side drawer.
  - Gear placement: lower-left corner.
- Global indicators (paused banner, failed badge) are anchored in the right pane header.
- Dense mode:
  - Controlled by a sticky toggle (persists across sessions).
  - Changes padding + typography scale only (no behavioral changes like auto-collapsing sections).

### Claude's Discretion
- Exact selected/active styling for inline citations (beyond underline) and whether any scroll/spotlight behavior is used.
- Exact layout details inside the center Review tab (sub-tabs vs stacked sections) as long as it preserves the decided pane roles and behaviors.

</decisions>

<specifics>
## Specific Ideas

- Right pane is a collector of citing spans-of-interest: clicking inline citations adds/activates spans there.
- Source placement is bibliographic/record-level ("placed in the citation tree"), not per-citation-instance.
- Strict auto-placement policy: silently auto-place only on high confidence; otherwise require attention.
- No popups/toasts for background completion; rely on status colors + progress bars.

</specifics>

<deferred>
## Deferred Ideas

- Full node-graph navigation surface and recursive retrieval-oriented UI (Phase 8.1). A "2-node radius mini-graph" concept came up; treat any graph-like navigation beyond a compact citation-tree/list widget as Phase 8.1.
- Multi-session selection/creation at startup, plus explicit "save/export session" affordances (not in Phase 8 boundary as written).

</deferred>

---

*Phase: 08-workspace-fast-path-review-ux*
*Context gathered: 2026-02-02*
