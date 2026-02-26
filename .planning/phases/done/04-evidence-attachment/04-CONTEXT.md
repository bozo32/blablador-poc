# Phase 4: Evidence Attachment - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Attach cited PDFs to claims so each claim is anchored to a specific document, with attached content parsed and ready for downstream evidence retrieval. Attachment selection, parsing, auto-matching, queue management, and readiness metadata are in scope; evidence matching/ranking/review remain later phases.

</domain>

<decisions>
## Implementation Decisions

### Attachment surface & interactions
- The entire claim card is the primary drop surface; on drag-over the attachment affordance brightens while the rest of the card dims, and other claims drop to low opacity to clarify targeting.
- Keyboard shortcuts must exist for attach/detach, but exact mappings and scope remain open; a fallback “Attach PDF” modal (with its own drag/drop) covers non-DnD users.
- Dropping onto a populated claim prompts before replacing; multi-file drops land in a dedicated queue panel first, then auto-match to citing claims with toast confirmation when a claim receives an auto-assigned attachment.
- Queue accepts stacked PDFs, attempts silent auto-conversion of non-PDFs (inline status icon reports convert success/failure), and maintains extra files per claim; ambiguous matches trigger a resolving pop-up and unmatched files prompt immediately for manual assignment.
- The queue lives in its own collapsible side/bottom panel (think VS Code pane) that lists every cited file for the document with statuses (unprocessed, converted, resolved, entailed). It shows batch-level progress for multi-drop jobs, collapses when finished, and leaves a persistent summary chip (with error counts when needed) that reopens the panel on click.
- Auto-matching may attach one source to multiple claims referencing the same work. Attachments are shared objects but each claim keeps a per-claim override so re-uploading on one claim does not overwrite others. If auto-match fails entirely the user is prompted right away. Detaching removes the file from the queue instead of recycling it.
- Auto-matching prefers covering every citing claim for a work rather than picking a single target. Shared attachments update globally only when explicitly chosen; otherwise claims maintain their overrides.

### Parsing visibility & messaging
- While upload + OCR/parsing runs, the claim shows a full-card skeleton; we keep a single status indicator for the pipeline but swap the text (e.g., “Uploading…”, “Parsing…”) as phases advance. Upload progress itself lives in the queue panel, not on the claim.
- Auto retry runs once with an inline countdown message; after that the user gets manual retry plus a link that opens a right-side diagnostics drawer. Successful retries return quietly to the normal state.
- Errors expose a short summary plus diagnostics link; queue highlight calls attention when parsing is done but matching is pending. Auto-conversion/matching progress is shown in the queue, with claim cards updating only when data is ready.

### Attachment persistence & revisit flow
- Attachment state, queued files, history, and errors persist across navigation and sessions. Claim panels stay collapsed until users open them, but uploads continue in the background; leaving mid-upload resumes automatically on return with a persistent banner noting background work.
- A per-claim attachment timeline records the last five events (attach, convert, auto-match, detach, etc.); shared attachments show a summary link on each participating claim.
- Queue items persist between sessions, as do pending conversions/matching jobs, which auto-resume next launch. Returning users see an activity digest summarizing auto-matches performed while they were away.
- Editing a claim triggers an inline revalidation prompt near the attachment badge, pauses downstream use until confirmed, and this prompt is the only blocker (no modal). Attachment histories persist but trim beyond five events.

### Evidence readiness metadata
- Parsed attachments capture full TEI structure, sentence/page offsets, vector embeddings, and retrieval hints. Metadata explicitly maps which claims tie to which sections/spans and stores confidence scores for bindings.
- Metadata lives in a shared vector store for reuse downstream, plus canonical attachment records. Power users can open a command-palette action to view and optionally export metadata (JSON), including version history if the file is reprocessed.
- Staleness is tracked with timestamp comparisons; stale metadata blocks downstream services until refreshed. Version history is maintained and surfaced through the same power-user entry point.

### Claude's Discretion
- Exact keyboard shortcut mapping and whether shortcuts are global or claim-scoped.
- Visual styling for badges, loaders, queue panel chrome, and timeline treatments.
- Retry backoff strategy beyond the first automatic attempt, queue collapse animation, and how the persistent chip is styled.
- Thumbnail generation specifics and any advanced conversion heuristics beyond silent conversion + status icons.
</decisions>

<specifics>
## Specific Ideas

- Queue panel should feel like a collapsible VS Code side pane with a header (“Queue”) that expands/contracts, lists each cited file across the document, and shows statuses (unprocessed, converted, resolved, entailed) with activity indicators.
- Batch completion collapses the queue automatically, leaving a summary chip that keeps error counts visible and reopens the panel when clicked.

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within this phase scope.

</deferred>

---

*Phase: 04-evidence-attachment*
*Context gathered: 2026-01-26*
