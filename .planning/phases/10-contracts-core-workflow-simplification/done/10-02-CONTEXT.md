# Phase 10-02: Core Workflow Simplification (Happy Path) - Context

**Gathered:** 2026-02-21
**Status:** Ready for research/planning

<domain>
## Phase Boundary

Implement a single, predictable happy-path workflow with a thin orchestrator/state machine over the Phase 10-01 stage contracts.

In scope:
- On document deposit: ingest/extract runs automatically; reviewers can read progressively while background work proceeds.
- When a reviewer finalizes claimspan segmentation (declares they are chasing it): start cited-work processing in the background through NLI.
- Clear surfaces to trigger/observe runs, poll/stream status, fetch artifacts, and record an assessment.
- Timeouts and durations at boundaries; retries; per-target cancellation; predictable states.
- Remove/disable confusing mode switches: single default behavior for the happy path.

Out of scope (defer if it comes up): fully automated downloading of missing cited works (stub only), and new “capability” expansions beyond the happy path.

</domain>

<decisions>
## Implementation Decisions

### Canonical objects and terminology
- Use product-facing terms `citespan` (canonical) and `claimspan` (user-defined selection(s) within a citespan).
- Claimspans are reviewer-scoped; different reviewers may define different claimspans for the same citespan.
- Pipeline runs and run history are reviewer-scoped.

### Entrypoint and run granularity
- Deposit of a citing document auto-starts ingestion/extraction; reviewer can begin reading before ingestion/extraction is fully complete (progressive UX with clear processing states).
- When a reviewer finalizes claimspan segmentation (declares “chasing”), the system auto-starts background processing for the relevant cited work(s) and runs through NLI.
- Run identity: claimspan-centric (runs are “about” a claimspan within a citing work) and created at claimspan finalization.
- Target scope: process all citation-derived targets for the whole parent unit (not limited to a single marker).
- Multiple targets run in parallel (best effort; no user-visible cap).
- Reruns always mint a new `run_id`.
- If inputs change (e.g., claimspan edited after finalize), automatically start a new run.
- If a claimspan is split into multiple claimspans, each new claimspan starts its own runs.
- Resume behavior: within a run, resume missing targets/stages when they become runnable (e.g., after a cited PDF is deposited).

### Requested cited works and acquisition
- Requested works list is derived from consolidated bibliography items linked to in-text markers.
- Left-pane requested-works queue is the primary UX for “what PDFs do I need next”.
- Queue ordering is citation order.
- Queue states visible at a glance: Requested -> Available -> Processing -> Done.
- PDF matching:
  - Auto-match is allowed with fuzzy metadata matching when confidence is high.
  - If a deposited PDF could match multiple requested works, prompt the user to choose.
  - If a deposited PDF matches none, allow manual assignment to a requested work.
  - When auto-match is not confident/ambiguous, allow resolving later (unassigned until user resolves).
- Bib -> ingested-work linking:
  - Auto-link when confident.
  - If ambiguous (multiple ingested works match), prompt the user to choose.
- If a claimspan has no identifiable citation mapping, mark it as “needs target” and allow the reviewer to select a target from the requested works.

### Orchestration behavior
- Execution is background by default; reviewer keeps reading while processing runs.
- Status/updates are stage + per-target (not just stage-level).
- Streaming: use server-sent events (SSE) for per-target updates; fall back to polling on disconnect.
- Running state is represented via a separate run-status view (not solely inferred from artifacts).
- Timeouts: retry first; if still timing out, record partial outcome and continue with other work.
- Failures: auto-retry a small fixed number of times within the same run_id; after retries are exhausted, record final stage outcome.
- Persist intermediate progress artifacts (debug-only; normal reviewer UX stays clean).
- Error isolation: if one target fails/timeouts, continue other targets and complete with mixed results.
- Global background pause is obeyed.
- Cancellation:
  - Cancel is supported per-target.
  - Cancel attempts best-effort cleanup and discards partial results.
  - Canceled is a distinct state in the run status UI (not treated as an error).

### Reviewer UX expectations (10-02 includes full queue UX)
- Right-pane claimspans accumulate and show status clearly.
- Status indicators use compact icons with drilldown.
- Completion coloring follows traffic-light semantics: green=good, orange=partial/warnings, red=error.
- If a claimspan has multiple cited works, show multiple work-status icons (one per cited work), each clickable.
- Drilldown default shows a human summary first (errors/warnings + what to do), with raw artifacts available behind it.
- Quiet updates: status changes update inline; no toast notifications by default.
- Mode switches: single default behavior for the happy path; hide advanced toggles.

### Assessment output and persistence
- “Complete” for a claimspan/target means an assessment is recorded (not merely that NLI finished).
- No auto-assessment: reviewer decides.
- Assessment granularity: both per cited work and an overall claimspan rollup.
- Assessment labels: Supports / Contradicts / Inconsistent / Silent.
- Evidence selection allows multiple spans per cited work.
- Completion gating: allow partial completion (claimspan can be marked complete even if some works are pending/unassessed).
- Persistence is mirrored:
  - Pipeline artifacts are persisted (10-01 contract store).
  - Existing evidence stores remain in use, and reviewer selections/judgments are persisted there as well.
  - Additionally write an assessment artifact via pipeline contracts for replay/audit.

### Claude's Discretion
- Exact confidence thresholding heuristics for “auto-match when confident”.
- Exact run-status payload shape (as long as it supports per-target + per-stage state and durations).
- Exact icon set and drilldown layout details (within the constraints above).

</decisions>

<specifics>
## Specific Ideas

- When a claimspan has multiple cited works, each cited work gets its own status icon; icons can be clicked to view retrieval info.
- If any UI elements described above are too large for the 10-02 implementation slice, capture them explicitly in planning as required UX to land by the end of Phase 10.

</specifics>

<deferred>
## Deferred Ideas

- Automatic cited-work download (“trigger scripts for automated download”) is a stub for future development; for 10-02, the system should support a user-driven requested-works queue and manual association.

</deferred>

---

*Phase: 10-contracts-core-workflow-simplification*
*Context gathered: 2026-02-21*
