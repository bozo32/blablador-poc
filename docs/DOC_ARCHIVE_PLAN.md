# Documentation Archive and Move Proposal (Non-Destructive)

This proposal defines how to clean documentation structure without moving or deleting files during active bug hunts.

## Current Constraint

- Active Reading/Chasing bug hunt is in progress.
- Physical moves/deletions are deferred to reduce churn and avoid context breaks.

## Policy (Now)

1. Keep all files in current locations.
2. Classify docs via index and governance matrix.
3. Add metadata labels to clarify normative vs reference status.

## Proposed Future Structure (After Bug-Hunt Stabilization)

Target structure:

- `docs/contracts/`
  - identity/scope/visibility and other invariant behavior contracts
- `docs/protocols/`
  - workflow/runbook/operator protocols
- `docs/spec/`
  - functional and technical spec docs
- `docs/strategy/`
  - rationale and directional documents
- `docs/archive/`
  - superseded non-normative docs that should remain discoverable

Planning and execution records stay in `.planning/` but should be surfaced through curated indexes.

## Candidate Moves (Deferred)

- Move `docs/WORKFLOW_STAGE_BACKEND_NOTES.md` -> `docs/archive/` after extracting any still-relevant details into protocol/spec docs.
- Keep `docs/GRAPH_COMPACTION_ROLLBACK_RUNBOOK.md` in protocols/runbooks (active).
- Keep `docs/EU_LIBRARY_CITATION_WALKING_PATH.md` in strategy (active direction).

## Preconditions for Move Execution

1. Bug-hunt pass complete for Reading/Chasing reliability.
2. Current phase verification docs updated and green.
3. No unresolved doc conflict marked "high" in `docs/DOC_GOVERNANCE_MATRIX.md`.

## Migration Sequence (When Enabled)

1. Copy files into target folders (no delete).
2. Update links from `docs/START_HERE.md` and `docs/DOC_INDEX.md`.
3. Run link/path validation.
4. Remove original files in a follow-up commit only after one stable cycle.

## Success Criteria

- One mandatory entry (`START_HERE`) remains valid.
- Authority order remains explicit and unchanged.
- Historical context remains accessible through archive and planning indexes.
