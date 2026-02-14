# Working Notes (Casual Mode)

<!-- SNAPSHOT:START -->
Updated: `2026-02-14T10:00:38.646243Z`
Branch: `feature/span-claim-graph-rebuild`
HEAD: `d50bb56`

```text
## feature/span-claim-graph-rebuild...origin/feature/span-claim-graph-rebuild [ahead 37]
 M .planning/NOTES.md
 M backend/grobid_client.py
 M backend/main.py
 M frontend/ui.py
?? .opencode/command/notes-snapshot.md
?? .opencode/command/notes.md
?? scripts/update_notes_snapshot.py
```

```text
.planning/NOTES.md       |  4 ++++
 backend/grobid_client.py | 33 ++++++++++++++++++++++-----------
 backend/main.py          | 18 +++++++++++++++++-
 frontend/ui.py           |  6 +++++-
 4 files changed, 48 insertions(+), 13 deletions(-)
```

```text
d50bb56 (HEAD -> feature/span-claim-graph-rebuild) docs: add lightweight recovery workflow
892ffd5 chore(attach): avoid cross-thread graph_store
409333d chore(ingest): align resolution typing and schema
dca818d feat(ingest): queue uploads for background processing
9f1451d chore(git): ignore macOS metadata
```
<!-- SNAPSHOT:END -->

Use this as a lightweight, resumable scratchpad when you're not running a full
GSD plan.

Update it at the start/end of a session so context loss is cheap.

## Goal

- What are we trying to change and why?

## Current State

- Branch: `...`
- Last known green: `pytest -q` (pass/fail) at <timestamp>
- Current behavior (1-3 bullets):

## What Changed (So Far)

- Files touched:
- New/changed endpoints, flags, settings:
- Data migrations or backfills:

## Known Risks / Loose Ends

- Concurrency / idempotency:
- State drift (schemas vs persisted JSON):
- UI caching / session state:

## Next 3 Actions

1.
2.
3.

## How To Verify

- Commands:
- Manual checks:
- Links (PR, issue, docs):

## Rollback Plan

- If this goes sideways, revert commit(s):
- Or disable feature via setting:
