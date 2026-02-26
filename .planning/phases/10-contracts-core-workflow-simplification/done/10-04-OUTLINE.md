phase: 10-contracts-core-workflow-simplification
plan: 10-04
title: Durable Decisions (Evidence Decision Events)

# 10-04: Durable Decisions (Evidence Decision Events)

## Objective

Make reviewer decisions durable and API-backed so UI session state is not the
source of truth.

## Scope

- Evidence candidate decision events (pin/accept/reject/clear)
- Idempotency keys for writes
- (Optional) optimistic concurrency control per claim

## Success Criteria

- Decisions persist across restarts and are replayable.
- Decisions remain attached to stable IDs (span selectors / attempt ids).
