# Phase 10: Contracts + Core Workflow Simplification (INSERTED)

## Why This Phase Exists

The current system works end-to-end, but key workflow stages are coupled by
ad-hoc dict payloads, process globals, and implicit side effects.

For a local POC where "working features" matter more than robustness, the
highest leverage simplification is to make stage boundaries explicit so
components can be swapped (e.g. ColBERT reranker) without rewiring the rest of
the workflow.

## Primary Goal

Define and adopt a minimal set of stage contracts (typed artifacts) so the core
workflow becomes:

- easy to understand
- easy to extend (add-ons attach at clear boundaries)
- easy to verify (a single E2E script exercises the contracts)

## Scope (Mirrors `.planning/PRIORITIES.md`)

Phase 10 covers Priorities 1-7:

1) Contracts arc
2) Core workflow simplification
3) Graph as navigation
4) Durable decision/event layer
5) Demo + trace replay
6) Optional add-ons
7) Deprioritized ops/cloud direction

## Exit Criteria

- A single, documented “happy path” for citespan -> candidates -> rerank -> NLI
  -> assessment uses explicit contracts between stages.
- A small orchestrator/state machine executes stages and records durations and
  timeouts at boundaries (without introducing distributed-systems complexity).
- Graph/Cytoscape becomes navigation (selection routes into the main workspace)
  rather than a dead-end tab.
