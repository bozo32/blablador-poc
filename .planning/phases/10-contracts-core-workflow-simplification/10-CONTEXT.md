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

## Documentation Governance Decisions (2026-03-03)

### Phase Boundary (within Phase 10)

This discussion locks HOW docs express and enforce the existing contract/protocol
model. It does not add new product capability.

### Implementation Decisions

- Authority order is fixed as: Contracts > Protocol > Spec.
- Single-entry startup is mandatory via `docs/START_HERE.md`.
- Historical planning docs stay in place during active bug hunts; use indexing
  and classification first, defer physical reshuffle.
- Contract vs protocol split is fixed: contract docs define invariant behavioral
  requirements, protocol docs define verification/workflow execution.
- Distributed/cloud guidance belongs in spec/rationale docs, not behavior
  contracts.

### Claude's Discretion

- Naming and shape of compact/use-case packs can evolve as long as mandatory
  entry and authority order remain unchanged.
- Index formatting and cross-link density are implementation details.

### Specific Ideas Captured

- Session startup should support commands like:
  - `read docs/OS_ERIN-compact.md`
  - `read docs/OS_ERIN-usecase-<mode>.md`
- Docs should be organized as a coherent "pile" with a single road in and clear
  guardrails for conflict resolution.

### Deferred Ideas

- Physical archive/move/delete pass for legacy docs is deferred until current
  bug-hunt stabilization completes.
