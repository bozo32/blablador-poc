---
title: Planning Index / Priority Order
updated: 2026-02-26
---

# Planning Index / Priority Order

This repo has a lot of historical planning docs under `.planning/`.

For ongoing /gsd-managed development, treat *this file* as the canonical
"what to do next" ordering, and treat other planning docs as reference
material.

## Priority Order (POC-first)

0) Phase 10: Demo + trace replay (10-05)
- Goal: reproducible E2E workflow (seed corpus + trace replay).
- Source: `.planning/phases/10-contracts-core-workflow-simplification/10-05-OUTLINE.md`

1) Contracts arc (make stage boundaries explicit + swappable)
- Goal: define stable, versioned artifacts/contracts between stages so add-ons
  like ColBERT can swap in/out without touching callers.
- Primary scope: citespan -> candidates -> ranked -> NLI -> decision.
- Sources:
  - `.planning/architecture/SPAN_GRAPH_MODEL.md`
  - `.planning/architecture/GRAPH_WORKBOARD_CORE.md`
  - `.planning/V2-REPLUMBING-PLAN.md` (spine artifact patterns)

2) Core workflow simplification (happy-path UX + minimal orchestration)
- Goal: keep the core reviewer workflow maximally simple; orchestration is a
  small state machine over stage artifacts (timeouts at boundaries; record
  durations; avoid hidden globals).
- Sources: `.planning/ROADMAP.md` (Phase 8), `.planning/REQUIREMENTS.md`

3) Make the graph a navigation surface (not a tab)
- Goal: graph selections route into chase/workspace state; support recursive
  retrieval across a citation tree.
- Sources: `.planning/ROADMAP.md` (Phase 8.1), `.planning/REQUIREMENTS.md`

4) Durable decision/event layer (UI not the source of truth)
- Goal: evidence candidate decisions as durable events + idempotency.
- Sources: `.planning/REQUIREMENTS.md` (DEC-01/02/03)

5) Demo + trace replay (repeatable, regression-friendly)
- Goal: packaged corpus + seeded artifacts + record/replay traces.
- Sources: `.planning/ROADMAP.md` (Phase 8.2), `.planning/REQUIREMENTS.md`

6) Optional add-ons (only if they help the POC)
- Hosted HF inference mode (ML-02)
- Tri-level assessment (VAL-05..08)
- Neighborhood contradiction search / cherry-pick signals

7) V2 / ops / cloud direction (deprioritize for local POC)
- Works Manager, model gateway, capability discovery, cloud/multi-user specs.
- Sources: `.planning/V2-PLANNING.md`, `.planning/architecture/*_SPEC.md`

## Phase Mapping

- Priority 0: Phase 10 (continue execution)
- Phase 09.3 is complete; reference lives under `.planning/phases/done/09.3-spine-everywhere-legacy-removal/`

## How To Use This With GSD

- Use `/gsd-plan-phase` to plan the next small chunk from this list.
- Use `/gsd-execute-phase` to implement it.
- Use `/gsd-verify-work` for a crash-resumable verification log.

If you need context, the refreshed codebase map lives in `.planning/codebase/`.
