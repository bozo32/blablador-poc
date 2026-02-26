phase: 10-contracts-core-workflow-simplification
plan: 10-02
title: Core Workflow Simplification (Happy Path)

# 10-02: Core Workflow Simplification (Happy Path)

## Objective

Make the core reviewer workflow maximally simple and predictable, with a thin
orchestrator/state machine over stage artifacts (not a new distributed system).

## Scope

- One “happy path” flow for a local POC:
  - ingest -> extract -> citespans -> candidates -> rerank -> NLI -> assessment
- Timeouts at boundaries (HTTP calls) and duration recording per stage.
- Remove/disable confusing mode switches that are not meant to work in the POC.

## Deliverables

- A single “run” entrypoint that advances stage-by-stage and writes artifacts.
- Clear API endpoints to:
  - trigger a run
  - poll stage state
  - fetch the latest artifact per stage

## Success Criteria

- A reviewer can run the happy path on a small corpus without manual
  restarts/retries.
- Add-ons (e.g. ColBERT) attach by implementing a single interface and
  producing the same contract output.
