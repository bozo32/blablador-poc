phase: 10-contracts-core-workflow-simplification
plan: 10-03
title: Graph As Navigation (Cytoscape Routing)

# 10-03: Graph As Navigation (Cytoscape Routing)

## Objective

Turn the graph from a partial “representation tab” into a navigation surface:
selecting nodes/edges routes into the main workspace (document/span/claim).

## Scope

- Define a small selection contract (already `{type,id}`) and map it to
  workspace navigation state.
- Add recursive retrieval hooks (follow citations from cited sources).

## Success Criteria

- Selecting a node/edge updates the active context deterministically.
- The graph is not a dead-end: the user can continue work from a selection.
