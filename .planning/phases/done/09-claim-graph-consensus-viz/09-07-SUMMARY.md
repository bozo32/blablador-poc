---
phase: 09-claim-graph-consensus-viz
plan: 07
subsystem: ui
tags: [cytoscape, streamlit-component, poc, surfing]

requires:
  - phase: 09-claim-graph-consensus-viz/09-06
    provides: Multi-user compare mode
provides:
  - Cytoscape.js renderer embedded via a minimal Streamlit component contract
  - In-repo POC dataset demonstrating stable anchors vs plural segmentation vs heat
  - Surfing (POC) mode with layer toggles, canonical-only view, selection-driven inspector

key-files:
  created:
    - frontend/components/cytoscape_panel.py
    - frontend/components/cytoscape_component/index.html
    - data/poc_graph.json
    - README_poc.md
  modified:
    - frontend/ui.py
    - frontend/state_keys.py

verification:
  automated:
    - python -m py_compile frontend/ui.py frontend/state_keys.py frontend/components/cytoscape_panel.py
    - python -c "import json; json.load(open('data/poc_graph.json'))"
  manual:
    - Pending: 09-07-PLAN.md checkpoint walkthrough

commits:
  - 90ea743 feat(ui): add Cytoscape Streamlit component
  - d3cf9ee feat(poc): add in-repo Cytoscape demo dataset
  - 8c5c695 feat(ui): add Surfing POC mode with Cytoscape
completed: 2026-02-09
---

# Phase 9 Plan 07: Cytoscape Surfing POC Summary

Added a Cytoscape-based Surfing (POC) mode that keeps the canvas quiet by default (no long on-graph labels), while still supporting pan/zoom + selection and an inspector that explains the architectural model:

- Stable anchors: `Work` + `CiteAnchor`
- Plurality begins at segmentation: per-reviewer `ClaimAtom`s
- Coordination without adjudication: `ClaimAnchor` as a coordination object
- Attributable evidence links: `Assertion` -> `EvidenceSpan`
- Heat: computed from disagreement diversity and volume; treated as attention routing, not correctness

Manual verification is still required per `.planning/phases/09-claim-graph-consensus-viz/09-07-PLAN.md`.
