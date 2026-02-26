---
phase: 09-claim-graph-consensus-viz
research: 07
status: draft
created: 2026-02-07
---

# Phase 09-07 Research: Cytoscape in Streamlit

## What we need (functional)

- Embed Cytoscape.js in the center pane (Surfing tab).
- Render a small subgraph (10-30 nodes; 30-80 edges) from an in-repo JSON dataset.
- Capture selection events (node/edge click) back into Python to drive the inspector.
- Support programmatic focus (center on a selected CiteAnchor / selection) and UI toggles that hide/show layers.
- Deterministic, demo-friendly layout (dagre preferred; COSE as fallback).

## Options

### Option A: Custom Streamlit component (recommended)

Use Streamlit's component API (React) to host Cytoscape.js and report selection with `Streamlit.setComponentValue`.

Pros:
- Full control over styling, layout, overlays, and interaction.
- Selection + programmatic centering are first-class.
- Doesn't constrain us to vis-network limitations.

Cons:
- Adds a Node build step (but we can vendor compiled assets into repo).

Notes:
- We can commit the component's built frontend assets to avoid requiring Node for end users.
- Streamlit components are stable for this use case.

### Option B: Third-party component package

There are community packages for Cytoscape + Streamlit, but they tend to be unmaintained and/or limit event handling.

Pros:
- Less code.

Cons:
- Risky dependency; hard to evolve for our inspector + heat overlays.

### Option C: `st.components.v1.html` + CDN Cytoscape

Pros:
- No build.

Cons:
- Hard to reliably send selection back to Python without a real Streamlit component.
- Ends up re-implementing component plumbing anyway.

### Option D: Custom Streamlit component with *no build* (static HTML + CDN libs)

Use Streamlit's component API, but ship a static `index.html` that loads:

- `streamlit-component-lib` from a pinned CDN URL
- `cytoscape` (and any layout plugins) from pinned CDN URLs

Pros:
- No Node build step; simplest possible Streamlit wiring.
- Still gets reliable Python<->JS messaging (`Streamlit.setComponentValue`).
- Keeps Cytoscape as the long-term durable renderer without adopting a third-party wrapper.

Cons:
- Requires network access to fetch the pinned CDN assets (acceptable for this POC).

## Recommended approach

Implement a custom component (Option D preferred for the POC; Option A acceptable later if we want fully vendored assets):

- Python wrapper: `frontend/components/cytoscape_panel.py`
- Frontend: `frontend/components/cytoscape_component/` (Streamlit component scaffold)

Component API:

Input props:
- `elements`: Cytoscape elements (nodes + edges)
- `style`: stylesheet rules
- `layout`: `{name: 'dagre'|'cose'|... , ...}`
- `selection`: current selection `{type, id}` (optional)
- `focus`: `{nodeIds: [], fit: bool}` (optional)
- `options`: toggles + rendering flags

Output value (component -> Python):
- `{type: 'node'|'edge', id: string}` or `{}`

Why this fits Phase 09:
- Current `streamlit-agraph` renderer already uses a `{type,id}` selection contract. We can preserve this to keep the inspector logic stable while we swap renderers.

## Layout choice

- Use `dagre` for deterministic, readable demos.
- Provide `cose-bilkent` optional but not necessary for the POC.

## Anchors / drift handling

For the POC dataset, we can include token positions as hints, but:

- For "stable anchors", the spec requires quote selectors and/or fingerprints.
- The UI should display these as "stable anchor features" rather than pretending offsets are stable.

## Minimal build strategy

- Prefer a no-build static HTML component for the POC.
- If we later need offline/airgapped installs, vendor built assets and pin dependencies.
