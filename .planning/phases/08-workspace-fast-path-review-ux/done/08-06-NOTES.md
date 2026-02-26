# 08-06 Notes: Graph Relationship Model (For Phase 8.1)

This note captures the organizing model needed for recursive citation chasing.
It is written during Phase 8 execution so the concepts don't get lost.

## Goal

When we start walking citation trees back and forth, the user-facing organizing
relationships should map cleanly to a single underlying graph representation:

- parents / children (one hop)
- ancestors / descendants (multi-hop)
- siblings (share a parent)
- cousins (relevant to the same claim/topic but not directly connected by cites)

## Core Representation

Persist a graph as nodes + edges (and treat "relationship views" as queries).

### Nodes

- `document`
  - id: `doc:{doc_id}`
  - keys: `doc_id`, `filename`, `sha256`
  - optional: `title`, `year`, `authors`

- `work`
  - id: `work:{work_id}` (OpenAlex id / DOI normalized / local bib id)
  - keys: `doi`, `openalex_id`, `bib_target_id` (e.g., `R1`, `b4`)
  - note: allow multiple identifiers; use a canonical id + aliases

- `claim`
  - id: `claim:{claim_id}`
  - keys: `claim_id`, `claim_text`, `created_at`
  - optional: `doc_id` (where the claim originated)

- `evidence_span`
  - id: `span:{attachment_id}:{span_id}` (stable span ids)
  - keys: `attachment_id`, `span_id`, `sentence_id` (if available)

- `attachment`
  - id: `att:{attachment_id}`
  - keys: `attachment_id`, `filename`, `sha256` (if available), `status`
  - optional: `doc_id` (citing doc context), `target_id`, `citation_index`

### Edges

- `cites`
  - from: `document` -> `work` (or `document` -> `document` when resolved)
  - props: `citation_index`, `target_id`, `callout_text`, `sentence_id`

- `ingests`
  - from: `document` -> `attachment` (optional; expresses this attachment came
    from this citing doc session)

- `resolves_to`
  - from: `work` -> `work` (alias -> canonical)
  - props: `source` (crossref/openalex/local)

- `supports` / `contradicts` / `neutral`
  - from: `claim` -> `evidence_span`
  - props: `confidence`, `model`, `run_id`

- `span_in`
  - from: `evidence_span` -> `attachment` (or to `document` if attachment maps)
  - props: `page`, `bbox` (if available)

- `about`
  - from: `claim` -> `work` (optional; the claim is about validating a cited work)
  - props: `target_id`, `citation_index`

## Relationship Views (Queries)

Define these as graph queries, not bespoke data models:

- `parents(node)`
  - `document`: outgoing `cites` targets (works/docs)
  - `work`: incoming `cites` from documents (depending on view)

- `children(node)`
  - `document`: incoming `cites` (docs that cite this doc/work)
  - `work`: documents that cite the work (incoming `cites`)

- `ancestors(node, depth=N)`
  - N-hop traversal along `cites` (direction depends on whether user means
    "references" vs "cited-by")
  - requirements: cycle detection + depth cap

- `descendants(node, depth=N)`
  - reverse traversal of the above

- `siblings(node)`
  - nodes that share the same parent at distance 1
  - e.g., two works cited by the same document section; or two documents that
    cite the same parent work

- `cousins(node)`
  - "relevant to the same claim" but not connected by `cites`
  - implement as: nodes connected via `claim` clusters / shared evidence spans /
    shared normalized work identifiers
  - ranking signal: shared claim_id(s), shared work_id, shared embeddings

## Phase 8 Prereqs (So 8.1 Is Easy)

To make the queries reliable later, Phase 8 components should persist:

- stable `doc_id` for citing documents
- stable `target_id` + `citation_index` when placing source-bin items
- stable `sentence_id` in document body (08-01 fallback already handles this)
- stable `span_id` for attachments (already used for PDF jumping)
- identifier aliases for works (doi/openalex/local)

---

## UI Contract: Quarto/Cosmo (Phase 8 Cleanup)

This captures the agreed styling + layout direction so we can resume after a
`/clear` without losing intent.

### Target Feel

- Quarto / Bootswatch "Cosmo"-like: light, crisp, modern, text-forward.
- Subtle shadows only; borders over heavy drop-shadows; high contrast.

### Key Insight

Most "messiness" is structural: the app mixes epistemic modes (Reading vs
Chasing/Validation vs Graph) in the same center surface. Fix by making center
mode explicit and suppressing irrelevant affordances per mode.

### CSS Strategy (Streamlit-safe)

- Avoid relying on Streamlit internal DOM class names for component semantics.
- Use wrapper divs emitted by `st.markdown(..., unsafe_allow_html=True)`:
  - `.bb-card`, `.bb-section-header`, `.bb-toolbar`
  - `.bb-primary` / `.bb-danger` wrappers around Streamlit buttons
  - `.bb-sticky` wrapper for the right rail (export/judgment)

### Proposed Files

- Add: `frontend/assets/cosmo_streamlit.css`
  - tokens + primitives:
    - `--primary #2780E3`, `--success #3FB618`, `--warning #FF7518`, `--danger #FF0039`
    - `--text #2C3E50`, `--muted #6C757D`, `--border #DEE2E6`, `--bg #FFFFFF`, `--bg-subtle #F8F9FA`
    - `--radius 10px`, `--shadow-sm 0 1px 6px rgba(0,0,0,.06)`
  - `.bb-card`, `.bb-section-header`, `.bb-badge*`, `.bb-chip*`, `.bb-evidence*`, `.bb-toolbar`
- Loader: keep `frontend/assets/workspace.css` for layout scaffolding, but load
  Cosmo CSS after it so tokens/primitives win.

### Center Panel Mode Contract

- Wrap the center column content in exactly one of:
  - `.bb-center bb-mode-reading`
  - `.bb-center bb-mode-chasing`
  - `.bb-center bb-mode-graph`

- Add sticky center header:
  - `.bb-center-header` with `Mode: Reading | Chasing | Graph`
  - This prevents disorientation when content changes drastically.

### Mode Suppression Rules (high-level)

- Reading:
  - text-first; hide `.bb-toolbar`, `.bb-chiprow`, `.bb-badge` in center
  - keep citation chips for navigation
- Chasing:
  - object-first; show cards, toolbars, evidence list
  - show claim focus banner persistently
- Graph:
  - canvas-first; hide cards/headers/toolbars in center, show only canvas + small overlay
  - keep controls in right rail if cross-mode

### Where to Implement (code anchors)

- `frontend/ui.py`
  - `render_evidence_panel()`
    - Wrap judgment controls + rerun controls + filters into `.bb-card` blocks
    - Make exactly one primary action per block (e.g. Save judgment)
    - Move run history/diagnostics into a collapsed section in the right rail
  - `draw_ingestion_panel(center=..., right=...)`
    - Introduce center-mode wrapper + sticky header
    - Ensure mode switching is explicit and stable

### UX Requirements

- One primary action per panel (avoid multiple equally loud buttons).
- Right rail is sticky; center is scrollable.
- Export block uses modern download buttons; no dark blocks.
- Maintain accessibility: visible focus, readable placeholder/text, no hidden toggles.
