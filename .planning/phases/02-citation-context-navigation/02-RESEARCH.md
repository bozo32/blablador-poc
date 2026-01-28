# Phase 02: Citation Context Navigation - Research

**Researched:** 2026-01-23
**Domain:** Citation context retrieval, citation graph navigation (Streamlit + FastAPI)
**Confidence:** MEDIUM

## Summary

This phase centers on exposing citation context and citation graph navigation in the existing Streamlit + FastAPI stack. The backend already stores TEI/XML and extracted citations; the missing work is to create API endpoints that return sentence-level context around a selected callout and to query a citation graph from a standard public API so the UI can render a node graph and allow users to follow cited works.

The standard approach for citation graph metadata is to use the OpenAlex Works API: it provides `referenced_works` (outgoing references) and `cited_by_api_url` (incoming citations) for each work. For visualization in Streamlit, `st.graphviz_chart` is a supported built-in graph renderer, provided the `graphviz` Python package is installed.

**Primary recommendation:** Use FastAPI endpoints that read TEI data for context (sentence ± neighbors) and call OpenAlex for citation graph expansion; render the tree with Streamlit `st.graphviz_chart` using Graphviz DOT.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | Unpinned (environment.yml) | Backend API for context/citation graph endpoints | Existing backend uses FastAPI; consistent request/response model. |
| Streamlit | v1.53.0 (docs) | UI for callouts, context pane, graph view | Existing frontend is Streamlit; built-in graphviz chart support. |
| lxml | Unpinned (environment.yml) | TEI/XML parsing for citation context | Existing TEI extraction uses lxml; reliable XPath support. |
| Pydantic | >=2 (environment.yml) | API schemas for new endpoints | Existing backend uses Pydantic models. |
| OpenAlex Works API | 2026-01 docs | Citation graph metadata (references + cited-by) | Official API exposes referenced_works and cited_by_api_url. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| requests | Unpinned (environment.yml) | HTTP client for OpenAlex requests | Backend service calls to OpenAlex. |
| graphviz | >=0.19.0 (Streamlit docs) | Build DOT graphs for Streamlit | Required for `st.graphviz_chart`. |
| pandas | Unpinned (environment.yml) | Tabular display of citations (existing) | Keep for dataframes if needed in the UI. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `st.graphviz_chart` | Custom Streamlit component (D3/vis) | More interactivity but higher implementation and maintenance cost. |

**Installation:**
```bash
pip install graphviz
```

## Architecture Patterns

### Recommended Project Structure
```
backend/
├── main.py                     # FastAPI routes
├── citation_context.py         # TEI context extraction helpers
├── citation_graph.py           # OpenAlex client + graph shaping
├── schemas.py                  # Pydantic models for new responses
└── ingestion_store.py          # TEI storage access
frontend/
├── ingestion_api.py            # HTTP helpers for new endpoints
└── ui.py                       # Context pane + graph view
```

### Pattern 1: TEI-based Context Extraction
**What:** Locate the citation callout in TEI, then return the current sentence plus its neighboring sentences (prev/next) with cleaned text.
**When to use:** NAV-01 (context view) and NAV-02 (load cited context into validation pane).
**Example:**
```python
# Source: https://docs.streamlit.io/develop/api-reference/charts/st.graphviz_chart
# (Graphviz example provides DOT-based layout for the citation tree UI)
graph = graphviz.Digraph()
graph.edge("source", "cited")
st.graphviz_chart(graph)
```

### Pattern 2: OpenAlex-driven Citation Graph
**What:** Use DOI/OpenAlex ID to fetch work metadata; use `referenced_works` for outgoing edges and `cited_by_api_url` for incoming edges.
**When to use:** NAV-03 (citation tree).
**Example:**
```python
# Source: https://docs.openalex.org/api-entities/works/get-a-single-work
work = requests.get(
    "https://api.openalex.org/works/https://doi.org/10.7717/peerj.4375",
    params={"api_key": OPENALEX_API_KEY},
    timeout=10,
).json()
referenced = work.get("referenced_works", [])
```

### Anti-Patterns to Avoid
- **Parsing TEI without sentence boundaries:** Leads to misaligned context; always use `<s>` sentence nodes for prev/next context.
- **Client-side OpenAlex calls from Streamlit:** Credentials and rate limits should be handled server-side.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Citation graph expansion | Custom crawler for citation relationships | OpenAlex Works API | Provides referenced_works + cited_by_api_url with stable IDs. |
| Graph layout | Custom layout engine | Graphviz + `st.graphviz_chart` | Provides DOT layout and is supported by Streamlit. |
| TEI XML parsing | Regex/string slicing | lxml + XPath | Handles TEI namespaces and hierarchy safely. |

**Key insight:** Citation graphs and TEI parsing have edge cases (missing IDs, mixed structures) that are already handled by established libraries/services.

## Common Pitfalls

### Pitfall 1: Missing or Empty Context for Callouts
**What goes wrong:** Some `<ref type="bibr">` elements have no sentence context or target ID.
**Why it happens:** TEI extraction may omit `xml:id` or sentence nodes for certain citations.
**How to avoid:** If no sentence context is found, return a structured "context unavailable" response and render the inline message (per decisions).
**Warning signs:** `target_id` is null or `sentence` is empty in extraction output.

### Pitfall 2: OpenAlex API Key Requirement and Rate Limits
**What goes wrong:** Citation graph fetches start failing with 403/429 after 2026-02-13.
**Why it happens:** OpenAlex requires an API key and enforces credit limits.
**How to avoid:** Add a backend setting for the API key and handle 429 errors with retry/backoff and UI toasts.
**Warning signs:** Responses include 403 or 429 when requesting `works` or `cited_by_api_url`.

### Pitfall 3: Graph Explosion on Deep Trees
**What goes wrong:** The citation graph becomes too large to render or interpret.
**Why it happens:** `referenced_works` and `cited_by_api_url` can return many nodes.
**How to avoid:** Enforce depth and per-level node caps; use stub nodes labeled "data unavailable" for missing branches.
**Warning signs:** Graph rendering delays or memory spikes.

## Code Examples

Verified patterns from official sources:

### Streamlit Graphviz Chart
```python
# Source: https://docs.streamlit.io/develop/api-reference/charts/st.graphviz_chart
import streamlit as st
import graphviz

graph = graphviz.Digraph()
graph.edge("root", "ref1")
graph.edge("root", "cited-by-1")
st.graphviz_chart(graph)
```

### OpenAlex Work Lookup by DOI
```python
# Source: https://docs.openalex.org/api-entities/works/get-a-single-work
import requests

resp = requests.get(
    "https://api.openalex.org/works/https://doi.org/10.7717/peerj.4375",
    params={"api_key": "YOUR_API_KEY"},
    timeout=10,
)
work = resp.json()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Unauthenticated OpenAlex usage | OpenAlex API key required for all requests | 2026-02-13 (OpenAlex docs) | Requires configuration of API key in backend settings. |
| Crossref-only metadata | OpenAlex Works API for references + cited-by | Current | Enables citation tree with incoming/outgoing links. |

**Deprecated/outdated:**
- OpenAlex unauthenticated access: deprecated due to API key requirement (effective 2026-02-13).

## Open Questions

1. **Where should the OpenAlex API key live in settings?**
   - What we know: OpenAlex now requires `api_key` for requests.
   - What's unclear: Whether to add a new backend env var (recommended) or reuse existing API settings.
   - Recommendation: Add `OPENALEX_API_KEY` to `backend/settings.py` and `.env`.

2. **Should citation graph responses be cached per DOI/OpenAlex ID?**
   - What we know: Citation graph calls can be frequent and rate-limited.
   - What's unclear: Whether caching is required in v1.
   - Recommendation: Add a simple in-memory cache keyed by OpenAlex ID to reduce repeated calls.

## Sources

### Primary (HIGH confidence)
- https://docs.streamlit.io/develop/api-reference/charts/st.graphviz_chart - Streamlit graphviz chart usage and dependency requirements
- https://docs.openalex.org/api-entities/works - OpenAlex Works overview (referenced_works, cited_by_api_url)
- https://docs.openalex.org/api-entities/works/work-object - Work object fields (referenced_works, cited_by_count)
- https://docs.openalex.org/api-entities/works/get-a-single-work - Work lookup by DOI
- https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication - API key requirement and rate limits

### Secondary (MEDIUM confidence)
- Repository codebase (FastAPI + Streamlit + lxml usage) - current architecture conventions

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - versions for FastAPI/lxml are unpinned in environment.yml; OpenAlex and Streamlit docs are current.
- Architecture: MEDIUM - patterns derived from repository structure and TEI parsing conventions.
- Pitfalls: MEDIUM - OpenAlex constraints are documented; TEI edge cases inferred from extraction behavior.

**Research date:** 2026-01-23
**Valid until:** 2026-02-22
