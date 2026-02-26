# Phase 3: Claim Selection + Editing - Research

**Researched:** 2026-01-24
**Domain:** Streamlit UI workflows for claim selection and editing
**Confidence:** MEDIUM

## Summary

This phase is centered on a Streamlit UI workflow: presenting clause-level claim candidates from a citing sentence, letting users queue and confirm claim parses, and editing claim text in a modal. The repo already uses Streamlit for the UI and FastAPI with Pydantic models for claim-related data, so the standard approach is to extend the existing Streamlit app with session-state-backed queues, expanders for each queued claim panel, and a modal editor powered by `st.dialog` for in-place edits.

Official Streamlit documentation confirms the primitives needed here: `st.dialog` for modal editors with independent reruns, `st.expander` for collapsible queue panels, and `st.text_area` with `on_change` callbacks plus `st.session_state` for autosave behavior. The main pitfalls are Streamlit execution model constraints (state updates after widget creation, only one dialog open, and expanders computing content even when collapsed) so planning should explicitly model state storage and UI reruns around these limitations.

**Primary recommendation:** Use `st.session_state` to store the claim queue and edits, render queued items via `st.expander`, and implement the modal editor with `@st.dialog` plus `st.text_area(..., on_change=...)` callbacks for autosave.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Streamlit | v1.53.0 (docs) | UI framework for the app | Existing UI stack; provides `st.dialog`, `st.expander`, and session state primitives. |
| FastAPI | Unpinned (env) | Backend API for claim data | Existing backend service for ingestion and claim-related endpoints. |
| Pydantic | >=2 (env) | Typed request/response models | Used in `backend/schemas.py` for claim segment structures. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Requests | Unpinned (env) | UI to backend HTTP calls | Existing `frontend/ingestion_api.py` uses Requests for API calls. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `st.dialog` | Custom Streamlit component modal | More control, but adds JS build and component maintenance overhead. |

**Installation:**
```bash
pip install streamlit fastapi pydantic requests
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/
├── ui.py                 # Streamlit entry point
├── claim_queue.py        # Queue rendering + status logic (new)
├── claim_editor.py       # Modal editor helpers (new)
└── ingestion_api.py      # Backend API wrappers
```

### Pattern 1: Session-State Claim Queue
**What:** Store queued claims, statuses, and edits in `st.session_state` keyed by citation + claim IDs.
**When to use:** Every interaction that should persist across reruns (queue order, processed status, edited text).
**Example:**
```python
# Source: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
if "claim_queue" not in st.session_state:
    st.session_state["claim_queue"] = []

st.session_state.claim_queue.append(claim_id)
```

### Pattern 2: Modal Editor with `st.dialog`
**What:** Use `@st.dialog` to open a modal editor with autosave callbacks.
**When to use:** Editing a claim without leaving the queue context.
**Example:**
```python
# Source: https://docs.streamlit.io/develop/api-reference/execution-flow/st.dialog
@st.dialog("Edit claim", width="medium")
def edit_claim(claim_id):
    text = st.text_area("Claim text", key=f"claim-text-{claim_id}")
    if st.button("Close"):
        st.rerun()
```

### Pattern 3: Collapsible Queue Panels with `st.expander`
**What:** Render each queued claim inside a collapsible container.
**When to use:** Long queues where processed claims should collapse to save space.
**Example:**
```python
# Source: https://docs.streamlit.io/develop/api-reference/layout/st.expander
with st.expander("Smith (2020) - Pending", expanded=True):
    st.write(full_sentence)
```

### Anti-Patterns to Avoid
- **Widget state mutation after creation:** Streamlit forbids changing widget values via `st.session_state` after instantiation; set keys before widget creation.
- **Nested expanders for queue items:** Streamlit warns against nested expanders for responsive layout.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Modal editing | Custom HTML/CSS modal overlay | `st.dialog` | Built-in modal with rerun isolation and dismiss controls. |
| Collapsible panels | Custom show/hide state + Markdown | `st.expander` | Accessible, standard collapse behavior. |
| Autosave state | Global dicts or module globals | `st.session_state` + `on_change` | Survives reruns and aligns with Streamlit execution model. |

**Key insight:** Streamlit already provides modal, collapsible, and state primitives that match the UI requirements; custom implementations fight the rerun model and risk state loss.

## Common Pitfalls

### Pitfall 1: Dialog limitations
**What goes wrong:** Multiple dialogs or sidebar calls inside a dialog fail or behave unpredictably.
**Why it happens:** Streamlit allows only one dialog at a time and disallows `st.sidebar` in a dialog.
**How to avoid:** Ensure a single active dialog per run and keep dialog UI self-contained.
**Warning signs:** Dialogs not opening, sidebar errors, or stale dialog state.

### Pitfall 2: Widget state updates after instantiation
**What goes wrong:** `StreamlitAPIException` when updating widget state after creation.
**Why it happens:** Session State disallows modifying widget values once instantiated.
**How to avoid:** Initialize session state keys before creating widgets; update via callbacks.
**Warning signs:** Errors when setting `st.session_state` for text areas or inputs.

### Pitfall 3: Hidden expander still computes content
**What goes wrong:** Expensive computations run even when panel is collapsed.
**Why it happens:** Streamlit computes and sends expander content regardless of open/closed state.
**How to avoid:** Guard heavy operations before expander render or cache results.
**Warning signs:** Slow UI even with panels collapsed.

## Code Examples

Verified patterns from official sources:

### Autosave Text Area with Callback
```python
# Source: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
def on_claim_change(claim_id):
    st.session_state["claim_edits"][claim_id] = st.session_state[
        f"claim-text-{claim_id}"
    ]

st.text_area(
    "Claim",
    key=f"claim-text-{claim_id}",
    on_change=on_claim_change,
    args=(claim_id,),
)
```

### Modal Dialog for Editing
```python
# Source: https://docs.streamlit.io/develop/api-reference/execution-flow/st.dialog
@st.dialog("Edit claim", width="medium")
def edit_claim(claim_id):
    st.text_area("Claim text", key=f"claim-text-{claim_id}")
    if st.button("Done"):
        st.rerun()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full-script reruns for modal editing | `st.dialog` reruns the dialog independently | Streamlit v1.53.0 docs | Less disruptive UI updates for claim edits. |

**Deprecated/outdated:**
- Custom modal components: heavier maintenance and not needed for standard modal workflows.

## Open Questions

1. **Where should edited claims be persisted beyond the UI session?**
   - What we know: Streamlit session state is per-session and resets on reload.
   - What's unclear: Whether claim edits must be sent to the backend or stored in a future DB.
   - Recommendation: Plan for a backend persistence hook if edits must survive reloads.

## Sources

### Primary (HIGH confidence)
- https://docs.streamlit.io/develop/api-reference/execution-flow/st.dialog - modal dialog API and constraints
- https://docs.streamlit.io/develop/api-reference/widgets/st.text_area - text area widget + callbacks
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state - session state and callbacks
- https://docs.streamlit.io/develop/api-reference/layout/st.expander - collapsible panel container

### Secondary (MEDIUM confidence)
- .planning/PROJECT.md - repo stack context (Streamlit + FastAPI backend)
- environment.yml - package list and versions where pinned

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - Streamlit docs are authoritative; FastAPI/Pydantic versions not pinned.
- Architecture: HIGH - Patterns directly supported by Streamlit APIs.
- Pitfalls: HIGH - Documented Streamlit limitations from official docs.

**Research date:** 2026-01-24
**Valid until:** 2026-02-23
