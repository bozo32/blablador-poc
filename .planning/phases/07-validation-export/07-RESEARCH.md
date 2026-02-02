# Phase 07: Validation + Export - Research

**Researched:** 2026-02-02
**Domain:** Streamlit verdict capture + FastAPI persistence + export (JSON/CSV)
**Confidence:** HIGH

## Summary

Phase 07 should piggyback on the patterns introduced in Phase 06: per-claim state is keyed by `claim_id`, persisted on disk as JSON via a small backend “store” class and exposed through simple FastAPI endpoints. The Streamlit UI already has a natural “claim card” surface (the evidence review panel’s focused-claim header) and a natural “callout badge” surface (the citation chips/buttons rendered in the Document text view).

For planning, the critical repo-specific constraint is ID and mapping hygiene: this codebase has two claim-id shapes in use (`{row_id}:{segment_id}` and `cite:{doc_id}:{cite_idx}:{segment_id}`), and callouts are rendered from TEI-derived `citation_index/target_id/sentence_id` (see `backend/tei_body.py`). To make “callout validated/unvalidated” robust across reruns/reloads and across workflows, the persisted judgment payload must include enough provenance (doc/citation/sentence/callout identifiers) to reconstruct callout→claim relationships without relying on Streamlit session state.

**Primary recommendation:** Implement a Phase-06-style on-disk `JudgmentStore` keyed by `claim_id`, persist both verdict state (draft vs final) and callout provenance, and drive UI badges + export by joining `JudgmentStore` records with existing claim registry metadata and (optionally) Phase 06 evidence selections.

## Standard Stack

### Core
| Library/Tool | Version | Purpose | Why Standard |
|---|---:|---|---|
| FastAPI | repo | Backend endpoints for judgment CRUD/export | Already used in `backend/main.py` |
| Pydantic v2 | repo | Validate judgment payloads, defaulting/normalization | Already used in `backend/schemas.py` + stores |
| Streamlit | repo | UI for inline editing + download buttons | `frontend/ui.py` is the main UI |
| Python stdlib `json` | n/a | Persist store files + build JSON exports | Used in `backend/evidence_selection_store.py` |
| Python stdlib `csv` | n/a | CSV export (optional) | Avoid extra deps for CSV |

### Supporting
| Library/Tool | Version | Purpose | When to Use |
|---|---:|---|---|
| pandas | repo | Quick CSV generation for “flat” exports | If CSV needs quoting/escaping and consistent column order |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|---|---|---|
| On-disk JSON store | SQLite (extend `backend/claim_store.py`) | Higher schema effort; careful mapping between UI `claim_id` strings vs DB `claim_id` integers |

**Installation:** none (uses repo’s existing dependencies).

## Architecture Patterns

### Recommended Hook Points (Repo-Specific)

- **Backend persistence pattern:** mirror `backend/evidence_selection_store.py` (safe filename, `root_dir` derived from `settings.EVIDENCE_STORE_DIR.parent`, Pydantic validation, JSON `model_dump(mode="json")`).
  - Touchpoints: `backend/evidence_selection_store.py`, `backend/schemas.py`, `backend/main.py` (Phase 06 selection endpoints).

- **Frontend state + API pattern:** mirror `frontend/evidence_store.py` + `frontend/evidence_api.py`.
  - Touchpoints: `frontend/evidence_api.py` (request wrappers), `frontend/evidence_store.py` (session caching and save/sync methods).

- **Inline claim-card UI:** add judgment controls near the focused claim header inside `render_evidence_panel()`.
  - Touchpoints: `frontend/ui.py` inside `render_evidence_panel()` (see the existing “Overall source assessment” section saving Phase 06 selection via `store.save_selection(...)`).

- **Callout badge UI:** extend Document text callout rendering (chips + per-paragraph buttons) to show an outcome-specific indicator (icon+color) and make it clickable to open judgment details.
  - Touchpoints: `frontend/ui.py` Document text rendering loop (see `chip_class = "citation-chip"` and the button row rendering `doc-cite-btn::...`).

### Pattern 1: Store-Backed CRUD (Phase 06 precedent)
**What:** a tiny store class handles read/upsert and JSON file IO; API endpoints call store methods and return Pydantic models.
**When to use:** persisting per-claim judgments in `data/` without adding database schema.
**Example (existing):**

```python
# Source: backend/evidence_selection_store.py
class EvidenceSelectionStore:
    @property
    def root_dir(self) -> Path:
        base = getattr(self.settings, "EVIDENCE_STORE_DIR", None)
        if base is None:
            return Path("data") / "evidence_selections"
        return Path(base).parent / "evidence_selections"

    def read(self, claim_id: str) -> Optional[EvidenceSelectionPayload]:
        path = self._path_for_claim(claim_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EvidenceSelectionPayload.model_validate(payload)
```

### Pattern 2: Streamlit “Save then rerun”
**What:** UI controls update session state; “Save” triggers an API call and then `_rerun()` to refresh the view.
**When to use:** judgment editing inline on claim card; collapsing notes after save.
**Example (existing):**

```python
# Source: frontend/ui.py (evidence review)
if st.button("Save assessment", key=f"overall-save-{selected_claim}", type="primary"):
    store.save_selection(selected_claim, verdict=..., note=...)
    _rerun()
```

### Anti-Patterns to Avoid
- **Relying on Streamlit session state for export:** exports must work after reload; store judgment/provenance on disk and export from persisted records.
- **Mixing SQLite claim IDs with UI claim IDs:** `backend/claim_store.py` uses integer `claim_id` and separate `sentence_id/claim_index`; evidence/judgment flows currently key on string `claim_id`.

## Don't Hand-Roll

| Problem | Don’t Build | Use Instead | Why |
|---|---|---|---|
| Payload validation | ad-hoc dict checks in UI | Pydantic models in `backend/schemas.py` | Prevents malformed draft/final states; consistent normalization |
| File-safe claim IDs | custom path munging scattered in code | reuse `_SAFE_ID_RE` + `_safe_claim_id()` pattern | Avoid path traversal and OS-invalid filenames |
| Downloads | custom file server | `st.download_button` | Already used for JSON result download in `frontend/ui.py` |
| CSV quoting/escaping | manual string concat | `csv.DictWriter` or `pandas.DataFrame.to_csv` | Correctly handles commas/newlines/quotes |

**Key insight:** This repo already uses “store + Pydantic + simple endpoints” for Phase 06; extending that pattern to judgments keeps Phase 07 small and testable.

## Common Pitfalls

### Pitfall 1: “Validated” computed from the wrong field
**What goes wrong:** callouts show validated when a draft exists (or when Phase 06 evidence selection exists) rather than when a final verdict is set.
**Why it happens:** Phase 06 already saves a `verdict` in `data/evidence_selections/*.json`, but Phase 07’s definition of validated is: “final verdict set” (not merely “saved something”).
**How to avoid:** store an explicit judgment state (e.g., `status: draft|final`) and compute `validated = (status == "final" and verdict in {support, contradict, uncertain})`.
**Warning signs:** callout badge flips to validated immediately after saving notes without finalizing.

### Pitfall 2: Callout→claim mapping breaks across workflows
**What goes wrong:** clicking a callout status indicator can’t find the right claim(s).
**Why it happens:** claim IDs exist in two formats:
- CSV/segmentation flow: `claim_id = f"{row_id}:{segment_id}"` (see `frontend/claim_queue.py#L75`)
- Citation-chasing flow: `claim_id = f"cite:{doc_id}:{cite_idx}:{segment_id}"` (see `frontend/ui.py#L2729`)
**How to avoid:** persist callout provenance in the judgment record (doc_id, citation_index, target_id, sentence_id, callout text), and when routing from callout→judgment, match on those fields rather than trying to parse claim_id.
**Warning signs:** status indicators work for demo/one path but not the other.

### Pitfall 3: Draft/final UI state lost on rerun
**What goes wrong:** notes editor collapses/expands unpredictably; radio buttons reset.
**Why it happens:** Streamlit reruns wipe local variables; only `st.session_state` and persisted stores survive.
**How to avoid:**
- Hydrate judgment state at render start (like `EvidenceStore.sync_selection()` exists but is not currently used in `frontend/ui.py`).
- Keep a per-claim UI toggle key for “notes editor open” and force it closed after successful save.
**Warning signs:** saving verdict causes notes fields to reopen with old content.

### Pitfall 4: Safe filename collisions
**What goes wrong:** two distinct claim IDs map to the same JSON filename.
**Why it happens:** `_safe_claim_id()` replaces non `[a-zA-Z0-9._-]` chars with `_` (colons, slashes, spaces collapse).
**How to avoid:** include a short hash suffix (e.g., sha1 of full claim_id) in filenames, or nest by doc_id/citation_index.
**Warning signs:** judgments “overwrite” between claims that differ only by separators.

### Pitfall 5: Per-callout CSV shape ambiguity
**What goes wrong:** nested JSON export is fine, but CSV export becomes unusable (arrays/dicts in cells).
**Why it happens:** “per-callout” naturally groups multiple claims.
**How to avoid:** define CSV as a flattened row-per-(callout, claim) join; reserve nested grouping for JSON.
**Warning signs:** CSV contains stringified JSON blobs with inconsistent keys.

## Code Examples

### Example 1: Judgment payload shape (recommended)

```python
# Source pattern: backend/schemas.py + backend/evidence_selection_store.py

# Required by LOCKED decisions:
# - one verdict per claim
# - support/contradict/uncertain
# - draft/unreviewed allowed; "validated" only when final verdict set

JudgmentStatus = Literal["draft", "final"]
JudgmentVerdict = Literal["support", "contradict", "uncertain"]

class JudgmentNotes(BaseModel):
    rationale: Optional[str] = None
    caveats: Optional[str] = None
    followups: Optional[str] = None

class JudgmentPayload(BaseModel):
    claim_id: str
    updated_at: Optional[str] = None
    status: JudgmentStatus = "draft"
    verdict: Optional[JudgmentVerdict] = None
    notes: Optional[JudgmentNotes] = None
    # provenance for callout mapping + export
    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    sentence_id: Optional[str] = None
    callout: Optional[str] = None
```

### Example 2: Streamlit download buttons (existing pattern)

```python
# Source: frontend/ui.py#L3574
st.download_button(
    "Download all results as JSON",
    data=json.dumps(payload, indent=2),
    file_name="citation_support_results.json",
    mime="application/json",
)
```

### Example 3: Where to compute callout badge state

```python
# Source of callout data: backend/tei_body.py (segments include citation_index/target_id/sentence_id)
# Source of claim ids for a citation: frontend/ui.py#L2729 (claim_id = cite:{doc_id}:{cite_idx}:{segment_id})

# Recommended aggregation for callout indicator:
# - if no claims for callout: show "unvalidated"
# - if any claim has final verdict: show outcome styling based on that claim
# - if multiple claims disagree: show "uncertain" styling (or an "mixed" visual, but verdict remains per-claim)
```

## State of the Art (Repo-Current)

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| Keep reviewer choices only in session | Persist per-claim JSON selections on disk | Phase 06 (2026-01-31) | Enables Phase 07 to survive reloads and export reliably |

**Notable repo facts to plan around:**
- TEI span anchors for PDF navigation use `sentence_id` as `span_id` (Phase 06 decision; see `.planning/phases/06-evidence-review-selection/06-01-SUMMARY.md`).
- Document-body callout segments already include `sentence_id` (see `backend/tei_body.py#L78`).

## Open Questions

1. **How should a single callout badge behave when multiple claims exist for that citation index?**
   - What we know: claim creation in the chasing flow creates multiple `claim_id`s per `cite_idx` (`frontend/ui.py#L2729`).
   - What's unclear: whether the callout indicator should represent “all claims validated” vs “any claim validated”, and how to style disagreements.
   - Recommendation: treat callout badge as validated if *any* claim for that callout has a final verdict; clicking should open a filtered view listing all claims for that callout.

2. **Should Phase 07 reuse Phase 06’s `EvidenceSelectionPayload.verdict` as the final verdict?**
   - What we know: Phase 06 already persists `verdict` as `support|contradict|uncertain|none` keyed by `claim_id`.
   - What's unclear: Phase 07’s “draft vs final” semantics and structured notes don’t exist in Phase 06 payload.
   - Recommendation: keep a separate judgment payload so “final” is explicit, and optionally include/merge evidence selection data in verbose export.

## Sources

### Primary (HIGH confidence)
- `backend/evidence_selection_store.py` - On-disk JSON store pattern, safe filenames, root dir derivation.
- `backend/main.py` - Evidence selection endpoints (`/claims/{claim_id}/evidence/selection`).
- `backend/tei_body.py` - Callout segments and provenance fields (`citation_index`, `target_id`, `sentence_id`).
- `frontend/ui.py` - Claim registration for citations (`claim_id = cite:{doc_id}:{cite_idx}:{segment_id}`), citation chip rendering, and `st.download_button` usage.
- `frontend/evidence_store.py` + `frontend/evidence_api.py` - Session caching + API wrapper pattern.

### Secondary (MEDIUM confidence)
- `.planning/phases/06-evidence-review-selection/06-01-SUMMARY.md` - Phase 06 decisions that Phase 07 depends on.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all components exist in-repo and are already used.
- Architecture: HIGH - Phase 06 provides a direct template (store + endpoints + Streamlit save/rerun).
- Pitfalls: HIGH - derived from observed repo workflows and ID formats.

**Research date:** 2026-02-02
**Valid until:** 2026-03-02
