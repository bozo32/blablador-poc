# Span Anchoring + Work-Universal Locators (Spec)

This spec defines how to represent and resolve text spans robustly when:

- extraction is re-run (different parser versions)
- two users ingest slightly different PDFs of the same underlying article
- sentence/paragraph boundaries drift

Goal: spans should be *work-universal* where possible, while still being fast to
render for a specific extraction attempt.

## Current Stack: What Exists Already

### A) Spine locators table + endpoints

- Postgres table `locators` exists (project-scoped), storing opaque JSON:
  - `backend/db/migrate.py`
  - helpers: `backend/spine/locators.py`
  - endpoints:
    - `POST /spine/locators`
    - `GET /spine/locators/{locator_id}`
    - `GET /spine/document-versions/{document_version_id}/locators`

Important: locators are currently keyed to `document_version_id`.

### B) Quote selectors utility

- `backend/text_selectors.py` supports:
  - `build_anchor_quote(exact/prefix/suffix)`
  - `resolve_quote_selector()` (best-effort)

This is a good foundation for work-universal anchoring.

### C) Selector JSON already appears in some span-graph plumbing

- `backend/span_graph_store.py` stores `selector_json` for spans.

## The Problem

Attempt-local offsets are not universal:

- (page_index, char_start, char_end) will drift across:
  - OCR vs text-layer
  - different PDF builds
  - different parsers (GROBID vs fallback)
  - normalization differences

Therefore every span must carry at least one *content-addressed* selector.

## Proposed Standard: Multi-Selector Locator Bundle

Every locator payload should include a bundle of selectors.

### Selector Types

1) **Quote selector** (work-universal primary)

- Fields:
  - `exact`: short exact quote
  - `prefix`: short context before
  - `suffix`: short context after (optional)

2) **Position selector** (attempt-local fast path)

- Fields:
  - `attempt_id`: the attempt this offset was measured against
  - `container`: where offsets apply (e.g. `primary.body`, `fallback.page[3]`)
  - `start`, `end`: character offsets in the container's normalized text

3) **Token fingerprint selector** (re-anchoring helper)

- Fields:
  - `hash`: sha256 of normalized token window
  - `window_tokens`: N used
  - `normalize_version`: identifier for normalization rules

4) **Optional geometry selector** (when available)

- Fields:
  - `page_index`
  - `bbox`: [x0,y0,x1,y1] in PDF coordinate space
  - `dpi` if produced from raster/OCR

### Locator Payload Shape (Recommended)

Locator type examples:

- `type: text_span`
- `type: cite_span`
- `type: reference_entry`

Example payload_json for `text_span`:

```json
{
  "schema_version": 1,
  "work_id": "<work_id>",
  "selectors": {
    "quote": {"exact": "...", "prefix": "...", "suffix": null},
    "position": {"attempt_id": "<attempt_id>", "container": "fallback.body", "start": 1234, "end": 1402},
    "token": {"hash": "<sha256>", "window_tokens": 48, "normalize_version": "v1"},
    "geometry": null
  }
}
```

Notes:

- `work_id` should be carried even if the locator row is document-version
  scoped today.
- `position` is optional but recommended whenever we can compute it.

## Resolution Rules (How We Map A Locator To A New Attempt)

We resolve a locator against a specific attempt (which implies a specific PDF
version and extraction output).

Algorithm (deterministic, best-effort):

1) If `position.attempt_id == target_attempt_id`, use offsets directly.
2) Otherwise, try `quote.exact` search in the target attempt's normalized text.
   - If a single match: accept.
   - If multiple matches: disambiguate using prefix/suffix and token fingerprint.
3) If no exact match: fuzzy match using token windows (bounded; record low
   confidence).
4) If still unresolved: mark unresolved; UI should degrade (show text-only
   context, no highlight).

Persist the resolved mapping:

- Cache per `(locator_id, attempt_id)`:
  - resolved offsets
  - confidence
  - resolved_at

## Work-Universal vs Document-Version Scoped

Today `locators.document_version_id` forces locators to be tied to a specific
PDF version.

To make locators truly work-universal, we need one of:

Option A (recommended): introduce **work-scoped locators**

- Allow `document_version_id` to be nullable OR add `work_id` column.
- Document-version specific selectors (geometry/position) remain optional.

Option B: keep locators doc-version scoped, but treat `quote` as the portable
portion and re-create locators per version.

Option A is cleaner for multi-user cloud collaboration.

## Artifact Requirements (So Locators Can Resolve)

Every attempt must produce a normalized text container addressable by `container`
in position selectors. Minimum:

- `primary` path: TEI-derived normalized body text (or a deterministic join of
  paragraphs)
- `fallback` path: `fallback.body.txt` normalized exactly once

Normalization rules must be stable and versioned.

## Failure Modes + Expected Behavior

- Different PDF builds: quote selector should still resolve most of the time;
  confidence may drop.
- OCR introduces errors: quote resolution may fail; token-fingerprint fuzzy may
  partially rescue.
- Duplicate text: use prefix/suffix + token fingerprint to disambiguate.
- Empty extraction: locator remains unresolved; do not block the pipeline.

## Roadmap Hook

- Near-term: start writing quote selectors for all stored spans (citations,
  evidence selections) and store them in locator payloads.
- Medium-term: add work-scoped locators + `(locator_id, attempt_id)` resolution
  cache.
