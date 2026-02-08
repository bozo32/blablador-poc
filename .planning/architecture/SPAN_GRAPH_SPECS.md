# Specifications: Span-First Graph (Implementable MVP)

This spec translates `.planning/architecture/SPAN_GRAPH_MODEL.md` into an implementable set of storage structures, API surfaces, and UI behaviors.

It is written to be:
- adequate with respect to the architectural conditions
- implementable incrementally without breaking the existing workflow

## 0. Scope and Sequencing

We do this in three increments:

1) Persist `Work` + `Span` + `ClaimSpan` in the graph store (no UI changes required).
2) Record reviewer `Assertion`s pointing to `Span` as evidence (UI can stay claim-centric at first).
3) Switch Graph/Evidence UI to span-first exploration; keep a compatibility adapter for existing claim IDs during transition.

## 1. Storage (SQLite)

### 1.1 Tables

#### `works`
- `work_id TEXT PRIMARY KEY`
- `doi TEXT NULL`
- `openalex_id TEXT NULL`
- `title TEXT NULL`
- `authors_json TEXT NULL`
- `year TEXT NULL`
- `abstract TEXT NULL`
- `abstract_source TEXT NULL` (e.g., `openalex`)
- `abstract_embedding_json TEXT NULL` (optional; start with null)
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

Indexes:
- `works(doi)`
- `works(openalex_id)`

#### `work_cites`
Work-level citation edges (bibliographic network), typically from external metadata.

- `citing_work_id TEXT NOT NULL`
- `cited_work_id TEXT NOT NULL`
- `source TEXT NOT NULL` (e.g., `openalex`)
- `created_at TEXT NOT NULL`

PK:
- `(citing_work_id, cited_work_id, source)`

Indexes:
- `work_cites(citing_work_id)`
- `work_cites(cited_work_id)`

#### `ingests`
- `ingest_id TEXT PRIMARY KEY` (uuid)
- `sha256 TEXT NOT NULL`
- `filename TEXT NOT NULL`
- `aliases_json TEXT NOT NULL` (default `[]`)
- `work_id TEXT NULL` (FK to `works`)
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

Indexes:
- `ingests(sha256)`

#### `spans`
- `span_id TEXT PRIMARY KEY` (deterministic)
- `work_id TEXT NULL`
- `ingest_id TEXT NULL`
- `kind TEXT NOT NULL` (`citation_window|evidence_excerpt|other`)
- `selector_json TEXT NOT NULL` (quote selector)
- `window_fingerprint TEXT NULL`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

Constraints:
- exactly one of `work_id`/`ingest_id` must be non-null (enforced in code)

Indexes:
- `spans(work_id)`
- `spans(ingest_id)`

#### `span_cites`
Represents `Span -> Work` citation edges.
- `span_id TEXT NOT NULL`
- `cited_work_id TEXT NOT NULL`
- `reference_id TEXT NULL` (local bib id like b4)
- `citation_index INTEGER NULL`
- `created_at TEXT NOT NULL`

PK:
- `(span_id, cited_work_id)`

Indexes:
- `span_cites(cited_work_id)`

#### `span_cite_roles`
Reviewer-attributed role annotations for citations.

- `span_id TEXT NOT NULL`
- `cited_work_id TEXT NOT NULL`
- `reviewer_uid TEXT NOT NULL`
- `role TEXT NOT NULL` (`evidentiary|background|reputational|unknown`)
- `updated_at TEXT NOT NULL`

PK:
- `(span_id, cited_work_id, reviewer_uid)`

Indexes:
- `span_cite_roles(reviewer_uid)`

#### `claim_spans`
- `claim_span_id TEXT PRIMARY KEY`
- `span_id TEXT NOT NULL`
- `order_index INTEGER NOT NULL`
- `selector_json TEXT NULL` (optional; can be derived later)
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`

Index:
- `claim_spans(span_id, order_index)`

#### `claim_atoms`
- `claim_atom_id TEXT PRIMARY KEY` (uuid)
- `text TEXT NOT NULL`
- `created_by TEXT NOT NULL` (reviewer uid)
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`
- `supersedes_id TEXT NULL` (optional)

Indexes:
- `claim_atoms(created_by)`

#### `claim_span_atoms`
Many-to-many between `ClaimSpan` and `ClaimAtom`.
- `claim_span_id TEXT NOT NULL`
- `claim_atom_id TEXT NOT NULL`
- `created_at TEXT NOT NULL`

PK:
- `(claim_span_id, claim_atom_id)`

#### `assertions`
- `assertion_id TEXT PRIMARY KEY` (uuid)
- `reviewer_uid TEXT NOT NULL`
- `verdict TEXT NOT NULL` (`support|contradict|neutral|uncertain`)
- `confidence REAL NULL`
- `comment TEXT NULL`
- `claim_atom_id TEXT NULL`
- `claim_span_id TEXT NULL` (MVP can use this until atoms are present)
- `evidence_span_id TEXT NULL`
- `evidence_work_id TEXT NULL`
- `created_at TEXT NOT NULL`

Constraints (enforced in code):
- exactly one of `claim_atom_id`/`claim_span_id` is set
- at least one of `evidence_span_id`/`evidence_work_id` is set

Indexes:
- `assertions(reviewer_uid)`
- `assertions(claim_span_id)`
- `assertions(claim_atom_id)`
- `assertions(evidence_span_id)`

#### `neighborhood_runs`
Represents a neighborhood search (C13) for potential contradiction/cherry-picking detection.

- `run_id TEXT PRIMARY KEY` (uuid)
- `created_by TEXT NULL` (reviewer uid or `system`)
- `context_work_id TEXT NULL`
- `context_span_id TEXT NULL`
- `context_claim_span_id TEXT NULL`
- `context_claim_atom_id TEXT NULL`
- `method TEXT NOT NULL` (`bib_coupling|co_citation|references`)
- `params_json TEXT NOT NULL`
- `created_at TEXT NOT NULL`

Indexes:
- `neighborhood_runs(context_work_id)`
- `neighborhood_runs(context_span_id)`

#### `neighborhood_candidates`
- `run_id TEXT NOT NULL`
- `candidate_work_id TEXT NOT NULL`
- `bib_intersection INTEGER NULL`
- `abstract_score REAL NULL`
- `rank INTEGER NULL`
- `detail_json TEXT NULL`

PK:
- `(run_id, candidate_work_id)`

Indexes:
- `neighborhood_candidates(candidate_work_id)`

### 1.2 Deterministic ID Functions

- `span_id = sha256(work_id_or_ingest_id + kind + selector.exact + selector.prefix + window_fingerprint)`
- `claim_span_id = sha256(span_id + order_index + selector.exact + selector.prefix)`

Selector JSON is normalized (whitespace-collapse) before hashing.

## 2. API

### 2.1 Create / Update structure

1) `POST /spans/upsert`
- Input: `{ work_id|ingest_id, kind, selector, window_fingerprint }`
- Output: `{ span_id }`

2) `POST /spans/{span_id}/cites`
- Input: `{ cited_work_id, reference_id?, citation_index? }` (allow batch)

2b) `PUT /spans/{span_id}/cites/{cited_work_id}/role`
- Input: `{ reviewer_uid, role }`

3) `POST /spans/{span_id}/claim-spans`
- Input: list of `{ order_index, selector?, text? }`
- Output: created/updated `claim_span_id`s

4) `POST /claim-spans/{claim_span_id}/atoms`
- Input: `{ text, reviewer_uid }` creates a `claim_atom_id` and links

### 2.2 Assertions

5) `POST /assertions`
- Input: `{ reviewer_uid, verdict, confidence?, comment?, claim_span_id|claim_atom_id, evidence_span_id?|evidence_work_id? }`

6) `GET /claim-spans/{claim_span_id}/assertions?reviewer_uid=`

### 2.3 Exploration

7) `GET /works/{work_id}/spans?kind=citation_window`

8) `GET /spans/{span_id}` includes:
- cited works
- claim spans
- computed statuses per reviewer (optional; see Status lattice)

9) `GET /graph/work-subgraph?work_id=&hops=&...`
Work-level derived edges with expand-to-span support.

10) `GET /graph/span-subgraph?span_id=&...`
Span->claim span->evidence expansion.

### 2.4 Neighborhood search (C13)

11) `POST /neighborhood/search`
- Input:
  - `context`: one of `work_id|span_id|claim_span_id|claim_atom_id`
  - `method`: `bib_coupling|co_citation|references`
  - `bib`: parameters (e.g., max candidates, min intersection)
  - `abstract_filter`: `{ query_text, min_score }` (optional)
- Output: `{ run_id, candidates: [...] }`

v2 note (not implemented in v1):
- Allow the user to curate the abstract filter query by editing/selecting "statements" derived from the candidate work's title/abstract/keywords.

12) `GET /neighborhood/{run_id}`
- Output: run metadata + candidates

## 3. UI Behaviors

### 3.1 Exploration affordances

- Work graph shows derived `Work->Work` edges. Clicking expands to contributing `Span` anchors.
- Span inspector shows:
  - the anchored citation window
  - the list of cited works
  - claim spans and their per-reviewer status (`unknown|supported|contradicted|contested|not_supported`)
- Claim span inspector shows:
  - assertions grouped by reviewer
  - evidence spans/works referenced

### 3.2 Evidence placement

Attachments should become `(citing_work_id, cited_work_id)` or `(span_id, cited_work_id)` scoped rather than per-claim.
During transition:
- keep current per-claim attachment pipeline
- add a span-scoped attachment layer and have per-claim views reference it

## 4. Migration / Compatibility

### 4.1 Adapter: `cite:...` claim IDs

Current claim IDs (`cite:{doc}:{idx}:{reviewer?}:{seg}`) map to:
- `span_id` via `citation_anchor` (quote selector + fingerprint)
- `claim_span_id` via `order_index` and/or segment selector

Maintain:
- evidence rerun endpoints accepting legacy claim IDs

Add:
- resolver that can translate legacy claim IDs to `(span_id, claim_span_id)`.

### 4.2 Minimal viable cutover

- Persist spans/claim spans whenever segmentation is saved.
- Write assertions alongside existing judgment store writes.
- Once stable, switch Graph tab to operate on `span_id`/`work_id` rather than on `claim:{doc}:{sentence}:{idx}`.

## 5. Validation Checklist

- Multi-cited span renders as one span with N cited works; user can mark which are evidentiary vs neutral.
- Two-claim span supports AND adequacy; each claim can be satisfied by different cited works.
- Contradiction yields `contested`; contested propagates to span adequacy.
- `unknown` means not yet assessed.
- `not_supported` means assessed but no support evidence exists.
- Reviewer disagreement yields `contested` in consensus views.
- Secondary citation: assertion can point at a span in Work B even if Work C is missing.
- Two reviewers can disagree without overwrites; compare view remains meaningful.
- Neighborhood search: for a given span/claim, the system returns nearby works via bibliographic intersection and ranks them with an abstract filter; reviewers can attach contradictory assertions to candidates.
