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
