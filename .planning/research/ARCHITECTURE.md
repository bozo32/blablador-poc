# Architecture Research

**Domain:** Citation integrity workflows for academic PDF validation
**Researched:** 2026-01-23
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                User Layer                                  │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────────────┐   │
│  │ PDF Upload & │   │ Citation Tree UI │   │ Claim Review & Judgment UI │   │
│  │ Doc Viewer   │   │ (graph + PDF)    │   │ (evidence panel)            │   │
│  └──────┬───────┘   └───────┬──────────┘   └───────────┬───────────────┘   │
│         │                   │                          │                   │
├─────────┴───────────────────┴──────────────────────────┴──────────────────┤
│                           Orchestration API Layer                           │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐ │
│  │ Workflow Orchestr. │  │ Auth + Project State│  │ Evidence/Claim API    │ │
│  └─────────┬──────────┘  └───────────┬─────────┘  └──────────┬───────────┘ │
│            │                         │                        │            │
├────────────┴─────────────────────────┴────────────────────────┴───────────┤
│                         Processing & Retrieval Layer                        │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │ GROBID PDF │  │ Claim Parser  │  │ Retrieval +   │  │ NLI/Validation  │ │
│  │ Extraction │  │ + Segmentation│  │ Reranker      │  │ Scoring         │ │
│  └────┬───────┘  └───────┬───────┘  └───────┬───────┘  └─────────┬───────┘ │
│       │                  │                  │                    │         │
├───────┴──────────────────┴──────────────────┴────────────────────┴────────┤
│                              Data & Index Layer                             │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Object Store │  │ Relational   │  │ Citation Graph │  │ Vector/ColBERT│ │
│  │ (PDF + XML)  │  │ DB (state)   │  │ Store          │  │ Index         │ │
│  └──────────────┘  └──────────────┘  └────────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| PDF ingestion + storage | Accept PDFs, store original assets, track versions | Object storage + checksum metadata |
| GROBID extraction | Convert PDF to structured XML/TEI, extract citations and contexts | GROBID web service + batch client |
| Reference resolution | Normalize citations to identifiers (DOI, PMID) | Resolver service + Crossref lookup |
| Citation graph service | Build and serve citation tree, deduplicate nodes | Graph tables + caching layer |
| Claim parsing | Split text into claims, link to citation contexts | NLP service + rule layer |
| Retrieval + ranking | Retrieve candidate evidence (BM25/vector) and rerank | Dense index + ColBERT reranker |
| NLI validation | Score entailment/contradiction for each claim-evidence pair | NLI model service |
| Review UI | Human verification, store judgments, show provenance | Streamlit or web SPA |
| Audit/provenance | Persist model outputs, user judgments, and evidence snapshots | Relational DB + evidence snapshot store |

## Recommended Project Structure

```
services/
├── api/                     # FastAPI orchestration + project state
│   ├── routes/              # Upload, claims, evidence, judgments
│   ├── workflows/           # Pipeline orchestration + job dispatch
│   └── schemas/             # Shared request/response models
├── pdf-extraction/          # GROBID client + post-processing
│   ├── extractors/          # XML/TEI parsing, reference resolution
│   └── pipelines/           # Batch + async ingestion
├── retrieval/               # Indexing + retrieval + reranking
│   ├── indexers/             # Build/search vector/keyword indexes
│   └── rerankers/            # ColBERT interaction service
├── validation/              # NLI scoring + calibration
│   └── scorers/              # Entailment/contradiction scoring
apps/
├── ui/                      # Streamlit (or future SPA) UI
packages/
└── shared/                  # Shared types, IDs, and data contracts
infra/
└── compose/                 # Local infra orchestration
```

### Structure Rationale

- **services/** keeps pipeline responsibilities isolated so extraction, retrieval, and validation can evolve independently.
- **packages/shared/** prevents contract drift for claim IDs, citation IDs, and evidence records.
- **infra/** makes it easy to run GROBID/ColBERT locally while keeping deployment configurations separate.

## Architectural Patterns

### Pattern 1: Asynchronous pipeline with persisted checkpoints

**What:** Persist outputs between extraction, parsing, retrieval, and validation steps to allow replay and UI-first feedback.
**When to use:** Anytime PDF processing and retrieval take longer than a few seconds.
**Trade-offs:** More storage and schema versioning effort; easier debugging and reprocessing.

**Example:**
```typescript
// Pseudocode: store checkpoints as immutable versions
const extraction = await grobid.extract(pdfId)
await stores.extractedDocs.save(pdfId, extraction, { version: "grobid-0.8.2" })
```

### Pattern 2: Evidence snapshotting for reproducibility

**What:** Store the exact evidence passage and metadata used in each judgment.
**When to use:** Required for auditability and to compare model vs human decisions.
**Trade-offs:** Storage duplication; avoids downstream changes when indexes update.

**Example:**
```typescript
const snapshot = await evidenceStore.snapshot(passageId, indexVersion)
await judgments.save({ claimId, passageId, snapshot })
```

### Pattern 3: Graph-backed citation navigation

**What:** Maintain a normalized citation graph separate from document text stores.
**When to use:** Required for fast tree navigation and cross-document reuse.
**Trade-offs:** Extra normalization work; improves navigation performance and dedup.

## Data Flow

### Request Flow (PDF ingestion)

```
User Upload
    ↓
API Upload Handler → Object Store
    ↓
Workflow Orchestrator → GROBID Extraction → Reference Resolution
    ↓
Citation Graph + Structured Doc Store
```

### Request Flow (Claim validation)

```
User selects claim
    ↓
Claim Parser → Retrieval/Ranking → NLI Scoring
    ↓
Evidence Results → UI Review → Judgment Store
```

### Key Data Flows

1. **Citation tree navigation:** Citation graph serves tree nodes → fetch PDF assets → open document view.
2. **Claim-to-evidence linkage:** Claim segmenter outputs claim IDs → retrieval pipeline returns passages → NLI scores attached.
3. **Human verification loop:** User judgment updates evidence status → stored for model calibration and audit.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-1k users | Single API + batch pipeline is sufficient; async jobs recommended. |
| 1k-100k users | Separate extraction + retrieval services; cache citation graph queries. |
| 100k+ users | Dedicated indexing cluster, precomputed citation graphs, background backfill jobs. |

### Scaling Priorities

1. **First bottleneck:** PDF extraction throughput; mitigate with GROBID worker pool.
2. **Second bottleneck:** Retrieval latency; mitigate with prebuilt indexes and cached top-k.

## Anti-Patterns

### Anti-Pattern 1: Treating extraction as synchronous UI work

**What people do:** Block upload request until PDF extraction and citation parsing finish.
**Why it's wrong:** Leads to timeouts and poor UX; retries duplicate work.
**Do this instead:** Async workflow + status updates + resumable processing.

### Anti-Pattern 2: Mixing model scores and human judgments without provenance

**What people do:** Overwrite model scores with user labels in the same field.
**Why it's wrong:** Loses traceability and makes audit impossible.
**Do this instead:** Separate model output tables and human judgment records with timestamps.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GROBID | HTTP batch + web service | Provides PDF extraction and citation contexts. |
| Crossref/DOI resolver | HTTP lookup | Normalize citation metadata and identifiers. |
| ColBERT | Local service or RPC | Late-interaction reranking for retrieval. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| API ↔ Extraction | Queue + status polling | Avoid blocking user requests. |
| API ↔ Retrieval | RPC/HTTP | Keep retrieval isolated for index scaling. |
| API ↔ Validation | RPC/HTTP | Separate model scoring from orchestration logic. |

## Build Order Implications

1. **Document ingestion + extraction** must land first; downstream components depend on structured text and citations.
2. **Citation graph + reference resolution** enables tree navigation and is a dependency for claim linking.
3. **Claim parsing** depends on structured doc text and citation contexts.
4. **Retrieval + reranking** requires indexed corpora; build after claim segmentation.
5. **NLI validation + judgment storage** should follow retrieval to store evidence snapshots.

## Sources

- https://github.com/grobidOrg/grobid (PDF extraction, citation contexts, web service)
- https://grobid.readthedocs.io/en/latest/Grobid-service/ (GROBID service API)
- https://github.com/stanford-futuredata/ColBERT (late-interaction retrieval + indexing)
- https://fastapi.tiangolo.com/ (FastAPI orchestration layer)

---
*Architecture research for: citation integrity workflows*
*Researched: 2026-01-23*
