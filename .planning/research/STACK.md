# Stack Research

**Domain:** Citation integrity and claim validation workflows for academic PDFs
**Researched:** 2026-01-23
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| GROBID | 0.8.2 | Scholarly PDF parsing, citation extraction, TEI XML output | De facto standard for structured reference and metadata extraction from academic PDFs; supports citation parsing and layout-aware output. Confidence: HIGH (official release). |
| PostgreSQL | 17.2 | System of record for papers, claims, user judgments, and citation graph edges | Reliable relational store with strong JSON and recursive query support for citation trees; keeps workflows and provenance auditable. Confidence: HIGH (official release notes). |
| pgvector | 0.8.1 | Vector storage and similarity search inside PostgreSQL | Keeps embeddings co-located with metadata for joins and provenance; simpler ops for MVP while enabling hybrid retrieval. Confidence: HIGH (official tags). |
| PyTorch | 2.10.0 | Model runtime for claim extraction, NLI scoring, and reranking | Primary research ML runtime with broad model support and strong GPU acceleration. Confidence: HIGH (official release). |
| Transformers | 4.57.6 | Model library for NLI, claim extraction, and classification | Most comprehensive transformer model ecosystem; integrates with PyTorch for fine-tuning and inference. Confidence: HIGH (official release). |
| PDF.js | 5.4.530 | In-browser PDF rendering for citation/claim review | Most widely used open-source PDF viewer; supports annotation overlays and text-layer alignment. Confidence: HIGH (official release). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| spaCy | 3.8.11 | Sentence segmentation and linguistic preprocessing | Fast segmentation/tokenization to seed claim extraction and chunking before model inference. Confidence: HIGH (official release). |
| scispaCy | 0.6.2 | Scientific NER and abbreviation handling | When domain-specific entities (genes, chemicals, methods) improve claim grounding. Confidence: HIGH (official release). |
| sentence-transformers | 5.2.0 | Dense embeddings and cross-encoder reranking | For semantic retrieval and reranking pipelines feeding evidence validation. Confidence: HIGH (official release). |
| Qdrant | 1.16.3 | Dedicated vector database | Use when embeddings exceed Postgres performance limits or need hybrid + filtered search at scale. Confidence: HIGH (official release). |
| OpenAlex API | Current (no explicit version) | Open citation graph + metadata enrichment | Use to expand citation trees and normalize metadata when DOI coverage is incomplete. Confidence: MEDIUM (official docs, no versioning). |
| Crossref REST API | Current (no explicit version) | DOI metadata and reference resolution | Use for DOI normalization, publisher metadata, and citation matching. Confidence: MEDIUM (official docs, no versioning). |
| Semantic Scholar Graph API | Current (no explicit version) | Citation graph and paper metadata | Use for additional citation context and paper metadata enrichment. Confidence: MEDIUM (official docs, no versioning). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Docker + Docker Compose | Run GROBID, vector DB, and model services | Prefer official images; pin container versions to match the stack above. |
| pytest | Automated pipeline and model regression tests | Keep golden outputs for citation extraction and claim validation workflows. |

## Installation

```bash
# Core
pip install torch==2.10.0 transformers==4.57.6

# Supporting
pip install spacy==3.8.11 scispacy==0.6.2 sentence-transformers==5.2.0

# Vector store (Postgres extension)
# pgvector is installed at the database level; use the v0.8.1 release tag for builds.
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| PostgreSQL + pgvector | Qdrant | If embedding volume is large enough to justify a dedicated vector DB or you need built-in hybrid search at scale. |
| GROBID | Rule-based PDF text extraction (pdftotext/pdfminer.six) | Only for non-scholarly PDFs where citation structure is irrelevant and fast text extraction is sufficient. |
| PDF.js | PSPDFKit / Adobe PDF Embed | When you need commercial-grade annotation UX or enterprise compliance. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Raw text extraction only (pdftotext/pdfminer.six) for citations | Loses structured references and citation markers required for integrity workflows | GROBID 0.8.2 for TEI/XML with citation structure |
| Storing embeddings without relational metadata | Hard to trace provenance and join claims to evidence | PostgreSQL 17.2 + pgvector 0.8.1 (or Qdrant + Postgres) |
| Heuristic-only claim detection | Misses nuanced claims and introduces brittle rules | Transformer-based models with spaCy pre-segmentation |

## Stack Patterns by Variant

**If MVP or single-node deployment:**
- Use PostgreSQL 17.2 + pgvector 0.8.1
- Because it simplifies ops and keeps all metadata + embeddings in one system

**If high-scale retrieval (10M+ chunks or heavy multi-tenant usage):**
- Use Qdrant 1.16.3 for vectors plus PostgreSQL 17.2 for metadata
- Because dedicated vector DBs provide better scaling and hybrid search features

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| scispaCy 0.6.2 | spaCy 3.7.x-3.8.x | scispaCy release notes state compatibility with spaCy 3.7 and 3.8 (model packages may pin spaCy). |

## Sources

- https://github.com/kermitt2/grobid/releases — GROBID 0.8.2 release
- https://github.com/explosion/spaCy/releases — spaCy 3.8.11 release
- https://github.com/allenai/scispacy/releases — scispaCy 0.6.2 release
- https://github.com/huggingface/transformers/releases — Transformers 4.57.6 release
- https://github.com/huggingface/sentence-transformers/releases — Sentence-Transformers 5.2.0 release
- https://github.com/pytorch/pytorch/releases — PyTorch 2.10.0 release
- https://github.com/qdrant/qdrant/releases — Qdrant 1.16.3 release
- https://github.com/pgvector/pgvector/tags — pgvector 0.8.1 tag
- https://www.postgresql.org/docs/release/17.2/ — PostgreSQL 17.2 release notes
- https://github.com/mozilla/pdf.js/releases — PDF.js 5.4.530 release
- https://docs.openalex.org/ — OpenAlex API documentation
- https://www.crossref.org/documentation/retrieve-metadata/rest-api/ — Crossref REST API docs
- https://api.semanticscholar.org/api-docs/graph — Semantic Scholar Graph API docs

---
*Stack research for: Citation integrity and claim validation workflows for academic PDFs*
*Researched: 2026-01-23*
