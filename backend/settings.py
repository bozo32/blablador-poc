# backend/settings.py

from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict
from typing import Any, Literal, List, Tuple
from pathlib import Path

NLI_BATCH_SIZE = 50
TROLL_PAY_MARGIN = 0.7


class AppSettings(PydanticBaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    # selection for which mode the script will run, classic (nli) or hybrid
    PIPELINE_MODE: Literal["classic", "hybrid"] = Field("classic")

    # max number of entailing / contradicting candidates to show
    NLI_CANDIDATES_SHOWN: int = Field(3)

    # for NLI parallel processing
    NLI_BATCH_SIZE: int = Field(50)

    # margin used to determine what is ambiguous
    TROLL_PAY_MARGIN: float = Field(0.7)

    # — FastAPI backend URL for the frontend to call
    BACKEND_URL: str = Field(
        "http://localhost:8000",
        description="URL of the FastAPI citation-support service",
    )

    # — Blablador credentials
    API_KEY: str = Field("...", description="Your Blablador API key")
    API_BASE: str = Field(
        "https://api.helmholtz-blablador.fz-juelich.de",
        description="Base URL of the Blablador API",
    )

    DEFAULT_LLM_MODEL: str = Field(
        "alias-huge",
        description="Default Blablador model alias used by the UI",
    )

    # — GROBID extraction defaults
    GROBID_URL: str = Field(
        "http://localhost:8070",
        description="Base URL for the GROBID service",
    )
    GROBID_TIMEOUT: int = Field(
        120,
        description="Timeout (seconds) for GROBID extraction requests",
    )

    GROBID_MAX_CONCURRENT: int = Field(
        1,
        description="Max concurrent GROBID requests (client-side throttle)",
    )

    GROBID_RETRY_503: int = Field(
        3,
        description="Retry count for transient GROBID 503 errors",
    )

    GROBID_CONSOLIDATE_CITATIONS: bool = Field(
        False,
        description="Ask GROBID to consolidate citations (calls external services)",
    )
    GROBID_CONSOLIDATE_HEADER: bool = Field(
        False,
        description="Ask GROBID to consolidate header (calls external services)",
    )

    INGEST_PIPELINE_WORKERS: int = Field(
        2,
        ge=1,
        description="Background workers for ingestion (extract+resolve)",
    )
    INGEST_PIPELINE_QUEUE_MAX: int = Field(
        256,
        ge=1,
        description="Max queued ingestion jobs before dropping",
    )

    CROSSREF_MAILTO: str = Field(
        "",
        description="Contact email for Crossref REST API requests",
    )
    CROSSREF_API_URL: str = Field(
        "https://api.crossref.org/works",
        description="Base URL for Crossref REST API",
    )

    OPENALEX_API_URL: str = Field(
        "https://api.openalex.org/works",
        description="Base URL for OpenAlex Works API",
    )
    OPENALEX_API_KEY: str = Field(
        "",
        description="OpenAlex API key for citation graph expansion",
    )

    # — Embedding/Retrieval defaults
    EMBED_MODEL: str = Field(
        "intfloat/multilingual-e5-base",
        description="HF repo path for the embedding model",
    )
    MAX_SENTENCES: int = Field(
        1000,
        description="How many FAISS candidates to pull before thresholding",
    )
    FAISS_MIN_SCORE: float = Field(0.2, description="Minimum FAISS similarity score")

    # — Reranker defaults
    RERANKER_MODEL: str = Field("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANKER_TOP_K: int = Field(10)

    # — NLI defaults
    NLI_MODEL: str = Field("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    NLI_THRESHOLD: float = Field(0.5)

    # — Optional Hugging Face Inference API (Phase 08-08)
    # When enabled and a token is present, supported steps may use remote HF
    # inference instead of local Transformers pipelines.
    HF_REMOTE_INFERENCE: bool = Field(
        False,
        description="Use Hugging Face Inference API for supported demo steps",
    )
    HF_API_TOKEN: str = Field(
        "",
        description="Hugging Face access token (optional, kept local)",
    )

    # — Hybrid pipeline configuration —

    HYBRID_WINDOW_SIZE: int = Field(
        3,
        description="Number of sentences per sliding window",
    )
    HYBRID_STRIDE: int = Field(1, description="Stride for windowing sentences")
    HYBRID_MAX_PARAGRAPH_TOKENS: int = Field(
        256,
        description="Max tokens per paragraph/window",
    )
    HYBRID_PARAGRAPH_WINDOW_OVERLAP: float = Field(
        0.5,
        description="Window overlap fraction for long paragraphs",
    )
    HYBRID_FILTER_LOGIC: str = Field(
        "none",
        description="Which filters to apply: 'sbert', 'bm25', 'none' or 'both'",
    )
    HYBRID_SBERT_THRESHOLD: float = Field(0.40, description="SBERT cosine threshold")
    HYBRID_BM25_THRESHOLD: float = Field(0.00, description="BM25 threshold")

    HYBRID_ENABLE_COREF: bool = Field(True, description="Enable coreference patching")

    HYBRID_COREF_MODEL: str = Field(
        "biu-nlp/f-coref",  # Default to the well-maintained HF f-coref model
        validation_alias="HYBRID_COREF_MODEL",
        description="HuggingFace repo or local path for f-coref model",
    )

    HYBRID_COREF_DEVICE: str = Field(
        "cpu",  # or 'cuda' if you want GPU (and have one)
        validation_alias="HYBRID_COREF_DEVICE",
        description="Device for f-coref model inference (cpu/cuda)",
    )

    HYBRID_RERANK_MODEL: str = Field(
        "ColBERT",
        description="Which reranker: 'ColBERT' or 'SPLADE'",
    )
    HYBRID_RERANK_TOP_K: int = Field(10, description="Top-k windows after reranking")
    HYBRID_MASK_DISCOURSE: bool = Field(True, description="Mask discourse markers")
    HYBRID_DISCOURSE_MARKERS: List[str] = Field(
        default_factory=lambda: [
            "however",
            "therefore",
            "furthermore",
            "thus",
            "consequently",
            "meanwhile",
        ],
        description="List of discourse markers to mask",
    )

    HYBRID_NLI_TOP_K: int = Field(10, description="Number of windows to pass to NLI")
    HYBRID_NLI_MODEL: str = Field(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    )
    HYBRID_NLI_THRESHOLD: float = Field(0.5, description="NLI entailment threshold")

    HYBRID_VIS_COLOR_PRO: str = Field("green", description="Color for entail tokens")
    HYBRID_VIS_COLOR_CONTRA: str = Field("red", description="Color for contra tokens")
    HYBRID_VIS_COLOR_DEFAULT: str = Field(
        "black",
        description="Default color for other tokens",
    )
    HYBRID_AUDIT_MODE: bool = Field(True, description="Enable audit/debug output")

    # ---  Retrieval cut-off settings  ------------------------------------
    RETRIEVAL_K: int = Field(1500)  # max hits per claim

    RETRIEVAL_TAU: float = Field(0.90)
    # keep ≥ τ·best
    RETRIEVAL_MAX: int = Field(60)  # safety cap

    # --- ColBERT -----------------------------------------------------------------
    COLBERT_MODE: str = Field("external")  # "internal" | "external" | "off"
    COLBERT_API_URL: str = Field("http://localhost:7001")
    COLBERT_ROOT: Path = Field(
        Path(__file__).parent.parent / "data" / "colbert",
        description="Where ColBERT indexes are stored",
    )
    COLBERT_INDEX_PATH: Path = Field(
        Path(__file__).parent.parent
        / "data"
        / "experiments"
        / "default"
        / "indexes"
        / "default",
        description="Where ColBERT index files are actually stored",
    )
    INGESTION_DIR: Path = Field(
        Path(__file__).parent.parent / "data" / "ingestion",
        description="Root directory for local ingestion storage",
    )
    ATTACHMENT_DIR: Path = Field(
        Path(__file__).parent.parent / "data" / "attachments",
        description="Root directory for claim attachment storage",
    )
    CLAIM_DB_PATH: Path = Field(
        Path(__file__).parent.parent / "data" / "claims.db",
        description="SQLite path for storing confirmed claim parses",
    )
    GRAPH_DB_PATH: Path = Field(
        Path(__file__).parent.parent / "data" / "graph.db",
        description="SQLite path for the document/claim knowledge graph",
    )
    COLBERT_DIM: int = Field(128)
    COLBERT_MAXLEN: int = Field(180)
    COLBERT_TOP_K: int = Field(20)
    SHOW_SALIENCE: bool = Field(
        True,
        description="Show ColBERT token-salience colouring in the UI",
    )
    DEFAULT_REVIEWER_UID: str = Field(
        "",
        description="Default reviewer UID/email for claim confirmations",
    )

    # --- Evidence pipeline parameters --------------------------------------
    EVIDENCE_MAX_WINDOWS: int = Field(
        500,
        description="Maximum deterministic windows to load per claim",
    )
    EVIDENCE_WINDOW_SIZE: int = Field(
        3,
        description="Number of sentences per deterministic window",
    )
    EVIDENCE_WINDOW_STRIDE: int = Field(
        1,
        description="Stride to use when rolling attachment sentences",
    )
    EVIDENCE_MAX_CANDIDATES: int = Field(
        25,
        description="Maximum evidence candidates returned to the UI",
    )
    EVIDENCE_BM25_MIN_SCORE: float = Field(
        0.0,
        description="Minimum BM25 score required for deterministic seeds",
    )
    EVIDENCE_SEED_LIMIT: int = Field(
        50,
        description="Upper bound on deterministic BM25 seeds",
    )

    EVIDENCE_STORE_DIR: Path = Field(
        Path(__file__).parent.parent / "data" / "evidence_runs",
        description="Root directory for persisted evidence runs",
    )
    EVIDENCE_HISTORY_DEPTH: int = Field(
        5,
        description="Maximum historical runs to retain per claim",
        ge=1,
    )
    EVIDENCE_RERUN_TIMEOUT_SECONDS: int = Field(
        300,
        description="Maximum seconds a rerun job may execute before marked stale",
    )
    EVIDENCE_RERUN_WORKERS: int = Field(
        1,
        ge=1,
        description="Maximum concurrent evidence rerun jobs",
    )


# ---------------------------------------------------------------------------
#  Execution profiles (Phase 08-07)
# ---------------------------------------------------------------------------

# Keep the registry intentionally small. Profiles are referenced by name from
# frontend rerun requests via `advanced_settings.profile`.
EXECUTION_PROFILES: dict[str, dict[str, Any]] = {
    # Mirrors current defaults: classic pipeline with cross-encoder reranking.
    "Fast/Local": {
        "PIPELINE_MODE": "classic",
        "RERANKER_MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "RERANKER_TOP_K": 10,
        # Explicitly disable ColBERT so the profile routing is deterministic.
        "COLBERT_MODE": "off",
    },
    # Re-enable the ColBERT reranker path via hybrid pipeline configuration.
    "Best/Local": {
        "PIPELINE_MODE": "hybrid",
        "HYBRID_RERANK_MODEL": "ColBERT",
        "HYBRID_RERANK_TOP_K": 10,
        "COLBERT_MODE": "external",
    },
}

DEFAULT_EXECUTION_PROFILE = "Fast/Local"


def resolve_execution_profile(profile: str | None) -> Tuple[str | None, dict[str, Any]]:
    """Return (resolved_name, overrides) for a requested profile."""
    if not profile:
        return None, {}
    requested = str(profile).strip()
    if not requested:
        return None, {}
    if requested in EXECUTION_PROFILES:
        return requested, dict(EXECUTION_PROFILES[requested])
    lowered = requested.casefold()
    for name in EXECUTION_PROFILES:
        if name.casefold() == lowered:
            return name, dict(EXECUTION_PROFILES[name])
    return None, {}


def apply_execution_profile(
    base_settings: "AppSettings", profile: str | None
) -> Tuple["AppSettings", str | None, dict[str, Any]]:
    """Return (effective_settings, resolved_profile, overrides).

    This is intended for per-run overrides (e.g., evidence reruns) and MUST NOT
    mutate the global settings object.
    """
    resolved, overrides = resolve_execution_profile(profile)
    if not overrides:
        return base_settings, resolved, {}
    try:
        return base_settings.model_copy(update=overrides), resolved, dict(overrides)
    except Exception:
        payload = dict(base_settings.model_dump())
        payload.update(overrides)
        return AppSettings(**payload), resolved, dict(overrides)


# ---------------------------------------------------------------------------
#  Canonical, application‑wide settings object
# ---------------------------------------------------------------------------
settings = AppSettings()  # type: ignore[call-arg]
