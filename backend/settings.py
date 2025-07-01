# backend/settings.py

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal, List

NLI_BATCH_SIZE = 50
TROLL_PAY_MARGIN = 0.7


class Settings(BaseSettings):
    ## selection for which mode the script will run, classic (nli) or hybrid
    PIPELINE_MODE: Literal["classic", "hybrid"] = Field("classic", env="PIPELINE_MODE")

    ## max number of entailing / contradicting candidates to show
    NLI_CANDIDATES_SHOWN: int = Field(3, env="NLI_CANDIDATES_SHOWN")

    ## for NLI parallel processing
    NLI_BATCH_SIZE: int = Field(50, env="NLI_BATCH_SIZE")

    ## margin used to determine what is ambiguous
    TROLL_PAY_MARGIN: float = Field(0.7, env="TROLL_PAY_MARGIN")

    # — FastAPI backend URL for the frontend to call
    BACKEND_URL: str = Field(
        "http://localhost:8000",
        env="BACKEND_URL",
        description="URL of the FastAPI citation-support service",
    )

    # — Blablador credentials
    API_KEY: str = Field("...", env="API_KEY", description="Your Blablador API key")
    API_BASE: str = Field(
        "https://api.helmholtz-blablador.fz-juelich.de",
        env="API_BASE",
        description="Base URL of the Blablador API",
    )

    # — Embedding/Retrieval defaults
    EMBED_MODEL: str = Field(
        "intfloat/multilingual-e5-base",
        env="EMBED_MODEL",
        description="HF repo path for the embedding model",
    )
    MAX_SENTENCES: int = Field(
        1000,
        env="MAX_SENTENCES",
        description="How many FAISS candidates to pull before thresholding",
    )
    FAISS_MIN_SCORE: float = Field(
        0.2, env="FAISS_MIN_SCORE", description="Minimum FAISS similarity score"
    )

    # — Reranker defaults
    RERANKER_MODEL: str = Field(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", env="RERANKER_MODEL"
    )
    RERANKER_TOP_K: int = Field(10, env="RERANKER_TOP_K")

    # — NLI defaults
    NLI_MODEL: str = Field(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", env="NLI_MODEL"
    )
    NLI_THRESHOLD: float = Field(0.5, env="NLI_THRESHOLD")

    # — Hybrid pipeline configuration —

    HYBRID_WINDOW_SIZE: int = Field(
        3,
        env="HYBRID_WINDOW_SIZE",
        description="Number of sentences per sliding window",
    )
    HYBRID_STRIDE: int = Field(
        1, env="HYBRID_STRIDE", description="Stride for windowing sentences"
    )
    HYBRID_MAX_PARAGRAPH_TOKENS: int = Field(
        256,
        env="HYBRID_MAX_PARAGRAPH_TOKENS",
        description="Max tokens per paragraph/window",
    )
    HYBRID_PARAGRAPH_WINDOW_OVERLAP: float = Field(
        0.5,
        env="HYBRID_PARAGRAPH_WINDOW_OVERLAP",
        description="Window overlap fraction for long paragraphs",
    )

    HYBRID_USE_SBERT: bool = Field(
        True, env="HYBRID_USE_SBERT", description="Enable SBERT filtering"
    )
    HYBRID_USE_BM25: bool = Field(
        True, env="HYBRID_USE_BM25", description="Enable BM25 filtering"
    )
    HYBRID_FILTER_LOGIC: str = Field(
        "both",
        env="HYBRID_FILTER_LOGIC",
        description="Which filters to apply: 'sbert', 'bm25', or 'both'",
    )
    HYBRID_SBERT_THRESHOLD: float = Field(
        0.10, env="HYBRID_SBERT_THRESHOLD", description="SBERT cosine threshold"
    )
    HYBRID_BM25_THRESHOLD: float = Field(
        0.20, env="HYBRID_BM25_THRESHOLD", description="BM25 threshold"
    )

    HYBRID_ENABLE_COREF: bool = Field(
        True, env="HYBRID_ENABLE_COREF", description="Enable coreference patching"
    )
    
    HYBRID_COREF_MODEL: str = Field(
    "biu-nlp/f-coref",  # Default to the well-maintained HF f-coref model
        env="HYBRID_COREF_MODEL",
        description="HuggingFace repo or local path for f-coref model (default: biu-nlp/f-coref)"
    )

    HYBRID_COREF_DEVICE: str = Field(
        "cpu",  # or 'cuda' if you want GPU (and have one)
        env="HYBRID_COREF_DEVICE",
        description="Device for f-coref model inference (cpu/cuda)"
    )

    HYBRID_RERANK_MODEL: str = Field(
        "ColBERT",
        env="HYBRID_RERANK_MODEL",
        description="Which reranker: 'ColBERT' or 'SPLADE'",
    )
    HYBRID_RERANK_TOP_K: int = Field(
        20, env="HYBRID_RERANK_TOP_K", description="Top-k windows after reranking"
    )
    HYBRID_MASK_DISCOURSE: bool = Field(
        True, env="HYBRID_MASK_DISCOURSE", description="Mask discourse markers"
    )
    HYBRID_DISCOURSE_MARKERS: List[str] = Field(
        default_factory=lambda: [
            "however",
            "therefore",
            "furthermore",
            "thus",
            "consequently",
            "meanwhile",
        ],
        env="HYBRID_DISCOURSE_MARKERS",
        description="List of discourse markers to mask",
    )

    HYBRID_NLI_TOP_K: int = Field(
        10, env="HYBRID_NLI_TOP_K", description="Number of windows to pass to NLI"
    )
    HYBRID_NLI_MODEL: str = Field(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        env="HYBRID_NLI_MODEL",
    )
    HYBRID_NLI_THRESHOLD: float = Field(
        0.5, env="HYBRID_NLI_THRESHOLD", description="NLI entailment threshold"
    )

    HYBRID_VIS_COLOR_PRO: str = Field(
        "green", env="HYBRID_VIS_COLOR_PRO", description="Color for entail tokens"
    )
    HYBRID_VIS_COLOR_CONTRA: str = Field(
        "red", env="HYBRID_VIS_COLOR_CONTRA", description="Color for contra tokens"
    )
    HYBRID_VIS_COLOR_DEFAULT: str = Field(
        "black",
        env="HYBRID_VIS_COLOR_DEFAULT",
        description="Default color for other tokens",
    )
    HYBRID_AUDIT_MODE: bool = Field(
        False, env="HYBRID_AUDIT_MODE", description="Enable audit/debug output"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
