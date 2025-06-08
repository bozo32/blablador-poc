# backend/settings.py

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # — FastAPI backend URL for the frontend to call
    BACKEND_URL: str = Field(
        "http://localhost:8000",
        env="BACKEND_URL",
        description="URL of the FastAPI citation-support service",
    )

    # — Blablador credentials
    API_KEY: str = Field(..., env="API_KEY", description="Your Blablador API key")
    API_BASE: str = Field(
        ..., env="API_BASE", description="Base URL of the Blablador API"
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
