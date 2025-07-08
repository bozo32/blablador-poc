# backend/pipeline_registry.py

from backend import retriever
from backend.settings import AppSettings


def get_pipeline(cfg: AppSettings):
    """
    Return the build_all function for the desired pipeline mode.
    """
    if cfg.PIPELINE_MODE == "classic":
        return retriever.build_all
    elif cfg.PIPELINE_MODE == "hybrid":
        # We'll implement HybridPipeline in the next step.
        from backend.hybrid import HybridPipeline

        return HybridPipeline.build_all
    else:
        raise ValueError(f"Unknown PIPELINE_MODE: {cfg.PIPELINE_MODE}")
