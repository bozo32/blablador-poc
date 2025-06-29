# backend/pipeline_registry.py

from backend import retriever, nli
from backend.settings import Settings


def get_pipeline(cfg: Settings):
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
