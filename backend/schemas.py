from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Evidence(BaseModel):
    quote: str
    location: str
    assessment: str

class SegmentResult(BaseModel):
    segment_id: str
    claim: str
    quoted_evidence: List[Evidence] = Field(default_factory=list)

class Segment(BaseModel):
    segment_id: str
    claim: str

class Settings(BaseModel):
    embed_model: Optional[str] = Field(None, description="Embedding model alias or HF path")
    max_sentences: Optional[int] = Field(None, description="Maximum number of sentences to retrieve via FAISS")
    faiss_min_score: Optional[float] = Field(None, description="Minimum FAISS similarity score threshold")
    nli_model: Optional[str] = Field(None, description="Local NLI model name")
    llm_model: Optional[str] = Field(None, description="LLM model alias for remote calls")
    nli_threshold: Optional[float] = Field(None, description="Minimum NLI confidence threshold")
    data_dir: Optional[str] = Field(None, description="Path to the folder containing CSV and TEI files")
    api_key: Optional[str] = Field(None, description="API key for Blablador service")
    base_url: Optional[str] = Field(None, description="Base URL for the Blablador API endpoint")

class SentencePayload(BaseModel):
    row_id: int
    citing_title: str
    citing_id: str
    original_sentence: str
    segments: List[Segment]
    settings: Optional[Settings] = Field(default=None, description="Optional runtime settings for embedding, FAISS, NLI, and LLM models")

class SentenceResult(BaseModel):
    text: str
    segments: List[SegmentResult]

class CitingPaperResult(BaseModel):
    title: str
    id: str
    sentences: List[SentenceResult]

class CitedPaperResult(BaseModel):
    title: str
    id: str
    doi: str
    citing_papers: List[CitingPaperResult]

class SegmentRequest(BaseModel):
    folder: str
    row: int
    citing_title: str
    citing_id: str
    original_sentence: str
    segments: List[Segment]
    settings: Settings