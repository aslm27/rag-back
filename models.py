"""Pydantic models for the API request/response schemas."""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class Citation(BaseModel):
    """A single source citation from the retrieved evidence."""
    document_name: str
    section_title: str
    page_number: int
    chunk_id: str
    source_url: str
    quote: str = Field(description="Short exact quote from the chunk that supports the claim")


class GroundedAnswer(BaseModel):
    """Structured answer grounded in retrieved clinical evidence."""
    recommendation: str = Field(
        default="",
        description="Direct, concise clinical recommendation based ONLY on the evidence",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Bullet-point excerpts grounded in the retrieved text",
    )
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["High", "Medium", "Low", "Insufficient Evidence"] = "Insufficient Evidence"
    disclaimer: str = Field(
        default="This system supports — never replaces — clinical judgment. Outputs are guideline-grounded, not diagnostic."
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Filled only when confidence is Insufficient Evidence or the query is out of scope",
    )


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    query: str = Field(..., min_length=1, max_length=2000, description="Clinical question")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    method: Literal["semantic", "bm25", "hybrid"] = Field(
        default="hybrid", description="Retrieval method"
    )


class EvidenceChunk(BaseModel):
    """A retrieved evidence chunk returned alongside the answer."""
    chunk_id: str
    document_name: str
    publisher: str
    source_url: str
    page_number: int
    section_title: str
    text: str
    similarity_score: float


class ChatResponse(BaseModel):
    """Full response to a chat query."""
    answer: GroundedAnswer
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    query: str
    method: str
    k: int


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""
    name: str
    publisher: str
    source_url: str
    chunk_count: int
    page_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    topic: str
    total_chunks: int
    total_documents: int
    embedding_backend: str
    index_loaded: bool
