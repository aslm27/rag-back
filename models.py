from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


Confidence = Literal["High", "Medium", "Low", "Insufficient Evidence"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
DocumentStatus = Literal["uploaded", "queued", "ingesting", "ready", "failed"]


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
    recommendation: str = Field(default="")
    supporting_evidence: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = "Insufficient Evidence"
    disclaimer: str = Field(
        default="This system supports — never replaces — clinical judgment. Outputs are guideline-grounded, not diagnostic."
    )
    refusal_reason: Optional[str] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    method: Literal["semantic", "bm25", "hybrid"] = "hybrid"


class ConversationMessageRequest(ChatRequest):
    project_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_name: str
    publisher: str
    source_url: str
    page_number: int
    section_title: str
    text: str
    similarity_score: float


class ChatResponse(BaseModel):
    answer: GroundedAnswer
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    query: str
    method: str
    k: int
    refusal: bool = False
    pipeline: list[str] = Field(default_factory=list)
    request_id: str = ""


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Project(BaseModel):
    id: str
    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DocumentInfo(BaseModel):
    name: str
    publisher: str = "UNKNOWN"
    source_url: str = ""
    chunk_count: int = 0
    page_count: int = 0


class ProjectDocument(BaseModel):
    id: str
    project_id: str
    filename: str
    status: DocumentStatus
    size_bytes: int = 0
    content_type: str = "application/pdf"
    created_at: datetime
    updated_at: datetime
    ingested_at: Optional[datetime] = None
    job_id: Optional[str] = None
    chunk_count: int = 0
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document: ProjectDocument


class IngestResponse(BaseModel):
    document_id: str
    job_id: str
    status: JobStatus


class Job(BaseModel):
    id: str
    type: str
    status: JobStatus
    project_id: Optional[str] = None
    document_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    method: Literal["semantic", "bm25", "hybrid"] = "hybrid"


class RetrieveResponse(BaseModel):
    query: str
    method: str
    k: int
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    safety_threshold: float
    max_similarity: float
    safe_to_generate: bool


class EvaluationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    answer: Optional[GroundedAnswer] = None
    expected_chunk_ids: list[str] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    query: str
    citation_precision: float
    citation_recall: Optional[float] = None
    grounded: bool
    notes: list[str] = Field(default_factory=list)


class MessageResponse(ChatResponse):
    conversation_id: str
    message_id: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    created_at: str
    updated_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


class HealthResponse(BaseModel):
    status: str = "ok"
    topic: str
    total_chunks: int
    total_documents: int
    embedding_backend: str
    index_loaded: bool


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    engine_loaded: bool
    index_loaded: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: Optional[str] = None
