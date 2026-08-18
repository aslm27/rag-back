import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import (
    CORS_ORIGINS,
    DEFAULT_SEARCH_K,
    DEFAULT_SEARCH_METHOD,
    RAW_PDFS_DIR,
    SIMILARITY_THRESHOLD,
    TOPIC,
    VECTOR_STORE_DIR,
)
from db import new_id, store, utcnow
from models import (
    ChatRequest,
    ChatResponse,
    ConversationMessageRequest,
    Citation,
    DocumentInfo,
    DocumentUploadResponse,
    EvaluationRequest,
    EvaluationResponse,
    EvidenceChunk,
    GroundedAnswer,
    HealthResponse,
    IngestResponse,
    Job,
    MessageResponse,
    Project,
    ProjectCreate,
    ProjectDocument,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from rag_engine import build_bm25, engine, validate_grounded_answer


API_KEY = os.getenv("API_KEY", "")
bearer = HTTPBearer(auto_error=False)


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Optional API-key guard: local development is open, production can require a key."""
    if not API_KEY:
        return
    token = credentials.credentials if credentials else x_api_key
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Authentication required.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    engine.load()
    yield


app = FastAPI(
    title="Clinical Evidence RAG API",
    description="Project-scoped evidence retrieval and grounded answer generation.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _as_project(row: dict[str, Any]) -> Project:
    return Project.model_validate(row)


def _as_document(row: dict[str, Any]) -> ProjectDocument:
    return ProjectDocument.model_validate(row)


def _as_job(row: dict[str, Any]) -> Job:
    return Job.model_validate(row)


def _project_or_404(project_id: str) -> dict[str, Any]:
    project = store.get("projects", project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _document_or_404(document_id: str) -> dict[str, Any]:
    document = store.get("documents", document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


def _evidence_from_rows(rows: pd.DataFrame) -> list[EvidenceChunk]:
    evidence: list[EvidenceChunk] = []
    for _, row in rows.iterrows():
        evidence.append(EvidenceChunk(
            chunk_id=str(row["chunk_id"]),
            document_name=str(row.get("document_name", "")),
            publisher=str(row.get("publisher", "UNKNOWN")),
            source_url=str(row.get("source_url", "")),
            page_number=int(row.get("page_number", 0)),
            section_title=str(row.get("section_title", "General")),
            text=str(row.get("text", "")),
            similarity_score=float(row.get("similarity_score", 0.0)),
        ))
    return evidence


def _rows_for_ids(ids: list[str], scores: list[float], project_id: str | None = None, k: int = 5) -> pd.DataFrame:
    if engine.metadata_df is None or not len(engine.metadata_df):
        return pd.DataFrame()
    lookup = engine.metadata_df.set_index("chunk_id")
    selected: list[dict[str, Any]] = []
    for chunk_id, score in zip(ids, scores):
        if chunk_id not in lookup.index:
            continue
        row = lookup.loc[chunk_id]
        if project_id and "project_id" in row.index and str(row.get("project_id")) != str(project_id):
            continue
        item = row.to_dict()
        item["chunk_id"] = str(chunk_id)
        item["similarity_score"] = float(score)
        selected.append(item)
        if len(selected) >= k:
            break
    return pd.DataFrame(selected)


def _retrieve(query: str, k: int, method: str, project_id: str | None = None) -> pd.DataFrame:
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Engine not loaded yet.")
    if engine.metadata_df is None or not len(engine.metadata_df) or engine.vector_index is None:
        raise HTTPException(status_code=400, detail="No documents have been ingested.")
    search_k = len(engine.metadata_df)
    try:
        ids, scores = engine.search(query=query, k=search_k, method=method)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc
    return _rows_for_ids(ids, scores, project_id=project_id, k=k)


def _retrieve_response(query: str, k: int, method: str, project_id: str | None = None) -> tuple[pd.DataFrame, float, bool]:
    rows = _retrieve(query, k, method, project_id)
    max_similarity = float(rows["similarity_score"].max()) if len(rows) else 0.0
    return rows, max_similarity, max_similarity >= SIMILARITY_THRESHOLD


def _log_retrieval(request_id: str, query: str, rows: pd.DataFrame, method: str, k: int, max_similarity: float, safe: bool, project_id: str | None = None, conversation_id: str | None = None) -> None:
    run = store.insert("retrieval_runs", {
        "request_id": request_id,
        "project_id": project_id,
        "conversation_id": conversation_id,
        "query": query,
        "method": method,
        "requested_k": k,
        "safety_threshold": SIMILARITY_THRESHOLD,
        "max_similarity": max_similarity,
        "safe_to_generate": safe,
    })
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        store.insert("retrieval_chunks", {
            "retrieval_run_id": run["id"],
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row.get("document_id")) if row.get("document_id") else None,
            "rank": rank,
            "similarity_score": float(row.get("similarity_score", 0.0)),
            "excerpt": str(row.get("text", ""))[:500],
            "metadata": {"document_name": str(row.get("document_name", "")), "page_number": int(row.get("page_number", 0))},
        })


def _run_pipeline(query: str, k: int, method: str, *, request_id: str, project_id: str | None = None, conversation_id: str | None = None) -> tuple[GroundedAnswer, pd.DataFrame, list[str]]:
    started = time.perf_counter()
    stages = ["request_received", "auth_validated", "input_validated"]
    store.log_pipeline(request_id=request_id, stage="request_received", project_id=project_id, conversation_id=conversation_id)
    store.log_pipeline(request_id=request_id, stage="auth_validated", project_id=project_id, conversation_id=conversation_id)
    store.log_pipeline(request_id=request_id, stage="input_validated", project_id=project_id, conversation_id=conversation_id)

    rows, max_similarity, safe = _retrieve_response(query, k, method, project_id)
    stages.append("context_retrieved")
    store.log_pipeline(request_id=request_id, stage="context_retrieved", project_id=project_id, conversation_id=conversation_id, details={"count": len(rows)})
    _log_retrieval(request_id, query, rows, method, k, max_similarity, safe, project_id, conversation_id)
    stages.append("safety_threshold_checked")
    store.log_pipeline(request_id=request_id, stage="safety_threshold_checked", status="ok" if safe else "blocked", project_id=project_id, conversation_id=conversation_id, details={"threshold": SIMILARITY_THRESHOLD, "max_similarity": max_similarity})

    if not safe:
        answer = GroundedAnswer(refusal_reason="Retrieved evidence similarity is below the safety threshold.")
        stages.append("refusal_returned")
        store.log_pipeline(request_id=request_id, stage="refusal_returned", status="blocked", project_id=project_id, conversation_id=conversation_id, latency_ms=int((time.perf_counter() - started) * 1000))
        return answer, rows, stages

    stages.append("generation_started")
    store.log_pipeline(request_id=request_id, stage="generation_started", project_id=project_id, conversation_id=conversation_id)
    try:
        answer, _ = engine.generate(query=query, k=k, method=method, rows_override=rows)
    except Exception as exc:
        store.log_pipeline(request_id=request_id, stage="error", status="error", project_id=project_id, conversation_id=conversation_id, details={"error": str(exc)})
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    answer = validate_grounded_answer(answer, rows)
    stages.extend(["answer_validated", "citations_validated"])
    store.log_pipeline(request_id=request_id, stage="answer_validated", project_id=project_id, conversation_id=conversation_id)
    store.log_pipeline(request_id=request_id, stage="citations_validated", project_id=project_id, conversation_id=conversation_id, details={"citation_count": len(answer.citations)})
    final_stage = "refusal_returned" if answer.refusal_reason or answer.confidence == "Insufficient Evidence" else "answer_returned"
    stages.append(final_stage)
    store.log_pipeline(request_id=request_id, stage=final_stage, status="blocked" if final_stage == "refusal_returned" else "ok", project_id=project_id, conversation_id=conversation_id, latency_ms=int((time.perf_counter() - started) * 1000))
    return answer, rows, stages


def _chat_response(answer: GroundedAnswer, rows: pd.DataFrame, query: str, method: str, k: int, request_id: str, stages: list[str]) -> ChatResponse:
    return ChatResponse(answer=answer, evidence=_evidence_from_rows(rows), query=query, method=method, k=k, refusal=bool(answer.refusal_reason or answer.confidence == "Insufficient Evidence"), pipeline=stages, request_id=request_id)


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    metadata = engine.metadata_df
    return HealthResponse(
        status="ok",
        topic=TOPIC,
        total_chunks=len(metadata) if metadata is not None else 0,
        total_documents=int(metadata["document_name"].nunique()) if metadata is not None and len(metadata) else 0,
        embedding_backend=engine.embedder.backend if engine.embedder else "not loaded",
        index_loaded=engine.vector_index is not None,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    checks = {"engine_loaded": engine.loaded, "index_loaded": engine.vector_index is not None, "persistence_available": True}
    is_ready = engine.loaded and (engine.vector_index is not None or not (engine.metadata_df is not None and len(engine.metadata_df)))
    return ReadyResponse(status="ready" if is_ready else "not_ready", engine_loaded=engine.loaded, index_loaded=engine.vector_index is not None, checks=checks, detail=None if is_ready else "Engine or vector index is not loaded.")


@app.post("/api/v1/projects", response_model=Project, dependencies=[Depends(require_auth)])
async def create_project(req: ProjectCreate) -> Project:
    now = utcnow()
    return _as_project(store.insert("projects", {"id": new_id(), "name": req.name, "description": req.description, "metadata": req.metadata, "owner_id": None, "created_at": now, "updated_at": now}))


@app.get("/api/v1/projects", response_model=list[Project], dependencies=[Depends(require_auth)])
async def list_projects() -> list[Project]:
    return [_as_project(row) for row in store.list("projects")]


@app.post("/api/v1/projects/{project_id}/documents", response_model=DocumentUploadResponse, dependencies=[Depends(require_auth)])
async def upload_project_document(project_id: str, file: UploadFile = File(...)) -> DocumentUploadResponse:
    _project_or_404(project_id)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    document_id = new_id()
    project_dir = RAW_PDFS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    path = project_dir / f"{document_id}_{safe_name}"
    with path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    size = path.stat().st_size
    now = utcnow()
    row = store.insert("documents", {"id": document_id, "project_id": project_id, "filename": safe_name, "storage_path": str(path), "content_type": file.content_type or "application/pdf", "size_bytes": size, "status": "uploaded", "created_at": now, "updated_at": now})
    return DocumentUploadResponse(document=_as_document(row))


@app.get("/api/v1/projects/{project_id}/documents", response_model=list[ProjectDocument], dependencies=[Depends(require_auth)])
async def list_project_documents(project_id: str) -> list[ProjectDocument]:
    _project_or_404(project_id)
    return [_as_document(row) for row in store.list("documents", {"project_id": project_id})]


@app.post("/api/v1/documents/{document_id}/ingest", response_model=IngestResponse, dependencies=[Depends(require_auth)])
async def ingest_document(document_id: str) -> IngestResponse:
    document = _document_or_404(document_id)
    job_id = new_id()
    now = utcnow()
    job = store.insert("ingestion_jobs", {"id": job_id, "type": "document_ingest", "project_id": document["project_id"], "document_id": document_id, "status": "queued", "created_at": now, "updated_at": now})
    store.update("documents", document_id, {"status": "ingesting", "job_id": job_id, "error": None})
    try:
        from pdf_pipeline import ingest_pdfs

        project_dir = RAW_PDFS_DIR / str(document["project_id"])
        records, _ = ingest_pdfs(project_dir, {})
        target_prefix = f"{document_id}_"
        records = [record for record in records if str(record.get("document_name", "")).startswith(target_prefix) or record.get("document_name") == document["filename"]]
        if not records:
            raise ValueError("No chunks produced from PDF.")
        for record in records:
            record["document_id"] = document_id
            record["project_id"] = document["project_id"]
            record["document_name"] = document["filename"]
        existing = engine.metadata_df if engine.metadata_df is not None and len(engine.metadata_df) else pd.DataFrame()
        if len(existing) and "document_id" in existing.columns:
            existing = existing[existing["document_id"].astype(str) != str(document_id)]
        combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if len(existing) else pd.DataFrame(records)
        embs = engine.embedder.encode(combined["text"].tolist())
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(VECTOR_STORE_DIR / "index.faiss"))
        combined.to_json(VECTOR_STORE_DIR / "metadata.json", orient="records", indent=2)
        engine.metadata_df = combined
        engine.vector_index = index
        engine.bm25_index = build_bm25(combined)
        for record in records:
            store.insert("document_chunks", {"document_id": document_id, "project_id": document["project_id"], "chunk_id": record["chunk_id"], "page_number": int(record.get("page_number", 0)), "section_title": record.get("section_title", "General"), "text_content": record.get("text", ""), "token_count": record.get("token_count"), "metadata": {"publisher": record.get("publisher", "UNKNOWN"), "source_url": record.get("source_url", "")}})
        result = {"chunk_count": len(records), "total_chunks": len(combined)}
        store.update("documents", document_id, {"status": "ready", "chunk_count": len(records), "ingested_at": utcnow(), "error": None})
        store.update("ingestion_jobs", job_id, {"status": "succeeded", "result": result, "finished_at": utcnow(), "error": None})
        return IngestResponse(document_id=document_id, job_id=job_id, status="succeeded")
    except Exception as exc:
        store.update("documents", document_id, {"status": "failed", "error": str(exc)})
        store.update("ingestion_jobs", job_id, {"status": "failed", "finished_at": utcnow(), "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.get("/api/v1/documents/{document_id}/status", response_model=ProjectDocument, dependencies=[Depends(require_auth)])
async def document_status(document_id: str) -> ProjectDocument:
    return _as_document(_document_or_404(document_id))


@app.get("/api/v1/jobs/{job_id}", response_model=Job, dependencies=[Depends(require_auth)])
async def job_status(job_id: str) -> Job:
    job = store.get("ingestion_jobs", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _as_job(job)


@app.post("/api/v1/projects/{project_id}/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_auth)])
async def retrieve_debug(project_id: str, req: RetrieveRequest) -> RetrieveResponse:
    _project_or_404(project_id)
    request_id = str(uuid.uuid4())
    rows, max_similarity, safe = _retrieve_response(req.query, req.k, req.method, project_id)
    _log_retrieval(request_id, req.query, rows, req.method, req.k, max_similarity, safe, project_id)
    return RetrieveResponse(query=req.query, method=req.method, k=req.k, evidence=_evidence_from_rows(rows), safety_threshold=SIMILARITY_THRESHOLD, max_similarity=max_similarity, safe_to_generate=safe)


@app.post("/api/v1/conversations/{conversation_id}/messages", response_model=MessageResponse, dependencies=[Depends(require_auth)])
async def conversation_message(conversation_id: str, request: ConversationMessageRequest) -> MessageResponse:
    request_id = str(uuid.uuid4())
    existing = store.get("conversations", conversation_id)
    actual_conversation_id = conversation_id if existing else new_id()
    if not existing:
        store.insert("conversations", {"id": actual_conversation_id, "project_id": request.project_id, "user_id": None, "title": request.query[:80], "metadata": request.metadata})
    elif request.project_id and not existing.get("project_id"):
        store.update("conversations", actual_conversation_id, {"project_id": request.project_id})
    project_id = request.project_id or (existing or {}).get("project_id")
    if project_id:
        _project_or_404(project_id)
    store.insert("messages", {"conversation_id": actual_conversation_id, "role": "user", "content": request.query, "request_id": request_id})
    answer, rows, stages = _run_pipeline(request.query, request.k, request.method, request_id=request_id, project_id=project_id, conversation_id=actual_conversation_id)
    response = _chat_response(answer, rows, request.query, request.method, request.k, request_id, stages)
    message_id = new_id()
    store.insert("messages", {"id": message_id, "conversation_id": actual_conversation_id, "role": "assistant", "content": answer.recommendation, "answer": answer.model_dump(mode="json"), "refusal": response.refusal, "request_id": request_id})
    return MessageResponse(**response.model_dump(), conversation_id=actual_conversation_id, message_id=message_id)


@app.post("/api/v1/projects/{project_id}/evaluations", response_model=EvaluationResponse, dependencies=[Depends(require_auth)])
async def evaluate(project_id: str, req: EvaluationRequest) -> EvaluationResponse:
    _project_or_404(project_id)
    answer = req.answer or GroundedAnswer()
    cited = {citation.chunk_id for citation in answer.citations if citation.chunk_id}
    expected = set(req.expected_chunk_ids)
    valid = cited & expected if expected else cited
    precision = len(valid) / len(cited) if cited else 0.0
    recall = len(valid) / len(expected) if expected else None
    grounded = bool(answer.recommendation and cited and not answer.refusal_reason and answer.confidence != "Insufficient Evidence")
    notes = []
    if not cited:
        notes.append("The answer contains no citations.")
    if expected and not expected.issubset(cited):
        notes.append("One or more expected chunks were not cited.")
    result = EvaluationResponse(query=req.query, citation_precision=precision, citation_recall=recall, grounded=grounded, notes=notes)
    store.insert("evaluations", {"project_id": project_id, "query": req.query, "answer": answer.model_dump(mode="json"), "expected_chunk_ids": list(expected), "citation_precision": precision, "citation_recall": recall, "grounded": grounded, "notes": notes})
    return result


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
async def legacy_chat(req: ChatRequest) -> ChatResponse:
    request_id = str(uuid.uuid4())
    answer, rows, stages = _run_pipeline(req.query, req.k, req.method, request_id=request_id)
    return _chat_response(answer, rows, req.query, req.method, req.k, request_id, stages)


@app.get("/api/documents", response_model=list[DocumentInfo], dependencies=[Depends(require_auth)])
async def legacy_documents() -> list[DocumentInfo]:
    if engine.metadata_df is None or not len(engine.metadata_df):
        return []
    docs = []
    for name, group in engine.metadata_df.groupby("document_name"):
        docs.append(DocumentInfo(name=str(name), publisher=str(group["publisher"].iloc[0]) if "publisher" in group.columns else "UNKNOWN", source_url=str(group["source_url"].iloc[0]) if "source_url" in group.columns else "", chunk_count=len(group), page_count=int(group["page_number"].nunique())))
    return docs


@app.post("/api/upload", dependencies=[Depends(require_auth)])
async def legacy_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    project = store.list("projects")
    project_id = project[0]["id"] if project else create_project(ProjectCreate(name="Default project")).id
    upload = await upload_project_document(project_id, file)
    ingest = await ingest_document(upload.document.id)
    return {"status": ingest.status, "file": upload.document.filename, "total_chunks": upload.document.chunk_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
