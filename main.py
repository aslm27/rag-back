"""FastAPI application — Clinical Evidence RAG API."""
import json
import shutil
from contextlib import asynccontextmanager

import psycopg2
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from config import (
    CORS_ORIGINS, TOPIC, RAW_PDFS_DIR, VECTOR_STORE_DIR,
    JWT_SECRET_KEY,
)
from db import close_db_pool, init_db_pool
from auth import (
    create_access_token,
    create_user,
    get_current_user,
    get_user_by_email,
    get_user_with_password,
    hash_password,
    verify_password,
)
from models import (
    AuthResponse,
    ChatRequest, ChatResponse, EvidenceChunk,
    DocumentInfo, HealthResponse,
    LoginRequest,
    RegisterRequest,
    UserProfile,
)
from rag_engine import engine


# ── Lifespan ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes on startup."""
    init_db_pool()
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    # Ensure data dirs exist
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    engine.load()
    try:
        yield
    finally:
        close_db_pool()


# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Evidence RAG API",
    description="AI-powered clinical evidence retrieval and synthesis for metabolic health guidelines.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """Register a new user and return access token."""
    email = str(req.email).strip().lower()
    full_name = req.full_name.strip() if req.full_name else None

    if get_user_by_email(email) is not None:
        raise HTTPException(status_code=409, detail="Email is already registered.")

    hashed = hash_password(req.password)

    try:
        user = create_user(email=email, hashed_password=hashed, full_name=full_name)
    except psycopg2.IntegrityError:
        raise HTTPException(status_code=409, detail="Email is already registered.")

    token = create_access_token(user_id=user.id, email=user.email)
    return AuthResponse(access_token=token, user=user)


@app.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Authenticate user and return access token."""
    email = str(req.email).strip().lower()
    row = get_user_with_password(email)
    if row is None or not verify_password(req.password, str(row["hashed_password"])):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = UserProfile(
        id=str(row["id"]),
        email=str(row["email"]),
        full_name=row.get("full_name"),
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )
    token = create_access_token(user_id=user.id, email=user.email)
    return AuthResponse(access_token=token, user=user)


@app.get("/auth/me", response_model=UserProfile)
async def me(current_user: UserProfile = Depends(get_current_user)):
    """Get current authenticated user."""
    return current_user


@app.get("/api/health", response_model=HealthResponse)
async def health(current_user: UserProfile = Depends(get_current_user)):
    """Health check with system stats."""
    return HealthResponse(
        status="ok",
        topic=TOPIC,
        total_chunks=len(engine.metadata_df) if engine.metadata_df is not None else 0,
        total_documents=engine.metadata_df["document_name"].nunique() if engine.metadata_df is not None and len(engine.metadata_df) > 0 else 0,
        embedding_backend=engine.embedder.backend if engine.embedder else "not loaded",
        index_loaded=engine.vector_index is not None,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: UserProfile = Depends(get_current_user)):
    """
    Main RAG endpoint: retrieves evidence and generates a grounded answer.
    """
    print(f"[chat] user_id={current_user.id} method={req.method} k={req.k}")

    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Engine not loaded yet.")

    if engine.metadata_df is None or len(engine.metadata_df) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Upload PDFs first via /api/upload.",
        )

    try:
        answer, evidence_rows = engine.generate(
            query=req.query,
            k=req.k,
            method=req.method,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    evidence = []
    for _, r in evidence_rows.iterrows():
        evidence.append(EvidenceChunk(
            chunk_id=str(r["chunk_id"]),
            document_name=str(r["document_name"]),
            publisher=str(r.get("publisher", "UNKNOWN")),
            source_url=str(r.get("source_url", "")),
            page_number=int(r["page_number"]),
            section_title=str(r["section_title"]),
            text=str(r["text"]),
            similarity_score=float(r.get("similarity_score", 0.0)),
        ))

    return ChatResponse(
        answer=answer,
        evidence=evidence,
        query=req.query,
        method=req.method,
        k=req.k,
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
async def list_documents(current_user: UserProfile = Depends(get_current_user)):
    """List all ingested documents with metadata."""
    if engine.metadata_df is None or len(engine.metadata_df) == 0:
        return []

    docs = []
    for doc_name, group in engine.metadata_df.groupby("document_name"):
        docs.append(DocumentInfo(
            name=str(doc_name),
            publisher=str(group["publisher"].iloc[0]) if "publisher" in group.columns else "UNKNOWN",
            source_url=str(group["source_url"].iloc[0]) if "source_url" in group.columns else "",
            chunk_count=len(group),
            page_count=int(group["page_number"].nunique()),
        ))
    return docs


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: UserProfile = Depends(get_current_user),
):
    """Upload a new PDF and re-ingest."""
    print(f"[upload] user_id={current_user.id} file={file.filename}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save file
    save_path = RAW_PDFS_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Re-ingest
    from pdf_pipeline import ingest_pdfs
    import pandas as pd
    import faiss
    from rag_engine import build_bm25

    # Load existing sources config
    config_path = VECTOR_STORE_DIR / "config.json"
    sources = {}
    if config_path.exists():
        with open(config_path) as cf:
            cfg = json.load(cf)
            sources = cfg.get("sources", {})

    records, _ = ingest_pdfs(RAW_PDFS_DIR, sources)
    if not records:
        raise HTTPException(status_code=400, detail="No chunks produced from PDF.")

    meta_df = pd.DataFrame(records)
    embs = engine.embedder.encode(meta_df["text"].tolist())
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)

    # Persist
    faiss.write_index(idx, str(VECTOR_STORE_DIR / "index.faiss"))
    meta_df.to_json(VECTOR_STORE_DIR / "metadata.json", orient="records", indent=2)

    # Update engine state
    engine.metadata_df = meta_df
    engine.vector_index = idx
    engine.bm25_index = build_bm25(meta_df)

    return {
        "status": "ok",
        "file": file.filename,
        "uploaded_by": current_user.id,
        "total_chunks": len(meta_df),
        "total_documents": meta_df["document_name"].nunique(),
    }


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
