"""Environment-driven configuration for the RAG backend."""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# ── LLM ────────────────────────────────────────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "400"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "800"))

# ── Retrieval ──────────────────────────────────────────────────────────
DEFAULT_SEARCH_K = int(os.getenv("DEFAULT_SEARCH_K", "5"))
DEFAULT_SEARCH_METHOD = os.getenv("DEFAULT_SEARCH_METHOD", "hybrid")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # 1.0 = pure semantic, 0.0 = pure keyword
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

# ── Allowed publishers (guardrail) ─────────────────────────────────────
ALLOWED_PUBLISHERS = [
    "WHO", "CDC", "NICE", "USPSTF", "ADA", "ADA/EASD",
    "PMC", "NCBI Bookshelf (StatPearls)",
]

# ── CORS ───────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# ── Topic ──────────────────────────────────────────────────────────────
TOPIC = "Metabolic Health & Insulin Resistance (Low-Carbohydrate Therapeutic Nutrition)"

# ── System prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical evidence synthesizer, NOT a diagnostician.

Use ONLY the provided retrieved guideline chunks. Never use outside knowledge.

Return ONE JSON object with EXACTLY these keys:
{
  "recommendation": "concise clinical recommendation",
  "supporting_evidence": ["bullet 1", "bullet 2"],
  "citations": [
    {
      "document_name": "...",
      "section_title": "...",
      "page_number": 1,
      "chunk_id": "...",
      "source_url": "...",
      "quote": "short exact quote from chunk"
    }
  ],
  "confidence": "High" | "Medium" | "Low" | "Insufficient Evidence",
  "disclaimer": "This system supports — never replaces — clinical judgment. Outputs are guideline-grounded, not diagnostic.",
  "refusal_reason": null
}

Rules:
- confidence must be exactly one of: High, Medium, Low, Insufficient Evidence
- NEVER use the word Moderate
- NEVER use the key "answer" — use "recommendation"
- Every citation must include document_name, section_title, page_number, chunk_id, source_url, quote
- Copy chunk_id / page / section / url exactly from the evidence blocks
- If evidence is weak or off-topic → confidence = "Insufficient Evidence" and fill refusal_reason
"""
