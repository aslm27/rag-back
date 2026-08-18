"""Core RAG engine: embedding, search, hybrid retrieval, and grounded generation.

Extracted and refactored from the Colab notebook's Day 1 + Day 2 cells.
"""
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, HYBRID_ALPHA,
    SIMILARITY_THRESHOLD, SYSTEM_PROMPT,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    VECTOR_STORE_DIR,
)
from models import Citation, GroundedAnswer


# ═══════════════════════════════════════════════════════════════════════
#  Embedding Model
# ═══════════════════════════════════════════════════════════════════════
class EmbeddingModel:
    """Wraps sentence-transformers with a deterministic offline fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, fallback_dim: int = EMBEDDING_DIM):
        self.dim = fallback_dim
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"[warn] sentence-transformers unavailable ({type(e).__name__}) -> hashing fallback.")
            self.model = None
            self.backend = "hashing-fallback"

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.backend == "sentence-transformers":
            embs = np.asarray(
                self.model.encode(texts, batch_size=batch_size, show_progress_bar=False),
                dtype="float32",
            )
        else:
            embs = np.array([self._hash_embed(t) for t in texts], dtype="float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.clip(norms, 1e-10, None)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-zA-Z]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec


# ═══════════════════════════════════════════════════════════════════════
#  Search Functions
# ═══════════════════════════════════════════════════════════════════════
def semantic_search(
    query: str,
    index: faiss.Index,
    meta: pd.DataFrame,
    model: EmbeddingModel,
    k: int = 10,
) -> tuple[list[str], list[float]]:
    q_emb = model.encode([query])
    scores, idxs = index.search(q_emb, k)
    return meta.iloc[idxs[0]]["chunk_id"].tolist(), scores[0].tolist()


def build_bm25(meta: pd.DataFrame) -> BM25Okapi:
    tokenized = [re.findall(r"[a-zA-Z]+", t.lower()) for t in meta["text"]]
    return BM25Okapi(tokenized)


def keyword_search_bm25(
    query: str,
    bm25: BM25Okapi,
    meta: pd.DataFrame,
    k: int = 10,
) -> tuple[list[str], list[float]]:
    scores = bm25.get_scores(re.findall(r"[a-zA-Z]+", query.lower()))
    top_idx = np.argsort(scores)[::-1][:k]
    return meta.iloc[top_idx]["chunk_id"].tolist(), scores[top_idx].tolist()


def hybrid_search(
    query: str,
    index: faiss.Index,
    meta: pd.DataFrame,
    model: EmbeddingModel,
    bm25: BM25Okapi,
    k: int = 10,
    alpha: float = HYBRID_ALPHA,
) -> tuple[list[str], list[float]]:
    """alpha=1.0 -> pure semantic, alpha=0.0 -> pure keyword."""
    sem_ids, sem_scores = semantic_search(query, index, meta, model, k=len(meta))
    kw_ids, kw_scores = keyword_search_bm25(query, bm25, meta, k=len(meta))

    def normalize(scores):
        arr = np.array(scores, dtype="float32")
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

    sem_norm = dict(zip(sem_ids, normalize(sem_scores)))
    kw_norm = dict(zip(kw_ids, normalize(kw_scores)))

    all_ids = set(sem_ids) | set(kw_ids)
    blended = {
        cid: alpha * sem_norm.get(cid, 0) + (1 - alpha) * kw_norm.get(cid, 0)
        for cid in all_ids
    }
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:k]
    return [cid for cid, _ in ranked], [s for _, s in ranked]


# ═══════════════════════════════════════════════════════════════════════
#  Context Builder
# ═══════════════════════════════════════════════════════════════════════
def build_context(chunks_df: pd.DataFrame) -> str:
    """Format retrieved rows into a numbered evidence block for the LLM."""
    blocks = []
    for i, (_, r) in enumerate(chunks_df.iterrows()):
        blocks.append(
            f"[Evidence {i + 1}]\n"
            f"chunk_id: {r['chunk_id']}\n"
            f"document: {r['document_name']}\n"
            f"section: {r['section_title']}\n"
            f"page: {r['page_number']}\n"
            f"source_url: {r['source_url']}\n"
            f"text: {r['text'][:1200]}\n"
        )
    return "\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════════
#  Safe JSON Parser
# ═══════════════════════════════════════════════════════════════════════
def safe_parse_grounded(raw: str, evidence_rows: pd.DataFrame | None = None, metadata_df: pd.DataFrame | None = None) -> GroundedAnswer:
    """Robustly parse LLM output into a GroundedAnswer, tolerating common format issues."""
    cleaned = raw.strip()

    # Extract JSON from markdown fences if present
    if "```" in cleaned:
        parts = cleaned.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                cleaned = p
                break

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return GroundedAnswer(
            recommendation="",
            supporting_evidence=[],
            citations=[],
            confidence="Insufficient Evidence",
            refusal_reason=f"Invalid JSON from model: {e}",
        )

    # recommendation (model sometimes uses "answer")
    recommendation = str(
        data.get("recommendation") or data.get("answer") or data.get("summary") or ""
    ).strip()

    # supporting evidence
    supporting = data.get("supporting_evidence") or data.get("evidence") or []
    if not supporting and data.get("citations"):
        supporting = [
            c.get("quote", "") for c in data["citations"]
            if isinstance(c, dict) and c.get("quote")
        ]
    supporting_evidence = [str(x).strip() for x in supporting if str(x).strip()]

    # confidence normalization
    conf_raw = str(data.get("confidence", "Insufficient Evidence")).strip()
    conf_map = {
        "high": "High",
        "medium": "Medium",
        "moderate": "Medium",
        "low": "Low",
        "insufficient evidence": "Insufficient Evidence",
        "insufficient": "Insufficient Evidence",
    }
    conf = conf_map.get(conf_raw.lower(), conf_raw)
    if conf not in {"High", "Medium", "Low", "Insufficient Evidence"}:
        conf = "Medium" if recommendation else "Insufficient Evidence"

    # build lookup from retrieved rows
    lookup = {}
    if evidence_rows is not None and len(evidence_rows):
        for _, r in evidence_rows.iterrows():
            lookup[str(r["chunk_id"])] = r
    if metadata_df is not None:
        for _, r in metadata_df.iterrows():
            cid = str(r["chunk_id"])
            if cid not in lookup:
                lookup[cid] = r

    # parse citations
    citations = []
    for c in data.get("citations") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("chunk_id", "")).strip()
        meta = lookup.get(cid)

        try:
            citations.append(Citation(
                document_name=str(c.get("document_name") or (meta["document_name"] if meta is not None else "unknown")),
                section_title=str(c.get("section_title") or (meta["section_title"] if meta is not None else "General")),
                page_number=int(c.get("page_number") or (meta["page_number"] if meta is not None else 0)),
                chunk_id=cid or (str(meta["chunk_id"]) if meta is not None else ""),
                source_url=str(c.get("source_url") or (meta["source_url"] if meta is not None else "")),
                quote=str(c.get("quote", "") or "")[:300],
            ))
        except Exception:
            continue

    return GroundedAnswer(
        recommendation=recommendation,
        supporting_evidence=supporting_evidence,
        citations=citations,
        confidence=conf,
        disclaimer=str(data.get(
            "disclaimer",
            "This system supports — never replaces — clinical judgment. Outputs are guideline-grounded, not diagnostic.",
        )),
        refusal_reason=data.get("refusal_reason"),
    )


# ═══════════════════════════════════════════════════════════════════════
#  RAG Engine (stateful)
# ═══════════════════════════════════════════════════════════════════════
class RAGEngine:
    """Encapsulates the full RAG pipeline state and operations."""

    def __init__(self):
        self.embedder: EmbeddingModel | None = None
        self.vector_index: faiss.Index | None = None
        self.metadata_df: pd.DataFrame | None = None
        self.bm25_index: BM25Okapi | None = None
        self.llm_client: OpenAI | None = None
        self.loaded = False

    def load(self):
        """Load all indexes and models from the vector store."""
        store_dir = Path(VECTOR_STORE_DIR)

        # Load embedder
        print("[rag] Loading embedding model...")
        self.embedder = EmbeddingModel()
        print(f"[rag] Embedding backend: {self.embedder.backend} | dim={self.embedder.dim}")

        # Load FAISS index
        index_path = store_dir / "index.faiss"
        if index_path.exists():
            print("[rag] Loading FAISS index...")
            self.vector_index = faiss.read_index(str(index_path))
            print(f"[rag] FAISS index: {self.vector_index.ntotal} vectors")
        else:
            print("[rag] No FAISS index found -- will build on first ingestion.")
            self.vector_index = None

        # Load metadata
        meta_path = store_dir / "metadata.json"
        if meta_path.exists():
            print("[rag] Loading metadata...")
            self.metadata_df = pd.read_json(meta_path)
            print(f"[rag] {len(self.metadata_df)} chunks loaded")
        else:
            print("[rag] No metadata found.")
            self.metadata_df = pd.DataFrame()

        # Build BM25 index
        if self.metadata_df is not None and len(self.metadata_df) > 0:
            print("[rag] Building BM25 index...")
            self.bm25_index = build_bm25(self.metadata_df)
        else:
            self.bm25_index = None

        # Initialize LLM client
        if LLM_API_KEY:
            self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            print(f"[rag] LLM client ready ({LLM_MODEL} via {LLM_BASE_URL})")
        else:
            print("[warn] No LLM_API_KEY set -- chat will return retrieval-only results.")
            self.llm_client = None

        self.loaded = True
        print("[rag] Engine ready.")

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
    ) -> tuple[list[str], list[float]]:
        """Run search using the specified method."""
        if method == "semantic":
            return semantic_search(query, self.vector_index, self.metadata_df, self.embedder, k)
        elif method == "bm25":
            return keyword_search_bm25(query, self.bm25_index, self.metadata_df, k)
        else:  # hybrid
            return hybrid_search(
                query, self.vector_index, self.metadata_df,
                self.embedder, self.bm25_index, k,
            )

    def _llm_call(self, system: str, user: str) -> str:
        """Call the LLM and return raw text output."""
        resp = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def generate(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
    ) -> tuple[GroundedAnswer, pd.DataFrame]:
        """Full RAG pipeline: retrieve → build context → LLM → parse."""
        # 1. Retrieve
        ids, scores = self.search(query, k, method)
        rows = (
            self.metadata_df[self.metadata_df["chunk_id"].isin(ids)]
            .set_index("chunk_id")
            .loc[ids]
            .reset_index()
        )
        rows["similarity_score"] = scores

        # 2. Confidence gate
        max_score = float(rows["similarity_score"].max()) if len(rows) else 0.0
        if max_score < SIMILARITY_THRESHOLD:
            return GroundedAnswer(
                recommendation="",
                supporting_evidence=[],
                citations=[],
                confidence="Insufficient Evidence",
                refusal_reason="Retrieved evidence similarity is below the safety threshold.",
            ), rows

        # 3. Build prompt
        context = build_context(rows)
        user_prompt = (
            f"Clinical question:\n{query}\n\n"
            f"Retrieved evidence (only source of truth):\n{context}\n\n"
            f"Produce a GroundedAnswer JSON."
        )

        # 4. Call LLM (or return retrieval-only if no client)
        if self.llm_client is None:
            return GroundedAnswer(
                recommendation="LLM not configured. Review the retrieved evidence below.",
                supporting_evidence=[r["text"][:200] + "…" for _, r in rows.head(3).iterrows()],
                citations=[],
                confidence="Low",
                refusal_reason="No LLM_API_KEY configured on the server.",
            ), rows

        raw = self._llm_call(SYSTEM_PROMPT, user_prompt)

        # 5. Parse
        answer = safe_parse_grounded(raw, evidence_rows=rows, metadata_df=self.metadata_df)
        return answer, rows


# Singleton engine instance
engine = RAGEngine()
