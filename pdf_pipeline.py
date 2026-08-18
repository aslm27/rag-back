"""PDF extraction, cleaning, and section-aware chunking pipeline.

Extracted from the Colab notebook's Day 1 cells.
"""
import re
import uuid
import hashlib
from pathlib import Path

from config import CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS, ALLOWED_PUBLISHERS


# ── Tokenizer ──────────────────────────────────────────────────────────
def _load_tokenizer():
    """tiktoken needs to download its vocab file on first use."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"[warn] tiktoken unavailable ({type(e).__name__}) -> ~4-chars/token estimate.")
        return None


_ENC = _load_tokenizer()


def count_tokens(text: str) -> int:
    """Token count via tiktoken when available, else ~4-characters-per-token estimate."""
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


# ── PDF Extraction ─────────────────────────────────────────────────────
def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """Extract raw text per page from a PDF, keeping 1-indexed page number."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return [
        {"page_number": i + 1, "text": page.extract_text() or ""}
        for i, page in enumerate(reader.pages)
    ]


# ── Text Cleaning ──────────────────────────────────────────────────────
def clean_page_text(text: str, running_headers: list[str] | None = None) -> str:
    """Strip common PDF extraction artifacts from one page of text."""
    running_headers = running_headers or []
    # de-hyphenate line-wrapped words
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # collapse single newlines into spaces, keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # strip known repeated running headers/footers
    for header in running_headers:
        text = text.replace(header, "")
    # collapse excess whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[dict], running_headers: list[str] | None = None) -> list[dict]:
    return [
        {"page_number": p["page_number"], "text": clean_page_text(p["text"], running_headers)}
        for p in pages
    ]


# ── Section Detection ─────────────────────────────────────────────────
SECTION_HEADER_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{2,60}|[A-Z][A-Z ]{4,60})$"
)


def split_into_sections(page_text: str) -> list[tuple[str, str]]:
    """Split a cleaned page into (section_title, body) blocks."""
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    sections, current_title, current_body = [], "General", []

    for line in lines:
        if SECTION_HEADER_RE.match(line):
            if current_body:
                sections.append((current_title, " ".join(current_body)))
            current_title, current_body = line, []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, " ".join(current_body)))
    return sections


# ── Chunking ───────────────────────────────────────────────────────────
def chunk_pages(
    pages: list[dict],
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> list[dict]:
    """Section-aware chunking: merges small sections, splits large ones."""
    chunks: list[dict] = []
    buffer_text, buffer_title, buffer_page = "", None, None

    def flush():
        nonlocal buffer_text, buffer_title, buffer_page
        if buffer_text.strip():
            chunks.append({
                "section_title": buffer_title,
                "page_number": buffer_page,
                "text": buffer_text.strip(),
            })
        buffer_text, buffer_title, buffer_page = "", None, None

    for page in pages:
        for title, body in split_into_sections(page["text"]):
            body_tokens = count_tokens(body)

            if body_tokens > max_tokens:
                # split on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", body)
                sub_text = ""
                for sent in sentences:
                    if count_tokens(sub_text + " " + sent) > max_tokens and sub_text:
                        chunks.append({
                            "section_title": title,
                            "page_number": page["page_number"],
                            "text": sub_text.strip(),
                        })
                        sub_text = sent
                    else:
                        sub_text = (sub_text + " " + sent).strip()
                if sub_text:
                    chunks.append({
                        "section_title": title,
                        "page_number": page["page_number"],
                        "text": sub_text.strip(),
                    })
                continue

            if buffer_title not in (None, title) and count_tokens(buffer_text) >= min_tokens:
                flush()
            if buffer_title is None:
                buffer_title, buffer_page = title, page["page_number"]
            buffer_text = (buffer_text + " " + body).strip()

            if count_tokens(buffer_text) >= min_tokens:
                flush()

    flush()
    return chunks


# ── Full Ingestion Pipeline ────────────────────────────────────────────
def build_chunk_records(
    all_chunks_by_doc: dict,
    sources: dict | None = None,
) -> list[dict]:
    """Build chunk records with metadata for all documents."""
    records = []
    sources = sources or {}
    for doc_name, chunks in all_chunks_by_doc.items():
        source_meta = sources.get(doc_name, {"publisher": "UNKNOWN", "source_url": ""})
        for c in chunks:
            records.append({
                "chunk_id": str(uuid.uuid4()),
                "document_name": doc_name,
                "publisher": source_meta.get("publisher", "UNKNOWN"),
                "source_url": source_meta.get("source_url", ""),
                "page_number": c["page_number"],
                "section_title": c["section_title"],
                "text": c["text"],
                "token_count": count_tokens(c["text"]),
            })
    return records


def ingest_pdfs(pdf_dir: str | Path, sources: dict | None = None) -> tuple[list[dict], dict]:
    """
    Full pipeline: find PDFs → extract → clean → chunk → build records.
    Returns (chunk_records, all_documents_pages).
    """
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    found = sorted(p.name for p in pdf_dir.glob("*.pdf"))
    if not found:
        return [], {}

    all_documents_pages = {}
    for fname in found:
        raw_pages = extract_pdf_pages(str(pdf_dir / fname))
        all_documents_pages[fname] = clean_pages(raw_pages)

    all_chunks_by_doc = {
        doc: chunk_pages(pages) for doc, pages in all_documents_pages.items()
    }

    records = build_chunk_records(all_chunks_by_doc, sources)
    return records, all_documents_pages
