from __future__ import annotations

from pathlib import Path

from rag.chunking import chunk_text
from rag.pdf_extract import extract_pdf_text
from rag.store import get_rag_store


def ingest_pdf_bytes(
    pdf_bytes: bytes,
    *,
    source_name: str,
    data_dir: Path,
    replace: bool = False,
) -> dict:
    store = get_rag_store(data_dir)
    if replace:
        store.clear()
    text = extract_pdf_text(pdf_bytes)
    if not text.strip():
        raise ValueError(
            "No text extracted from this PDF. It may be image-only: install Tesseract "
            "(brew install tesseract) for OCR on scanned pages."
        )
    chunks = chunk_text(text)
    added = store.add_text_chunks(chunks, source=source_name)
    stats = store.stats()
    return {
        "chunks_added": added,
        "chars_extracted": len(text),
        "total_chunks": stats["chunks"],
        "source": source_name,
    }


def retrieve_context(query: str, data_dir: Path, top_k: int = 5, max_chars: int = 2800) -> tuple[str, list[dict]]:
    store = get_rag_store(data_dir)
    hits = store.search(query, top_k=top_k)
    if not hits:
        return "", []
    parts: list[str] = []
    meta: list[dict] = []
    used = 0
    for text, score, src in hits:
        block = f"[{src} | sim={score:.3f}]\n{text}"
        if used + len(block) + 8 > max_chars:
            break
        parts.append(block)
        meta.append({"source": src, "score": score, "preview": text[:200]})
        used += len(block) + 8
    return "\n\n---\n\n".join(parts), meta
