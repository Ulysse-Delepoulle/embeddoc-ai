from typing import Callable

from app.utils.pdf_parser import parse_pdf, chunk_pages
from langchain.schema import Document
from app.database import get_vector_store

EMBED_BATCH = 32


def ingest_pdf(file_path: str, document_name: str) -> int:
    """Synchronous full ingest (used by the background thread via embed_chunks)."""
    chunks = chunk_pages(parse_pdf(file_path, document_name))
    return embed_chunks(chunks)


def embed_chunks(chunks: list, on_progress: Callable[[int], None] = None) -> int:
    """Embed and store pre-parsed chunks in batches.

    Calls on_progress(chunks_done) after each batch so callers can track progress.
    Returns total number of chunks stored.
    """
    vs = get_vector_store()
    total = len(chunks)

    for i in range(0, total, EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        docs = [
            Document(
                page_content=c["text"],
                metadata={"page_num": c["page_num"], "document_name": c["document_name"]},
            )
            for c in batch
        ]
        vs.add_documents(docs)
        if on_progress:
            on_progress(min(i + EMBED_BATCH, total))

    return total
