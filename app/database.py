import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: PGVector | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        device = settings.get_embedding_device()
        logger.info("Embedding device: %s", device)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = PGVector(
            connection=settings.get_database_url(),
            embeddings=get_embeddings(),
            collection_name=settings.collection_name,
            # Store page-level metadata (source, page_number, etc.) alongside vectors
            use_jsonb=True,
        )
    return _vector_store


@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    settings = get_settings()
    conn = psycopg2.connect(settings.get_database_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def save_document(filename: str, display_name: str) -> None:
    """Upsert a document's display name."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddoc_documents (filename, display_name)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE SET display_name = EXCLUDED.display_name
                """,
                (filename, display_name),
            )
        conn.commit()


def list_documents() -> list[dict]:
    """Return all documents ordered by insertion time."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, display_name FROM embeddoc_documents ORDER BY created_at")
            return [dict(row) for row in cur.fetchall()]


def delete_document_record(filename: str) -> None:
    """Remove a document's metadata row."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM embeddoc_documents WHERE filename = %s", (filename,))
        conn.commit()


def delete_document_chunks(document_name: str) -> int:
    """Delete all vector chunks belonging to a document. Returns number of rows deleted."""
    settings = get_settings()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                ) AND cmetadata->>'document_name' = %s
                """,
                (settings.collection_name, document_name),
            )
            deleted = cur.rowcount
        conn.commit()
    return deleted


def init_db() -> None:
    """Enable pgvector and create the LangChain PGVector tables on first startup."""
    settings = get_settings()
    db_url = settings.get_database_url()

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    logger.info("pgvector extension ready")

    get_vector_store().create_tables_if_not_exists()
    logger.info("Embeddings table ready (collection: %s)", settings.collection_name)

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddoc_documents (
                    filename     TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()
    logger.info("Documents metadata table ready")