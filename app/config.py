from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Anthropic
    anthropic_api_key: str

    # PostgreSQL
    postgres_user: str = "embeddoc"
    postgres_password: str = "embeddoc"
    postgres_db: str = "embeddoc"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Explicit override for managed databases; auto-built otherwise
    database_url: str = ""

    # RAG
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    claude_model: str = "claude-sonnet-4-6"
    collection_name: str = "document_embeddings"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_device: str = ""  # empty = auto-detect at runtime

    def get_embedding_device(self) -> str:
        if self.embedding_device:
            return self.embedding_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def get_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
