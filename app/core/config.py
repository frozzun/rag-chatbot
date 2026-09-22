"""Application Configuration using pydantic-settings."""

from functools import lru_cache
from typing import Any, Literal, Optional
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General Application
    app_name: str = Field(default="Production-FastAPI-RAG", description="Application name")
    app_env: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_DEBUG", "DEBUG"),
        description="Debug mode",
    )
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, str):
            if v.lower() in ("true", "1", "yes", "t", "dev", "development"):
                return True
            if v.lower() in ("false", "0", "no", "f", "release", "prod", "production", ""):
                return False
        return bool(v)

    # LLM Provider Selection
    llm_provider: Literal["mock", "gemini", "openai_compatible"] = Field(
        default="mock",
        description="LLM provider: mock (zero-dep test), gemini (Google SaaS), openai_compatible (local Ollama/vLLM or OpenAI)",
    )

    # Google Gemini Settings (Used when llm_provider='gemini')
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    gemini_model_name: str = Field(default="gemini-3.8-flash", description="Gemini text model")
    gemini_embedding_model: str = Field(default="gemini-embedding-001", description="Gemini embedding model")

    # OpenAI-Compatible / Local LLM Settings (Used when llm_provider='openai_compatible')
    openai_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for OpenAI-compatible endpoint (e.g., Ollama or vLLM)",
    )
    openai_api_key: str = Field(default="ollama", description="API Key for OpenAI-compatible endpoint")
    openai_model_name: str = Field(default="llama3.1:8b", description="Model name for chat completions")
    openai_embedding_model: str = Field(
        default="nomic-embed-text", description="Model name for embeddings"
    )

    # Vector Store (ChromaDB)
    chroma_persist_dir: str = Field(default="./data/chroma", description="Directory to persist ChromaDB files")
    chroma_collection_name: str = Field(default="rag_documents", description="ChromaDB collection name")

    # RAG Guardrails & Safety
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold. Below this, context fallback is triggered.",
    )
    max_context_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum estimated token limit for retrieved context in prompt.",
    )
    chunk_size: int = Field(default=500, gt=50, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, description="Text chunk overlap in characters")
    fallback_message: str = Field(
        default="해당 문서를 찾을 수 없습니다.",
        description="Safe fallback message when no relevant documents are found.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
