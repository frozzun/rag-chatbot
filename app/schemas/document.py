"""Document ingestion schemas and DTOs."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class TextIngestRequest(BaseModel):
    """Payload for direct text ingestion."""

    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field(..., min_length=5, description="Document body text")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Custom metadata key-values")


class DocumentIngestResponse(BaseModel):
    """Response returned upon document ingestion."""

    document_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title")
    content_hash: str = Field(description="SHA-256 hash of document content")
    chunks_count: int = Field(description="Number of created chunks")
    is_duplicate: bool = Field(default=False, description="Whether document already existed")
    message: str = Field(description="Status description")


class DocumentChunkInfo(BaseModel):
    """Information about an indexed chunk."""

    chunk_id: str
    chunk_index: int
    text: str
    content_hash: str
    document_id: str
    title: str


class DocumentListResponse(BaseModel):
    """Summary list of ingested documents."""

    total_chunks: int
    collection_name: str

