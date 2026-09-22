"""Chat and RAG query schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    """Chat or RAG search query request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override similarity score threshold"
    )


class RetrievedChunk(BaseModel):
    """Context chunk retrieved from vector store."""

    chunk_id: str
    text: str
    similarity_score: float
    document_id: Optional[str] = None
    title: Optional[str] = None


class ChatQueryResponse(BaseModel):
    """Non-streaming RAG query response."""

    answer: str = Field(description="Generated answer or fallback message")
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks")
    fallback_triggered: bool = Field(default=False, description="True if score was below threshold")
    latency_ms: float = Field(description="RAG pipeline execution latency in milliseconds")


class StreamChunkPayload(BaseModel):
    """Payload for Server-Sent Events (SSE) data field."""

    text: Optional[str] = Field(default=None, description="Incremental generated token chunk")
    done: bool = Field(default=False, description="True when generation is finished")
    fallback_triggered: bool = Field(default=False, description="True if fallback triggered")
    retrieved_chunks: Optional[list[RetrievedChunk]] = Field(
        default=None, description="Source context chunks included in final payload"
    )

