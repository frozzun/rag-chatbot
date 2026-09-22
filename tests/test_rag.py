"""Tests for RAG pipeline, similarity threshold fallback, and token budgeting."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_rag_query_with_match(client: AsyncClient):
    """Ingest a document and query for it."""
    # Ingest document
    doc_text = "Python AnyIO is an asynchronous networking and concurrency library that works on asyncio."
    await client.post(
        "/api/v1/documents/text",
        json={"title": "AnyIO Doc", "content": doc_text},
    )

    # Query with reasonable threshold
    query_payload = {
        "query": "Python AnyIO is an asynchronous networking library",
        "top_k": 3,
        "score_threshold": 0.5,
    }
    response = await client.post("/api/v1/chat", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["fallback_triggered"] is False
    assert len(data["data"]["answer"]) > 0
    assert len(data["data"]["retrieved_chunks"]) > 0


@pytest.mark.asyncio
async def test_rag_query_fallback_when_threshold_unmet(client: AsyncClient):
    """Verify fallback response is triggered when similarity is below threshold."""
    query_payload = {
        "query": "완전히 다른 내용의 쿼리 123456789xyz",
        "top_k": 3,
        "score_threshold": 0.999,  # Unattainably high threshold
    }
    response = await client.post("/api/v1/chat", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["fallback_triggered"] is True
    assert data["data"]["answer"] == "해당 문서를 찾을 수 없습니다."


def test_assemble_context_with_budget_truncation():
    """Verify that context chunks are truncated when exceeding max token budget."""
    from app.schemas.chat import RetrievedChunk
    from app.services.rag_service import assemble_context_with_budget

    chunks = [
        RetrievedChunk(chunk_id="c1", text="A" * 300, similarity_score=0.9, title="Doc1"),
        RetrievedChunk(chunk_id="c2", text="B" * 300, similarity_score=0.8, title="Doc2"),
        RetrievedChunk(chunk_id="c3", text="C" * 300, similarity_score=0.7, title="Doc3"),
    ]

    # Budget of 120 tokens (~360 chars) fits only Doc1
    context_text, selected = assemble_context_with_budget(chunks, max_tokens=120)
    assert len(selected) == 1
    assert selected[0].chunk_id == "c1"
    assert "Doc1" in context_text
    assert "Doc2" not in context_text

