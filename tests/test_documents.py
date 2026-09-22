"""Tests for document ingestion and idempotency."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_text_ingest_and_idempotency(client: AsyncClient):
    """Verify SHA-256 deduplication idempotency."""
    doc_payload = {
        "title": "FastAPI 아키텍처 가이드",
        "content": "FastAPI는 고성능 비동기 웹 프레임워크로, Starlette와 Pydantic을 기반으로 동작합니다.",
        "metadata": {"category": "architecture"},
    }

    # First ingestion
    resp1 = await client.post("/api/v1/documents/text", json=doc_payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["success"] is True
    assert data1["data"]["is_duplicate"] is False
    doc_id = data1["data"]["document_id"]
    content_hash = data1["data"]["content_hash"]
    assert len(content_hash) == 64  # SHA-256 length

    # Second ingestion with identical content -> should be detected as duplicate (Idempotency)
    resp2 = await client.post("/api/v1/documents/text", json=doc_payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["success"] is True
    assert data2["data"]["is_duplicate"] is True
    assert data2["data"]["document_id"] == doc_id
    assert data2["data"]["content_hash"] == content_hash


@pytest.mark.asyncio
async def test_file_upload_ingest(client: AsyncClient):
    """Verify file upload ingestion."""
    file_content = b"Docker and Kubernetes provide fault tolerance and isolated application execution."
    files = {"file": ("deployment.txt", file_content, "text/plain")}

    response = await client.post("/api/v1/documents/file", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["title"] == "deployment.txt"
    assert data["data"]["is_duplicate"] is False


@pytest.mark.asyncio
async def test_document_stats(client: AsyncClient):
    """Verify stats returns chunk count."""
    response = await client.get("/api/v1/documents/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["total_chunks"] >= 1

