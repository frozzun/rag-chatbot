"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness(client: AsyncClient):
    """Verify /api/v1/health returns success and X-Request-ID."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "alive"
    assert "X-Request-ID" in response.headers
    assert data["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_readiness(client: AsyncClient):
    """Verify /api/v1/health/ready returns vector store status."""
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "ready"
    assert data["data"]["vector_store"] == "chromadb"
    assert "total_vectors" in data["data"]

