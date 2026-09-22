"""Tests for consistent exception handling matching rules.md."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_validation_error_format(client: AsyncClient):
    """Verify 422 error matches standard error format."""
    # Invalid payload: missing required 'content' field
    invalid_payload = {"title": "No Content"}

    response = await client.post("/api/v1/documents/text", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert "request_id" in data["error"]
    assert data["error"]["request_id"] is not None


@pytest.mark.asyncio
async def test_empty_content_app_exception(client: AsyncClient):
    """Verify custom AppException returns standard error JSON."""
    # Length >= 5 passes Pydantic min_length, but all spaces triggers service InvalidRequestError
    payload = {"title": "Empty Doc", "content": "      "}

    response = await client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_REQUEST"
    assert "비어 있습니다" in data["error"]["message"]
    assert "request_id" in data["error"]
