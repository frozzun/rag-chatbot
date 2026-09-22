"""Tests for Server-Sent Events (SSE) streaming chat."""

import json
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_sse_chat_streaming(client: AsyncClient):
    """Verify SSE streaming format and event delivery."""
    # Ingest document
    await client.post(
        "/api/v1/documents/text",
        json={"title": "SSE Protocol", "content": "Server-Sent Events allow servers to push data to web clients."},
    )

    query_payload = {
        "query": "Server-Sent Events",
        "top_k": 2,
        "score_threshold": 0.3,
    }

    chunks_received = []
    async with client.stream("POST", "/api/v1/chat/stream", json=query_payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["Content-Type"]

        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_json = json.loads(line[6:])
            chunks_received.append(data_json)

    assert len(chunks_received) > 0
    # Final chunk should have done=True
    final_chunk = chunks_received[-1]
    assert final_chunk["done"] is True


@pytest.mark.asyncio
async def test_sse_client_disconnect_abort(client: AsyncClient):
    """Verify stream is cleanly aborted when client disconnects early."""
    query_payload = {
        "query": "Server-Sent Events",
        "top_k": 2,
        "score_threshold": 0.3,
    }

    # Simulate client disconnect by breaking after receiving first chunk
    async with client.stream("POST", "/api/v1/chat/stream", json=query_payload) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line and line.startswith("data: "):
                # Client abruptly drops connection
                break
    # Exited cleanly without unhandled exception

