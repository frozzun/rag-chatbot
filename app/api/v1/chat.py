"""Chat and RAG query endpoints (sync and SSE streaming)."""

import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from app.api.deps import get_rag_service
from app.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.schemas.common import ApiResponse
from app.services.rag_service import RAGService

logger = logging.getLogger("app.chat")

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ApiResponse[ChatQueryResponse])
async def query_chat(
    payload: ChatQueryRequest,
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> ApiResponse[ChatQueryResponse]:
    """Execute non-streaming RAG query."""
    response = await rag_service.answer_query(payload)
    return ApiResponse(
        success=True,
        data=response,
        request_id=getattr(request.state, "request_id", None),
    )


@router.post("/stream")
async def stream_chat(
    payload: ChatQueryRequest,
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """Execute SSE streaming RAG query with client disconnect detection."""
    request_id = getattr(request.state, "request_id", "unknown")

    async def sse_event_generator():
        try:
            async for chunk in rag_service.stream_query(payload):
                # Critical check per rules.md §2.2: Detect client disconnect to prevent billing leaks
                if await request.is_disconnected():
                    logger.warning(
                        f"Client disconnected during streaming (request_id={request_id}). Cancelling LLM generation."
                    )
                    break

                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as exc:
            logger.error(f"Exception during SSE streaming: {exc}", exc_info=True)
            error_payload = json.dumps(
                {"code": "STREAM_ERROR", "message": "스트리밍 생성 중 오류가 발생했습니다.", "request_id": request_id}
            )
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )

