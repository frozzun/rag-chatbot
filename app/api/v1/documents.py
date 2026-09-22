"""Document ingestion and management endpoints."""

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.requests import Request

import anyio
from app.api.deps import get_document_service, get_vector_store
from app.core.exceptions import InvalidRequestError
from app.schemas.common import ApiResponse
from app.schemas.document import (
    DocumentIngestResponse,
    DocumentListResponse,
    TextIngestRequest,
)
from app.services.document_service import DocumentService
from app.services.file_parser import parse_file_content_sync
from app.vectorstore.chromadb_client import ChromaVectorStore

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/text", response_model=ApiResponse[DocumentIngestResponse])
async def ingest_text(
    payload: TextIngestRequest,
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
) -> ApiResponse[DocumentIngestResponse]:
    """Ingest document from raw text with SHA-256 idempotency."""
    result = await document_service.ingest_document(
        title=payload.title,
        content=payload.content,
        metadata=payload.metadata,
    )
    return ApiResponse(
        success=True,
        data=result,
        request_id=getattr(request.state, "request_id", None),
    )


@router.post("/file", response_model=ApiResponse[DocumentIngestResponse])
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> ApiResponse[DocumentIngestResponse]:
    """Upload and ingest a file (PDF, Markdown, Text, etc.) with SHA-256 idempotency."""
    if not file.filename:
        raise InvalidRequestError("파일명이 유효하지 않습니다.")

    content_bytes = await file.read()
    if not content_bytes:
        raise InvalidRequestError("업로드된 파일이 비어 있습니다.")

    # Offload CPU-bound PDF/text parsing to worker thread to prevent event loop blocking
    text_content = await anyio.to_thread.run_sync(
        parse_file_content_sync, file.filename, content_bytes
    )

    result = await document_service.ingest_document(
        title=file.filename,
        content=text_content,
        metadata={"filename": file.filename, "content_type": file.content_type},
    )

    return ApiResponse(
        success=True,
        data=result,
        request_id=getattr(request.state, "request_id", None),
    )


@router.get("/stats", response_model=ApiResponse[DocumentListResponse])
async def get_document_stats(
    request: Request,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> ApiResponse[DocumentListResponse]:
    """Return total chunk count and collection status."""
    count = await vector_store.count_async()
    return ApiResponse(
        success=True,
        data=DocumentListResponse(
            total_chunks=count,
            collection_name=vector_store.collection_name,
        ),
        request_id=getattr(request.state, "request_id", None),
    )

