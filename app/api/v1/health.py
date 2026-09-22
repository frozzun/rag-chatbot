"""Health check and readiness probe endpoints."""

from fastapi import APIRouter, Depends
from starlette.requests import Request

from app.api.deps import get_vector_store
from app.core.config import Settings, get_settings
from app.schemas.common import ApiResponse
from app.vectorstore.chromadb_client import ChromaVectorStore

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=ApiResponse[dict])
async def liveness_probe(request: Request) -> ApiResponse[dict]:
    """Basic liveness probe."""
    return ApiResponse(
        success=True,
        data={"status": "alive", "service": "rag-chatbot"},
        request_id=getattr(request.state, "request_id", None),
    )


@router.get("/health/ready", response_model=ApiResponse[dict])
async def readiness_probe(
    request: Request,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> ApiResponse[dict]:
    """Readiness probe checking ChromaDB connection and LLM configuration."""
    total_vectors = await vector_store.count_async()
    return ApiResponse(
        success=True,
        data={
            "status": "ready",
            "llm_provider": settings.llm_provider,
            "vector_store": "chromadb",
            "total_vectors": total_vectors,
            "collection": settings.chroma_collection_name,
        },
        request_id=getattr(request.state, "request_id", None),
    )

