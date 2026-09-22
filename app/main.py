"""FastAPI application factory, lifespan management, and global error handling."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import v1_router
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware
from app.schemas.common import ApiErrorDetail, ApiErrorResponse
from app.services.embedding_client import create_embedding_client
from app.services.llm_client import create_llm_client
from app.vectorstore.chromadb_client import ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Modern lifespan context manager for setup and graceful shutdown of resources."""
    settings = get_settings()
    logger = setup_logging(settings.log_level)
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode (LLM: {settings.llm_provider})")

    # 1. Initialize Vector Store (ChromaDB)
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    vector_store.initialize_sync()
    app.state.vector_store = vector_store

    # 2. Initialize Clients
    embedding_client = create_embedding_client(settings)
    app.state.embedding_client = embedding_client

    llm_client = create_llm_client(settings)
    app.state.llm_client = llm_client

    yield

    # Graceful Shutdown
    logger.info("Executing graceful shutdown...")
    await embedding_client.close()
    await llm_client.close()
    vector_store.close()
    logger.info("All resources successfully closed.")


def create_app() -> FastAPI:
    """Application factory for FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Production-Ready FastAPI RAG Chatbot",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    # Register Routers
    app.include_router(v1_router)

    # Global Exception Handlers conforming strictly to rules.md §4.1
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        response_body = ApiErrorResponse(
            success=False,
            error=ApiErrorDetail(
                code=exc.code,
                message=exc.message,
                request_id=request_id,
                details=exc.details or None,
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_body.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        response_body = ApiErrorResponse(
            success=False,
            error=ApiErrorDetail(
                code="VALIDATION_ERROR",
                message="입력 데이터 유효성 검증에 실패했습니다.",
                request_id=request_id,
                details={"errors": exc.errors()},
            ),
        )
        return JSONResponse(
            status_code=422,
            content=response_body.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        response_body = ApiErrorResponse(
            success=False,
            error=ApiErrorDetail(
                code="HTTP_ERROR",
                message=str(exc.detail),
                request_id=request_id,
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_body.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        response_body = ApiErrorResponse(
            success=False,
            error=ApiErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="서버 내부 오류가 발생했습니다.",
                request_id=request_id,
            ),
        )
        return JSONResponse(
            status_code=500,
            content=response_body.model_dump(),
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    return app


app = create_app()

