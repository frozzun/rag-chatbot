"""FastAPI Dependency Injection providers."""

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.document_service import DocumentService
from app.services.embedding_client import BaseEmbeddingClient
from app.services.llm_client import BaseLLMClient
from app.services.rag_service import RAGService
from app.vectorstore.chromadb_client import ChromaVectorStore


def get_vector_store(request: Request) -> ChromaVectorStore:
    """Retrieve vector store instance cached in app.state."""
    return request.app.state.vector_store


def get_embedding_client(request: Request) -> BaseEmbeddingClient:
    """Retrieve embedding client cached in app.state."""
    return request.app.state.embedding_client


def get_llm_client(request: Request) -> BaseLLMClient:
    """Retrieve LLM client cached in app.state."""
    return request.app.state.llm_client


def get_document_service(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    embedding_client: BaseEmbeddingClient = Depends(get_embedding_client),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """Provide DocumentService with injected dependencies."""
    return DocumentService(
        vector_store=vector_store,
        embedding_client=embedding_client,
        settings=settings,
    )


def get_rag_service(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    embedding_client: BaseEmbeddingClient = Depends(get_embedding_client),
    llm_client: BaseLLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> RAGService:
    """Provide RAGService with injected dependencies."""
    return RAGService(
        vector_store=vector_store,
        embedding_client=embedding_client,
        llm_client=llm_client,
        settings=settings,
    )

