"""Pytest fixtures for FastAPI RAG application."""

import os
import shutil
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.embedding_client import MockEmbeddingClient
from app.services.llm_client import MockLLMClient
from app.vectorstore.chromadb_client import ChromaVectorStore


@pytest.fixture(scope="session")
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB during tests."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_chroma_dir) -> Settings:
    """Provide isolated test settings."""
    return Settings(
        app_name="Test-RAG-Chatbot",
        app_env="test",
        debug=True,
        log_level="DEBUG",
        llm_provider="mock",
        chroma_persist_dir=temp_chroma_dir,
        chroma_collection_name="test_collection",
        similarity_threshold=0.65,
        max_context_tokens=1000,
        chunk_size=200,
        chunk_overlap=20,
        fallback_message="해당 문서를 찾을 수 없습니다.",
    )


import pytest_asyncio


@pytest.fixture
def app_instance(test_settings):
    """Create FastAPI application with initialized dependencies."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings

    # Initialize store and clients
    store = ChromaVectorStore(
        persist_directory=test_settings.chroma_persist_dir,
        collection_name=test_settings.chroma_collection_name,
    )
    store.initialize_sync()
    app.state.vector_store = store
    app.state.embedding_client = MockEmbeddingClient(dimension=64)
    app.state.llm_client = MockLLMClient()

    yield app

    # Teardown
    store.close()


@pytest_asyncio.fixture
async def client(app_instance) -> AsyncClient:
    """Async HTTP client for testing endpoints."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
