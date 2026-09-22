"""Embedding client abstraction supporting Mock, Gemini, and OpenAI-compatible (Local LLM)."""

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import Optional
import httpx

from app.core.config import Settings
from app.core.exceptions import LLMServiceException

logger = logging.getLogger("app.embedding")


class BaseEmbeddingClient(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        pass

    async def close(self) -> None:
        """Cleanup any active network sessions."""
        pass


class MockEmbeddingClient(BaseEmbeddingClient):
    """Deterministic in-memory embedding client using token-frequency projection for semantic testing."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def _generate_vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        if not text:
            return vector

        # Tokenize by simple words and 3-char n-grams
        tokens = text.lower().split()
        if len(text) >= 3:
            tokens.extend([text[i : i + 3] for i in range(len(text) - 2)])

        for token in tokens:
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0

        # L2-normalize vector for cosine distance
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)


class GeminiEmbeddingClient(BaseEmbeddingClient):
    """Google Gemini embedding client using google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        if not api_key:
            raise LLMServiceException("Gemini API Key가 제공되지 않았습니다.")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            results: list[list[float]] = []
            # Batch embedding
            for text in texts:
                resp = await self.client.aio.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                if hasattr(resp, "embedding") and resp.embedding:
                    results.append(resp.embedding.values)
                elif hasattr(resp, "embeddings") and resp.embeddings:
                    results.append(resp.embeddings[0].values)
                else:
                    raise LLMServiceException("임베딩 결과를 추출할 수 없습니다.")
            return results
        except Exception as exc:
            logger.error(f"Gemini embedding error: {exc}", exc_info=True)
            raise LLMServiceException(f"Gemini 임베딩 생성 실패: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_texts([text])
        return res[0]


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    """OpenAI-compatible embedding client for local LLM (Ollama, vLLM, etc.)."""

    def __init__(self, base_url: str, api_key: str = "ollama", model_name: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._http_client.post(
                "/embeddings",
                json={"input": texts, "model": self.model_name},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]
        except Exception as exc:
            logger.error(f"OpenAI-compatible embedding error: {exc}", exc_info=True)
            raise LLMServiceException(f"로컬/OpenAI 임베딩 호출 실패: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_texts([text])
        return res[0]

    async def close(self) -> None:
        await self._http_client.aclose()


def create_embedding_client(settings: Settings) -> BaseEmbeddingClient:
    """Factory creating configured embedding client."""
    provider = settings.llm_provider
    if provider == "gemini":
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set. Falling back to MockEmbeddingClient.")
            return MockEmbeddingClient()
        return GeminiEmbeddingClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_embedding_model,
        )
    elif provider == "openai_compatible":
        return OpenAICompatibleEmbeddingClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
        )
    else:
        return MockEmbeddingClient()
