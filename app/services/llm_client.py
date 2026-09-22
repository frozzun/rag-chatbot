"""LLM client abstraction supporting Mock, Gemini, and OpenAI-compatible (Local LLM)."""

import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import httpx

from app.core.config import Settings
from app.core.exceptions import LLMServiceException

logger = logging.getLogger("app.llm")


class BaseLLMClient(ABC):
    """Abstract interface for LLM text generation and streaming."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate complete response text."""
        pass

    @abstractmethod
    async def stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Stream response tokens asynchronously."""
        pass

    async def close(self) -> None:
        """Cleanup any network resources."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock LLM client for offline tests and local development."""

    def __init__(self, prefix: str = "[Mock AI Answer] "):
        self.prefix = prefix

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return f"{self.prefix}질문에 대한 검색된 지식을 바탕으로 생성된 답변입니다. (Prompt length: {len(prompt)})"

    async def stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        full_text = await self.generate(prompt, system_prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            yield chunk


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini LLM client using google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        if not api_key:
            raise LLMServiceException("Gemini API Key가 제공되지 않았습니다.")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            from google.genai import types
            config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            logger.error(f"Gemini generate error: {exc}", exc_info=True)
            raise LLMServiceException(f"Gemini 답변 생성 실패: {exc}") from exc

    async def stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        try:
            from google.genai import types
            config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
            stream_resp = await self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            async for chunk in stream_resp:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.error(f"Gemini stream error: {exc}", exc_info=True)
            raise LLMServiceException(f"Gemini 스트리밍 생성 실패: {exc}") from exc


class OpenAICompatibleLLMClient(BaseLLMClient):
    """OpenAI-compatible LLM client for local LLM (Ollama, vLLM, etc.)."""

    def __init__(self, base_url: str, api_key: str = "ollama", model_name: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        try:
            resp = await self._http_client.post(
                "/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error(f"OpenAI-compatible generate error: {exc}", exc_info=True)
            raise LLMServiceException(f"로컬 LLM 답변 생성 실패: {exc}") from exc

    async def stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        messages = self._build_messages(prompt, system_prompt)
        try:
            async with self._http_client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if line.startswith("data: "):
                        payload_str = line[6:]
                        if payload_str == "[DONE]":
                            break
                        try:
                            payload = json.loads(payload_str)
                            delta = payload.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.error(f"OpenAI-compatible stream error: {exc}", exc_info=True)
            raise LLMServiceException(f"로컬 LLM 스트리밍 실패: {exc}") from exc

    async def close(self) -> None:
        await self._http_client.aclose()


def create_llm_client(settings: Settings) -> BaseLLMClient:
    """Factory creating configured LLM client."""
    provider = settings.llm_provider
    if provider == "gemini":
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set. Falling back to MockLLMClient.")
            return MockLLMClient()
        return GeminiLLMClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
        )
    elif provider == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_name,
        )
    else:
        return MockLLMClient()

