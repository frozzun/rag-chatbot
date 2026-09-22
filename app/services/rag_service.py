"""RAG Pipeline service with similarity fallback and token budgeting."""

import logging
import time
from typing import AsyncIterator, Optional

from app.core.config import Settings
from app.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    RetrievedChunk,
    StreamChunkPayload,
)
from app.services.embedding_client import BaseEmbeddingClient
from app.services.llm_client import BaseLLMClient
from app.vectorstore.chromadb_client import ChromaVectorStore

logger = logging.getLogger("app.rag")


SYSTEM_PROMPT_TEMPLATE = """당신은 신뢰할 수 있는 RAG 어시스턴트입니다.
반드시 제공된 [참고 문서]의 내용만을 근거로 정확하고 명확하게 답변하십시오.
문서에서 명확히 확인할 수 없는 내용은 추측하거나 지어내지 마십시오."""


def estimate_tokens(text: str) -> int:
    """Fast character-based heuristic token estimator (approx 3 chars per token for Korean/English mix)."""
    return max(1, len(text) // 3)


def assemble_context_with_budget(
    chunks: list[RetrievedChunk], max_tokens: int
) -> tuple[str, list[RetrievedChunk]]:
    """Assemble context text while respecting the token budget (Context Overflow Prevention)."""
    current_tokens = 0
    selected_chunks: list[RetrievedChunk] = []
    context_parts: list[str] = []

    for chunk in chunks:
        chunk_text = f"[{chunk.title or '문서'}] {chunk.text}"
        chunk_tokens = estimate_tokens(chunk_text)

        if current_tokens + chunk_tokens > max_tokens:
            # Check if we can partially fit or if we should stop
            remaining_budget = max_tokens - current_tokens
            if remaining_budget > 30 and not selected_chunks:
                # Truncate first chunk if even first chunk exceeds budget
                char_limit = remaining_budget * 3
                truncated_text = chunk.text[:char_limit] + "... (truncated)"
                context_parts.append(f"[{chunk.title or '문서'}] {truncated_text}")
                selected_chunks.append(chunk)
            break

        context_parts.append(chunk_text)
        selected_chunks.append(chunk)
        current_tokens += chunk_tokens

    assembled_text = "\n\n".join(context_parts)
    return assembled_text, selected_chunks


class RAGService:
    """Coordinates vector search, similarity threshold validation, and LLM generation."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_client: BaseEmbeddingClient,
        llm_client: BaseLLMClient,
        settings: Settings,
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.settings = settings

    async def _retrieve_and_evaluate(
        self, query: str, top_k: int, threshold: Optional[float] = None
    ) -> tuple[bool, list[RetrievedChunk], str]:
        """Query vector store and apply score threshold safety check."""
        active_threshold = threshold if threshold is not None else self.settings.similarity_threshold

        # 1. Embed user query
        query_vector = await self.embedding_client.embed_query(query)

        # 2. Vector search in ChromaDB
        search_res = await self.vector_store.query_async(query_vector, n_results=top_k)

        documents = search_res.get("documents", [[]])[0]
        distances = search_res.get("distances", [[]])[0]
        metadatas = search_res.get("metadatas", [[]])[0]
        ids = search_res.get("ids", [[]])[0]

        if not documents:
            logger.info("No documents found in vector store. Triggering fallback.")
            return True, [], self.settings.fallback_message

        retrieved_chunks: list[RetrievedChunk] = []
        for i, doc_text in enumerate(documents):
            dist = distances[i] if i < len(distances) else 1.0
            meta = metadatas[i] if i < len(metadatas) else {}
            chunk_id = ids[i] if i < len(ids) else f"chunk-{i}"

            # Convert cosine distance to similarity score: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc_text,
                    similarity_score=round(similarity, 4),
                    document_id=meta.get("document_id"),
                    title=meta.get("title"),
                )
            )

        # Sort chunks by similarity descending
        retrieved_chunks.sort(key=lambda x: x.similarity_score, reverse=True)

        # 3. Context Fallback Check: check highest score against threshold
        highest_score = retrieved_chunks[0].similarity_score if retrieved_chunks else 0.0
        if highest_score < active_threshold:
            logger.info(
                f"Highest similarity score ({highest_score:.4f}) is below threshold ({active_threshold:.4f}). "
                "Triggering Context Fallback to prevent hallucination."
            )
            return True, retrieved_chunks, self.settings.fallback_message

        # Filter out individual chunks that don't meet threshold
        valid_chunks = [c for c in retrieved_chunks if c.similarity_score >= active_threshold]
        if not valid_chunks:
            return True, retrieved_chunks, self.settings.fallback_message

        return False, valid_chunks, ""

    async def answer_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """Process user query and return complete answer with latency measurement."""
        start_time = time.perf_counter()

        fallback_triggered, chunks, fallback_msg = await self._retrieve_and_evaluate(
            request.query, request.top_k, request.score_threshold
        )

        if fallback_triggered:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatQueryResponse(
                answer=fallback_msg,
                retrieved_chunks=chunks,
                fallback_triggered=True,
                latency_ms=latency_ms,
            )

        # Context Window Overflow Prevention: Token budgeting
        context_text, budgeted_chunks = assemble_context_with_budget(
            chunks, self.settings.max_context_tokens
        )

        prompt = f"""[참고 문서]
{context_text}

[사용자 질문]
{request.query}

위 참고 문서를 바탕으로 사용자 질문에 답변하십시오."""

        answer = await self.llm_client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_TEMPLATE)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatQueryResponse(
            answer=answer,
            retrieved_chunks=budgeted_chunks,
            fallback_triggered=False,
            latency_ms=latency_ms,
        )

    async def stream_query(self, request: ChatQueryRequest) -> AsyncIterator[StreamChunkPayload]:
        """Stream RAG response tokens with fallback handling."""
        fallback_triggered, chunks, fallback_msg = await self._retrieve_and_evaluate(
            request.query, request.top_k, request.score_threshold
        )

        if fallback_triggered:
            # Yield fallback message as single stream delta
            yield StreamChunkPayload(text=fallback_msg, done=False, fallback_triggered=True)
            yield StreamChunkPayload(
                text=None,
                done=True,
                fallback_triggered=True,
                retrieved_chunks=chunks,
            )
            return

        context_text, budgeted_chunks = assemble_context_with_budget(
            chunks, self.settings.max_context_tokens
        )

        prompt = f"""[참고 문서]
{context_text}

[사용자 질문]
{request.query}

위 참고 문서를 바탕으로 사용자 질문에 답변하십시오."""

        # Stream generated tokens
        async for token in self.llm_client.stream(prompt=prompt, system_prompt=SYSTEM_PROMPT_TEMPLATE):
            yield StreamChunkPayload(text=token, done=False, fallback_triggered=False)

        # Final chunk with metadata
        yield StreamChunkPayload(
            text=None,
            done=True,
            fallback_triggered=False,
            retrieved_chunks=budgeted_chunks,
        )

