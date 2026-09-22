"""Document ingestion service with SHA-256 idempotency and async chunking."""

import hashlib
import logging
import uuid
from typing import Any, Optional
import anyio

from app.core.config import Settings
from app.core.exceptions import InvalidRequestError
from app.schemas.document import DocumentIngestResponse
from app.services.embedding_client import BaseEmbeddingClient
from app.vectorstore.chromadb_client import ChromaVectorStore

logger = logging.getLogger("app.document")


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def recursive_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """CPU-bound text chunking logic designed to run in worker thread."""
    if not text.strip():
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


class DocumentService:
    """Handles idempotent document indexing, chunking, and embedding."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_client: BaseEmbeddingClient,
        settings: Settings,
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.settings = settings

    async def ingest_document(
        self,
        title: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DocumentIngestResponse:
        """Idempotently ingest a text document into the vector store."""
        if not content or not content.strip():
            raise InvalidRequestError("문서 내용이 비어 있습니다.")

        # 1. Compute SHA-256 hash
        content_hash = compute_sha256(content)

        # 2. Check for duplicate ingestion (Idempotency guarantee)
        existing = await self.vector_store.find_by_hash_async(content_hash)
        if existing:
            doc_id = existing["metadata"].get("document_id", existing["id"])
            logger.info(
                f"Document with hash '{content_hash[:8]}...' already exists as ID '{doc_id}'. Skipping re-indexing."
            )
            return DocumentIngestResponse(
                document_id=str(doc_id),
                title=title,
                content_hash=content_hash,
                chunks_count=int(existing["metadata"].get("total_chunks", 1)),
                is_duplicate=True,
                message="이미 동일한 내용의 문서가 인덱싱되어 있습니다 (멱등성 보장).",
            )

        # 3. Offload CPU-bound chunking to worker thread
        chunks = await anyio.to_thread.run_sync(
            recursive_chunk_text,
            content,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        if not chunks:
            raise InvalidRequestError("청킹된 유효한 텍스트 블록이 없습니다.")

        document_id = f"doc-{uuid.uuid4().hex[:12]}"
        total_chunks = len(chunks)

        logger.info(
            f"Ingesting document '{title}' ({document_id}) into {total_chunks} chunks (hash={content_hash[:8]})."
        )

        # 4. Generate embeddings asynchronously
        embeddings = await self.embedding_client.embed_texts(chunks)

        # 5. Prepare chunk metadata and IDs
        ids: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []
        user_meta = metadata or {}

        for idx, _ in enumerate(chunks):
            chunk_id = f"{document_id}#c{idx}"
            ids.append(chunk_id)
            meta = {
                **user_meta,
                "document_id": document_id,
                "title": title,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "content_hash": content_hash,
            }
            chunk_metadatas.append(meta)

        # 6. Store in ChromaDB offloaded to worker thread
        await self.vector_store.add_chunks_async(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas,
        )

        logger.info(f"Successfully indexed document '{document_id}' with {total_chunks} chunks.")

        return DocumentIngestResponse(
            document_id=document_id,
            title=title,
            content_hash=content_hash,
            chunks_count=total_chunks,
            is_duplicate=False,
            message="문서가 성공적으로 청킹 및 인덱싱되었습니다.",
        )

