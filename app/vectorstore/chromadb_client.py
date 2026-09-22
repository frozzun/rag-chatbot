"""ChromaDB Vector Store wrapper with anyio thread offloading."""

import logging
import os
from typing import Any, Optional
import anyio
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import VectorStoreException

logger = logging.getLogger("app.vectorstore")


class ChromaVectorStore:
    """Thread-safe, async-friendly wrapper for ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "rag_documents",
        mode: str = "embedded",
        server_host: str = "localhost",
        server_port: int = 8000,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.mode = mode
        self.server_host = server_host
        self.server_port = server_port
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    def initialize_sync(self) -> None:
        """Synchronous initialization called during lifespan startup."""
        try:
            if self.mode == "http":
                logger.info(
                    f"Connecting to standalone ChromaDB server at {self.server_host}:{self.server_port}"
                )
                self._client = chromadb.HttpClient(
                    host=self.server_host,
                    port=self.server_port,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            else:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

            # Use cosine similarity: distance is 1 - cosine_similarity
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB ({self.mode} mode) ready for collection '{self.collection_name}'"
            )
        except Exception as exc:
            logger.error(f"Failed to initialize ChromaDB: {exc}", exc_info=True)
            raise VectorStoreException(
                message=f"ChromaDB 초기화에 실패했습니다: {exc}",
                details={"mode": self.mode, "persist_directory": self.persist_directory},
            ) from exc

    def close(self) -> None:
        """Close client and release resources during lifespan shutdown."""
        logger.info("Shutting down ChromaDB client session")
        self._collection = None
        self._client = None

    # ---- Internal Sync Operations (To be run in worker threads) ----

    def _add_chunks_sync(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if self._collection is None:
            raise VectorStoreException("ChromaDB 컬렉션이 초기화되지 않았습니다.")
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def _find_by_hash_sync(self, content_hash: str) -> Optional[dict[str, Any]]:
        if self._collection is None:
            raise VectorStoreException("ChromaDB 컬렉션이 초기화되지 않았습니다.")
        results = self._collection.get(
            where={"content_hash": content_hash},
            limit=1,
            include=["metadatas"],
        )
        if results and results.get("ids") and len(results["ids"]) > 0:
            return {
                "id": results["ids"][0],
                "metadata": results["metadatas"][0] if results.get("metadatas") else {},
            }
        return None

    def _query_sync(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> dict[str, Any]:
        if self._collection is None:
            raise VectorStoreException("ChromaDB 컬렉션이 초기화되지 않았습니다.")
        count = self._collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

        actual_n = min(n_results, count)
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "distances", "metadatas"],
        )

    def _count_sync(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def _delete_by_document_id_sync(self, document_id: str) -> int:
        if self._collection is None:
            return 0
        existing = self._collection.get(where={"document_id": document_id})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    # ---- Asynchronous Public Methods with anyio offloading ----

    async def add_chunks_async(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add chunks to ChromaDB offloaded to a worker thread."""
        try:
            await anyio.to_thread.run_sync(
                self._add_chunks_sync, ids, embeddings, documents, metadatas
            )
        except Exception as exc:
            logger.error(f"Error adding chunks to ChromaDB: {exc}", exc_info=True)
            raise VectorStoreException(f"벡터 스토어에 청크를 추가하는 중 오류가 발생했습니다: {exc}") from exc

    async def find_by_hash_async(self, content_hash: str) -> Optional[dict[str, Any]]:
        """Check if document with given SHA-256 hash exists in worker thread."""
        try:
            return await anyio.to_thread.run_sync(self._find_by_hash_sync, content_hash)
        except Exception as exc:
            logger.error(f"Error finding hash in ChromaDB: {exc}", exc_info=True)
            raise VectorStoreException(f"문서 해시 검색 중 오류가 발생했습니다: {exc}") from exc

    async def query_async(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> dict[str, Any]:
        """Query nearest vector neighbors offloaded to a worker thread."""
        try:
            return await anyio.to_thread.run_sync(self._query_sync, query_embedding, n_results)
        except Exception as exc:
            logger.error(f"Error querying ChromaDB: {exc}", exc_info=True)
            raise VectorStoreException(f"벡터 유사도 검색 중 오류가 발생했습니다: {exc}") from exc

    async def count_async(self) -> int:
        """Count total vectors in collection in worker thread."""
        return await anyio.to_thread.run_sync(self._count_sync)

    async def delete_by_document_id_async(self, document_id: str) -> int:
        """Delete all chunks belonging to document_id in worker thread."""
        return await anyio.to_thread.run_sync(self._delete_by_document_id_sync, document_id)
