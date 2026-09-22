"""Custom business exceptions for consistent error handling."""

from typing import Any, Optional


class AppException(Exception):
    """Base application exception with error code, message, and HTTP status code."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentNotFoundError(AppException):
    """Raised when requested document or chunk cannot be found."""

    def __init__(self, message: str = "문서를 찾을 수 없습니다.", details: Optional[dict[str, Any]] = None):
        super().__init__(
            code="DOCUMENT_NOT_FOUND",
            message=message,
            status_code=404,
            details=details,
        )


class DuplicateDocumentError(AppException):
    """Raised when a document with identical hash already exists."""

    def __init__(self, message: str = "동일한 내용의 문서가 이미 등록되어 있습니다.", details: Optional[dict[str, Any]] = None):
        super().__init__(
            code="DUPLICATE_DOCUMENT",
            message=message,
            status_code=409,
            details=details,
        )


class VectorStoreException(AppException):
    """Raised when ChromaDB or vector store operations fail."""

    def __init__(self, message: str = "벡터 스토어 처리 중 오류가 발생했습니다.", details: Optional[dict[str, Any]] = None):
        super().__init__(
            code="VECTOR_STORE_ERROR",
            message=message,
            status_code=503,
            details=details,
        )


class LLMServiceException(AppException):
    """Raised when LLM generation or embedding client fails."""

    def __init__(self, message: str = "LLM 서비스 호출 중 오류가 발생했습니다.", details: Optional[dict[str, Any]] = None):
        super().__init__(
            code="LLM_SERVICE_ERROR",
            message=message,
            status_code=502,
            details=details,
        )


class ContextOverflowException(AppException):
    """Raised when input query or context exceeds processing limits."""

    def __init__(self, message: str = "컨텍스트 윈도우 한도를 초과했습니다.", details: Optional[dict[str, Any]] = None):
        super().__init__(
            code="CONTEXT_OVERFLOW",
            message=message,
            status_code=400,
            details=details,
        )


class InvalidRequestError(AppException):
    """Raised for general invalid input validation."""

    def __init__(self, message: str = "잘못된 요청입니다.", details: Optional[dict[str, Any]] = None):
        super().__init__(
            code="INVALID_REQUEST",
            message=message,
            status_code=400,
            details=details,
        )

