"""Common standard response schemas."""

from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    """Standard successful API response envelope."""

    success: bool = Field(default=True, description="Success status")
    data: DataT = Field(description="Response data payload")
    request_id: Optional[str] = Field(default=None, description="Unique request tracing ID")


class ApiErrorDetail(BaseModel):
    """Structured error information."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    request_id: Optional[str] = Field(default=None, description="Unique request tracing ID")
    details: Optional[dict[str, Any]] = Field(default=None, description="Additional context or validation details")


class ApiErrorResponse(BaseModel):
    """Standard error API response envelope matching rules.md."""

    success: bool = Field(default=False, description="Failure status")
    error: ApiErrorDetail = Field(description="Error details")

