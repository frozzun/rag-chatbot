"""Structured JSON Logging with PII and API Key Masking."""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

# Regex patterns for masking sensitive information
_SENSITIVE_PATTERNS = [
    (re.compile(r"(AIza[0-9A-Za-z-_]{35})"), "[MASKED_API_KEY]"),
    (re.compile(r"(sk-[a-zA-Z0-9]{20,})"), "[MASKED_API_KEY]"),
    (re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"), "[MASKED_EMAIL]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), r"\1[MASKED_TOKEN]"),
]


def mask_sensitive_data(text: str) -> str:
    """Mask known sensitive token and PII patterns in string."""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_sensitive_data(record.getMessage()),
        }

        # Include custom attributes if present
        for key in ("request_id", "latency_ms", "status_code", "path", "method", "client_ip"):
            val = getattr(record, key, None)
            if val is not None:
                log_payload[key] = val

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, ensure_ascii=False)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure root logger with structured JSON formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(stream_handler)

    # Suppress verbose noisy third-party loggers
    logging.getLogger("uvicorn.access").handlers = [stream_handler]
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger

