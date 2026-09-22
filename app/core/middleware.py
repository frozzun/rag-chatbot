"""Middleware for request ID propagation and structured latency logging."""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.middleware")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Generates X-Request-ID and logs request lifecycle with latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            # Error handler will produce response, but log here if unhandled escapes
            logger.error(
                f"Unhandled exception during request processing: {exc}",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": status_code,
                },
                exc_info=True,
            )
            raise
        finally:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            # Skip high-frequency health probes in normal logs if desired, or log with INFO
            logger.info(
                f"Completed {request.method} {request.url.path} with status {status_code} in {latency_ms}ms",
                extra={
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                    "status_code": status_code,
                    "path": request.url.path,
                    "method": request.method,
                    "client_ip": request.client.host if request.client else None,
                },
            )

