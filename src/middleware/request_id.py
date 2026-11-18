"""Request ID middleware (mints uuid7 + echoes X-Request-ID)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Set ``request.state.request_id`` and echo ``X-Request-ID``."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get("x-request-id", "").strip()
        request_id = incoming or str(uuid.uuid7())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
