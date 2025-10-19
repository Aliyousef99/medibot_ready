import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

TRACE_ID_CTX_VAR: ContextVar[str] = ContextVar("trace_id", default="")


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a trace_id to every request and response.
    The trace_id is also stored in a context variable for logging.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = str(uuid.uuid4())
        TRACE_ID_CTX_VAR.set(trace_id)
        request.state.trace_id = trace_id

        response = await call_next(request)

        # Propagate trace id to client; header names are case-insensitive
        response.headers["x-trace-id"] = trace_id
        return response
