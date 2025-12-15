from typing import Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from backend.middleware.tracing import TRACE_ID_CTX_VAR


def status_to_code(status_code: int) -> str:
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
    }
    return mapping.get(status_code, f"HTTP_{status_code}")


async def handle_http_exception(request: Request, exc: HTTPException):
    trace_id = TRACE_ID_CTX_VAR.get()
    detail: Any = exc.detail
    message = detail if isinstance(detail, str) else "HTTP error"
    body = {"code": status_to_code(exc.status_code), "message": message, "trace_id": trace_id}
    if detail is not None:
        body["details"] = detail
    return JSONResponse(status_code=exc.status_code, content=body)


async def handle_unhandled_exception(request: Request, exc: Exception):
    trace_id = TRACE_ID_CTX_VAR.get()
    body = {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred",
        "details": str(exc),
        "trace_id": trace_id,
    }
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)
