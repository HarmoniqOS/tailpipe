"""Bearer-token gate.

Every request needs `Authorization: Bearer <INGEST_TOKEN>`, except /health.
Browser-friendly: the token may also arrive as a ?token= query param, so a plain
clickable link (no custom headers) can reach read-only pages like /live and
/graph. If INGEST_TOKEN is unset the gate is open (local dev only).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import INGEST_TOKEN


class TokenAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/health" or not INGEST_TOKEN:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        # browser-friendly: allow the token as a query param so a plain clickable
        # link (no custom headers) can reach read-only pages like /live
        qtoken = request.query_params.get("token", "")
        if auth != f"Bearer {INGEST_TOKEN}" and qtoken != INGEST_TOKEN:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)
