"""Browser pages — the 3D knowledge graph and the live edge-typing view.

The HTML/JS lives in ./templates as static files; these handlers just inject the
access token (passed as ?token= so a plain clickable link, with no custom
headers, can reach the read-only pages) and serve them.
"""

from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response

_TEMPLATES = Path(__file__).parent / "templates"
_GRAPH_HTML = (_TEMPLATES / "graph.html").read_text(encoding="utf-8")
_LIVE_HTML = (_TEMPLATES / "live.html").read_text(encoding="utf-8")


async def graph_page(request: Request):
    """Browser force-graph visualization of the knowledge graph. Clickable link,
    token via ?token= query param. 3d toggle via ?d=2 for the flat view."""
    tok = request.query_params.get("token", "")
    html = _GRAPH_HTML.replace("__TOKEN__", tok)
    return Response(html, media_type="text/html")


async def live_page(request: Request):
    """Browser-viewable live progress page (auto-refreshes). Clickable link,
    no scripts to run — token passes via ?token= query param."""
    tok = request.query_params.get("token", "")
    html = _LIVE_HTML.replace("__TOKEN__", tok)
    return Response(html, media_type="text/html")
