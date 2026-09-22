"""
Tailpipe Memory Core — app assembly.

One container, the whole spine:
  POST /ingest   — receives Collector schema-v1 records into SQLite (raw + FTS index)
  /mcp           — MCP server (streamable HTTP): search_memory, get_conversation,
                   memory_status + knowledge-graph tools
  GET  /graph    — interactive 3D knowledge-graph visualization
  GET  /health   — liveness

Storage layout (bind-mount DATA_DIR, default /data):
  /data/index/memory.db   — SQLite: conversations, messages, FTS5, vectors, graph
  /data/raw/              — append-only NDJSON of every ingested record (canonical)

This module only wires things together. The pieces live in:
  config      — environment knobs
  db          — connection + schema
  embeddings  — embedder + background embed worker
  search      — hybrid retrieval primitives
  ingest      — /ingest + record storage
  ledger      — extraction job queue + curation endpoints
  graph       — entities, edges, typing, communities, viz feed
  pages       — /graph + /live browser pages
  api         — read/utility JSON API
  auth        — bearer-token middleware
  mcp_tools   — MCP server + agent-facing tools
"""

import contextlib
import threading

from starlette.applications import Starlette
from starlette.routing import Mount, Route

from .db import init_db
from .embeddings import embed_worker, _embed_stop
from .auth import TokenAuth
from .mcp_tools import mcp
from .ingest import ingest
from .ledger import (
    ledger_seed, ledger_claim, ledger_submit, ledger_complete, ledger_status,
    ledger_mention_names, ledger_mentions_by_name, ledger_search, ledger_delete,
    ledger_conversations_for_name, ledger_purge,
)
from .graph import (
    entities_build, entities_list, edges_build, graph_edges_dump, edges_for_typing,
    set_edge_types, typing_status, graph_full, rename_community, graph_set_communities,
)
from .pages import graph_page, live_page
from .api import (
    api_attribute, api_stats, api_conversations, api_search, health, get_record,
    embed_pending, embed_batch_in,
)

init_db()


@contextlib.asynccontextmanager
async def lifespan(_app):
    # FastMCP's streamable HTTP transport needs its session manager task
    # group running for the life of the server. The embed worker runs for the
    # same lifetime, continuously vectorizing unembedded messages.
    worker = threading.Thread(target=embed_worker, daemon=True, name="embed-worker")
    worker.start()
    try:
        async with mcp.session_manager.run():
            yield
    finally:
        _embed_stop.set()


app = Starlette(
    routes=[
        Route("/ingest", ingest, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/record", get_record, methods=["GET"]),
        Route("/api/attribute", api_attribute, methods=["POST"]),
        Route("/api/ledger/seed", ledger_seed, methods=["POST"]),
        Route("/api/ledger/claim", ledger_claim, methods=["POST"]),
        Route("/api/ledger/submit", ledger_submit, methods=["POST"]),
        Route("/api/ledger/complete", ledger_complete, methods=["POST"]),
        Route("/api/ledger/status", ledger_status, methods=["GET"]),
        Route("/api/ledger/mention_names", ledger_mention_names, methods=["GET"]),
        Route("/api/ledger/mentions_by_name", ledger_mentions_by_name, methods=["GET"]),
        Route("/api/ledger/search", ledger_search, methods=["GET"]),
        Route("/api/ledger/delete", ledger_delete, methods=["POST"]),
        Route("/api/ledger/conversations_for_name", ledger_conversations_for_name, methods=["GET"]),
        Route("/api/ledger/purge", ledger_purge, methods=["POST"]),
        Route("/api/entities/build", entities_build, methods=["POST"]),
        Route("/api/entities/list", entities_list, methods=["GET"]),
        Route("/api/entities/build_edges", edges_build, methods=["POST"]),
        Route("/api/graph/edges", graph_edges_dump, methods=["GET"]),
        Route("/api/graph/edges_for_typing", edges_for_typing, methods=["GET"]),
        Route("/api/graph/set_edge_types", set_edge_types, methods=["POST"]),
        Route("/api/graph/typing_status", typing_status, methods=["GET"]),
        Route("/api/graph/full", graph_full, methods=["GET"]),
        Route("/api/graph/rename_community", rename_community, methods=["POST"]),
        Route("/graph", graph_page, methods=["GET"]),
        Route("/live", live_page, methods=["GET"]),
        Route("/api/graph/set_communities", graph_set_communities, methods=["POST"]),
        Route("/api/stats", api_stats, methods=["GET"]),
        Route("/api/conversations", api_conversations, methods=["GET"]),
        Route("/api/search", api_search, methods=["GET"]),
        Route("/embed_pending", embed_pending, methods=["GET"]),
        Route("/embed_batch", embed_batch_in, methods=["POST"]),
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)
app.add_middleware(TokenAuth)
