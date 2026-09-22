"""Read/utility HTTP API — stats, conversation browsing, search, ownership,
raw-record reads, and the external-embedding handoff.

These are the JSON endpoints the UI and the extraction pipeline call directly
(the MCP tools in mcp_tools.py cover the agent-facing surface).
"""

import json

import sqlite_vec

from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import DB_PATH, EXCLUDED_OWNERS, EMBED_MODEL, EMBED_DIM, EMBED_MAX_CHARS
from .db import db, _db_lock
from .search import _date_bounds, _fts_hits, _vec_hits


async def api_attribute(request: Request):
    """Batch-set conversation ownership. Body: {"owner": "...", "keys": [...]}.
    Owner changes take effect immediately in search/retrieval via the
    privacy partition (EXCLUDED_OWNERS)."""
    body = await request.json()
    owner = body.get("owner", "").strip()
    keys = body.get("keys", [])
    if not owner or not isinstance(keys, list):
        return JSONResponse({"error": "need owner and keys[]"}, status_code=400)
    with _db_lock:
        conn = db()
        try:
            updated = 0
            for key in keys:
                cur = conn.execute("UPDATE conversations SET owner=? WHERE key=?", (owner, key))
                updated += cur.rowcount
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "owner": owner, "requested": len(keys), "updated": updated})


async def api_stats(_request: Request):
    """Aggregate stats for the UI: totals, per-source, monthly density, embedding coverage."""
    conn = db()
    try:
        sources = [dict(r) for r in conn.execute(
            "SELECT source, COUNT(*) AS conversations, SUM(message_count) AS messages, "
            "MIN(created_at) AS first, MAX(updated_at) AS last FROM conversations GROUP BY source")]
        monthly = [dict(r) for r in conn.execute(
            "SELECT substr(created_at, 1, 7) AS month, source, COUNT(*) AS n "
            "FROM conversations WHERE created_at IS NOT NULL GROUP BY month, source ORDER BY month")]
        embeddable = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE text != ''").fetchone()["n"]
        embedded = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()["n"]
    finally:
        conn.close()
    return JSONResponse({"sources": sources, "monthly": monthly,
                         "embedded": embedded, "embeddable": embeddable})


async def api_conversations(request: Request):
    """Paged conversation metadata for the UI list panel."""
    p = request.query_params
    limit = min(int(p.get("limit", "50")), 200)
    offset = int(p.get("offset", "0"))
    sql = "SELECT key, source, title, created_at, updated_at, message_count FROM conversations WHERE owner NOT IN ({})".format(
        ",".join("?" * len(EXCLUDED_OWNERS)) or "''")
    params = [*EXCLUDED_OWNERS]
    for field, clause in (("source", " AND source = ?"), ("after", " AND updated_at >= ?"),
                           ("before", " AND created_at <= ?")):
        if p.get(field):
            sql += clause
            params.append(p[field])
    sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    conn = db()
    try:
        rows = [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()
    return JSONResponse({"conversations": rows})


async def api_search(request: Request):
    """Structured hybrid search for the UI (same engine as the MCP tool)."""
    p = request.query_params
    query = p.get("q", "").strip()
    if not query:
        return JSONResponse({"results": []})
    limit = min(int(p.get("limit", "12")), 25)
    after, before = _date_bounds(p.get("after", ""), p.get("before", ""))
    source = p.get("source", "")
    conn = db()
    try:
        fts = _fts_hits(conn, query, source, 25, after, before)
        try:
            vec = _vec_hits(conn, query, source, 25, after, before)
        except Exception:
            vec = []
        scores: dict = {}
        for rank, key in enumerate(fts):
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        for rank, key in enumerate(vec):
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
        results = []
        for conv_key, native_id in ranked:
            r = conn.execute(
                """SELECT m.role, m.created_at, substr(m.text, 1, 400) AS excerpt, c.title, c.source
                   FROM messages m JOIN conversations c ON c.key = m.conv_key
                   WHERE m.conv_key = ? AND m.native_id = ?""",
                (conv_key, native_id)).fetchone()
            if not r:
                continue
            results.append({
                "conv_key": conv_key, "native_id": native_id, "role": r["role"],
                "created_at": r["created_at"], "excerpt": r["excerpt"],
                "title": r["title"], "source": r["source"],
                "via": ("both" if (conv_key, native_id) in fts and (conv_key, native_id) in vec
                        else "keyword" if (conv_key, native_id) in fts else "semantic"),
            })
    finally:
        conn.close()
    return JSONResponse({"results": results})


async def health(_request: Request):
    return JSONResponse({"ok": True, "db": DB_PATH.exists()})


async def get_record(request: Request):
    """Return the full stored schema-v1 record for a conversation key
    (source:native_id). Powers the extraction pipeline's replay reads."""
    key = request.query_params.get("key", "")
    conn = db()
    try:
        row = conn.execute("SELECT record, owner FROM conversations WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    if row["owner"] in EXCLUDED_OWNERS:
        return JSONResponse({"error": "excluded by privacy partition"}, status_code=403)
    return JSONResponse(json.loads(row["record"]))


async def embed_pending(request: Request):
    """Hand out unembedded messages so an external worker (PC GPU/CPU) can
    bulk-embed the backlog. Same model required: BAAI/bge-small-en-v1.5."""
    limit = min(int(request.query_params.get("limit", "256")), 1024)
    conn = db()
    try:
        rows = conn.execute("""
            SELECT m.conv_key, m.native_id, substr(m.text, 1, ?) AS text FROM messages m
            LEFT JOIN vec_map v ON v.conv_key = m.conv_key AND v.native_id = m.native_id
            WHERE v.vec_id IS NULL AND m.text != '' LIMIT ?
        """, (EMBED_MAX_CHARS, limit)).fetchall()
    finally:
        conn.close()
    return JSONResponse({"model": EMBED_MODEL, "dim": EMBED_DIM,
                         "items": [dict(r) for r in rows]})


async def embed_batch_in(request: Request):
    """Accept externally computed vectors. INSERT OR IGNORE dedups against the
    NAS-side worker racing on the same rows."""
    items = await request.json()
    accepted = 0
    with _db_lock:
        conn = db()
        try:
            for it in items:
                emb = it.get("embedding") or []
                if len(emb) != EMBED_DIM:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO vec_map (conv_key, native_id) VALUES (?,?)",
                    (it["conv_key"], it["native_id"]),
                )
                if cur.lastrowid:
                    conn.execute(
                        "INSERT INTO message_vecs (rowid, embedding) VALUES (?,?)",
                        (cur.lastrowid, sqlite_vec.serialize_float32([float(x) for x in emb])),
                    )
                    accepted += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "accepted": accepted})
