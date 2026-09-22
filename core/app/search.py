"""Hybrid retrieval primitives — keyword (FTS5) + semantic (vector) lookups.

Both return `(conv_key, native_id)` tuples so callers (the /api/search route and
the search_memory MCP tool) can fuse them with reciprocal-rank fusion. The
privacy partition (EXCLUDED_OWNERS) is enforced here so nothing sealed can leak
into a result set.
"""

import sqlite3

import sqlite_vec

from .config import EXCLUDED_OWNERS
from .embeddings import embedder


def _date_bounds(after: str, before: str):
    """Normalize date-only inputs to full-day ISO bounds; ISO strings compare
    lexicographically at this granularity."""
    a = f"{after}T00:00:00" if len(after) == 10 else after
    b = f"{before}T23:59:59.999999" if len(before) == 10 else before
    return a, b


def _fts_hits(conn, query: str, source: str, k: int, after: str = "", before: str = "") -> list:
    sql = """
        SELECT m.conv_key, m.native_id FROM messages_fts
        JOIN messages m ON m.conv_key = messages_fts.conv_key AND m.native_id = messages_fts.native_id
        JOIN conversations c ON c.key = m.conv_key
        WHERE messages_fts MATCH ? AND c.owner NOT IN ({owners})
    """.format(owners=",".join("?" * len(EXCLUDED_OWNERS)) or "''")
    params = [query, *EXCLUDED_OWNERS]
    if source:
        sql += " AND c.source = ?"
        params.append(source)
    if after:
        sql += " AND m.created_at >= ?"
        params.append(after)
    if before:
        sql += " AND m.created_at <= ?"
        params.append(before)
    sql += " ORDER BY bm25(messages_fts) LIMIT ?"
    params.append(k)
    try:
        return [(r["conv_key"], r["native_id"]) for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []  # FTS query-syntax error (stray quotes etc.) — vector side still answers


def _vec_hits(conn, query: str, source: str, k: int, after: str = "", before: str = "") -> list:
    qvec = list(embedder().embed([query]))[0]
    rows = conn.execute(
        "SELECT rowid, distance FROM message_vecs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (sqlite_vec.serialize_float32([float(x) for x in qvec]), k * 4),
    ).fetchall()
    out = []
    for r in rows:
        m = conn.execute(
            """SELECT v.conv_key, v.native_id, c.owner, c.source, msg.created_at FROM vec_map v
               JOIN conversations c ON c.key = v.conv_key
               JOIN messages msg ON msg.conv_key = v.conv_key AND msg.native_id = v.native_id
               WHERE v.vec_id = ?""",
            (r["rowid"],),
        ).fetchone()
        if not m or m["owner"] in EXCLUDED_OWNERS:
            continue
        if source and m["source"] != source:
            continue
        if after and (m["created_at"] or "") < after:
            continue
        if before and (m["created_at"] or "~") > before:
            continue
        out.append((m["conv_key"], m["native_id"]))
        if len(out) >= k:
            break
    return out
