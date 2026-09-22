"""Extraction ledger — the job queue + curation surface for the knowledge layer.

Workers claim conversations in global chronological order (life-order replay),
submit per-chunk mention/assertion extractions, and mark them complete. The
search/curation endpoints here let you audit and surgically correct what was
extracted (find, delete, purge) before it's promoted into the graph.
"""

import json
from datetime import datetime, timezone

from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import EXCLUDED_OWNERS
from .db import db, _db_lock


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def ledger_seed(request: Request):
    """Seed extraction jobs. Body: {"keys": [...]} for a pilot subset, or
    {"all": true} for every non-excluded conversation."""
    body = await request.json()
    with _db_lock:
        conn = db()
        try:
            if body.get("all"):
                keys = [r["key"] for r in conn.execute(
                    "SELECT key FROM conversations WHERE owner NOT IN ({})".format(
                        ",".join("?" * len(EXCLUDED_OWNERS)) or "''"), [*EXCLUDED_OWNERS])]
            else:
                keys = body.get("keys", [])
            added = 0
            for k in keys:
                cur = conn.execute("INSERT OR IGNORE INTO ledger_jobs (conv_key) VALUES (?)", (k,))
                added += cur.rowcount
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "seeded": added, "total_requested": len(keys)})


async def ledger_claim(request: Request):
    """Atomically claim the next pending job in GLOBAL CHRONOLOGICAL order
    (life-order replay). Body: {"worker": "name"}. Stale claims (>2h) are
    reclaimed."""
    body = await request.json()
    worker = body.get("worker", "anon")
    now_iso = now()
    with _db_lock:
        conn = db()
        try:
            conn.execute(
                "UPDATE ledger_jobs SET status='pending', worker=NULL WHERE status='claimed' "
                "AND claimed_at < datetime('now', '-2 hours')")
            row = conn.execute("""
                SELECT j.conv_key FROM ledger_jobs j
                JOIN conversations c ON c.key = j.conv_key
                WHERE j.status='pending'
                ORDER BY c.created_at ASC LIMIT 1""").fetchone()
            if not row:
                conn.commit()
                return JSONResponse({"done": True})
            conn.execute("UPDATE ledger_jobs SET status='claimed', worker=?, claimed_at=? WHERE conv_key=?",
                         (worker, now_iso, row["conv_key"]))
            conn.commit()
            key = row["conv_key"]
        finally:
            conn.close()
    return JSONResponse({"conv_key": key})


async def ledger_submit(request: Request):
    """Store one chunk's extraction. Body: {conv_key, chunk_idx, engine,
    valid, payload:{mentions, assertions, state}}."""
    b = await request.json()
    payload = b.get("payload") or {}
    with _db_lock:
        conn = db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO extractions (conv_key, chunk_idx, engine, valid, payload, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (b["conv_key"], b["chunk_idx"], b.get("engine"), 1 if b.get("valid", True) else 0,
                 json.dumps(payload, ensure_ascii=False), now()))
            conn.execute("DELETE FROM mentions WHERE conv_key=? AND chunk_idx=?", (b["conv_key"], b["chunk_idx"]))
            conn.execute("DELETE FROM assertions WHERE conv_key=? AND chunk_idx=?", (b["conv_key"], b["chunk_idx"]))
            for m in payload.get("mentions", []):
                conn.execute(
                    "INSERT INTO mentions (conv_key, chunk_idx, name, kind, context, speaker, evidence) VALUES (?,?,?,?,?,?,?)",
                    (b["conv_key"], b["chunk_idx"], m.get("name"), m.get("kind"), m.get("context"),
                     m.get("speaker"), json.dumps(m.get("evidence_ids", []))))
            for a in payload.get("assertions", []):
                conn.execute(
                    "INSERT INTO assertions (conv_key, chunk_idx, type, statement, speaker, rationale, status, acceptance, solves, owner, evidence) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (b["conv_key"], b["chunk_idx"], a.get("type"), a.get("statement"), a.get("speaker"),
                     a.get("rationale"), a.get("status"), a.get("acceptance"), a.get("solves"),
                     a.get("owner"), json.dumps(a.get("evidence_ids", []))))
            conn.execute("UPDATE ledger_jobs SET chunks_done=? WHERE conv_key=?",
                         (b["chunk_idx"] + 1, b["conv_key"]))
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True})


async def ledger_complete(request: Request):
    b = await request.json()
    status = "failed" if b.get("failed") else "done"
    with _db_lock:
        conn = db()
        try:
            conn.execute("UPDATE ledger_jobs SET status=?, completed_at=? WHERE conv_key=?",
                         (status, now(), b["conv_key"]))
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True})


async def ledger_mention_names(request: Request):
    """Aggregated mention names by frequency — the worksheet head. Excludes
    partition-owned conversations."""
    limit = min(int(request.query_params.get("limit", "200")), 1000)
    conn = db()
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT m.name, COUNT(*) AS mentions,
                   COUNT(DISTINCT m.conv_key) AS conversations,
                   GROUP_CONCAT(DISTINCT m.kind) AS kinds
            FROM mentions m
            JOIN conversations c ON c.key = m.conv_key
            WHERE c.owner NOT IN ({owners}) AND m.name IS NOT NULL AND m.name != ''
            GROUP BY LOWER(m.name)
            ORDER BY mentions DESC LIMIT ?
        """.format(owners=",".join("?" * len(EXCLUDED_OWNERS)) or "''"),
            [*EXCLUDED_OWNERS, limit])]
    finally:
        conn.close()
    return JSONResponse({"names": rows})


async def ledger_mentions_by_name(request: Request):
    """All mentions for a name, with context + conversation date, for
    context-signature clustering (Ron-paradox split detection)."""
    name = request.query_params.get("name", "")
    conn = db()
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT m.id, m.context, m.kind, m.speaker, m.conv_key, c.title, c.created_at
            FROM mentions m
            JOIN conversations c ON c.key = m.conv_key
            WHERE LOWER(m.name) = LOWER(?) AND c.owner NOT IN ({owners})
        """.format(owners=",".join("?" * len(EXCLUDED_OWNERS)) or "''"),
            [name, *EXCLUDED_OWNERS])]
    finally:
        conn.close()
    return JSONResponse({"name": name, "count": len(rows), "mentions": rows})


async def ledger_search(request: Request):
    """Find assertions/mentions by keyword in their text — for auditing and
    surgical correction. Returns rows with ids + provenance."""
    q = request.query_params.get("q", "").strip()
    if not q:
        return JSONResponse({"error": "q required"}, status_code=400)
    like = f"%{q}%"
    conn = db()
    try:
        assertions = [dict(r) for r in conn.execute("""
            SELECT a.id, a.type, a.statement, a.speaker, a.conv_key, a.chunk_idx,
                   a.evidence, c.title, c.created_at
            FROM assertions a JOIN conversations c ON c.key = a.conv_key
            WHERE a.statement LIKE ? ORDER BY c.created_at LIMIT 100
        """, (like,))]
        mentions = [dict(r) for r in conn.execute("""
            SELECT m.id, m.name, m.kind, m.context, m.speaker, m.conv_key, m.chunk_idx,
                   m.evidence, c.title, c.created_at
            FROM mentions m JOIN conversations c ON c.key = m.conv_key
            WHERE m.name LIKE ? OR m.context LIKE ? ORDER BY c.created_at LIMIT 100
        """, (like, like))]
    finally:
        conn.close()
    return JSONResponse({"assertions": assertions, "mentions": mentions})


async def ledger_delete(request: Request):
    """Delete specific ledger rows. Body: {assertions:[ids], mentions:[ids]}."""
    b = await request.json()
    a_ids = b.get("assertions", [])
    m_ids = b.get("mentions", [])
    with _db_lock:
        conn = db()
        try:
            da = conn.executemany("DELETE FROM assertions WHERE id=?", [(i,) for i in a_ids]).rowcount
            dm = conn.executemany("DELETE FROM mentions WHERE id=?", [(i,) for i in m_ids]).rowcount
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "deleted_assertions": len(a_ids), "deleted_mentions": len(m_ids)})


async def ledger_conversations_for_name(request: Request):
    """Which conversations contain a given entity, ranked by mention density —
    to identify which source threads to seal."""
    name = request.query_params.get("name", "")
    like = f"%{name}%"
    conn = db()
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT x.conv_key, c.title, c.source, c.created_at, c.message_count, x.hits
            FROM (
                SELECT conv_key, COUNT(*) AS hits FROM mentions WHERE name LIKE ? GROUP BY conv_key
                UNION ALL
                SELECT conv_key, COUNT(*) AS hits FROM assertions WHERE statement LIKE ? GROUP BY conv_key
            ) x JOIN conversations c ON c.key = x.conv_key
            GROUP BY x.conv_key ORDER BY SUM(x.hits) DESC
        """, (like, like))]
    finally:
        conn.close()
    return JSONResponse({"name": name, "conversations": rows})


async def ledger_purge(request: Request):
    """Remove ledger rows by entity name and/or by conversation. Body:
    {name: "...", conv_keys: [...]}. Name match deletes assertions whose
    statement mentions it + mentions of it; conv_keys deletes all rows from
    those conversations."""
    b = await request.json()
    name = b.get("name", "").strip()
    conv_keys = b.get("conv_keys", [])
    with _db_lock:
        conn = db()
        try:
            da = dm = 0
            if name:
                like = f"%{name}%"
                da += conn.execute("DELETE FROM assertions WHERE statement LIKE ?", (like,)).rowcount
                dm += conn.execute("DELETE FROM mentions WHERE name LIKE ? OR context LIKE ?", (like, like)).rowcount
            for k in conv_keys:
                da += conn.execute("DELETE FROM assertions WHERE conv_key=?", (k,)).rowcount
                dm += conn.execute("DELETE FROM mentions WHERE conv_key=?", (k,)).rowcount
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "deleted_assertions": da, "deleted_mentions": dm})


async def ledger_status(_request: Request):
    conn = db()
    try:
        by = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM ledger_jobs GROUP BY status")}
        counts = {
            "mentions": conn.execute("SELECT COUNT(*) AS n FROM mentions").fetchone()["n"],
            "assertions": conn.execute("SELECT COUNT(*) AS n FROM assertions").fetchone()["n"],
            "chunks": conn.execute("SELECT COUNT(*) AS n FROM extractions").fetchone()["n"],
            "invalid_chunks": conn.execute("SELECT COUNT(*) AS n FROM extractions WHERE valid=0").fetchone()["n"],
        }
    finally:
        conn.close()
    return JSONResponse({"jobs": by, "ledger": counts})
