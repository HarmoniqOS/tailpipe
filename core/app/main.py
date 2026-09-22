"""
Tailpipe Memory Core.

One container, the whole spine:
  POST /ingest   — receives Collector schema-v1 records into SQLite (raw + FTS index)
  /mcp           — MCP server (streamable HTTP): search_memory, get_conversation,
                   memory_status + knowledge-graph tools
  GET  /graph    — interactive 3D knowledge-graph visualization
  GET  /health   — liveness

Storage layout (bind-mount DATA_DIR, default /data):
  /data/index/memory.db   — SQLite: conversations, messages, FTS5, vectors, graph
  /data/raw/              — append-only NDJSON of every ingested record (canonical)
"""

import json
import os
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

import sqlite_vec

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "index" / "memory.db"
RAW_DIR = DATA_DIR / "raw"
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")

# Embeddings: small CPU model, same model for corpus and queries.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ONNX, N100-friendly
EMBED_DIM = 384
EMBED_MAX_CHARS = 2000                   # embed the same projection FTS indexes
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(DATA_DIR / "index" / "fastembed"))

_embedder = None
_embedder_lock = threading.Lock()


def embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(EMBED_MODEL)
        return _embedder

# Owners excluded from search/retrieval by default (privacy partition).
# "sealed" is always excluded — the tier for personal/sensitive threads that
# stay in the raw archive but are off every search, agent, and graph path.
# Owners whose conversations are kept in the raw archive but excluded from all
# search / agent / graph surfaces (privacy partition). "sealed" is always
# excluded; add others via EXCLUDED_OWNERS in .env (comma-separated).
EXCLUDED_OWNERS = {o.strip() for o in os.environ.get("EXCLUDED_OWNERS", "").split(",") if o.strip()}
EXCLUDED_OWNERS.add("sealed")

_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS conversations (
        key TEXT PRIMARY KEY,               -- source:native_id
        source TEXT NOT NULL,
        native_id TEXT NOT NULL,
        title TEXT,
        model TEXT,
        owner TEXT DEFAULT 'unknown',
        created_at TEXT,
        updated_at TEXT,
        message_count INTEGER,
        ingested_at TEXT,
        record JSON
    );
    CREATE TABLE IF NOT EXISTS messages (
        conv_key TEXT NOT NULL,
        native_id TEXT NOT NULL,
        role TEXT,
        created_at TEXT,
        on_active_path INTEGER,
        content_hash TEXT,
        text TEXT,
        PRIMARY KEY (conv_key, native_id)
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        text, conv_key UNINDEXED, native_id UNINDEXED, tokenize='porter unicode61'
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_key);
    CREATE TABLE IF NOT EXISTS vec_map (
        vec_id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_key TEXT NOT NULL,
        native_id TEXT NOT NULL,
        UNIQUE (conv_key, native_id)
    );
    CREATE TABLE IF NOT EXISTS ledger_jobs (
        conv_key TEXT PRIMARY KEY,
        status TEXT DEFAULT 'pending',      -- pending | claimed | done | failed
        worker TEXT,
        claimed_at TEXT,
        chunks_done INTEGER DEFAULT 0,
        completed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS extractions (
        conv_key TEXT NOT NULL,
        chunk_idx INTEGER NOT NULL,
        engine TEXT,
        valid INTEGER DEFAULT 1,
        payload JSON,                        -- full mention/assertion/state output
        created_at TEXT,
        PRIMARY KEY (conv_key, chunk_idx)
    );
    CREATE TABLE IF NOT EXISTS mentions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_key TEXT, chunk_idx INTEGER,
        name TEXT, kind TEXT, context TEXT, speaker TEXT,
        evidence JSON
    );
    CREATE TABLE IF NOT EXISTS assertions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_key TEXT, chunk_idx INTEGER,
        type TEXT, statement TEXT, speaker TEXT, rationale TEXT,
        status TEXT, acceptance TEXT, solves TEXT, owner TEXT,
        evidence JSON
    );
    CREATE INDEX IF NOT EXISTS idx_mentions_name ON mentions(name);
    CREATE INDEX IF NOT EXISTS idx_assertions_type ON assertions(type);
    CREATE TABLE IF NOT EXISTS entities (
        entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical TEXT UNIQUE, type TEXT, mentions INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS entity_aliases (
        alias TEXT, entity_id INTEGER, PRIMARY KEY (alias, entity_id)
    );
    CREATE TABLE IF NOT EXISTS entity_map (
        mention_id INTEGER PRIMARY KEY, entity_id INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_entity_aliases ON entity_aliases(alias);
    CREATE INDEX IF NOT EXISTS idx_entity_map_eid ON entity_map(entity_id);
    CREATE TABLE IF NOT EXISTS edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        src_entity INTEGER, dst_entity INTEGER, relation TEXT,
        weight INTEGER DEFAULT 1, evidence JSON,
        first_seen TEXT, last_seen TEXT, verified INTEGER DEFAULT 0,
        UNIQUE (src_entity, dst_entity, relation)
    );
    CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_entity);
    CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_entity);
    CREATE TABLE IF NOT EXISTS communities (community_id INTEGER PRIMARY KEY, label TEXT, size INTEGER);
    CREATE TABLE IF NOT EXISTS entity_community (entity_id INTEGER PRIMARY KEY, community_id INTEGER);
    CREATE INDEX IF NOT EXISTS idx_ecomm ON entity_community(community_id);
    """)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS message_vecs USING vec0(embedding float[{EMBED_DIM}])"
    )
    # migrations: edge-typing columns (added 2026-09-19)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(edges)")}
    for col, decl in (("verified_relation", "TEXT"), ("temporality", "TEXT"),
                      ("as_of", "TEXT"), ("direction", "TEXT"), ("verified_at", "TEXT")):
        if col not in have:
            conn.execute(f"ALTER TABLE edges ADD COLUMN {col} {decl}")
    # chunk indexes speed the entity->mention->assertion drill-down join
    conn.execute("CREATE INDEX IF NOT EXISTS idx_assertions_chunk ON assertions(conv_key, chunk_idx)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(conv_key, chunk_idx)")
    conn.commit()
    conn.close()


# ── Background embedding worker ───────────────────────────────────────────────
# Embedding is always async: ingest stays fast; this thread continuously picks
# up messages that have text but no vector (covers backfill and live ingest
# with the same loop). Progress is visible via memory_status.

_embed_stop = threading.Event()


def _embed_batch() -> int:
    conn = db()
    try:
        rows = conn.execute("""
            SELECT m.conv_key, m.native_id, m.text FROM messages m
            LEFT JOIN vec_map v ON v.conv_key = m.conv_key AND v.native_id = m.native_id
            WHERE v.vec_id IS NULL AND m.text != '' LIMIT 64
        """).fetchall()
    finally:
        conn.close()
    if not rows:
        return 0

    texts = [r["text"][:EMBED_MAX_CHARS] for r in rows]
    vectors = list(embedder().embed(texts))

    with _db_lock:
        conn = db()
        try:
            for r, vec in zip(rows, vectors):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO vec_map (conv_key, native_id) VALUES (?,?)",
                    (r["conv_key"], r["native_id"]),
                )
                if cur.lastrowid:
                    conn.execute(
                        "INSERT INTO message_vecs (rowid, embedding) VALUES (?,?)",
                        (cur.lastrowid, sqlite_vec.serialize_float32([float(x) for x in vec])),
                    )
            conn.commit()
        finally:
            conn.close()
    return len(rows)


def embed_worker():
    while not _embed_stop.is_set():
        try:
            n = _embed_batch()
        except Exception:
            n = 0  # transient failure — back off and retry
        if n == 0:
            _embed_stop.wait(15.0)


def flat_text(content: list) -> str:
    parts = []
    for b in content or []:
        if b.get("type") in ("text", "thinking"):
            if b.get("text"):
                parts.append(b["text"])
            parts.extend(b.get("summaries", []))
    return "\n".join(p for p in parts if p)


def ingest_record(record: dict, force: bool = False) -> dict:
    conv = record["conversation"]
    key = f"{conv['source']}:{conv['native_id']}"
    now = datetime.now(timezone.utc).isoformat()

    # Canonical raw append (one NDJSON line per ingest event, partitioned by month)
    raw_file = RAW_DIR / f"{conv['source'].replace('.', '_')}-{now[:7]}.ndjson"
    with open(raw_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    owner = (conv.get("attribution") or {}).get("owner", "unknown")
    messages = record.get("messages", [])

    with _db_lock:
        conn = db()
        try:
            # Newest wins: an older snapshot (e.g. an official export backfilling
            # history) must not regress a fresher live capture. Raw archive above
            # keeps every version regardless.
            existing = conn.execute(
                "SELECT updated_at, message_count FROM conversations WHERE key=?", (key,)
            ).fetchone()
            if existing and existing["updated_at"] and conv.get("updated_at") and not force:
                if conv["updated_at"] <= existing["updated_at"]:
                    return {"key": key, "messages": existing["message_count"],
                            "owner": owner, "skipped": "stale (newer version already stored)"}
            conn.execute(
                """INSERT INTO conversations (key, source, native_id, title, model, owner,
                       created_at, updated_at, message_count, ingested_at, record)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                       title=excluded.title, model=excluded.model, owner=excluded.owner,
                       updated_at=excluded.updated_at, message_count=excluded.message_count,
                       ingested_at=excluded.ingested_at, record=excluded.record""",
                (key, conv["source"], conv["native_id"], conv.get("title"), conv.get("model"),
                 owner, conv.get("created_at"), conv.get("updated_at"), len(messages), now,
                 json.dumps(record, ensure_ascii=False)),
            )
            # Replace message rows + FTS + vector entries for this conversation
            # (idempotent upsert; vectors regenerate via the embed worker)
            conn.execute("DELETE FROM messages WHERE conv_key=?", (key,))
            conn.execute("DELETE FROM messages_fts WHERE conv_key=?", (key,))
            conn.execute(
                "DELETE FROM message_vecs WHERE rowid IN (SELECT vec_id FROM vec_map WHERE conv_key=?)", (key,)
            )
            conn.execute("DELETE FROM vec_map WHERE conv_key=?", (key,))
            for m in messages:
                text = flat_text(m.get("content"))
                # Attachment/file extracted text is part of the searchable record
                extracted = [
                    a["extracted_content"]
                    for a in (m.get("attachments") or []) + (m.get("files") or [])
                    if isinstance(a, dict) and a.get("extracted_content")
                ]
                if extracted:
                    text = "\n".join([text, *extracted]) if text else "\n".join(extracted)
                conn.execute(
                    "INSERT OR REPLACE INTO messages (conv_key, native_id, role, created_at, on_active_path, content_hash, text) VALUES (?,?,?,?,?,?,?)",
                    (key, m["native_id"], m.get("role"), m.get("created_at"),
                     1 if m.get("on_active_path") else 0, m.get("content_hash"), text),
                )
                if text:
                    conn.execute(
                        "INSERT INTO messages_fts (text, conv_key, native_id) VALUES (?,?,?)",
                        (text, key, m["native_id"]),
                    )
            conn.commit()
        finally:
            conn.close()

    return {"key": key, "messages": len(messages), "owner": owner}


# ── HTTP endpoints ────────────────────────────────────────────────────────────

async def ingest(request: Request):
    record = await request.json()
    if record.get("schema_version") != 1:
        return JSONResponse({"error": "unsupported schema_version"}, status_code=400)
    force = request.query_params.get("force") in ("1", "true")
    try:
        result = ingest_record(record, force=force)
    except (KeyError, TypeError) as e:
        return JSONResponse({"error": f"malformed record: {e}"}, status_code=400)
    return JSONResponse({"ok": True, **result})


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


async def entities_build(request: Request):
    """Build the entity layer. Body:
      {entities:[{canonical,type,aliases:[...]}], split_map:{mention_id:canonical}}
    Wipes+rebuilds entities/aliases/map. Auto-maps mentions by UNAMBIGUOUS alias
    (alias in exactly one entity); applies split_map for the ambiguous ones."""
    b = await request.json()
    ents = b.get("entities", [])
    split_map = b.get("split_map", {})
    with _db_lock:
        conn = db()
        try:
            conn.executescript("DELETE FROM entities; DELETE FROM entity_aliases; DELETE FROM entity_map;")
            canon_id = {}
            for e in ents:
                cur = conn.execute("INSERT INTO entities (canonical, type, mentions) VALUES (?,?,?)",
                                   (e["canonical"], e.get("type"), e.get("mentions", 0)))
                canon_id[e["canonical"]] = cur.lastrowid
            # alias -> set of entity_ids
            alias_ids = {}
            for e in ents:
                eid = canon_id[e["canonical"]]
                for a in e.get("aliases", []):
                    alias_ids.setdefault(a.lower(), set()).add(eid)
                    conn.execute("INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?,?)", (a, eid))
            # auto-map mentions by unambiguous alias
            mapped = 0
            for alias_l, eids in alias_ids.items():
                if len(eids) == 1:
                    eid = next(iter(eids))
                    mapped += conn.execute(
                        "INSERT OR REPLACE INTO entity_map (mention_id, entity_id) "
                        "SELECT id, ? FROM mentions WHERE LOWER(name)=?", (eid, alias_l)).rowcount
            # apply split assignments (override)
            split_applied = 0
            for mid, canonical in split_map.items():
                eid = canon_id.get(canonical)
                if eid:
                    conn.execute("INSERT OR REPLACE INTO entity_map (mention_id, entity_id) VALUES (?,?)",
                                 (int(mid), eid))
                    split_applied += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "entities": len(ents), "auto_mapped": mapped,
                         "split_applied": split_applied})


async def edges_build(_request: Request):
    """Generate candidate edges: within each chunk, the mapped entities are
    connected by that chunk's assertions (typed by assertion type). Weight =
    number of supporting chunks; evidence = assertion ids. Directional edges
    only where the assertion type implies direction; otherwise both."""
    with _db_lock:
        conn = db()
        try:
            conn.execute("DELETE FROM edges")
            # entities present per chunk (via mapped mentions)
            rows = conn.execute("""
                SELECT m.conv_key, m.chunk_idx, em.entity_id, c.created_at
                FROM mentions m
                JOIN entity_map em ON em.mention_id = m.id
                JOIN conversations c ON c.key = m.conv_key
            """).fetchall()
            chunk_ents = defaultdict(set)
            chunk_date = {}
            for r in rows:
                k = (r["conv_key"], r["chunk_idx"])
                chunk_ents[k].add(r["entity_id"])
                chunk_date[k] = (r["created_at"] or "")[:10]

            # assertions per chunk (type + id)
            arows = conn.execute("SELECT id, conv_key, chunk_idx, type FROM assertions").fetchall()
            chunk_asserts = defaultdict(list)
            for a in arows:
                chunk_asserts[(a["conv_key"], a["chunk_idx"])].append((a["id"], a["type"]))

            agg = defaultdict(lambda: {"w": 0, "ev": [], "first": "9999", "last": "0", "rel": Counter()})
            for k, ents in chunk_ents.items():
                if len(ents) < 2:
                    continue
                asserts = chunk_asserts.get(k, [])
                if not asserts:
                    continue
                date = chunk_date.get(k, "")
                el = sorted(ents)
                for i in range(len(el)):
                    for j in range(i + 1, len(el)):
                        pair = (el[i], el[j])
                        rec = agg[pair]
                        rec["w"] += 1
                        rec["ev"].extend(a[0] for a in asserts[:3])
                        rec["rel"].update(a[1] for a in asserts)
                        if date and date < rec["first"]:
                            rec["first"] = date
                        if date and date > rec["last"]:
                            rec["last"] = date

            written = 0
            for (s, d), rec in agg.items():
                rel = rec["rel"].most_common(1)[0][0] if rec["rel"] else "related"
                conn.execute(
                    "INSERT OR REPLACE INTO edges (src_entity, dst_entity, relation, weight, evidence, first_seen, last_seen) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (s, d, rel, rec["w"], json.dumps(rec["ev"][:20]), rec["first"], rec["last"]))
                written += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "edges": written})


async def graph_edges_dump(_request: Request):
    """Dump all edges (src, dst, weight) for external community detection."""
    conn = db()
    try:
        rows = [[r["src_entity"], r["dst_entity"], r["weight"]]
                for r in conn.execute("SELECT src_entity, dst_entity, weight FROM edges")]
    finally:
        conn.close()
    return JSONResponse({"edges": rows})


async def edges_for_typing(request: Request):
    """Serve edges + their supporting assertion text so a model can upgrade the
    generic relation to a semantic one and stamp temporality. Params:
    min_weight (default 2), limit (default 200), offset (0), only_untyped (1)."""
    q = request.query_params
    min_w = int(q.get("min_weight", "2"))
    limit = min(int(q.get("limit", "200")), 1000)
    offset = int(q.get("offset", "0"))
    only_untyped = q.get("only_untyped", "1") != "0"
    conn = db()
    try:
        where = "WHERE weight >= ?"
        params = [min_w]
        if only_untyped:
            where += " AND (verified IS NULL OR verified = 0)"
        edge_rows = conn.execute(
            f"SELECT edge_id, src_entity, dst_entity, relation, weight, "
            f"first_seen, last_seen FROM edges {where} "
            f"ORDER BY weight DESC LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
        out = []
        for e in edge_rows:
            src = conn.execute("SELECT canonical, type FROM entities WHERE entity_id=?",
                               (e["src_entity"],)).fetchone()
            dst = conn.execute("SELECT canonical, type FROM entities WHERE entity_id=?",
                               (e["dst_entity"],)).fetchone()
            if not src or not dst:
                continue
            # chunks where BOTH entities are mapped (real co-occurrence, not
            # chunk-bag noise): intersect each entity's (conv_key, chunk_idx) set
            def _chunks(eid):
                return {(r["conv_key"], r["chunk_idx"]) for r in conn.execute(
                    "SELECT DISTINCT m.conv_key, m.chunk_idx FROM mentions m "
                    "JOIN entity_map em ON em.mention_id=m.id WHERE em.entity_id=?", (eid,))}
            shared = _chunks(e["src_entity"]) & _chunks(e["dst_entity"])
            # assertions from those shared chunks, prefer ones naming either entity
            aliases = [a["alias"].lower() for a in conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id IN (?,?)",
                (e["src_entity"], e["dst_entity"]))]
            cand = []
            for (ck, cx) in list(shared)[:60]:
                for a in conn.execute(
                        "SELECT type, statement, speaker, rationale FROM assertions "
                        "WHERE conv_key=? AND chunk_idx=?", (ck, cx)).fetchall():
                    txt = ((a["statement"] or "") + " " + (a["rationale"] or "")).lower()
                    score = sum(1 for al in aliases if al and al in txt)
                    cand.append((score, {"type": a["type"], "speaker": a["speaker"],
                                          "statement": (a["statement"] or "")[:240],
                                          "rationale": (a["rationale"] or "")[:120]}))
            cand.sort(key=lambda x: -x[0])
            ev = [c[1] for c in cand[:7]]
            out.append({
                "edge_id": e["edge_id"],
                "src": {"name": src["canonical"], "type": src["type"]},
                "dst": {"name": dst["canonical"], "type": dst["type"]},
                "relation": e["relation"], "weight": e["weight"],
                "first_seen": e["first_seen"], "last_seen": e["last_seen"],
                "evidence": ev})
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM edges {where}", params).fetchone()["n"]
    finally:
        conn.close()
    return JSONResponse({"edges": out, "returned": len(out), "matching_total": total})


async def set_edge_types(request: Request):
    """Write verified edge typings. Body: {typings:[{edge_id, relation,
    direction, temporality, as_of}]}. direction: forward(src->dst) |
    backward(dst->src) | undirected. temporality: enduring | state."""
    b = await request.json()
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with _db_lock:
        conn = db()
        try:
            for t in b.get("typings", []):
                conn.execute(
                    "UPDATE edges SET verified_relation=?, direction=?, temporality=?, "
                    "as_of=?, verified=1, verified_at=? WHERE edge_id=?",
                    (t.get("relation"), t.get("direction"), t.get("temporality"),
                     t.get("as_of"), now, t["edge_id"]))
                n += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "typed": n})


async def typing_status(request: Request):
    """Live edge-typing progress as JSON. ?min_weight defaults to 2."""
    min_w = int(request.query_params.get("min_weight", "2"))
    conn = db()
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM edges WHERE weight>=?",
                             (min_w,)).fetchone()["n"]
        typed = conn.execute("SELECT COUNT(*) AS n FROM edges WHERE weight>=? AND verified=1",
                             (min_w,)).fetchone()["n"]
        rels = {r["verified_relation"]: r["n"] for r in conn.execute(
            "SELECT verified_relation, COUNT(*) AS n FROM edges WHERE verified=1 "
            "GROUP BY verified_relation ORDER BY n DESC")}
        temps = {r["temporality"]: r["n"] for r in conn.execute(
            "SELECT temporality, COUNT(*) AS n FROM edges WHERE verified=1 GROUP BY temporality")}
        recent = []
        for r in conn.execute(
                "SELECT src_entity, dst_entity, verified_relation, direction, temporality, "
                "as_of, weight, verified_at FROM edges WHERE verified=1 "
                "ORDER BY verified_at DESC LIMIT 15"):
            recent.append({
                "src": _entity_name(conn, r["src_entity"]),
                "dst": _entity_name(conn, r["dst_entity"]),
                "relation": r["verified_relation"], "direction": r["direction"],
                "temporality": r["temporality"], "as_of": r["as_of"], "weight": r["weight"]})
        # rate: typings in the last 5 minutes
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        recent5 = conn.execute(
            "SELECT COUNT(*) AS n FROM edges WHERE verified=1 AND verified_at>=?",
            (cutoff,)).fetchone()["n"]
    finally:
        conn.close()
    rate = recent5 / 300.0
    remaining = max(0, total - typed)
    eta_s = int(remaining / rate) if rate > 0 else None
    return JSONResponse({"total": total, "typed": typed, "remaining": remaining,
                         "pct": round(100 * typed / total, 1) if total else 0,
                         "rate_per_s": round(rate, 3), "eta_seconds": eta_s,
                         "relations": rels, "temporality": temps, "recent": recent})


async def graph_full(request: Request):
    """Full node+link dump for the force-graph visualization. Nodes carry
    type/mentions/community; links carry the typed relation + weight +
    temporality. ?min_weight (default 3) trims density. ?depth explodes each
    entity into its detail: depth>=1 adds a sample of raw mentions, depth>=2 also
    adds the assertions (decisions/problems/solutions/facts) from those chunks."""
    min_w = int(request.query_params.get("min_weight", "3"))
    depth = int(request.query_params.get("depth", "0"))
    per = min(int(request.query_params.get("per", "16")), 80)
    conn = db()
    try:
        comm_label = {r["community_id"]: r["label"]
                      for r in conn.execute("SELECT community_id, label FROM communities")}
        ent_comm = {r["entity_id"]: r["community_id"]
                    for r in conn.execute("SELECT entity_id, community_id FROM entity_community")}
        # only include nodes that actually appear on a surviving link
        links, node_ids = [], set()
        for r in conn.execute(
                "SELECT src_entity, dst_entity, relation, verified_relation, direction, "
                "temporality, as_of, weight FROM edges WHERE weight>=?", (min_w,)):
            s, d = r["src_entity"], r["dst_entity"]
            rel = r["verified_relation"] or r["relation"]
            # orient so source is the semantic subject
            if r["direction"] == "backward":
                s, d = d, s
            links.append({"source": s, "target": d, "relation": rel,
                          "weight": r["weight"], "temporality": r["temporality"],
                          "as_of": r["as_of"], "directed": r["direction"] in ("forward", "backward")})
            node_ids.add(r["src_entity"]); node_ids.add(r["dst_entity"])
        nodes = []
        for r in conn.execute("SELECT entity_id, canonical, type, mentions FROM entities"):
            if r["entity_id"] not in node_ids:
                continue
            cid = ent_comm.get(r["entity_id"])
            nodes.append({"id": r["entity_id"], "name": r["canonical"], "type": r["type"],
                          "mentions": r["mentions"], "community": cid,
                          "theme": comm_label.get(cid, "")})

        total_pieces = None
        if depth >= 1 and node_ids:
            ph = ",".join("?" * len(node_ids))
            tm = conn.execute("SELECT COUNT(*) AS n FROM entity_map").fetchone()["n"]
            total_pieces = tm
            # level 1: up to `per` raw mentions per entity (window-limited, never 100K rows)
            for r in conn.execute(f"""
                SELECT entity_id, mid, mname, kind FROM (
                    SELECT em.entity_id AS entity_id, m.id AS mid, m.name AS mname, m.kind AS kind,
                           ROW_NUMBER() OVER (PARTITION BY em.entity_id ORDER BY m.id) AS rn
                    FROM entity_map em JOIN mentions m ON m.id = em.mention_id
                    WHERE em.entity_id IN ({ph})
                ) WHERE rn <= ?
            """, (*node_ids, per)).fetchall():
                mid = f"m{r['mid']}"
                cid = ent_comm.get(r["entity_id"])
                nodes.append({"id": mid, "name": (r["mname"] or "")[:48], "type": r["kind"] or "mention",
                              "mentions": 1, "community": cid, "theme": comm_label.get(cid, ""),
                              "piece": True, "kind": "mention"})
                links.append({"source": r["entity_id"], "target": mid, "relation": "mention",
                              "weight": 1, "temporality": "enduring", "as_of": None,
                              "directed": False, "piece": True})
        if depth >= 2 and node_ids:
            ph = ",".join("?" * len(node_ids))
            total_pieces = conn.execute("SELECT COUNT(*) AS n FROM assertions").fetchone()["n"]
            # level 2: the actual knowledge — assertions from the chunks each
            # entity appears in (decisions/problems/solutions/facts/...)
            for r in conn.execute(f"""
                SELECT entity_id, aid, statement, atype FROM (
                    SELECT entity_id, aid, statement, atype,
                           ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY aid) AS rn FROM (
                        SELECT DISTINCT em.entity_id AS entity_id, a.id AS aid,
                               a.statement AS statement, a.type AS atype
                        FROM entity_map em JOIN mentions m ON m.id = em.mention_id
                        JOIN assertions a ON a.conv_key = m.conv_key AND a.chunk_idx = m.chunk_idx
                        WHERE em.entity_id IN ({ph})
                    )
                ) WHERE rn <= ?
            """, (*node_ids, per)).fetchall():
                aid = f"a{r['aid']}"
                cid = ent_comm.get(r["entity_id"])
                nodes.append({"id": aid, "name": (r["statement"] or "")[:80], "type": r["atype"] or "claim",
                              "mentions": 1, "community": cid, "theme": comm_label.get(cid, ""),
                              "piece": True, "kind": "assertion"})
                links.append({"source": r["entity_id"], "target": aid, "relation": r["atype"] or "claim",
                              "weight": 1, "temporality": "enduring", "as_of": None,
                              "directed": False, "piece": True})
    finally:
        conn.close()
    return JSONResponse({"nodes": nodes, "links": links, "depth": depth,
                         "total_pieces": total_pieces,
                         "communities": [{"id": k, "label": v} for k, v in sorted(comm_label.items())]})


async def rename_community(request: Request):
    """Rename a theme/community label (for demo-friendly tags). Body:
    {community_id, label}."""
    b = await request.json()
    with _db_lock:
        conn = db()
        try:
            conn.execute("UPDATE communities SET label=? WHERE community_id=?",
                         (b.get("label", ""), b["community_id"]))
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True})


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


_LIVE_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Edge Typing - live</title>
<style>
 body{background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:24px}
 h1{font-size:18px;color:#58a6ff;margin:0 0 4px}
 .sub{color:#8b949e;font-size:12px;margin-bottom:16px}
 .bar{height:22px;background:#21262d;border-radius:6px;overflow:hidden;margin:10px 0}
 .fill{height:100%;background:linear-gradient(90deg,#238636,#2ea043);transition:width .5s}
 .big{font-size:15px;color:#e6edf3}
 .stats{display:flex;gap:24px;flex-wrap:wrap;margin:14px 0}
 .stat b{color:#e6edf3;font-size:20px;display:block}
 .stat span{color:#8b949e;font-size:12px}
 table{border-collapse:collapse;width:100%;margin-top:14px;font-size:13px}
 td{padding:3px 10px 3px 0;white-space:nowrap}
 .state{color:#d29922}.enduring{color:#8b949e}
 .rel{color:#58a6ff}.pill{color:#7ee787}
 .chips span{display:inline-block;background:#21262d;border-radius:10px;padding:2px 9px;margin:2px;font-size:12px}
</style></head><body>
<h1>Knowledge Graph &mdash; Edge Typing</h1>
<div class=sub>tiiny Qwen3.6-35B &middot; auto-refreshing every 4s</div>
<div class=big id=hdr>loading&hellip;</div>
<div class=bar><div class=fill id=fill style=width:0%></div></div>
<div class=stats id=stats></div>
<div class=chips id=rels></div>
<div class=chips id=temps></div>
<table id=recent></table>
<script>
const TOKEN="__TOKEN__";
function fmt(s){if(s==null)return"--";const h=Math.floor(s/3600),m=Math.floor((s%3600)/60);return h+"h "+m+"m"}
async function tick(){
 try{
  const r=await fetch("/api/graph/typing_status?token="+encodeURIComponent(TOKEN));
  const d=await r.json();
  document.getElementById("hdr").textContent=d.typed+" / "+d.total+" edges typed ("+d.pct+"%)";
  document.getElementById("fill").style.width=d.pct+"%";
  document.getElementById("stats").innerHTML=
   "<div class=stat><b>"+d.remaining+"</b><span>remaining</span></div>"+
   "<div class=stat><b>"+d.rate_per_s+"/s</b><span>current rate</span></div>"+
   "<div class=stat><b>"+fmt(d.eta_seconds)+"</b><span>ETA</span></div>";
  document.getElementById("rels").innerHTML="<span>relations:</span> "+
   Object.entries(d.relations).map(([k,v])=>"<span class=rel>"+k+" "+v+"</span>").join("");
  document.getElementById("temps").innerHTML="<span>temporality:</span> "+
   Object.entries(d.temporality).map(([k,v])=>"<span class=pill>"+(k||"?")+" "+v+"</span>").join("");
  document.getElementById("recent").innerHTML="<tr><td colspan=4 style=color:#8b949e>&mdash; last 15 typed &mdash;</td></tr>"+
   d.recent.map(e=>{const a={forward:"&rarr;",backward:"&larr;",undirected:"&harr;"}[e.direction]||"&mdash;";
    const asof=e.as_of?(" @"+e.as_of):"";const cls=e.temporality=="state"?"state":"enduring";
    return "<tr><td>"+e.src+" "+a+" "+e.dst+"</td><td class=rel>"+e.relation+"</td><td class='"+cls+"'>"+e.temporality+asof+"</td><td style=color:#6e7681>w="+e.weight+"</td></tr>"}).join("");
 }catch(e){document.getElementById("hdr").textContent="(waiting for server…)"}
}
tick();setInterval(tick,4000);
</script></body></html>"""


_GRAPH_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Tailpipe - Knowledge Graph</title>
<style>
 html,body{margin:0;height:100%;background:#05070d;overflow:hidden;font:13px/1.4 -apple-system,Segoe UI,sans-serif;color:#c9d1d9}
 #graph{position:fixed;inset:0}
 #hud{position:fixed;top:0;left:0;padding:16px 18px;z-index:10;max-width:340px;pointer-events:none}
 #hud h1{font-size:16px;margin:0 0 2px;color:#e6edf3;letter-spacing:.3px;text-shadow:0 0 12px #1f6feb}
 #hud .s{color:#8b949e;font-size:11px}
 #legend{margin-top:14px;pointer-events:auto}
 #legend .row{display:flex;align-items:center;gap:7px;margin:3px 0;font-size:12px}
 #legend .dot{width:11px;height:11px;border-radius:50%;box-shadow:0 0 8px currentColor;flex:none}
 #legend .lbl{outline:none;cursor:text;border-bottom:1px dashed transparent;padding:0 1px}
 #legend .lbl:hover,#legend .lbl:focus{border-bottom-color:#3a4a66}
 #legend .x{margin-left:auto;cursor:pointer;color:#5b6472;font-size:14px;padding:0 4px;user-select:none;opacity:0}
 #legend .row:hover .x{opacity:1}
 #legend .x:hover{color:#ff7b72}
 #legend .restore{color:#4d7fd6;font-size:11px;margin-top:5px;cursor:pointer;pointer-events:auto}
 #legend .restore:hover{color:#79b0ff}
 #legend .hint{color:#4b5563;font-size:10px;margin:2px 0 6px}
 #ctl{position:fixed;top:16px;right:18px;z-index:10;text-align:right}
 #ctl input[type=text]{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;padding:6px 9px;width:180px;outline:none}
 #ctl input[type=text]:focus{border-color:#1f6feb}
 #glowbox{margin-top:8px;color:#8b949e;font-size:11px}
 #glowbox input{vertical-align:middle;width:96px}
 .dens{margin-top:8px}
 .dens button{background:#161b22;border:1px solid #30363d;color:#8b949e;border-radius:5px;padding:4px 9px;margin-left:4px;cursor:pointer;font-size:11px}
 .dens button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
 #tip{position:fixed;z-index:20;pointer-events:none;background:#0d1117ee;border:1px solid #30363d;border-radius:7px;padding:7px 10px;display:none;max-width:260px;box-shadow:0 4px 20px #000a}
 #tip b{color:#e6edf3}#tip .t{color:#58a6ff}#tip .th{color:#7ee787;font-size:11px}
 #count{position:fixed;bottom:12px;left:18px;z-index:10;color:#6e7681;font-size:11px}
</style></head><body>
<div id=graph></div>
<div id=hud>
 <h1>Tailpipe &mdash; Your Cognitive Graph</h1>
 <div class=s>your conversations, resolved into entities, edges &amp; themes</div>
 <div class=hint id=lhint>click a name to rename &middot; hover a row and hit &times; to hide the tag &middot; click a node to lock focus</div>
 <div id=legend></div>
</div>
<div id=ctl>
 <input id=search type=text placeholder="find an entity&hellip;" autocomplete=off>
 <div class=dens>
   density: <button data-w=10>10+</button><button data-w=5>5+</button>
   <button data-w=3 class=on>3+</button><button data-w=2>2+</button>
 </div>
 <div class=dens>depth: <button id=lvl>&#9698; drill into pieces</button></div>
 <div id=glowbox>glow <input id=glow type=range min=0 max=2.5 step=0.05 value=1.15></div>
</div>
<div id=tip></div>
<div id=count>loading&hellip;</div>
<script type="importmap">
{ "imports": {
  "three": "https://esm.sh/three@0.179.0",
  "three/": "https://esm.sh/three@0.179.0/"
}}
</script>
<script type="module">
import ForceGraph3D from "https://esm.sh/3d-force-graph@1.80.0?external=three";
import { UnrealBloomPass } from "https://esm.sh/three@0.179.0/examples/jsm/postprocessing/UnrealBloomPass.js";
import SpriteText from "https://esm.sh/three-spritetext@1.9?external=three";

const TOKEN="__TOKEN__";
const PALETTE=["#4d96ff","#ff5a6a","#ffd93d","#63e6a0","#b892ff","#ff9f45","#31e0d4","#ff5ec4","#4de1ff","#ffa9a9"];
const el=document.getElementById.bind(document);
const DIM_N="#0a0d18", DIM_L="#060a14";
let Graph, bloom, RAW=null, curW=3, depth=0, fitted=false, labelCut=Infinity;
let hover=null, pinned=null; const hlN=new Set(), hlL=new Set();
// demo controls: rename + hide theme tags from the legend, remembered per browser.
// legHidden only affects the upper-left list — the nodes always stay in the graph.
let legHidden=new Set(JSON.parse(localStorage.getItem("hg_leghidden")||"[]"));
let overrides=JSON.parse(localStorage.getItem("hg_labels")||"{}");
const themeLabel=c=> overrides[c.id] ?? ((c.label||"").split(" / ").slice(0,3).join(" / ")||("theme "+c.id));

const baseColor=n=> n.community==null ? "#7b8698" : PALETTE[n.community % PALETTE.length];
const nColor=n=> hover ? (hlN.has(n.id)?baseColor(n):DIM_N) : baseColor(n);
const lColor=l=> hover ? (hlL.has(l)?"#9fe8ff":DIM_L)
                       : ((l.source&&l.source.community!=null)?PALETTE[l.source.community%PALETTE.length]:"#2b466e");
const pCount=l=> hover ? (hlL.has(l)?4:0) : (depth>0?0:1);  // idle flow in macro; off when exploded (too many)
const lWidth=l=> (hover&&hlL.has(l)) ? Math.max(1.4,Math.sqrt(l.weight)/2.5)
                                     : Math.max(0.4,Math.sqrt(l.weight)/4.5);

function applyFocus(node){
  hover=node||null; hlN.clear(); hlL.clear();
  if(node){
    hlN.add(node.id);
    (window._links||[]).forEach(l=>{
      const s=l.source.id??l.source, t=l.target.id??l.target;
      if(s===node.id||t===node.id){ hlL.add(l); hlN.add(s); hlN.add(t); }
    });
    showTip(node);
  } else { el("tip").style.display="none"; }
  Graph.nodeColor(nColor).linkColor(lColor).linkWidth(lWidth).linkDirectionalParticles(pCount);
}
// hover previews only when nothing is pinned; a click locks the focus so it
// survives the rotation. Click the same node (or the background) to release.
function setHover(node){ if(!pinned) applyFocus(node); }
function pin(node){
  if(pinned && pinned.id===node.id){ pinned=null; applyFocus(null); return; }
  pinned=node; Graph.controls().autoRotate=false; applyFocus(node);
  const dist=90, r=Math.hypot(node.x,node.y,node.z)||1;
  Graph.cameraPosition({x:node.x*(1+dist/r),y:node.y*(1+dist/r),z:node.z*(1+dist/r)},node,1000);
}
function clearFocus(){ pinned=null; applyFocus(null); }

function renderLegend(){
  const present=new Set(RAW.nodes.map(n=>n.community));
  const box=el("legend"); box.innerHTML="";
  const comms=RAW.communities.filter(c=>present.has(c.id));
  comms.filter(c=>!legHidden.has(c.id)).forEach(c=>{
    const col=PALETTE[c.id%PALETTE.length];
    const row=document.createElement("div"); row.className="row";
    const dot=document.createElement("span"); dot.className="dot";
    dot.style.color=col; dot.style.background=col;
    const lbl=document.createElement("span"); lbl.className="lbl"; lbl.contentEditable="true";
    lbl.spellcheck=false; lbl.textContent=themeLabel(c);
    lbl.addEventListener("keydown",e=>{ if(e.key==="Enter"){e.preventDefault(); lbl.blur();} });
    lbl.addEventListener("blur",()=>saveLabel(c.id,lbl.textContent.trim()));
    const x=document.createElement("span"); x.className="x"; x.textContent="\\u00d7";
    x.title="hide this tag from the list (nodes stay)";
    x.addEventListener("click",()=>{ legHidden.add(c.id);
      localStorage.setItem("hg_leghidden",JSON.stringify([...legHidden])); renderLegend(); });
    row.append(dot,lbl,x); box.appendChild(row);
  });
  const nHidden=comms.filter(c=>legHidden.has(c.id)).length;
  if(nHidden){
    const f=document.createElement("div"); f.className="restore";
    f.textContent="show "+nHidden+" hidden tag"+(nHidden>1?"s":"");
    f.addEventListener("click",()=>{ legHidden.clear();
      localStorage.setItem("hg_leghidden","[]"); renderLegend(); });
    box.appendChild(f);
  }
}

function saveLabel(cid,label){
  overrides[cid]=label; localStorage.setItem("hg_labels",JSON.stringify(overrides));
  fetch("/api/graph/rename_community?token="+encodeURIComponent(TOKEN),
    {method:"POST",headers:{"content-type":"application/json"},
     body:JSON.stringify({community_id:cid,label})}).catch(()=>{});
}

function applyView(){   // nodes always fully shown; legHidden only touches the list
  if(Graph){
    // weaker repulsion when exploded so the thousands of pieces stay clustered
    // near their entity instead of blowing apart into a thin scatter
    Graph.d3Force("charge").strength(depth>=2 ? -14 : depth===1 ? -26 : -170);
    Graph.linkOpacity(depth>0?0.13:0.32);
  }
  Graph.graphData({nodes:RAW.nodes, links:RAW.links});
  window._nodes=RAW.nodes; window._links=RAW.links;
  const pieces=RAW.nodes.filter(n=>n.piece).length, ents=RAW.nodes.length-pieces;
  if(depth>0){
    const kind=depth>=2?"claims":"mentions";
    el("count").textContent=ents.toLocaleString()+" entities  ·  "+pieces.toLocaleString()+
      " pieces shown  ·  of "+(RAW.total_pieces||0).toLocaleString()+" "+kind+" in the graph";
  } else {
    const themes=new Set(RAW.nodes.map(n=>n.community));
    el("count").textContent=ents+" entities  ·  "+RAW.links.length+" connections  ·  "+themes.size+" themes";
  }
}

let loadSeq=0;
async function load(w){
  const mySeq=++loadSeq;   // drop stale responses if the user clicks again mid-load
  el("count").textContent=depth>0?"exploding into pieces…":"loading…";
  const url="/api/graph/full?min_weight="+w+(depth>0?"&depth="+depth:"")+"&token="+encodeURIComponent(TOKEN);
  const r=await fetch(url);
  const data=await r.json();
  if(mySeq!==loadSeq) return;   // a newer load started — this one is stale
  RAW=data;
  const maxM=Math.max(1,...RAW.nodes.map(n=>n.mentions||1));
  RAW.nodes.forEach(n=>{ n.val=1+7*Math.sqrt((n.mentions||1)/maxM); });
  // always-label the top ~45 hubs so the graph is readable, not just dots
  const byM=[...RAW.nodes].sort((a,b)=>(b.mentions||0)-(a.mentions||0));
  labelCut = byM.length>45 ? (byM[44].mentions||0) : 0;
  if(!Graph){
    Graph=ForceGraph3D({controlType:"orbit"})(el("graph"))
      .width(window.innerWidth).height(window.innerHeight)
      .backgroundColor("#02030a")
      .showNavInfo(false)
      .nodeColor(nColor)
      .nodeVal("val")
      .nodeOpacity(0.9)
      .nodeResolution(8)
      .nodeThreeObjectExtend(true)
      .nodeThreeObject(n=>{
        if((n.mentions||0)<labelCut) return null;
        const s=new SpriteText(n.name);
        s.color="#dbe6ff"; s.textHeight=Math.min(6,2.8+n.val/4);
        s.fontWeight="600"; s.material.depthWrite=false; s.material.opacity=0.85;
        s.position.y=-(n.val+4);
        return s;
      })
      .linkColor(lColor)
      .linkOpacity(0.32)
      .linkWidth(lWidth)
      .linkDirectionalParticles(pCount)
      .linkDirectionalParticleColor(()=>"#6fd8ff")
      .linkDirectionalParticleWidth(1.0)
      .linkDirectionalParticleSpeed(l=>0.004+Math.min(0.01,l.weight/4000))
      .onNodeHover(setHover)
      .onNodeClick(pin)
      .onBackgroundClick(clearFocus)
      .onEngineStop(()=>{ if(!fitted){fitted=true; Graph.zoomToFit(1400,90);} });
    Graph.d3Force("charge").strength(-170);
    Graph.d3Force("link").distance(l=> l.piece ? 7 : (24+40/(l.weight||1)));  // pieces hug their entity
    Graph.d3VelocityDecay(0.24);                 // keeps a slow living drift
    // bloom = the glow, but restrained so the core doesn't blow out to white.
    // threshold means only the brighter nodes bloom; strength driven by the slider.
    bloom=new UnrealBloomPass();
    bloom.strength=+el("glow").value; bloom.radius=0.7; bloom.threshold=0.2;
    Graph.postProcessingComposer().addPass(bloom);
    const ctr=Graph.controls();
    ctr.autoRotate=true; ctr.autoRotateSpeed=0.65;   // alive on load, stops when grabbed
    el("graph").addEventListener("pointerdown",()=>{ ctr.autoRotate=false; });
  }
  fitted=false; pinned=null; hover=null; hlN.clear(); hlL.clear();
  renderLegend();
  applyView();
}

el("glow").addEventListener("input",e=>{ if(bloom) bloom.strength=+e.target.value; });

// keep the canvas locked to the viewport (fixes fullscreen / window-resize dead space)
addEventListener("resize",()=>{ if(Graph) Graph.width(window.innerWidth).height(window.innerHeight); });

el("lvl").addEventListener("click",()=>{
  depth=(depth+1)%3;   // macro -> mentions -> +claims -> macro
  el("lvl").innerHTML = depth===0 ? "&#9698; drill into pieces"
                      : depth===1 ? "&#9698; go deeper: claims"
                      : "&#9650; back to macro";
  el("lvl").classList.toggle("on", depth>0);
  load(curW);
});

function showTip(n){
  const t=el("tip");
  if(!n){t.style.display="none";return;}
  t.innerHTML="<b>"+n.name+"</b> <span class=t>"+n.type+"</span><br>"+
    "<span class=th>"+(n.theme? n.theme.split(" / ").slice(0,3).join(" / "):"")+"</span><br>"+
    (n.mentions||0)+" mentions";
  t.style.display="block";
}
addEventListener("mousemove",e=>{const t=el("tip");if(t.style.display==="block"){t.style.left=(e.clientX+14)+"px";t.style.top=(e.clientY+14)+"px";}});

el("search").addEventListener("keydown",e=>{
  if(e.key!=="Enter")return;
  const q=e.target.value.trim().toLowerCase();
  const hit=(window._nodes||[]).filter(n=>n.name.toLowerCase().includes(q))
    .sort((a,b)=>(b.mentions||0)-(a.mentions||0))[0];
  if(hit) pin(hit);   // search pins the match so it stays lit
});

document.querySelectorAll(".dens button[data-w]").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll(".dens button[data-w]").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); curW=+b.dataset.w; load(curW);
}));

load(curW);
</script></body></html>"""


async def graph_set_communities(request: Request):
    """Store community detection results. Body:
    {communities:[{community_id, label, entity_ids:[...]}]}."""
    b = await request.json()
    with _db_lock:
        conn = db()
        try:
            conn.executescript("DELETE FROM communities; DELETE FROM entity_community;")
            for c in b.get("communities", []):
                cid = c["community_id"]
                conn.execute("INSERT INTO communities (community_id, label, size) VALUES (?,?,?)",
                             (cid, c.get("label", ""), len(c.get("entity_ids", []))))
                for eid in c.get("entity_ids", []):
                    conn.execute("INSERT OR REPLACE INTO entity_community (entity_id, community_id) VALUES (?,?)",
                                 (eid, cid))
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "communities": len(b.get("communities", []))})


async def entities_list(request: Request):
    limit = min(int(request.query_params.get("limit", "300")), 2000)
    conn = db()
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT e.entity_id, e.canonical, e.type, e.mentions,
                   COUNT(DISTINCT em.mention_id) AS mapped
            FROM entities e LEFT JOIN entity_map em ON em.entity_id = e.entity_id
            GROUP BY e.entity_id ORDER BY e.mentions DESC LIMIT ?""", (limit,))]
    finally:
        conn.close()
    return JSONResponse({"entities": rows})


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# ── MCP server ────────────────────────────────────────────────────────────────

# LAN service reached by IP/hostname — loosen the default localhost-only
# DNS-rebinding guard to the addresses this box actually answers on.
# Real access control is the bearer token + LAN-only exposure.
# Set ALLOWED_HOSTS in .env to your box's LAN address(es), comma-separated.
_PORT = os.environ.get("PORT", "8080")
_ALLOWED_HOSTS = []
for _h in (h.strip() for h in os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")):
    if _h:
        _ALLOWED_HOSTS += [_h, f"{_h}:{_PORT}"]
mcp = FastMCP(
    "tailpipe-memory",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=_ALLOWED_HOSTS,
        allowed_origins=[f"http://{h}" for h in _ALLOWED_HOSTS],
    ),
)


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


@mcp.tool()
def search_memory(query: str, source: str = "", after: str = "", before: str = "", limit: int = 8) -> str:
    """Hybrid search (keyword + semantic) across all captured AI conversations
    (Claude.ai, ChatGPT, ...). Returns scored excerpts with conversation
    references. Use get_conversation with a ref to read full context.
    Optional after/before restrict by message date (ISO, e.g. "2025-03-01")."""
    limit = max(1, min(limit, 25))
    after, before = _date_bounds(after, before)
    conn = db()
    try:
        fts = _fts_hits(conn, query, source, 25, after, before)
        try:
            vec = _vec_hits(conn, query, source, 25, after, before)
        except Exception:
            vec = []  # embedder warming up or no vectors yet — degrade to FTS

        # Reciprocal rank fusion
        scores: dict = {}
        for rank, key in enumerate(fts):
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        for rank, key in enumerate(vec):
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]

        out = []
        for conv_key, native_id in ranked:
            r = conn.execute(
                """SELECT m.role, m.created_at, substr(m.text, 1, 300) AS excerpt,
                          c.title, c.source
                   FROM messages m JOIN conversations c ON c.key = m.conv_key
                   WHERE m.conv_key = ? AND m.native_id = ?""",
                (conv_key, native_id),
            ).fetchone()
            if not r:
                continue
            via = ("kw+sem" if (conv_key, native_id) in fts and (conv_key, native_id) in vec
                   else "kw" if (conv_key, native_id) in fts else "sem")
            out.append(
                f"[{r['source']}] {r['title'] or '(untitled)'} — {r['role']} @ {r['created_at'] or '?'} ({via})\n"
                f"  ref: {conv_key}#{native_id}\n"
                f"  {(r['excerpt'] or '').strip()}"
            )
    finally:
        conn.close()

    return "\n\n".join(out) if out else "No matches."


@mcp.tool()
def get_conversation(ref: str, window: int = 6) -> str:
    """Read full message context. ref = 'source:conversation_id' (whole active
    path, newest last) or 'source:conversation_id#message_id' (window of
    messages around that hit)."""
    conv_key, _, msg_id = ref.partition("#")
    conn = db()
    try:
        conv = conn.execute("SELECT * FROM conversations WHERE key=?", (conv_key,)).fetchone()
        if not conv:
            return f"Not found: {conv_key}"
        if conv["owner"] in EXCLUDED_OWNERS:
            return "This conversation is excluded from retrieval."
        rows = conn.execute(
            "SELECT * FROM messages WHERE conv_key=? AND on_active_path=1 ORDER BY created_at",
            (conv_key,),
        ).fetchall()
    finally:
        conn.close()

    if msg_id:
        idx = next((i for i, r in enumerate(rows) if r["native_id"] == msg_id), None)
        if idx is not None:
            lo = max(0, idx - window)
            rows = rows[lo: idx + window + 1]

    header = f"{conv['title'] or '(untitled)'} [{conv['source']}] — {len(rows)} messages shown"
    body = "\n\n".join(
        f"{r['role'].upper()} ({r['created_at'] or '?'}):\n{(r['text'] or '')[:2000]}" for r in rows
    )
    return f"{header}\n\n{body}"


@mcp.tool()
def list_conversations(after: str = "", before: str = "", source: str = "", limit: int = 20) -> str:
    """Browse conversations by time — no search term needed. Answers "what was
    I working on in <period>". Dates are ISO (e.g. "2025-03-01"); source
    optionally narrows (claude.ai, chatgpt.com, claude-code, gemini.google.com).
    Returns titles, refs, dates, and message counts, newest first."""
    limit = max(1, min(limit, 100))
    after, before = _date_bounds(after, before)
    sql = "SELECT key, source, title, created_at, updated_at, message_count FROM conversations WHERE owner NOT IN ({})".format(
        ",".join("?" * len(EXCLUDED_OWNERS)) or "''")
    params = [*EXCLUDED_OWNERS]
    if source:
        sql += " AND source = ?"
        params.append(source)
    if after:
        sql += " AND updated_at >= ?"
        params.append(after)
    if before:
        sql += " AND created_at <= ?"
        params.append(before)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    conn = db()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No conversations in that range."
    return "\n".join(
        f"[{r['source']}] {(r['created_at'] or '?')[:10]} -> {(r['updated_at'] or '?')[:10]}  "
        f"({r['message_count']} msgs)  {r['title'] or '(untitled)'}\n  ref: {r['key']}"
        for r in rows
    )


def _resolve_entity(conn, name: str):
    low = name.lower()
    # 1. exact canonical
    r = conn.execute("SELECT entity_id, canonical, type FROM entities WHERE LOWER(canonical)=?", (low,)).fetchone()
    if r:
        return r
    # 2. canonical whose first word IS the query ("Cody" -> "Cody Ford (...)"),
    #    most-mentioned wins. Checked before aliases so a short form maps to its
    #    OWN canonical, not to something that merely lists it as an alias.
    r = conn.execute(
        "SELECT entity_id, canonical, type FROM entities "
        "WHERE LOWER(canonical) LIKE ? ORDER BY mentions DESC LIMIT 1", (low + " %",)).fetchone()
    if r:
        return r
    # 3. exact alias (nicknames/abbreviations that aren't canonical prefixes)
    return conn.execute(
        "SELECT e.entity_id, e.canonical, e.type FROM entity_aliases a "
        "JOIN entities e ON e.entity_id=a.entity_id WHERE LOWER(a.alias)=? "
        "ORDER BY e.mentions DESC LIMIT 1", (low,)).fetchone()


def _entity_name(conn, eid):
    r = conn.execute("SELECT canonical FROM entities WHERE entity_id=?", (eid,)).fetchone()
    return r["canonical"] if r else "?"


def _rel_sentence(conn, r):
    """Render an edge as a plain sentence — 'SUBJECT relation OBJECT [as of DATE]' —
    readable aloud without knowing which entity was queried. Grammar over glyphs."""
    rel = r["verified_relation"] or r["relation"]
    src, dst = _entity_name(conn, r["src_entity"]), _entity_name(conn, r["dst_entity"])
    direction = r["direction"] if "direction" in r.keys() else None
    subj, obj = (dst, src) if direction == "backward" else (src, dst)
    tag = f" (as of {r['as_of']})" if r["temporality"] == "state" and r["as_of"] else ""
    return f"{subj} {rel} {obj}{tag}"


@mcp.tool()
def entity_neighborhood(entity: str, limit: int = 15) -> str:
    """What's connected to an entity in the knowledge graph — the people,
    projects, tools, and orgs it co-occurs with across all conversations,
    strongest first, with the active date span. Answers 'what's related to X'.
    """
    conn = db()
    try:
        e = _resolve_entity(conn, entity)
        if not e:
            return f"No entity found for '{entity}'. Try a different name."
        eid = e["entity_id"]
        rows = conn.execute(
            "SELECT src_entity, dst_entity, relation, verified_relation, direction, "
            "temporality, as_of, weight, first_seen, last_seen "
            "FROM edges WHERE src_entity=? OR dst_entity=? ORDER BY weight DESC LIMIT ?",
            (eid, eid, max(1, min(limit, 40)))).fetchall()
        if not rows:
            return f"{e['canonical']} ({e['type']}) has no graph connections yet."
        out = [f"{e['canonical']} ({e['type']}) — connected to:"]
        for r in rows:
            out.append(f"  {_rel_sentence(conn, r)}  "
                       f"[{r['weight']} shared contexts, {r['first_seen']}..{r['last_seen']}]")
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def entity_timeline(entity: str, limit: int = 25, speaker: str = "") -> str:
    """Chronological record of what happened with an entity — the dated
    decisions, problems, facts, advice, and lessons involving it, oldest
    first. Answers 'what happened with X over time' and 'how did X evolve'.
    Optional speaker filter: "user", "assistant" (or "agent"), or "both".
    """
    conn = db()
    try:
        e = _resolve_entity(conn, entity)
        if not e:
            return f"No entity found for '{entity}'."
        eid = e["entity_id"]
        # aboutness terms — canonical, its de-parenthesized core, and every alias
        terms = {e["canonical"].lower(), e["canonical"].split(" (")[0].strip().lower()}
        for row in conn.execute("SELECT alias FROM entity_aliases WHERE entity_id=?", (eid,)):
            terms.add((row["alias"] or "").lower())
        terms = {t for t in terms if t}
        spk = {"agent": "assistant"}.get(speaker.lower().strip(), speaker.lower().strip())
        spk_sql, params = "", [eid, *EXCLUDED_OWNERS]
        if spk in ("user", "assistant", "both"):
            spk_sql = " AND a.speaker=?"
            params.append(spk)
        rows = conn.execute("""
            SELECT DISTINCT a.type, a.statement, a.speaker, a.conv_key, a.evidence, c.created_at
            FROM assertions a
            JOIN mentions m ON m.conv_key=a.conv_key AND m.chunk_idx=a.chunk_idx
            JOIN entity_map em ON em.mention_id=m.id
            JOIN conversations c ON c.key=a.conv_key
            WHERE em.entity_id=? AND c.owner NOT IN ({owners}){spk}
            ORDER BY c.created_at LIMIT 400
        """.format(owners=",".join("?" * len(EXCLUDED_OWNERS)) or "''", spk=spk_sql),
            params).fetchall()
        out, n = [f"Timeline for {e['canonical']} ({e['type']}):"], 0
        for r in rows:
            # aboutness: keep only assertions that actually name the entity (as a
            # whole word — so "Ron" doesn't match "st_ron_g"/"f_ron_t"), not
            # everything that merely co-occurred in the same chunk
            stmt = r["statement"] or ""
            if not any(re.search(r"\b" + re.escape(t) + r"\b", stmt, re.I) for t in terms):
                continue
            ref = ""
            try:
                ids = json.loads(r["evidence"] or "[]")
                if ids:
                    ref = f"  ⟨{r['conv_key']}#{ids[0]}⟩"   # provenance ref
            except Exception:
                pass
            out.append(f"  [{(r['created_at'] or '?')[:10]}] ({r['type']}/{r['speaker']}) "
                       f"{r['statement'][:180]}{ref}")
            n += 1
            if n >= max(1, min(limit, 60)):
                break
        if n == 0:
            return f"No timeline entries clearly about {e['canonical']}."
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def what_connects(entity_a: str, entity_b: str) -> str:
    """How two entities relate: direct graph edges between them plus their
    shared connections (things both are linked to). Answers 'how are X and
    Y connected'."""
    conn = db()
    try:
        ea, eb = _resolve_entity(conn, entity_a), _resolve_entity(conn, entity_b)
        if not ea or not eb:
            return f"Could not resolve {'A' if not ea else 'B'} entity."
        a, b = ea["entity_id"], eb["entity_id"]
        direct = conn.execute(
            "SELECT src_entity, dst_entity, relation, verified_relation, direction, temporality, "
            "as_of, weight, first_seen, last_seen FROM edges "
            "WHERE (src_entity=? AND dst_entity=?) OR (src_entity=? AND dst_entity=?)",
            (a, b, b, a)).fetchall()

        def _weighted_neighbors(eid):
            d = {}
            for r in conn.execute(
                    "SELECT CASE WHEN src_entity=? THEN dst_entity ELSE src_entity END AS o, weight "
                    "FROM edges WHERE src_entity=? OR dst_entity=?", (eid, eid, eid)):
                d[r["o"]] = d.get(r["o"], 0) + r["weight"]
            return d
        wa, wb = _weighted_neighbors(a), _weighted_neighbors(b)
        shared = (set(wa) & set(wb)) - {a, b}
        out = [f"{ea['canonical']} <-> {eb['canonical']}:"]
        if direct:
            r = direct[0]
            out.append(f"  DIRECT: {_rel_sentence(conn, r)}  "
                       f"({r['weight']} shared contexts, {r['first_seen']}..{r['last_seen']})")
        else:
            out.append("  no direct edge")
        if shared:
            # strongest shared links first, not alphabetical
            ranked = sorted(shared, key=lambda s: -(wa[s] + wb[s]))
            names = [_entity_name(conn, s) for s in ranked[:15]]
            out.append(f"  both connect to: {', '.join(names)}")
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def list_themes() -> str:
    """The major themes of the archive — clusters of tightly-connected entities
    detected in the knowledge graph. Answers 'what are the big areas of my
    life/work'. Each theme lists its most-central entities."""
    conn = db()
    try:
        comms = conn.execute("SELECT community_id, label, size FROM communities ORDER BY size DESC").fetchall()
        if not comms:
            return "No themes computed yet."
        out = []
        for c in comms:
            tops = conn.execute("""
                SELECT e.canonical FROM entity_community ec
                JOIN entities e ON e.entity_id=ec.entity_id
                WHERE ec.community_id=? ORDER BY e.mentions DESC LIMIT 8""", (c["community_id"],)).fetchall()
            out.append(f"[{c['size']} entities] {c['label']}\n    {', '.join(t['canonical'] for t in tops)}")
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def theme_of(entity: str) -> str:
    """Which theme/cluster an entity belongs to, and its fellow members.
    Answers 'what's in the same area as X'."""
    conn = db()
    try:
        e = _resolve_entity(conn, entity)
        if not e:
            return f"No entity found for '{entity}'."
        row = conn.execute("""SELECT c.community_id, c.label FROM entity_community ec
                              JOIN communities c ON c.community_id=ec.community_id
                              WHERE ec.entity_id=?""", (e["entity_id"],)).fetchone()
        if not row:
            return f"{e['canonical']} isn't assigned to a theme."
        members = conn.execute("""SELECT e.canonical FROM entity_community ec
            JOIN entities e ON e.entity_id=ec.entity_id
            WHERE ec.community_id=? ORDER BY e.mentions DESC LIMIT 20""", (row["community_id"],)).fetchall()
        return (f"{e['canonical']} belongs to theme: {row['label']}\n  members: "
                + ", ".join(m["canonical"] for m in members))
    finally:
        conn.close()


@mcp.tool()
def memory_status() -> str:
    """Memory core freshness: per-source conversation counts, last ingest time,
    and semantic-index (embedding) coverage."""
    conn = db()
    try:
        rows = conn.execute(
            "SELECT source, COUNT(*) AS n, SUM(message_count) AS msgs, MAX(ingested_at) AS last FROM conversations GROUP BY source"
        ).fetchall()
        embeddable = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE text != ''").fetchone()["n"]
        embedded = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()["n"]
        excluded = conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE owner IN ({})".format(
                ",".join("?" * len(EXCLUDED_OWNERS)) or "''"
            ), [*EXCLUDED_OWNERS],
        ).fetchone()
    finally:
        conn.close()
    lines = [f"{r['source']}: {r['n']} conversations, {r['msgs']} messages, last ingest {r['last']}" for r in rows]
    lines.append(f"excluded from retrieval (privacy partition): {excluded['n']} conversations")
    pct = (100 * embedded // embeddable) if embeddable else 0
    lines.append(f"semantic index: {embedded}/{embeddable} messages embedded ({pct}%)")
    return "\n".join(lines) or "Memory core is empty."


# ── App assembly ──────────────────────────────────────────────────────────────

init_db()


import contextlib


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
