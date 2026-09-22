"""SQLite connection + schema.

`db()` opens a connection with sqlite-vec loaded; `init_db()` creates the whole
schema (idempotent) and runs migrations. `_db_lock` serializes writers — SQLite
allows one writer at a time and the ingest/ledger/graph paths all mutate.
"""

import sqlite3
import threading

import sqlite_vec

from .config import DB_PATH, RAW_DIR, EMBED_DIM

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
