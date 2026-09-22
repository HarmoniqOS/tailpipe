"""Semantic embeddings + the background embed worker.

Embedding is always async: ingest stays fast; a single worker thread
continuously picks up messages that have text but no vector (covers backfill
and live ingest with the same loop). Progress is visible via memory_status.
"""

import threading

import sqlite_vec

from .config import EMBED_MODEL, EMBED_MAX_CHARS
from .db import db, _db_lock

_embedder = None
_embedder_lock = threading.Lock()


def embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(EMBED_MODEL)
        return _embedder


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
