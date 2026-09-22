"""Ingest — receive Collector schema-v1 records into SQLite + the raw archive.

Every record is appended verbatim to the canonical NDJSON archive (nothing is
ever lost), then upserted into the searchable store. Newest-wins: an older
snapshot (e.g. an official export backfilling history) never regresses a fresher
live capture.
"""

import json
from datetime import datetime, timezone

from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import RAW_DIR
from .db import db, _db_lock


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
