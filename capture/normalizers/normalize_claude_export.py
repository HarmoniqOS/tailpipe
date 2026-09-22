"""
Claude.ai conversation normalizer — Tailpipe ingestion.

Maps a raw claude.ai conversation JSON (from the session-API detail endpoint,
?tree=True&rendering_mode=messages) into the Tailpipe schema-v1 format:
conversation metadata + message list with native IDs, parent pointers (branch
tree preserved), role mapping, content blocks, and per-message content hashes
for the dedup gate.

Configuration:
    TAILPIPE_URL   ingest endpoint (default: http://localhost:8080/ingest)
    INGEST_TOKEN   bearer token (optional)

Usage:
    python normalize_claude_export.py <raw_export.json> [out.json]
    python normalize_claude_export.py <raw_export.json> --ingest
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

SOURCE = "claude.ai"
ROLE_MAP = {"human": "user", "assistant": "assistant"}

TAILPIPE_URL = os.environ.get("TAILPIPE_URL", "http://localhost:8080/ingest")
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")


def normalize_block(block: dict) -> dict:
    """Keep every block faithfully; strip only nulls and UI-layout noise."""
    btype = block.get("type", "unknown")
    out = {"type": btype}

    if btype == "text":
        out["text"] = block.get("text", "")
        if block.get("citations"):
            out["citations"] = block["citations"]
    elif btype == "thinking":
        # claude.ai redacts raw thinking (thinking_hidden=true) but keeps
        # per-block summaries — capture those; they're free topic annotations.
        out["text"] = block.get("thinking") or block.get("text", "")
        if block.get("start_timestamp"):
            out["start_timestamp"] = block["start_timestamp"]
        if block.get("stop_timestamp"):
            out["stop_timestamp"] = block["stop_timestamp"]
        summaries = [
            s.get("summary", s) if isinstance(s, dict) else s
            for s in block.get("summaries") or []
        ]
        if summaries:
            out["summaries"] = summaries
        if block.get("thinking_hidden"):
            out["redacted"] = True
    elif btype == "tool_use":
        out["tool_name"] = block.get("name", "")
        out["input"] = block.get("input", {})
    elif btype == "tool_result":
        out["tool_name"] = block.get("name", "")
        out["content"] = block.get("content", [])
        out["is_error"] = bool(block.get("is_error"))
    else:
        # Unknown block type: keep raw so nothing is silently dropped
        out["raw"] = {k: v for k, v in block.items() if k != "type" and v is not None}

    return out


def flat_text(content: list) -> str:
    """Flatten content blocks to text for hashing and search seeding."""
    parts = []
    for b in content:
        if b["type"] in ("text", "thinking"):
            parts.append(b.get("text", ""))
            parts.extend(b.get("summaries", []))
        elif b["type"] == "tool_use":
            parts.append(f"[tool_use:{b.get('tool_name', '')}]")
        elif b["type"] == "tool_result":
            parts.append(f"[tool_result:{b.get('tool_name', '')}]")
    return "\n".join(p for p in parts if p)


def content_hash(role: str, text: str) -> str:
    return hashlib.sha256(f"{role}\x00{text}".encode("utf-8")).hexdigest()[:16]


def normalize(raw: dict) -> dict:
    messages = []
    for m in raw.get("chat_messages", []):
        content = [normalize_block(b) for b in m.get("content", [])]
        role = ROLE_MAP.get(m.get("sender", ""), m.get("sender", "unknown"))
        text = flat_text(content)
        messages.append({
            "native_id": m["uuid"],
            "parent_native_id": m.get("parent_message_uuid"),
            "role": role,
            "created_at": m.get("created_at"),
            "updated_at": m.get("updated_at"),
            "content": content,
            "content_hash": content_hash(role, text),
            "truncated": bool(m.get("truncated")),
            # Keep attachment/file objects whole — exports carry
            # extracted_content (pasted file text), which is corpus data.
            "attachments": m.get("attachments") or [],
            "files": m.get("files") or [],
        })

    # Branch analysis: parents with >1 child are edit/retry points; the active
    # path is the chain from current_leaf_message_uuid back to the root.
    children: dict = {}
    by_id = {m["native_id"]: m for m in messages}
    for m in messages:
        children.setdefault(m["parent_native_id"], []).append(m["native_id"])
    branch_points = {p: kids for p, kids in children.items() if p in by_id and len(kids) > 1}

    active_path = set()
    cursor = raw.get("current_leaf_message_uuid")
    while cursor and cursor in by_id:
        active_path.add(cursor)
        cursor = by_id[cursor]["parent_native_id"]
    if active_path:
        for m in messages:
            m["on_active_path"] = m["native_id"] in active_path
    else:
        # Official exports are flattened linear transcripts with no leaf
        # pointer — every message is the live path.
        for m in messages:
            m["on_active_path"] = True

    return {
        "schema_version": 1,
        "conversation": {
            "source": SOURCE,
            "native_id": raw["uuid"],
            "title": raw.get("name", ""),
            "summary": raw.get("summary", ""),
            "model": raw.get("model"),
            "project_native_id": raw.get("project_uuid"),
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
            "current_leaf_native_id": raw.get("current_leaf_message_uuid"),
            "sync": {"ingestor": "normalize_claude_export", "captured_via": "export"},
        },
        "messages": messages,
        "stats": {
            "message_count": len(messages),
            "branch_points": len(branch_points),
            "abandoned_messages": sum(1 for m in messages if not m["on_active_path"]),
        },
    }


def ingest(record: dict) -> dict:
    """POST a schema-v1 record to the configured Tailpipe ingest endpoint."""
    body = json.dumps(record).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    req = urllib.request.Request(TAILPIPE_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="raw claude.ai conversation JSON")
    ap.add_argument("output", nargs="?", help="output path (default: <input>.normalized.json)")
    ap.add_argument("--ingest", action="store_true", help="POST result to TAILPIPE_URL after writing")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_suffix(".normalized.json")

    raw = json.loads(src.read_text(encoding="utf-8"))
    norm = normalize(raw)
    out.write_text(json.dumps(norm, indent=2, ensure_ascii=False), encoding="utf-8")

    conv, stats = norm["conversation"], norm["stats"]
    roles: dict = {}
    blocks: dict = {}
    for m in norm["messages"]:
        roles[m["role"]] = roles.get(m["role"], 0) + 1
        for b in m["content"]:
            blocks[b["type"]] = blocks.get(b["type"], 0) + 1

    print(f"Normalized: {conv['title']}")
    print(f"  source={conv['source']}  native_id={conv['native_id']}")
    print(f"  model={conv['model']}  span={conv['created_at']} -> {conv['updated_at']}")
    print(f"  messages={stats['message_count']}  roles={roles}")
    print(f"  blocks={blocks}")
    print(f"  branch_points={stats['branch_points']}  abandoned_messages={stats['abandoned_messages']}")
    print(f"  -> {out}")

    if args.ingest:
        result = ingest(norm)
        print(f"  ingested: {result}")


if __name__ == "__main__":
    main()
