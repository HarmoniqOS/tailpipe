"""
ChatGPT conversation normalizer — Tailpipe ingestion.

Maps a raw chatgpt.com conversation JSON (from the session-API detail endpoint
/backend-api/conversation/{id}) into the same Tailpipe schema-v1 format as the
Claude normalizer: conversation metadata + message list with native IDs, parent
pointers (the `mapping` tree is preserved), role mapping, content blocks, and
per-message content hashes for the dedup gate.

Configuration:
    TAILPIPE_URL   ingest endpoint (default: http://localhost:8080/ingest)
    INGEST_TOKEN   bearer token (optional)

Usage:
    python normalize_chatgpt_export.py <raw_export.json> [out.json]
    python normalize_chatgpt_export.py <raw_export.json> --ingest
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "chatgpt.com"

TAILPIPE_URL = os.environ.get("TAILPIPE_URL", "http://localhost:8080/ingest")
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")


def iso(ts) -> str:
    """ChatGPT uses unix epoch floats; normalize to ISO 8601 UTC."""
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def normalize_content(content: dict) -> list:
    """Map ChatGPT content shapes onto the shared block vocabulary."""
    ctype = content.get("content_type", "unknown")

    if ctype == "text":
        return [{"type": "text", "text": "\n".join(p for p in content.get("parts", []) if isinstance(p, str))}]

    if ctype == "thoughts":
        # Reasoning summaries — ChatGPT's analog of Claude's thinking summaries.
        # Entries may carry `summary` and/or `content`; some are empty.
        summaries, texts = [], []
        for t in content.get("thoughts") or []:
            if isinstance(t, dict):
                if t.get("summary"):
                    summaries.append(t["summary"])
                if t.get("content"):
                    texts.append(t["content"])
        block = {"type": "thinking", "text": "\n".join(texts)}
        if summaries:
            block["summaries"] = summaries
        return [block]

    if ctype == "reasoning_recap":
        return [{"type": "thinking", "summaries": [content.get("content", "")], "text": ""}]

    if ctype == "code":
        return [{
            "type": "tool_use",
            "tool_name": content.get("response_format_name") or f"code:{content.get('language', '')}",
            "input": {"language": content.get("language"), "code": content.get("text", "")},
        }]

    if ctype == "execution_output":
        return [{"type": "tool_result", "tool_name": "code", "content": [{"type": "text", "text": content.get("text", "")}], "is_error": False}]

    if ctype == "multimodal_text":
        blocks = []
        for p in content.get("parts", []):
            if isinstance(p, str):
                blocks.append({"type": "text", "text": p})
            elif isinstance(p, dict):
                # Voice-mode transcripts are real speech — surface as text so
                # they index and read like any other message.
                if p.get("content_type") == "audio_transcription" and p.get("text"):
                    blocks.append({"type": "text", "text": p["text"], "modality": "voice"})
                else:
                    blocks.append({"type": p.get("content_type", "asset"), "raw": {k: v for k, v in p.items() if v is not None}})
        return blocks

    # Unknown content type: keep raw so nothing is silently dropped
    return [{"type": ctype, "raw": {k: v for k, v in content.items() if k != "content_type" and v is not None}}]


def flat_text(content: list) -> str:
    parts = []
    for b in content:
        if b["type"] in ("text", "thinking"):
            if b.get("text"):
                parts.append(b["text"])
            parts.extend(b.get("summaries", []))
        elif b["type"] == "tool_use":
            parts.append(f"[tool_use:{b.get('tool_name', '')}]")
        elif b["type"] == "tool_result":
            parts.append(f"[tool_result:{b.get('tool_name', '')}]")
    return "\n".join(p for p in parts if p)


def content_hash(role: str, text: str) -> str:
    return hashlib.sha256(f"{role}\x00{text}".encode("utf-8")).hexdigest()[:16]


def normalize(raw: dict) -> dict:
    mapping = raw.get("mapping", {})
    messages = []

    for node_id, node in mapping.items():
        msg = node.get("message")
        if not msg:
            continue  # synthetic root node
        role = msg.get("author", {}).get("role", "unknown")
        content = normalize_content(msg.get("content", {}))
        text = flat_text(content)
        messages.append({
            "native_id": node_id,
            "parent_native_id": node.get("parent"),
            "role": role,
            "created_at": iso(msg.get("create_time")),
            "updated_at": iso(msg.get("update_time")),
            "content": content,
            "content_hash": content_hash(role, text),
            "model": (msg.get("metadata") or {}).get("model_slug"),
            "hidden": bool((msg.get("metadata") or {}).get("is_visually_hidden_from_conversation")),
        })

    by_id = {m["native_id"]: m for m in messages}
    children: dict = {}
    for m in messages:
        children.setdefault(m["parent_native_id"], []).append(m["native_id"])
    branch_points = {p: kids for p, kids in children.items() if p in by_id and len(kids) > 1}

    active_path = set()
    cursor = raw.get("current_node")
    while cursor and cursor in mapping:
        active_path.add(cursor)
        cursor = mapping[cursor].get("parent")
    for m in messages:
        m["on_active_path"] = m["native_id"] in active_path

    return {
        "schema_version": 1,
        "conversation": {
            "source": SOURCE,
            "native_id": raw.get("conversation_id"),
            "title": raw.get("title", ""),
            "summary": "",
            "model": raw.get("default_model_slug"),
            "project_native_id": raw.get("gizmo_id"),
            "created_at": iso(raw.get("create_time")),
            "updated_at": iso(raw.get("update_time")),
            "current_leaf_native_id": raw.get("current_node"),
            "sync": {"ingestor": "normalize_chatgpt_export", "captured_via": "export"},
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
    ap.add_argument("input", help="raw chatgpt.com conversation JSON")
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
