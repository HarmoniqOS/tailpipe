"""
Claude Code transcript normalizer — Tailpipe schema-v1.

Claude Code writes each coding session as an append-only JSONL file at
`~/.claude/projects/<project-dir>/<session-id>.jsonl` — one JSON event per
line. This module turns a stream of those events into the same schema-v1 record
shape the other capture paths produce (see `capture/normalizers/`), so a whole
session becomes one conversation with a list of messages.

Only MAIN-session turns are kept: `user` / `assistant` events on the top-level
transcript. Sidechain events (`isSidechain: true`, i.e. subagents) and the
bookkeeping record types Claude Code interleaves (`queue-operation`,
`file-history-snapshot`, `mode`, `last-prompt`, ...) are ignored by design —
this captures what the human actually discussed with the agent.

The core `SessionAccumulator` is stream-friendly: feed it decoded events one at
a time (in file order) and it keeps a running view of the session that can be
serialized to a schema-v1 record at any point. This lets the watcher re-emit a
session's full record as it grows, without re-reading the whole file.

Secret redaction runs before any text leaves the record: an optional exact-match
list (env `TAILPIPE_REDACT_FILE`, one secret per line) plus a pattern sweep for
common credential shapes. Env-only — nothing is hardcoded.
"""

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

SOURCE = "claude-code"

# Record types that carry a conversational turn. Everything else on the
# transcript (queue-operation, file-history-snapshot, mode, last-prompt,
# attachment, ...) is bookkeeping and is skipped.
TURN_TYPES = ("user", "assistant")

# ── Secret redaction ──────────────────────────────────────────────────────────
# Two layers, both env-driven so nothing sensitive is baked into the code:
#   1. Exact-match against a caller-supplied secrets file (TAILPIPE_REDACT_FILE),
#      one secret per line — for tokens that live nowhere else on disk.
#   2. A pattern sweep for common credential shapes.

SECRET_PATTERNS = [
    ("bearer", re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]{16,}")),
    ("api-key", re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{20,}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("hex-key", re.compile(r"\b[0-9a-f]{40,}\b")),
    ("pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("url-cred", re.compile(r"(://[^/\s:]+):[^@/\s]+@")),
]


class Redactor:
    """Redacts secrets from text and nested JSON. Tracks counts for reporting."""

    def __init__(self, secrets_file: "str | os.PathLike | None" = None):
        self.counts: Counter = Counter()
        self.known: list = []
        if secrets_file:
            self.load_known_secrets(Path(secrets_file))

    def load_known_secrets(self, path: Path) -> None:
        if not path.exists():
            return
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                self.known.extend(
                    str(v) for v in data.values() if isinstance(v, str) and len(v) > 8
                )
                return
        except json.JSONDecodeError:
            pass
        for line in raw.splitlines():
            line = line.strip()
            if len(line) > 8 and not line.startswith("#"):
                self.known.append(line)

    def text(self, text: str) -> str:
        if not text:
            return text
        for secret in self.known:
            if secret in text:
                self.counts["known-credential"] += text.count(secret)
                text = text.replace(secret, "[REDACTED:known-credential]")
        for name, pattern in SECRET_PATTERNS:
            def sub(m, _name=name):
                self.counts[_name] += 1
                prefix = m.group(1) if m.groups() else ""
                return f"{prefix}[REDACTED:{_name}]"
            text = pattern.sub(sub, text)
        return text

    def deep(self, obj):
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, list):
            return [self.deep(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.deep(v) for k, v in obj.items()}
        return obj


# ── Content normalization ──────────────────────────────────────────────────────

def normalize_blocks(content, redactor: Redactor) -> list:
    """Map a Claude Code message's `content` to schema-v1 content blocks.

    `content` is either a plain string (simple user turns) or a list of typed
    blocks (text / thinking / tool_use / tool_result / ...). Unknown block types
    are preserved raw so nothing is silently dropped.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": redactor.text(content)}]
    blocks = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            blocks.append({"type": "text", "text": redactor.text(b.get("text", ""))})
        elif t == "thinking":
            blocks.append({"type": "thinking", "text": redactor.text(b.get("thinking", ""))})
        elif t == "tool_use":
            blocks.append({
                "type": "tool_use",
                "tool_name": b.get("name", ""),
                "input": redactor.deep(b.get("input", {})),
            })
        elif t == "tool_result":
            blocks.append({
                "type": "tool_result",
                "tool_name": str(b.get("tool_use_id", "")),
                "content": redactor.deep(b.get("content", [])),
                "is_error": bool(b.get("is_error")),
            })
        else:
            blocks.append({
                "type": t or "unknown",
                "raw": redactor.deep({k: v for k, v in b.items() if k != "type" and v is not None}),
            })
    return blocks


def flat_text(blocks: list) -> str:
    """Flatten text/thinking blocks for hashing and search seeding."""
    parts = [b["text"] for b in blocks if b["type"] in ("text", "thinking") and b.get("text")]
    return "\n".join(parts)


def content_hash(role: str, text: str) -> str:
    return hashlib.sha256(f"{role}\x00{text}".encode("utf-8")).hexdigest()[:16]


# Strip Claude Code's project-directory slug down to a human-ish name. The slug
# is the project's working directory with path separators collapsed to dashes,
# e.g. a home path like "/home/<user>/my-repo" becomes "-home-<user>-my-repo",
# and a Windows drive path becomes "<drive>--Users-<user>-my-repo". We drop the
# leading drive/home/user prefix so the title reads like the project, not a path.
_SLUG_PREFIX = re.compile(r"^[A-Za-z]?-+(?:Users|home)-[^-]+-?")


def project_name_from_slug(slug: str) -> str:
    return _SLUG_PREFIX.sub("", slug) or slug


class SessionAccumulator:
    """Accumulates the events of one Claude Code session into a schema-v1 record.

    Feed decoded JSON events (dicts) via `add_event` in file order. Call
    `to_record` to snapshot the session as a schema-v1 dict at any time. Because
    the ingest endpoint replaces a conversation's messages wholesale on each
    POST, the accumulator always holds the FULL message list — the file cursor
    only saves us from re-parsing bytes we've already seen.
    """

    def __init__(self, session_id: str, project_slug: str, redactor: Redactor):
        self.session_id = session_id
        self.project_slug = project_slug
        self.redactor = redactor
        self.messages: list = []
        self.title = ""
        self.models: Counter = Counter()
        self.cwds: Counter = Counter()
        self.branch = None
        self._seen_uuids: set = set()

    def add_event(self, rec: dict) -> bool:
        """Incorporate one decoded event. Returns True if it produced a message."""
        rtype = rec.get("type")
        if rtype == "ai-title":
            self.title = rec.get("aiTitle") or self.title
            return False
        if rtype not in TURN_TYPES or rec.get("isSidechain"):
            return False

        # Guard against re-processing the same turn (e.g. a re-read overlap).
        uuid = rec.get("uuid")
        if uuid and uuid in self._seen_uuids:
            return False

        msg = rec.get("message") or {}
        blocks = normalize_blocks(msg.get("content"), self.redactor)
        if not blocks:
            return False

        role = msg.get("role") or rtype
        if rec.get("cwd"):
            self.cwds[rec["cwd"]] += 1
        if msg.get("model"):
            self.models[msg["model"]] += 1
        if rec.get("gitBranch"):
            self.branch = rec["gitBranch"]

        if uuid:
            self._seen_uuids.add(uuid)
        self.messages.append({
            "native_id": uuid,
            "parent_native_id": rec.get("parentUuid"),
            "role": role,
            "created_at": rec.get("timestamp"),
            "updated_at": rec.get("timestamp"),
            "content": blocks,
            "content_hash": content_hash(role, flat_text(blocks)),
            "on_active_path": True,
            "git_branch": rec.get("gitBranch"),
            "model": msg.get("model"),
            "attachments": [],
            "files": [],
        })
        return True

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def to_record(self) -> "dict | None":
        """Serialize the accumulated session to a schema-v1 record, or None if
        no conversational turns have been seen yet."""
        if not self.messages:
            return None
        cwd = self.cwds.most_common(1)[0][0] if self.cwds else None
        project_name = project_name_from_slug(self.project_slug)
        session_title = self.redactor.text(self.title) or "untitled session"
        return {
            "schema_version": 1,
            "conversation": {
                "source": SOURCE,
                "native_id": self.session_id,
                "title": f"{project_name} / {session_title}",
                "summary": "",
                "model": self.models.most_common(1)[0][0] if self.models else None,
                "project_native_id": self.project_slug,
                "created_at": self.messages[0]["created_at"],
                "updated_at": self.messages[-1]["created_at"],
                "current_leaf_native_id": self.messages[-1]["native_id"],
                "attribution": {"owner": "unknown", "method": "machine-local", "confidence": 1.0},
                "sync": {
                    "ingestor": "cc_watcher",
                    "captured_via": "local-jsonl",
                    "cwd": cwd,
                    "git_branch": self.branch,
                },
            },
            "messages": self.messages,
            "stats": {
                "message_count": len(self.messages),
                "branch_points": 0,
                "abandoned_messages": 0,
            },
        }
