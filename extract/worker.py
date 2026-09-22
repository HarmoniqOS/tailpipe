"""
Tailpipe extraction worker — Stage 1 mention/assertion extraction.

Claims conversations from the memory core's job queue (global chronological
order — life-order replay), runs rolling extraction with carried state over
every active-path chunk, validates, and submits to the ledger. Multiple
workers can run simultaneously against the same queue.

Engines (set via EXTRACT_ENGINE env var):
    anthropic     — Anthropic API (forced tool-use / JSON schema)
    openai_compat — any OpenAI-compatible server (grammar-constrained JSON
                    via response_format); configure with OPENAI_COMPAT_URL,
                    OPENAI_COMPAT_KEY, OPENAI_COMPAT_MODEL

Config env vars:
    EXTRACT_ENGINE          — "anthropic" or "openai_compat" (required)
    ANTHROPIC_API_KEY       — required if EXTRACT_ENGINE=anthropic
    OPENAI_COMPAT_URL       — required if EXTRACT_ENGINE=openai_compat
    OPENAI_COMPAT_KEY       — API key for the compat server (may be empty)
    OPENAI_COMPAT_MODEL     — model name for the compat server
    TAILPIPE_URL            — memory core base URL (default: http://localhost:8080)
    INGEST_TOKEN            — bearer token for the memory core API

Usage:
    python -m extract.worker [--worker-name NAME] [--max-conversations N]
    python -u extract/worker.py                       # for live log output
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    # Running as a module: python -m extract.worker
    from .schema import PROMPT, OUTPUT_SCHEMA, flat, VALID_TYPES, VALID_SPEAKERS
except ImportError:
    # Running as a script: python extract/worker.py — make the package dir
    # importable so the sibling schema module resolves without a parent package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import PROMPT, OUTPUT_SCHEMA, flat, VALID_TYPES, VALID_SPEAKERS


# ── Config from env ───────────────────────────────────────────────────────────

TAILPIPE_URL   = os.environ.get("TAILPIPE_URL", "http://localhost:8080").rstrip("/")
INGEST_TOKEN   = os.environ.get("INGEST_TOKEN", "")
EXTRACT_ENGINE = os.environ.get("EXTRACT_ENGINE", "").strip()

# openai_compat
OPENAI_COMPAT_URL   = os.environ.get("OPENAI_COMPAT_URL", "").rstrip("/")
OPENAI_COMPAT_KEY   = os.environ.get("OPENAI_COMPAT_KEY", "")
OPENAI_COMPAT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "")

# anthropic
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL     = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

CHUNK_MESSAGES = 12

PROGRESS = Path(__file__).parent / "worker_progress.jsonl"


# ── Auth headers ──────────────────────────────────────────────────────────────

def _core_headers() -> dict:
    if not INGEST_TOKEN:
        return {}
    return {"Authorization": f"Bearer {INGEST_TOKEN}"}


# ── Progress log ──────────────────────────────────────────────────────────────

def feed(event: dict):
    """Append a progress event to the JSONL log and print it."""
    event["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(event, ensure_ascii=False)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ── LLM engine dispatch ───────────────────────────────────────────────────────

def call_engine(engine: str, prompt: str) -> str:
    """Call the configured engine and return raw text content."""
    if engine == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 6000,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [{
                    "name": "emit_extraction",
                    "description": "Emit the structured extraction.",
                    "input_schema": OUTPUT_SCHEMA,
                }],
                "tool_choice": {"type": "tool", "name": "emit_extraction"},
            },
            timeout=300,
        )
        r.raise_for_status()
        obj = next(
            (b["input"] for b in r.json().get("content", []) if b.get("type") == "tool_use"),
            None,
        )
        return json.dumps(obj) if obj else ""

    if engine == "openai_compat":
        if not OPENAI_COMPAT_URL:
            raise ValueError("OPENAI_COMPAT_URL is not set")
        if not OPENAI_COMPAT_MODEL:
            raise ValueError("OPENAI_COMPAT_MODEL is not set")
        headers = {"content-type": "application/json"}
        if OPENAI_COMPAT_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_COMPAT_KEY}"
        r = requests.post(
            f"{OPENAI_COMPAT_URL}/chat/completions",
            headers=headers,
            json={
                "model": OPENAI_COMPAT_MODEL,
                "max_tokens": 6000,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object", "schema": OUTPUT_SCHEMA},
                # Some servers (e.g. vLLM-style) accept chat_template_kwargs to
                # suppress chain-of-thought. Pass through if the server ignores it.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=1800,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content", "")

    raise ValueError(
        f"Unknown EXTRACT_ENGINE '{engine}'. "
        "Set EXTRACT_ENGINE to 'anthropic' or 'openai_compat'."
    )


# ── Validation ────────────────────────────────────────────────────────────────

def validate(obj) -> list:
    """Semantic checks the grammar can't express. Returns a list of problem strings."""
    if not isinstance(obj, dict) or not all(k in obj for k in ("mentions", "assertions", "state")):
        return ["missing top-level keys"]
    problems = []
    for a in obj.get("assertions", []):
        if a.get("type") not in VALID_TYPES:
            problems.append(f"bad type '{a.get('type')}'")
        if a.get("speaker") not in VALID_SPEAKERS:
            problems.append(f"bad speaker '{a.get('speaker')}' on assertion")
    for m in obj.get("mentions", []):
        if m.get("speaker") not in VALID_SPEAKERS:
            problems.append(f"bad speaker '{m.get('speaker')}' on mention")
    return problems


# ── Evidence ID resolution ────────────────────────────────────────────────────

def resolve_evidence(obj: dict, id_map: dict):
    """Map short chunk-local IDs (e.g. 'm3') to real message native_ids.
    Unknown IDs are silently dropped — they were hallucinated by the model."""
    for collection in ("mentions", "assertions"):
        for item in obj.get(collection, []):
            item["evidence_ids"] = [
                id_map[e] for e in item.get("evidence_ids", []) if e in id_map
            ]


# ── Per-conversation extraction ───────────────────────────────────────────────

def extract_conversation(engine: str, conv_key: str) -> bool:
    """Fetch, chunk, extract, validate, and submit one conversation.
    Returns True on full success, False if any chunk failed irrecoverably."""
    nh = _core_headers()
    rec = requests.get(
        f"{TAILPIPE_URL}/record", params={"key": conv_key}, headers=nh, timeout=120
    ).json()
    conv = rec["conversation"]
    msgs = [
        m for m in rec["messages"]
        if isinstance(m, dict) and m.get("on_active_path") and flat(m).strip()
    ]
    n_chunks = (len(msgs) + CHUNK_MESSAGES - 1) // CHUNK_MESSAGES
    feed({
        "event": "start", "conv": conv_key,
        "title": conv.get("title", "")[:70],
        "msgs": len(msgs), "chunks": n_chunks,
    })

    state = {"summary": "", "open_threads": [], "active_entities": []}
    for ci in range(n_chunks):
        window = msgs[ci * CHUNK_MESSAGES:(ci + 1) * CHUNK_MESSAGES]
        id_map = {}
        lines = []
        for mi, m in enumerate(window):
            short = f"m{ci * CHUNK_MESSAGES + mi + 1}"
            id_map[short] = m["native_id"]
            lines.append(f"[{short}] {m['role'].upper()}: {flat(m)}")
        chunk_date = (window[0].get("created_at") or conv.get("created_at") or "")[:10]
        prompt = PROMPT.format(
            state=json.dumps(state, ensure_ascii=False),
            chunk="\n\n".join(lines),
            date=chunk_date or "unknown",
        )

        t0 = time.time()
        obj, problems = None, ["not run"]
        for attempt in (1, 2):
            try:
                raw = call_engine(engine, prompt if attempt == 1 else
                                  prompt + f"\n\nYour previous output had errors: {problems}. Correct them.")
                # Robust JSON extraction: find the outermost balanced {} pair
                obj = _parse_json_object(raw)
                problems = validate(obj) if obj else ["empty response"]
            except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
                problems = [str(exc)[:120]]
                obj = None
            if obj and not problems:
                break

        valid = bool(obj) and not problems
        if obj:
            resolve_evidence(obj, id_map)
            obj["state"] = obj.get("state") or state
            state = obj["state"]
            requests.post(
                f"{TAILPIPE_URL}/api/ledger/submit", headers=nh, timeout=60,
                json={
                    "conv_key": conv_key,
                    "chunk_idx": ci,
                    "engine": engine,
                    "valid": valid,
                    "payload": obj,
                },
            )
        feed({
            "event": "chunk", "conv": conv_key, "chunk": ci + 1, "of": n_chunks,
            "valid": valid, "problems": problems if not valid else [],
            "mentions":   len(obj.get("mentions", []))   if obj else 0,
            "assertions": len(obj.get("assertions", [])) if obj else 0,
            "secs": round(time.time() - t0, 1),
        })
        if not obj:
            return False
    return True


def _parse_json_object(text: str):
    """Extract the first complete JSON object from text.
    Uses balanced-brace scanning so partial/trailing content is ignored."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if not EXTRACT_ENGINE:
        print("ERROR: EXTRACT_ENGINE is not set. Set it to 'anthropic' or 'openai_compat'.",
              file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Tailpipe extraction worker")
    ap.add_argument("--worker-name", default=None,
                    help="Identifier shown in ledger_jobs (default: <engine>-worker)")
    ap.add_argument("--max-conversations", type=int, default=0,
                    help="Stop after N conversations (0 = run until queue is empty)")
    args = ap.parse_args()

    worker = args.worker_name or f"{EXTRACT_ENGINE}-worker"
    nh = _core_headers()
    done = 0

    while True:
        claim = requests.post(
            f"{TAILPIPE_URL}/api/ledger/claim", headers=nh, timeout=30,
            json={"worker": worker},
        ).json()
        if claim.get("done"):
            feed({"event": "queue_empty", "worker": worker, "completed": done})
            print(f"queue empty — {done} conversations completed")
            break
        key = claim["conv_key"]
        try:
            ok = extract_conversation(EXTRACT_ENGINE, key)
        except Exception as exc:
            feed({"event": "error", "conv": key, "error": str(exc)[:200]})
            ok = False
        requests.post(
            f"{TAILPIPE_URL}/api/ledger/complete", headers=nh, timeout=30,
            json={"conv_key": key, "failed": not ok},
        )
        feed({"event": "complete", "conv": key, "ok": ok})
        done += 1
        if args.max_conversations and done >= args.max_conversations:
            feed({"event": "limit_reached", "worker": worker, "completed": done})
            break


if __name__ == "__main__":
    main()
