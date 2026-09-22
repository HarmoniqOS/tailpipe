"""
Extraction schema, prompt, and message-flattening helper.

This module is the single source of truth for:
  - OUTPUT_SCHEMA  — JSON Schema used for grammar-constrained decoding
                     (openai_compat) or Anthropic tool-use input_schema
  - PROMPT         — the rolling-state extraction prompt template
  - flat()         — message content → plain text (handles multi-block formats)
  - VALID_TYPES / VALID_SPEAKERS — vocabulary for post-extraction validation
"""

MAX_MSG_CHARS = 1200

# ── Vocabulary ────────────────────────────────────────────────────────────────

VALID_TYPES = {
    "decision", "advice", "problem", "solution",
    "fact", "preference", "lesson", "open_thread", "question",
}

# "both" = joint: a decision made together, a thread both parties own.
# Mentions are always user/assistant; assertions admit both.
VALID_SPEAKERS = {"user", "assistant", "both"}


# ── JSON Schema ───────────────────────────────────────────────────────────────
# Grammar-enforced by the engine (llama.cpp constrained decoding or Anthropic
# tool-use). Structure violations are impossible; the prompt carries semantics.

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":         {"type": "string"},
                    "kind":         {"enum": ["person", "org", "project", "product",
                                              "system", "concept", "place"]},
                    "context":      {"type": "string"},
                    "speaker":      {"enum": ["user", "assistant"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "kind", "context", "speaker", "evidence_ids"],
            },
        },
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type":         {"enum": ["decision", "advice", "problem", "solution",
                                              "fact", "preference", "lesson",
                                              "open_thread", "question"]},
                    "statement":    {"type": "string"},
                    "speaker":      {"enum": ["user", "assistant", "both"]},
                    "rationale":    {"type": "string"},
                    "status":       {"enum": ["made", "considered", "deferred",
                                              "open", "solved", "blocked", "n/a"]},
                    "acceptance":   {"enum": ["accepted", "rejected", "unclear", "n/a"]},
                    "solves":       {"type": "string"},
                    "owner":        {"enum": ["user", "assistant", "both", "n/a"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "type", "statement", "speaker", "rationale",
                    "status", "acceptance", "solves", "owner", "evidence_ids",
                ],
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "summary":         {"type": "string"},
                "open_threads":    {"type": "array", "items": {"type": "string"}},
                "active_entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "open_threads", "active_entities"],
        },
    },
    "required": ["mentions", "assertions", "state"],
}


# ── Prompt ────────────────────────────────────────────────────────────────────
# Template variables: {state} (JSON), {chunk} (formatted messages)

PROMPT = """You are the extraction stage of a personal memory pipeline. Extract structured evidence from this conversation chunk between a user and an AI assistant.

CRITICAL RULES:
- Every item must have "speaker": who said/did it — "user" or "assistant". Never merge their contributions.
- Every item must have "evidence_ids": the message IDs (like "m3") where it appears. Only cite messages in THIS chunk.
- Extract only what is explicitly present. No inference beyond the text. No invented names.
- Mentions commit to NO identity: record the name exactly as stated with local context. Do not decide which person a name refers to across conversations.
- For "advice" (assistant recommending something): check the user's NEXT messages for an acceptance signal.
- Attribution = ORIGINATOR, not echoer. Credit each item to whoever first stated it. If the user states a fact and the assistant merely repeats or confirms it, speaker="user". Use "assistant" only for claims the assistant introduced.
- Dates: resolve relative references ("today", "last week", "in February") against the CHUNK DATE shown below. When a fact states a value that legitimately changes over time (a price, count, status, valuation), record the SPECIFIC value and put its date in the statement — a later, different value is an update over time, NOT a contradiction.

Return ONLY a JSON object:
{{
  "mentions": [
    {{"name": "exact name as stated", "kind": "person|org|project|product|system|concept|place",
      "context": "one line of local context", "speaker": "user|assistant", "evidence_ids": ["m1"]}}
  ],
  "assertions": [
    {{"type": "decision", "statement": "...", "rationale": "stated reason or empty", "speaker": "user|assistant", "status": "made|considered|deferred", "evidence_ids": []}},
    {{"type": "advice", "statement": "...", "rationale": "...", "speaker": "assistant", "acceptance": "accepted|rejected|unclear", "evidence_ids": []}},
    {{"type": "problem", "statement": "...", "speaker": "user|assistant", "status": "open|solved|blocked", "evidence_ids": []}},
    {{"type": "solution", "statement": "...", "solves": "problem it addresses or empty", "speaker": "user|assistant", "evidence_ids": []}},
    {{"type": "fact", "statement": "...", "speaker": "user|assistant", "evidence_ids": []}},
    {{"type": "preference", "statement": "how the user wants things done", "speaker": "user", "evidence_ids": []}},
    {{"type": "lesson", "statement": "...", "speaker": "user|assistant", "evidence_ids": []}},
    {{"type": "open_thread", "statement": "...", "owner": "user|assistant|both", "evidence_ids": []}}
  ],
  "state": {{
    "summary": "2-3 sentence running summary INCLUDING prior state",
    "open_threads": ["still-unresolved items"],
    "active_entities": ["names currently in play"]
  }}
}}

Empty arrays are fine. Fewer, well-evidenced items beat many weak ones.
Field discipline: "status" applies only to decision/problem/solution, "acceptance" only to advice, "owner" only to open_thread — set "n/a" on every field that does not apply to the assertion type. Empty string for inapplicable text fields.

PRIOR STATE (from earlier in this conversation, empty if this is the first chunk):
{state}

CHUNK DATE: {date}

CONVERSATION CHUNK:
{chunk}"""


# ── Message flattening ────────────────────────────────────────────────────────

def flat(msg: dict) -> str:
    """Flatten a message's content blocks into plain text, respecting MAX_MSG_CHARS.

    Handles:
      - bare-string content blocks (older export formats)
      - {"type": "text", "text": "..."} blocks
      - {"type": "thinking", "summaries": [...]} blocks (rendered as brief note)
    """
    parts = []
    for block in msg.get("content", []) or []:
        if isinstance(block, str):
            if block.strip():
                parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
        elif block.get("type") == "thinking" and block.get("summaries"):
            parts.append("(thinking: " + "; ".join(str(s) for s in block["summaries"]) + ")")
    text = "\n".join(parts)
    return text[:MAX_MSG_CHARS] + (" [...truncated]" if len(text) > MAX_MSG_CHARS else "")
