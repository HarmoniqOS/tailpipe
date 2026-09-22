"""
Gemini batchexecute capture decoder — Tailpipe exploration tool.

Decodes a raw batchexecute response (captured via the console fetch/XHR hook)
into readable JSON, and heuristically extracts conversation turns from the
hNvQHb (conversation read) payload.

NOTE: This is an exploratory decoder, not a full normalizer. It does not
produce schema-v1 output. Gemini's batchexecute format is undocumented and
the heuristics here may break across API versions.

Usage:
    python parse_gemini_capture.py <gemini_capture.txt>
"""

import json
import re
import sys
from pathlib import Path


def decode_batchexecute(text: str) -> list:
    """Strip )]}' armor and length prefixes; return list of (rpcid, payload)."""
    text = text.lstrip()
    if text.startswith(")]}'"):
        text = text[4:]

    envelopes = []
    # Chunks are: a decimal length on its own line, then that many chars of JSON.
    # Simplest robust approach: find every top-level JSON array that starts with
    # [["wrb.fr" — split on lines that are pure digits and try to parse the rest.
    for m in re.finditer(r'\[\["wrb\.fr".*?\]\]\s*$', text, re.DOTALL | re.MULTILINE):
        pass  # regex over deeply nested arrays is unreliable; parse chunked instead

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            # Declared length is in UTF-16 code units — unreliable to count in
            # Python code points. Parse greedily: accumulate lines until the
            # chunk parses as JSON (chunks are usually a single line anyway).
            i += 1
            chunk = ""
            arr = None
            while i < len(lines):
                chunk += lines[i]
                i += 1
                try:
                    arr = json.loads(chunk)
                    break
                except json.JSONDecodeError:
                    chunk += "\n"
                    continue
            for entry in arr or []:
                if isinstance(entry, list) and entry and entry[0] == "wrb.fr":
                    rpcid = entry[1]
                    try:
                        payload = json.loads(entry[2]) if isinstance(entry[2], str) and entry[2] else entry[2]
                    except json.JSONDecodeError:
                        payload = entry[2]
                    envelopes.append((rpcid, payload))
        else:
            i += 1
    return envelopes


def walk_strings(node, path="$"):
    """Yield (path, string) for every meaningful string in the tree."""
    if isinstance(node, str):
        if len(node) > 40 and not node.startswith(("c_", "r_", "rc_", "$", "http")):
            yield path, node
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            yield from walk_strings(item, f"{path}[{idx}]")
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, f"{path}.{k}")


def extract_turns(payload) -> list:
    """
    Heuristic turn extraction from the hNvQHb payload.

    Observed layout: payload[0] is a list of turns, newest first. Each turn
    carries [conversation_id, response_id] pairs, the user query text, and
    model response candidate(s). We identify user text and candidate text by
    position of the long strings within each turn subtree.
    """
    turns = []
    turn_list = payload[0] if payload and isinstance(payload[0], list) else []
    for turn in turn_list:
        if not isinstance(turn, list):
            continue
        strings = [(p, s) for p, s in walk_strings(turn)]
        if not strings:
            continue
        # First long string in a turn is consistently the user query; the
        # longest remaining string is the selected model response.
        user_text = strings[0][1]
        rest = [s for _, s in strings[1:]]
        model_text = max(rest, key=len) if rest else ""
        ids = [x for x in _flatten(turn) if isinstance(x, str) and re.fullmatch(r"[cr]c?_[0-9a-f]+", x)]
        turns.append({
            "ids": list(dict.fromkeys(ids))[:3],
            "user": user_text,
            "model": model_text,
        })
    return turns


def _flatten(node):
    if isinstance(node, list):
        for item in node:
            yield from _flatten(item)
    else:
        yield node


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    text = src.read_text(encoding="utf-8", errors="replace")
    envelopes = decode_batchexecute(text)
    print(f"decoded envelopes: {[(rpc, 'payload' if p else 'empty') for rpc, p in envelopes]}")

    for rpcid, payload in envelopes:
        if rpcid != "hNvQHb" or not payload:
            continue
        out = src.with_suffix(".decoded.json")
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"hNvQHb payload decoded -> {out}")

        turns = extract_turns(payload)
        print(f"\nextracted {len(turns)} turns:")
        for t in turns[:6]:
            print(f"\n  ids: {t['ids']}")
            print(f"  USER:  {t['user'][:140]}")
            print(f"  MODEL: {t['model'][:140]}")


if __name__ == "__main__":
    main()
