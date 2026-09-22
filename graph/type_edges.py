"""
Tailpipe graph pipeline — Stage 3: edge typing.

The memory core's edge builder types every edge by the dominant assertion kind
in the chunks where two entities co-occur — so almost everything starts as
"fact" or "problem". This pass upgrades each edge to a semantic typed relation,
a direction (which entity is the subject), and a temporality (enduring fact vs
a point-in-time state).

The LLM reads the real supporting evidence for each edge and chooses from a
controlled relation vocabulary, guided by a strict JSON schema so the output
can be written back to the core without manual review.

Config env vars:
    TAILPIPE_URL          — memory core base URL (default: http://localhost:8080)
    INGEST_TOKEN          — bearer token for the memory core API
    EXTRACT_ENGINE        — "anthropic" or "openai_compat"
    ANTHROPIC_API_KEY     — required when EXTRACT_ENGINE=anthropic
    ANTHROPIC_MODEL       — defaults to claude-haiku-4-5-20251001
    OPENAI_COMPAT_URL     — required when EXTRACT_ENGINE=openai_compat
    OPENAI_COMPAT_KEY     — key for the compat server
    OPENAI_COMPAT_MODEL   — model name for the compat server

Usage:
    python -u -m graph.type_edges [--min-weight 2] [--batch 5] [--limit N] [--dry]

Options:
    --min-weight  Only type edges with at least this many shared contexts (default: 2).
    --batch       Edges per LLM call — larger batches amortise latency (default: 5).
    --page        Edges fetched per core request (default: 100).
    --limit       Stop after N typed edges (0 = all).
    --dry         Print typings to stdout without writing back to the core.

Progress is appended to graph/edge_typing_progress.jsonl.
A live status page is available at http://<TAILPIPE_URL>/api/graph/typing_status.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


# ── Config from env ───────────────────────────────────────────────────────────

TAILPIPE_URL   = os.environ.get("TAILPIPE_URL", "http://localhost:8080").rstrip("/")
INGEST_TOKEN   = os.environ.get("INGEST_TOKEN", "")
EXTRACT_ENGINE = os.environ.get("EXTRACT_ENGINE", "").strip()

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL     = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OPENAI_COMPAT_URL   = os.environ.get("OPENAI_COMPAT_URL", "").rstrip("/")
OPENAI_COMPAT_KEY   = os.environ.get("OPENAI_COMPAT_KEY", "")
OPENAI_COMPAT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "")

PROGRESS = Path(__file__).parent / "edge_typing_progress.jsonl"


# ── Relation vocabulary ───────────────────────────────────────────────────────
# Framed as "SUBJECT <relation> OBJECT". The model picks one relation and
# declares which entity is the subject via the direction field.

RELATIONS = {
    "part-of":         "SUBJECT is a component / subsystem / feature of OBJECT",
    "uses":            "SUBJECT uses / depends on / is built with OBJECT (tech, tool, service)",
    "works-on":        "SUBJECT (a person) works on / builds / owns OBJECT (a project)",
    "works-for":       "SUBJECT (a person) works for / is affiliated with OBJECT (an org)",
    "created-by":      "SUBJECT was created / authored / founded by OBJECT",
    "solved-by":       "SUBJECT (a problem) is solved / addressed by OBJECT (a solution/tool)",
    "decided-because": "SUBJECT (a decision/choice) was made because of OBJECT (a reason/factor)",
    "superseded-by":   "SUBJECT was replaced / deprecated / moved on from in favour of OBJECT",
    "alternative-to":  "SUBJECT and OBJECT are competing / comparable options (symmetric)",
    "discussed-with":  "SUBJECT and OBJECT are people who interact / discuss together (symmetric)",
    "family-of":       "SUBJECT and OBJECT are family — spouse / partner / parent / sibling / relative (symmetric)",
    "located-in":      "SUBJECT is located in / part of the place OBJECT",
    "related-to":      "generic association; no stronger relation is supported by the evidence",
}

VALID_REL  = set(RELATIONS)
VALID_DIR  = {"forward", "backward", "undirected"}
VALID_TEMP = {"enduring", "state"}

VOCAB_TEXT = "\n".join(f"  - {k}: {v}" for k, v in RELATIONS.items())

TYPING_SCHEMA = {
    "type": "object",
    "properties": {
        "typings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "edge_id":     {"type": "integer"},
                    "relation":    {"type": "string", "enum": sorted(VALID_REL)},
                    "direction":   {"type": "string", "enum": sorted(VALID_DIR)},
                    "temporality": {"type": "string", "enum": sorted(VALID_TEMP)},
                    "as_of":       {"type": ["string", "null"]},
                },
                "required": ["edge_id", "relation", "direction", "temporality"],
            },
        }
    },
    "required": ["typings"],
}

PROMPT_TEMPLATE = """\
You are typing edges in a personal knowledge graph. Each edge links a "src" \
entity and a "dst" entity that co-occur across conversations. For each edge, \
read the supporting evidence and decide the SINGLE best relation (as SUBJECT \
<relation> OBJECT), which entity is the SUBJECT, and the temporality.

RELATION VOCABULARY (pick exactly one per edge):
{vocab}

EVIDENCE BAR — do NOT infer strong relations from co-occurrence:
  works-for / works-on / created-by / part-of / family-of require an EXPLICIT
  statement in the evidence ("X works at Y", "X built Y", "X is Y's wife/husband").
  If the evidence only shows the two are discussed together or merely appear in the
  same context, DO NOT assert these — it would be a false claim. A person discussed
  near an org or project does NOT mean they work there or created it. When two
  PEOPLE co-occur with no explicit stated relationship, use "discussed-with";
  otherwise use "related-to". Under-claiming is always safer than a wrong strong
  claim.

DIRECTION — which entity is the SUBJECT of the relation:
  - forward   = SUBJECT is src, OBJECT is dst
  - backward  = SUBJECT is dst, OBJECT is src
  - undirected = symmetric relations only (alternative-to, discussed-with, family-of)
  For "part-of", the SUBJECT is the smaller/contained thing. Read the entity
  types carefully to decide which plays which role.

TEMPORALITY — does the RELATIONSHIP itself persist, or is it point-in-time?
  - enduring = structural / identity / architectural / authorship / historical
    facts that stay true: part-of, created-by, works-for, located-in, uses
    (a technology in the architecture), superseded-by. Default to enduring.
  - state = a transient condition at a moment in time that must NOT calcify
    into a permanent label: "currently blocked by", "considering switching to",
    "evaluating", "in the middle of migrating to", a deadline, a mood or
    status-in-progress. Only choose state when the evidence shows an active,
    changeable situation. If state, set "as_of" to the most recent evidence
    date (YYYY-MM-DD).

Return JSON: {{"typings":[{{"edge_id":<id>,"relation":"...","direction":"...",\
"temporality":"...","as_of":"YYYY-MM-DD or null"}}, ...]}} with exactly one \
entry per edge below. Base your choice ONLY on the evidence. If nothing \
stronger is supported, use "related-to" (usually enduring, undirected).

EDGES:
{edges}"""


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {INGEST_TOKEN}"} if INGEST_TOKEN else {}


# ── Progress log ──────────────────────────────────────────────────────────────

def feed(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_engine(prompt: str) -> str:
    if EXTRACT_ENGINE == "anthropic":
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
                "max_tokens": 4000,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=300,
        )
        r.raise_for_status()
        return next(
            (b["text"] for b in r.json().get("content", []) if b.get("type") == "text"),
            "",
        )

    if EXTRACT_ENGINE == "openai_compat":
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
                "max_tokens": 4000,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object", "schema": TYPING_SCHEMA},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=1800,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content", "")

    raise ValueError(
        f"Unknown EXTRACT_ENGINE '{EXTRACT_ENGINE}'. "
        "Set it to 'anthropic' or 'openai_compat'."
    )


# ── Batch helpers ─────────────────────────────────────────────────────────────

def _render_edge(e: dict) -> str:
    lines = [
        f"edge_id {e['edge_id']}: [{e['src']['type']}] {e['src']['name']}  <->  "
        f"[{e['dst']['type']}] {e['dst']['name']}   "
        f"(seen {e['first_seen']}..{e['last_seen']}, {e['weight']} shared contexts)"
    ]
    for ev in e["evidence"]:
        suffix = f" — {ev['rationale']}" if ev.get("rationale") else ""
        lines.append(
            f"    * ({ev['type']}/{ev.get('speaker')}) {ev['statement']}{suffix}"
        )
    if not e["evidence"]:
        lines.append("    * (no assertion text; type from entity names/types alone)")
    return "\n".join(lines)


# type-consistency guard: strong relations require matching endpoint types, so
# a model over-read can't produce "family-of" on an org or "works-for" a person.
_PROJECTISH = {"project", "product", "system", "tool"}
_FAMILY_RE = re.compile(
    r"\b(wife|husband|spouse|marri|fianc|mother|father|mom|dad|parent|brother|"
    r"sister|sibling|son|daughter|grand(mother|father|ma|pa)|aunt|uncle|cousin|"
    r"niece|nephew|family|relative|in-law)\w*", re.I)


def _type_ok(rel: str, subj_type: str, obj_type: str) -> bool:
    if rel == "family-of":  return subj_type == "person" and obj_type == "person"
    if rel == "works-for":  return subj_type == "person" and obj_type == "org"
    if rel == "works-on":   return subj_type == "person" and obj_type in _PROJECTISH
    return True


def _has_family_evidence(edge: dict) -> bool:
    """family-of must be backed by an explicit family word in the evidence —
    co-occurring people are not family."""
    for ev in edge.get("evidence", []):
        if _FAMILY_RE.search((ev.get("statement") or "") + " " + (ev.get("rationale") or "")):
            return True
    return False


def _validate(typings: list, want_ids: set, edges_by_id: dict) -> list:
    clean = []
    for t in typings:
        if t.get("edge_id") not in want_ids:
            continue
        if t.get("relation") not in VALID_REL:
            continue
        if t.get("direction") not in VALID_DIR:
            continue
        if t.get("temporality") not in VALID_TEMP:
            continue
        e = edges_by_id.get(t["edge_id"])
        if e:  # downgrade relations that don't hold up (type + family evidence)
            st, dt = e["src"]["type"], e["dst"]["type"]
            subj, obj = (dt, st) if t["direction"] == "backward" else (st, dt)
            bad = not _type_ok(t["relation"], subj, obj)
            if t["relation"] == "family-of" and not _has_family_evidence(e):
                bad = True   # co-occurring people are not family
            if bad:
                t["relation"] = "discussed-with" if (st == "person" and dt == "person") else "related-to"
                t["direction"] = "undirected"
        if t["temporality"] != "state":
            t["as_of"] = None
        clean.append({k: t.get(k) for k in ("edge_id", "relation", "direction",
                                              "temporality", "as_of")})
    return clean


def _type_batch(batch: list) -> list:
    prompt = PROMPT_TEMPLATE.format(
        vocab=VOCAB_TEXT,
        edges="\n\n".join(_render_edge(e) for e in batch),
    )
    want = {e["edge_id"] for e in batch}
    for attempt in (1, 2):
        try:
            text = _call_engine(
                prompt if attempt == 1
                else prompt + "\n\nReturn STRICT JSON with one entry per edge_id above."
            )
            # Balanced-brace extraction so partial JSON is tolerated.
            start = text.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            obj = json.loads(text[start:i + 1])
                            clean = _validate(obj.get("typings", []), want,
                                              {e["edge_id"]: e for e in batch})
                            if clean:
                                return clean
                            break
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as exc:
            feed({"event": "batch_error", "attempt": attempt, "error": str(exc)[:160]})
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tailpipe edge-typing pass")
    ap.add_argument("--min-weight", type=int, default=2,
                    help="Only type edges with at least this weight (default: 2)")
    ap.add_argument("--batch", type=int, default=5,
                    help="Edges per LLM call (default: 5)")
    ap.add_argument("--page", type=int, default=100,
                    help="Edges fetched per core request (default: 100)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N typed edges (0 = all)")
    ap.add_argument("--dry", action="store_true",
                    help="Print typings but do not write back to the core")
    args = ap.parse_args()

    nh = _headers()

    total_matching = requests.get(
        f"{TAILPIPE_URL}/api/graph/edges_for_typing",
        params={"min_weight": args.min_weight, "limit": 1},
        headers=nh, timeout=60,
    ).json()["matching_total"]

    feed({"event": "start", "min_weight": args.min_weight, "batch": args.batch,
          "untyped_total": total_matching, "limit": args.limit, "dry": args.dry})
    print(f"{total_matching} untyped edges at weight>={args.min_weight}; "
          f"batch={args.batch} dry={args.dry}", flush=True)

    typed = 0
    rel_counts, temp_counts = Counter(), Counter()
    page_offset = 0  # only advances past stuck pages (model keeps failing on them)

    while True:
        edges = requests.get(
            f"{TAILPIPE_URL}/api/graph/edges_for_typing",
            params={"min_weight": args.min_weight, "limit": args.page,
                    "offset": page_offset, "only_untyped": 1},
            headers=nh, timeout=180,
        ).json()["edges"]

        if not edges:
            break
        if args.dry:
            edges = edges[:args.limit or len(edges)]

        page_typed = 0
        for i in range(0, len(edges), args.batch):
            batch = edges[i:i + args.batch]
            t0 = time.time()
            typings = _type_batch(batch)

            if typings and not args.dry:
                requests.post(
                    f"{TAILPIPE_URL}/api/graph/set_edge_types",
                    headers=nh, timeout=60,
                    json={"typings": typings},
                )

            for t in typings:
                rel_counts[t["relation"]] += 1
                temp_counts[t["temporality"]] += 1

            typed += len(typings)
            page_typed += len(typings)

            for t in typings:
                e = next(x for x in batch if x["edge_id"] == t["edge_id"])
                arrow = {"forward": "->", "backward": "<-", "undirected": "<->"}[t["direction"]]
                feed({
                    "event": "edge", "edge_id": t["edge_id"],
                    "pair": f"{e['src']['name']} {arrow} {e['dst']['name']}",
                    "relation": t["relation"], "temporality": t["temporality"],
                    "as_of": t["as_of"], "weight": e["weight"],
                })

            feed({"event": "batch", "n": len(typings), "typed_total": typed,
                  "remaining": max(0, total_matching - typed),
                  "secs": round(time.time() - t0, 1)})
            print(f"  +{len(typings)}  total={typed}/{total_matching}  "
                  f"({round(time.time() - t0, 1)}s)  "
                  f"last: {batch[0]['src']['name']}..", flush=True)

            if args.limit and typed >= args.limit:
                feed({"event": "limit_reached", "typed": typed})
                _print_summary(typed, rel_counts, temp_counts)
                return

        if args.dry:
            break

        # If an entire page produced zero writes, step over it rather than
        # fetching the same stuck edges forever.
        if page_typed == 0:
            page_offset += len(edges)
            feed({"event": "page_stall", "advanced_offset_to": page_offset})

    feed({"event": "done", "typed": typed})
    _print_summary(typed, rel_counts, temp_counts)


def _print_summary(typed, rel_counts, temp_counts):
    print(f"\n=== typed {typed} edges ===", flush=True)
    print("relations:", dict(rel_counts.most_common()), flush=True)
    print("temporality:", dict(temp_counts), flush=True)


if __name__ == "__main__":
    main()
