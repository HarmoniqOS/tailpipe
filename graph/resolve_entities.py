"""
Tailpipe graph pipeline — Stage 1: entity resolution.

Pulls candidate mention names from the memory core, asks an LLM to resolve
each batch into canonical entities (or flag them as noise/splits), assembles
the results, then POSTs the final entity set to /api/entities/build.

Pipeline role:
    extract worker  ->  [ this script ]  ->  build_edges (core)
                                          ->  type_edges  ->  communities

The resolution happens in two passes so that medium-frequency names can be
merged into (or skipped in favour of) entities already identified in the
high-frequency pass:

    Pass 1 — names ranked 1 .. TIER1_END   (high-frequency)
    Pass 2 — names ranked TIER1_END+1 .. TIER2_END  (expansion)

Each batch is saved to a JSON file so the LLM pass is resumable without
re-querying the core.  Run the assembly sub-command after all batches are
done to merge, deduplicate, and POST to the core.

LLM contract (same engines as the extract worker):
    Input:  a JSON list of {name, mentions, dominant_kind, contexts:[...]}
    Output: a JSON list of resolution objects (one per input name):
        {
          "name": "<original name>",
          "canonical": "<normalised display name>",  -- present unless skip/split
          "type": "person|org|project|tool|place|concept|other",
          "skip": true,                               -- noise / stop-word
          "split": [                                  -- name refers to N distinct entities
            {"canonical": "...", "distinguishing_note": "..."},
            ...
          ],
          "note": "optional free-text annotation"
        }
    The model should return one object per name, in order. If a name clearly
    refers to an entity already in the tier-1 canonical list (passed in the
    system context), it should use that exact canonical string.

Config env vars (shared with the rest of the pipeline):
    TAILPIPE_URL          — memory core base URL (default: http://localhost:8080)
    INGEST_TOKEN          — bearer token for the memory core API
    EXTRACT_ENGINE        — "anthropic" or "openai_compat"
    ANTHROPIC_API_KEY     — required when EXTRACT_ENGINE=anthropic
    ANTHROPIC_MODEL       — defaults to claude-haiku-4-5-20251001
    OPENAI_COMPAT_URL     — required when EXTRACT_ENGINE=openai_compat
    OPENAI_COMPAT_KEY     — key for the compat server
    OPENAI_COMPAT_MODEL   — model name for the compat server

Usage:
    # Pull candidates and write batch files:
    python -m graph.resolve_entities pull [--tier1-end N] [--tier2-end N]

    # Resolve a single batch file with the LLM (run once per batch file,
    # or loop over all of them):
    python -m graph.resolve_entities resolve --batch entities_tier1_batch_0.json

    # Assemble all resolved batches and POST to the core:
    python -m graph.resolve_entities assemble

    # Or run everything end-to-end (pull -> resolve all -> assemble):
    python -m graph.resolve_entities run
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from fastembed import TextEmbedding


# ── Config from env ───────────────────────────────────────────────────────────

TAILPIPE_URL   = os.environ.get("TAILPIPE_URL", "http://localhost:8080").rstrip("/")
INGEST_TOKEN   = os.environ.get("INGEST_TOKEN", "")
EXTRACT_ENGINE = os.environ.get("EXTRACT_ENGINE", "").strip()

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL     = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OPENAI_COMPAT_URL   = os.environ.get("OPENAI_COMPAT_URL", "").rstrip("/")
OPENAI_COMPAT_KEY   = os.environ.get("OPENAI_COMPAT_KEY", "")
OPENAI_COMPAT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "")

# Rank window for the two resolution tiers.
TIER1_END = int(os.environ.get("TIER1_END", "180"))
TIER2_END = int(os.environ.get("TIER2_END", "350"))

# How many mention contexts to include per name (more = better LLM accuracy,
# slower + larger prompt).
MAX_CONTEXTS = 12

# LLM batch size: names per API call.
BATCH_SIZE = 20

# Names that are noise by definition and never worth resolving.
STOP_NAMES = {
    "user", "assistant", "the user", "the assistant", "ai",
    "the model", "the system", "it", "this", "that",
}

WORK_DIR = Path(__file__).parent / "resolution_work"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


# ── Core API helpers ──────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {INGEST_TOKEN}"} if INGEST_TOKEN else {}


def core_get(path: str) -> dict:
    req = urllib.request.Request(TAILPIPE_URL + path, headers=_headers())
    return json.load(urllib.request.urlopen(req, timeout=60))


def core_post(path: str, body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        TAILPIPE_URL + path, data=data,
        headers={**_headers(), "content-type": "application/json"},
        method="POST",
    )
    return json.load(urllib.request.urlopen(req, timeout=300))


# ── LLM engine dispatch ───────────────────────────────────────────────────────

RESOLUTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name":      {"type": "string"},
            "canonical": {"type": "string"},
            "type":      {"type": "string",
                          "enum": ["person", "org", "project", "tool",
                                   "place", "concept", "other"]},
            "skip":      {"type": "boolean"},
            "split":     {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical":          {"type": "string"},
                        "distinguishing_note": {"type": "string"},
                    },
                    "required": ["canonical"],
                },
            },
            "note":      {"type": "string"},
        },
        "required": ["name"],
    },
}

RESOLUTION_SYSTEM = """You are resolving entity names extracted from a personal \
conversation archive into a canonical knowledge graph. For each name:

- Set "canonical" to the normalised display form (e.g. "Acme Corp" not "acme corp").
- Set "type" to one of: person, org, project, tool, place, concept, other.
- Set "skip": true if the name is noise (pronouns, stop-words, role labels with
  no specific referent, extremely generic words).
- Set "split": [{canonical, distinguishing_note}, ...] if one name refers to
  multiple distinct entities across the conversations (the "Ron paradox").
  Include a distinguishing_note that uniquely identifies each sub-entity.
- If the name matches one of the existing canonicals listed at the end, use
  EXACTLY that canonical string so aliases merge correctly.
- Return a JSON array with one object per input name, in the same order."""


def _call_engine_text(prompt: str) -> str:
    """Call the configured LLM and return raw text."""
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
                "max_tokens": 4096,
                "temperature": 0,
                "system": RESOLUTION_SYSTEM,
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
                "max_tokens": 4096,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": RESOLUTION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "response_format": {"type": "json_object",
                                    "schema": {"type": "object",
                                               "properties": {"resolutions": RESOLUTION_SCHEMA},
                                               "required": ["resolutions"]}},
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


def _parse_array(text: str) -> list:
    """Extract the first complete JSON array or the 'resolutions' key from text."""
    text = text.strip()
    # openai_compat wraps in {"resolutions": [...]}
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "resolutions" in obj:
            return obj["resolutions"]
    except json.JSONDecodeError:
        pass
    # scan for a bare array
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return []
    return []


# ── Pull pass ─────────────────────────────────────────────────────────────────

def pull(tier1_end: int, tier2_end: int):
    """Fetch mention-name candidates and split into batch files."""
    WORK_DIR.mkdir(exist_ok=True)

    all_names = core_get(f"/api/ledger/mention_names?limit={tier2_end * 2}")["names"]
    # Filter stop words.
    all_names = [n for n in all_names if n["name"].strip().lower() not in STOP_NAMES]

    tier1 = all_names[:tier1_end]
    tier2 = all_names[tier1_end:tier2_end]

    def _enrich(names_slice, prefix):
        enriched = []
        for n in names_slice:
            data = core_get(f"/api/ledger/mentions_by_name?name={urllib.parse.quote(n['name'])}")
            ctxs, seen = [], set()
            for m in data["mentions"]:
                c = (m.get("context") or "").strip()
                key = c[:50].lower()
                if c and key not in seen:
                    seen.add(key)
                    ctxs.append(c[:180])
                if len(ctxs) >= MAX_CONTEXTS:
                    break
            enriched.append({
                "name": n["name"],
                "mentions": n["mentions"],
                "dominant_kind": (n.get("kinds") or "").split(",")[0],
                "contexts": ctxs,
            })
            print(f"  {n['name']}  ({n['mentions']} mentions, {len(ctxs)} contexts)")
        batches = [enriched[i:i + BATCH_SIZE] for i in range(0, len(enriched), BATCH_SIZE)]
        for bi, batch in enumerate(batches):
            out = WORK_DIR / f"{prefix}_batch_{bi}.json"
            out.write_text(json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> {len(enriched)} names in {len(batches)} batch files ({prefix}_batch_*.json)")
        return enriched

    print(f"=== Tier 1 ({len(tier1)} names, ranks 1-{tier1_end}) ===")
    tier1_enriched = _enrich(tier1, "entities_tier1")

    # Write the tier-1 canonical list so tier-2 resolution can merge into it.
    if (WORK_DIR / "entities_tier1_resolved.json").exists():
        existing = json.loads((WORK_DIR / "entities_tier1_resolved.json").read_text(encoding="utf-8"))
        t1_canonicals = sorted({r.get("canonical", "") for r in existing if r.get("canonical")})
        (WORK_DIR / "tier1_canonicals.json").write_text(
            json.dumps(t1_canonicals, ensure_ascii=False), encoding="utf-8"
        )
        print(f"-> tier1_canonicals.json ({len(t1_canonicals)} entries)")
    else:
        print("(tier-1 resolution not done yet; run 'resolve' for tier-1 batches first)")

    if tier2:
        print(f"\n=== Tier 2 ({len(tier2)} names, ranks {tier1_end+1}-{tier2_end}) ===")
        _enrich(tier2, "entities_tier2")


# ── Resolve pass ─────────────────────────────────────────────────────────────

def resolve_batch(batch_file: Path):
    """Resolve one batch file with the LLM. Writes resolved_<batch_file>."""
    out_path = WORK_DIR / f"resolved_{batch_file.name}"
    if out_path.exists():
        print(f"  already resolved: {out_path.name} — skipping")
        return

    batch = json.loads(batch_file.read_text(encoding="utf-8"))
    names_list = [e["name"] for e in batch]

    # Include tier-1 canonicals in the prompt when resolving tier-2 so that
    # names can be merged into already-identified entities.
    t1_path = WORK_DIR / "tier1_canonicals.json"
    t1_note = ""
    if t1_path.exists():
        t1 = json.loads(t1_path.read_text(encoding="utf-8"))
        if t1:
            t1_note = (
                "\n\nExisting canonical entities (merge into these where applicable):\n"
                + ", ".join(t1[:120])
            )

    prompt = (
        "Resolve each of the following entity names extracted from conversation archives.\n"
        "Return a JSON array with one resolution object per name, in order.\n\n"
        "NAMES:\n"
        + json.dumps(batch, ensure_ascii=False, indent=1)
        + t1_note
    )

    t0 = time.time()
    for attempt in (1, 2):
        try:
            raw = _call_engine_text(
                prompt if attempt == 1
                else prompt + "\n\nReturn a valid JSON array — one object per name in order."
            )
            results = _parse_array(raw)
            if results and len(results) == len(batch):
                break
            results = results or []  # may be partial; save what we got
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            print(f"  attempt {attempt} failed: {exc!s:.120}", flush=True)
            results = []

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {batch_file.name} -> {out_path.name}  "
          f"({len(results)}/{len(batch)} resolved, {round(time.time()-t0,1)}s)", flush=True)


def resolve_all():
    """Resolve all unresolved tier-1 batches, then update tier1_canonicals,
    then resolve all tier-2 batches."""
    tier1_batches = sorted(WORK_DIR.glob("entities_tier1_batch_*.json"))
    tier2_batches = sorted(WORK_DIR.glob("entities_tier2_batch_*.json"))

    print(f"=== Resolving {len(tier1_batches)} tier-1 batches ===")
    for f in tier1_batches:
        resolve_batch(f)

    # After tier-1 is resolved, rebuild the canonicals file for tier-2 context.
    _rebuild_tier1_canonicals()

    if tier2_batches:
        print(f"\n=== Resolving {len(tier2_batches)} tier-2 batches ===")
        for f in tier2_batches:
            resolve_batch(f)


def _rebuild_tier1_canonicals():
    resolved = []
    for f in sorted(WORK_DIR.glob("resolved_entities_tier1_batch_*.json")):
        resolved.extend(json.loads(f.read_text(encoding="utf-8")))
    canonicals = sorted({r.get("canonical", "") for r in resolved
                         if r.get("canonical") and not r.get("skip")})
    (WORK_DIR / "tier1_canonicals.json").write_text(
        json.dumps(canonicals, ensure_ascii=False), encoding="utf-8"
    )
    print(f"-> tier1_canonicals.json updated ({len(canonicals)} entries)")


# ── Assemble + POST ───────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def assemble():
    """Merge all resolved batch files into a unified entity set and POST to core."""
    # Gather raw mention counts from the pull files.
    counts = {}
    for src in WORK_DIR.glob("entities_tier*_batch_*.json"):
        for batch in json.loads(src.read_text(encoding="utf-8")):
            counts[batch["name"]] = batch["mentions"]

    # Load all resolved batches.
    resolutions = []
    for f in sorted(WORK_DIR.glob("resolved_entities_tier*_batch_*.json")):
        resolutions.extend(json.loads(f.read_text(encoding="utf-8")))

    if not resolutions:
        print("No resolved batches found. Run 'pull' then 'resolve' first.")
        return

    entities = defaultdict(lambda: {"type": None, "aliases": set(), "mentions": 0})
    splits, flags, skips = [], [], []

    for r in resolutions:
        name = r.get("name", "")
        if not name:
            continue
        if r.get("skip"):
            skips.append(name)
            continue
        if r.get("split"):
            parts = r["split"]
            splits.append({"name": name, "into": parts, "note": r.get("note", "")})
            for s in parts:
                c = _norm(s.get("canonical") or name)
                if not c:
                    continue
                ent = entities[c.lower()]
                ent["display"] = c
                ent["type"] = ent["type"] or "person"
                ent["aliases"].add(name)
                ent["mentions"] += counts.get(name, 0) // max(len(parts), 1)
            continue
        c = _norm(r.get("canonical") or name)
        ent = entities[c.lower()]
        ent["display"] = c
        ent["type"] = r.get("type") or ent["type"]
        ent["aliases"].add(name)
        ent["mentions"] += counts.get(name, 0)
        if r.get("note"):
            flags.append({"canonical": c, "name": name, "note": r["note"]})

    # Near-duplicate canonical detection (catches e.g. "WatchDog" vs "Watchdog Cyber").
    canon = sorted(entities.keys())
    dupes = [
        (entities[a]["display"], entities[b]["display"])
        for i, a in enumerate(canon)
        for b in canon[i + 1:]
        if (a in b or b in a) and abs(len(a) - len(b)) <= 8
    ]
    merges = [
        (e["display"], sorted(e["aliases"]))
        for e in entities.values()
        if len(e["aliases"]) > 1
    ]

    print(f"=== {len(entities)} unique entities from {len(resolutions)} resolved names ===")
    print(f"merges: {len(merges)}  |  splits: {len(splits)}  |  "
          f"skips: {len(skips)}  |  flags: {len(flags)}  |  near-dup: {len(dupes)}\n")

    for s in splits:
        outs = " | ".join(
            f"{x.get('canonical')} ({(x.get('distinguishing_note',''))[:45]})"
            for x in s["into"]
        )
        print(f"  split: {s['name']} -> {outs}")

    print("\n--- top merges (aliases collapsed to one node) ---")
    for disp, al in sorted(merges, key=lambda x: -len(x[1]))[:20]:
        aliases_str = ", ".join(a for a in al if a.lower() != disp.lower())
        print(f"  {disp}  <=  {aliases_str}")

    if dupes:
        print("\n--- near-duplicate canonicals (maybe merge?) ---")
        for a, b in dupes[:10]:
            print(f"  '{a}'  ~  '{b}'")

    # --- Split assignment via context embedding ---
    # Identify names that are shared across multiple canonicals (ambiguous names).
    alias_to_canons = defaultdict(set)
    for e in entities.values():
        for a in e["aliases"]:
            alias_to_canons[a].add(e["display"])
    split_names = {a: sorted(cs) for a, cs in alias_to_canons.items() if len(cs) > 1}

    split_map = {}
    if split_names:
        print(f"\n--- resolving {len(split_names)} ambiguous names via embedding ---")
        emb = embedder()
        for name, canons in split_names.items():
            def_vecs = np.array(list(emb.embed(canons)))
            def_vecs /= np.linalg.norm(def_vecs, axis=1, keepdims=True) + 1e-9
            data = core_get(f"/api/ledger/mentions_by_name?name={urllib.parse.quote(name)}")
            ms = [m for m in data["mentions"] if m.get("context")]
            if not ms:
                continue
            mvecs = np.array(list(emb.embed([m["context"][:200] for m in ms])))
            mvecs /= np.linalg.norm(mvecs, axis=1, keepdims=True) + 1e-9
            sims = mvecs @ def_vecs.T
            for m, row in zip(ms, sims):
                split_map[str(m["id"])] = canons[int(np.argmax(row))]
            dist = {c: sum(1 for _, row in zip(ms, sims) if canons[int(np.argmax(row))] == c)
                    for c in canons}
            print(f"  {name}: {dist}")

    # Build the final list, sorted by mention frequency.
    out = [
        {
            "canonical": e["display"],
            "type": e["type"],
            "mentions": e["mentions"],
            "aliases": sorted(e["aliases"]),
        }
        for e in entities.values()
        if e.get("display")
    ]
    out.sort(key=lambda x: -x["mentions"])

    # Save locally before posting (useful for inspection and re-runs).
    (WORK_DIR / "entities_assembled.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n-> entities_assembled.json ({len(out)} entities)")

    # POST to the memory core.
    payload = {"entities": out, "split_map": split_map}
    res = core_post("/api/entities/build", payload)
    print(f"-> /api/entities/build: {res}")

    # Trigger edge build.
    res2 = core_post("/api/entities/build_edges", {})
    print(f"-> /api/entities/build_edges: {res2}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tailpipe entity resolution pipeline")
    ap.add_argument("command", choices=["pull", "resolve", "assemble", "run"],
                    help=(
                        "pull — fetch candidates from core and write batch files; "
                        "resolve — call LLM on batch files; "
                        "assemble — merge resolved batches and POST to core; "
                        "run — pull + resolve + assemble end-to-end"
                    ))
    ap.add_argument("--batch", type=str, default=None,
                    help="Specific batch file to resolve (resolve sub-command only)")
    ap.add_argument("--tier1-end", type=int, default=TIER1_END,
                    help=f"Upper rank for tier-1 pass (default {TIER1_END})")
    ap.add_argument("--tier2-end", type=int, default=TIER2_END,
                    help=f"Upper rank for tier-2 expansion pass (default {TIER2_END})")
    args = ap.parse_args()

    if args.command == "pull":
        pull(args.tier1_end, args.tier2_end)

    elif args.command == "resolve":
        if args.batch:
            resolve_batch(WORK_DIR / args.batch)
        else:
            resolve_all()

    elif args.command == "assemble":
        assemble()

    elif args.command == "run":
        pull(args.tier1_end, args.tier2_end)
        resolve_all()
        assemble()


if __name__ == "__main__":
    main()
