"""MCP server + tools — the agent-facing surface.

Exposes the memory over MCP (streamable HTTP): hybrid search, conversation reads,
and the knowledge-graph tools (neighborhood, timeline, what-connects, themes,
status). `main.py` mounts `mcp.streamable_http_app()`; importing this module
registers every @mcp.tool() on that server.
"""

import json
import re

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .config import ALLOWED_HOSTS, EXCLUDED_OWNERS
from .db import db
from .search import _date_bounds, _fts_hits, _vec_hits
from .graph import _resolve_entity, _entity_name, _rel_sentence

mcp = FastMCP(
    "tailpipe-memory",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=ALLOWED_HOSTS,
        allowed_origins=[f"http://{h}" for h in ALLOWED_HOSTS],
    ),
)


@mcp.tool()
def search_memory(query: str, source: str = "", after: str = "", before: str = "", limit: int = 8) -> str:
    """Hybrid search (keyword + semantic) across all captured AI conversations
    (Claude.ai, ChatGPT, ...). Returns scored excerpts with conversation
    references. Use get_conversation with a ref to read full context.
    Optional after/before restrict by message date (ISO, e.g. "2025-03-01")."""
    limit = max(1, min(limit, 25))
    after, before = _date_bounds(after, before)
    conn = db()
    try:
        fts = _fts_hits(conn, query, source, 25, after, before)
        try:
            vec = _vec_hits(conn, query, source, 25, after, before)
        except Exception:
            vec = []  # embedder warming up or no vectors yet — degrade to FTS

        # Reciprocal rank fusion
        scores: dict = {}
        for rank, key in enumerate(fts):
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        for rank, key in enumerate(vec):
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]

        out = []
        for conv_key, native_id in ranked:
            r = conn.execute(
                """SELECT m.role, m.created_at, substr(m.text, 1, 300) AS excerpt,
                          c.title, c.source
                   FROM messages m JOIN conversations c ON c.key = m.conv_key
                   WHERE m.conv_key = ? AND m.native_id = ?""",
                (conv_key, native_id),
            ).fetchone()
            if not r:
                continue
            via = ("kw+sem" if (conv_key, native_id) in fts and (conv_key, native_id) in vec
                   else "kw" if (conv_key, native_id) in fts else "sem")
            out.append(
                f"[{r['source']}] {r['title'] or '(untitled)'} — {r['role']} @ {r['created_at'] or '?'} ({via})\n"
                f"  ref: {conv_key}#{native_id}\n"
                f"  {(r['excerpt'] or '').strip()}"
            )
    finally:
        conn.close()

    return "\n\n".join(out) if out else "No matches."


@mcp.tool()
def get_conversation(ref: str, window: int = 6) -> str:
    """Read full message context. ref = 'source:conversation_id' (whole active
    path, newest last) or 'source:conversation_id#message_id' (window of
    messages around that hit)."""
    conv_key, _, msg_id = ref.partition("#")
    conn = db()
    try:
        conv = conn.execute("SELECT * FROM conversations WHERE key=?", (conv_key,)).fetchone()
        if not conv:
            return f"Not found: {conv_key}"
        if conv["owner"] in EXCLUDED_OWNERS:
            return "This conversation is excluded from retrieval."
        rows = conn.execute(
            "SELECT * FROM messages WHERE conv_key=? AND on_active_path=1 ORDER BY created_at",
            (conv_key,),
        ).fetchall()
    finally:
        conn.close()

    if msg_id:
        idx = next((i for i, r in enumerate(rows) if r["native_id"] == msg_id), None)
        if idx is not None:
            lo = max(0, idx - window)
            rows = rows[lo: idx + window + 1]

    header = f"{conv['title'] or '(untitled)'} [{conv['source']}] — {len(rows)} messages shown"
    body = "\n\n".join(
        f"{r['role'].upper()} ({r['created_at'] or '?'}):\n{(r['text'] or '')[:2000]}" for r in rows
    )
    return f"{header}\n\n{body}"


@mcp.tool()
def list_conversations(after: str = "", before: str = "", source: str = "", limit: int = 20) -> str:
    """Browse conversations by time — no search term needed. Answers "what was
    I working on in <period>". Dates are ISO (e.g. "2025-03-01"); source
    optionally narrows (claude.ai, chatgpt.com, claude-code, gemini.google.com).
    Returns titles, refs, dates, and message counts, newest first."""
    limit = max(1, min(limit, 100))
    after, before = _date_bounds(after, before)
    sql = "SELECT key, source, title, created_at, updated_at, message_count FROM conversations WHERE owner NOT IN ({})".format(
        ",".join("?" * len(EXCLUDED_OWNERS)) or "''")
    params = [*EXCLUDED_OWNERS]
    if source:
        sql += " AND source = ?"
        params.append(source)
    if after:
        sql += " AND updated_at >= ?"
        params.append(after)
    if before:
        sql += " AND created_at <= ?"
        params.append(before)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    conn = db()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No conversations in that range."
    return "\n".join(
        f"[{r['source']}] {(r['created_at'] or '?')[:10]} -> {(r['updated_at'] or '?')[:10]}  "
        f"({r['message_count']} msgs)  {r['title'] or '(untitled)'}\n  ref: {r['key']}"
        for r in rows
    )


@mcp.tool()
def entity_neighborhood(entity: str, limit: int = 15) -> str:
    """What's connected to an entity in the knowledge graph — the people,
    projects, tools, and orgs it co-occurs with across all conversations,
    strongest first, with the active date span. Answers 'what's related to X'.
    """
    conn = db()
    try:
        e = _resolve_entity(conn, entity)
        if not e:
            return f"No entity found for '{entity}'. Try a different name."
        eid = e["entity_id"]
        rows = conn.execute(
            "SELECT src_entity, dst_entity, relation, verified_relation, direction, "
            "temporality, as_of, weight, first_seen, last_seen "
            "FROM edges WHERE src_entity=? OR dst_entity=? ORDER BY weight DESC LIMIT ?",
            (eid, eid, max(1, min(limit, 40)))).fetchall()
        if not rows:
            return f"{e['canonical']} ({e['type']}) has no graph connections yet."
        out = [f"{e['canonical']} ({e['type']}) — connected to:"]
        for r in rows:
            out.append(f"  {_rel_sentence(conn, r)}  "
                       f"[{r['weight']} shared contexts, {r['first_seen']}..{r['last_seen']}]")
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def entity_timeline(entity: str, limit: int = 25, speaker: str = "") -> str:
    """Chronological record of what happened with an entity — the dated
    decisions, problems, facts, advice, and lessons involving it, oldest
    first. Answers 'what happened with X over time' and 'how did X evolve'.
    Optional speaker filter: "user", "assistant" (or "agent"), or "both".
    """
    conn = db()
    try:
        e = _resolve_entity(conn, entity)
        if not e:
            return f"No entity found for '{entity}'."
        eid = e["entity_id"]
        # aboutness terms — canonical, its de-parenthesized core, and every alias
        terms = {e["canonical"].lower(), e["canonical"].split(" (")[0].strip().lower()}
        for row in conn.execute("SELECT alias FROM entity_aliases WHERE entity_id=?", (eid,)):
            terms.add((row["alias"] or "").lower())
        terms = {t for t in terms if t}
        spk = {"agent": "assistant"}.get(speaker.lower().strip(), speaker.lower().strip())
        spk_sql, params = "", [eid, *EXCLUDED_OWNERS]
        if spk in ("user", "assistant", "both"):
            spk_sql = " AND a.speaker=?"
            params.append(spk)
        rows = conn.execute("""
            SELECT DISTINCT a.type, a.statement, a.speaker, a.conv_key, a.evidence, c.created_at
            FROM assertions a
            JOIN mentions m ON m.conv_key=a.conv_key AND m.chunk_idx=a.chunk_idx
            JOIN entity_map em ON em.mention_id=m.id
            JOIN conversations c ON c.key=a.conv_key
            WHERE em.entity_id=? AND c.owner NOT IN ({owners}){spk}
            ORDER BY c.created_at LIMIT 400
        """.format(owners=",".join("?" * len(EXCLUDED_OWNERS)) or "''", spk=spk_sql),
            params).fetchall()
        out, n = [f"Timeline for {e['canonical']} ({e['type']}):"], 0
        for r in rows:
            # aboutness: keep only assertions that actually name the entity (as a
            # whole word — so "Ron" doesn't match "st_ron_g"/"f_ron_t"), not
            # everything that merely co-occurred in the same chunk
            stmt = r["statement"] or ""
            if not any(re.search(r"\b" + re.escape(t) + r"\b", stmt, re.I) for t in terms):
                continue
            ref = ""
            try:
                ids = json.loads(r["evidence"] or "[]")
                if ids:
                    ref = f"  ⟨{r['conv_key']}#{ids[0]}⟩"   # provenance ref
            except Exception:
                pass
            out.append(f"  [{(r['created_at'] or '?')[:10]}] ({r['type']}/{r['speaker']}) "
                       f"{r['statement'][:180]}{ref}")
            n += 1
            if n >= max(1, min(limit, 60)):
                break
        if n == 0:
            return f"No timeline entries clearly about {e['canonical']}."
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def what_connects(entity_a: str, entity_b: str) -> str:
    """How two entities relate: direct graph edges between them plus their
    shared connections (things both are linked to). Answers 'how are X and
    Y connected'."""
    conn = db()
    try:
        ea, eb = _resolve_entity(conn, entity_a), _resolve_entity(conn, entity_b)
        if not ea or not eb:
            return f"Could not resolve {'A' if not ea else 'B'} entity."
        a, b = ea["entity_id"], eb["entity_id"]
        direct = conn.execute(
            "SELECT src_entity, dst_entity, relation, verified_relation, direction, temporality, "
            "as_of, weight, first_seen, last_seen FROM edges "
            "WHERE (src_entity=? AND dst_entity=?) OR (src_entity=? AND dst_entity=?)",
            (a, b, b, a)).fetchall()

        def _weighted_neighbors(eid):
            d = {}
            for r in conn.execute(
                    "SELECT CASE WHEN src_entity=? THEN dst_entity ELSE src_entity END AS o, weight "
                    "FROM edges WHERE src_entity=? OR dst_entity=?", (eid, eid, eid)):
                d[r["o"]] = d.get(r["o"], 0) + r["weight"]
            return d
        wa, wb = _weighted_neighbors(a), _weighted_neighbors(b)
        shared = (set(wa) & set(wb)) - {a, b}
        out = [f"{ea['canonical']} <-> {eb['canonical']}:"]
        if direct:
            r = direct[0]
            out.append(f"  DIRECT: {_rel_sentence(conn, r)}  "
                       f"({r['weight']} shared contexts, {r['first_seen']}..{r['last_seen']})")
        else:
            out.append("  no direct edge")
        if shared:
            # strongest shared links first, not alphabetical
            ranked = sorted(shared, key=lambda s: -(wa[s] + wb[s]))
            names = [_entity_name(conn, s) for s in ranked[:15]]
            out.append(f"  both connect to: {', '.join(names)}")
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def list_themes() -> str:
    """The major themes of the archive — clusters of tightly-connected entities
    detected in the knowledge graph. Answers 'what are the big areas of my
    life/work'. Each theme lists its most-central entities."""
    conn = db()
    try:
        comms = conn.execute("SELECT community_id, label, size FROM communities ORDER BY size DESC").fetchall()
        if not comms:
            return "No themes computed yet."
        out = []
        for c in comms:
            tops = conn.execute("""
                SELECT e.canonical FROM entity_community ec
                JOIN entities e ON e.entity_id=ec.entity_id
                WHERE ec.community_id=? ORDER BY e.mentions DESC LIMIT 8""", (c["community_id"],)).fetchall()
            out.append(f"[{c['size']} entities] {c['label']}\n    {', '.join(t['canonical'] for t in tops)}")
        return "\n".join(out)
    finally:
        conn.close()


@mcp.tool()
def theme_of(entity: str) -> str:
    """Which theme/cluster an entity belongs to, and its fellow members.
    Answers 'what's in the same area as X'."""
    conn = db()
    try:
        e = _resolve_entity(conn, entity)
        if not e:
            return f"No entity found for '{entity}'."
        row = conn.execute("""SELECT c.community_id, c.label FROM entity_community ec
                              JOIN communities c ON c.community_id=ec.community_id
                              WHERE ec.entity_id=?""", (e["entity_id"],)).fetchone()
        if not row:
            return f"{e['canonical']} isn't assigned to a theme."
        members = conn.execute("""SELECT e.canonical FROM entity_community ec
            JOIN entities e ON e.entity_id=ec.entity_id
            WHERE ec.community_id=? ORDER BY e.mentions DESC LIMIT 20""", (row["community_id"],)).fetchall()
        return (f"{e['canonical']} belongs to theme: {row['label']}\n  members: "
                + ", ".join(m["canonical"] for m in members))
    finally:
        conn.close()


@mcp.tool()
def memory_status() -> str:
    """Memory core freshness: per-source conversation counts, last ingest time,
    and semantic-index (embedding) coverage."""
    conn = db()
    try:
        rows = conn.execute(
            "SELECT source, COUNT(*) AS n, SUM(message_count) AS msgs, MAX(ingested_at) AS last FROM conversations GROUP BY source"
        ).fetchall()
        embeddable = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE text != ''").fetchone()["n"]
        embedded = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()["n"]
        excluded = conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE owner IN ({})".format(
                ",".join("?" * len(EXCLUDED_OWNERS)) or "''"
            ), [*EXCLUDED_OWNERS],
        ).fetchone()
    finally:
        conn.close()
    lines = [f"{r['source']}: {r['n']} conversations, {r['msgs']} messages, last ingest {r['last']}" for r in rows]
    lines.append(f"excluded from retrieval (privacy partition): {excluded['n']} conversations")
    pct = (100 * embedded // embeddable) if embeddable else 0
    lines.append(f"semantic index: {embedded}/{embeddable} messages embedded ({pct}%)")
    return "\n".join(lines) or "Memory core is empty."
