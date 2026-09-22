"""Knowledge graph — entities, edges, typing, communities, and the viz feed.

The pipeline: resolve mentions into canonical entities, build weighted co-occurrence
edges from the ledger, upgrade those generic edges to typed/dated semantic relations,
cluster entities into communities (themes), and serve the whole thing as a node+link
feed for the 3D graph page. The `_resolve_entity`/`_entity_name`/`_rel_sentence`
helpers are shared with the MCP tools.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from starlette.requests import Request
from starlette.responses import JSONResponse

from .db import db, _db_lock


# ── Entity helpers (shared with the MCP tools) ────────────────────────────────

def _resolve_entity(conn, name: str):
    low = name.lower()
    # 1. exact canonical
    r = conn.execute("SELECT entity_id, canonical, type FROM entities WHERE LOWER(canonical)=?", (low,)).fetchone()
    if r:
        return r
    # 2. canonical whose first word IS the query ("Cody" -> "Cody Ford (...)"),
    #    most-mentioned wins. Checked before aliases so a short form maps to its
    #    OWN canonical, not to something that merely lists it as an alias.
    r = conn.execute(
        "SELECT entity_id, canonical, type FROM entities "
        "WHERE LOWER(canonical) LIKE ? ORDER BY mentions DESC LIMIT 1", (low + " %",)).fetchone()
    if r:
        return r
    # 3. exact alias (nicknames/abbreviations that aren't canonical prefixes)
    return conn.execute(
        "SELECT e.entity_id, e.canonical, e.type FROM entity_aliases a "
        "JOIN entities e ON e.entity_id=a.entity_id WHERE LOWER(a.alias)=? "
        "ORDER BY e.mentions DESC LIMIT 1", (low,)).fetchone()


def _entity_name(conn, eid):
    r = conn.execute("SELECT canonical FROM entities WHERE entity_id=?", (eid,)).fetchone()
    return r["canonical"] if r else "?"


def _rel_sentence(conn, r):
    """Render an edge as a plain sentence — 'SUBJECT relation OBJECT [as of DATE]' —
    readable aloud without knowing which entity was queried. Grammar over glyphs."""
    rel = r["verified_relation"] or r["relation"]
    src, dst = _entity_name(conn, r["src_entity"]), _entity_name(conn, r["dst_entity"])
    direction = r["direction"] if "direction" in r.keys() else None
    subj, obj = (dst, src) if direction == "backward" else (src, dst)
    tag = f" (as of {r['as_of']})" if r["temporality"] == "state" and r["as_of"] else ""
    return f"{subj} {rel} {obj}{tag}"


# ── Entity + edge build ───────────────────────────────────────────────────────

async def entities_build(request: Request):
    """Build the entity layer. Body:
      {entities:[{canonical,type,aliases:[...]}], split_map:{mention_id:canonical}}
    Wipes+rebuilds entities/aliases/map. Auto-maps mentions by UNAMBIGUOUS alias
    (alias in exactly one entity); applies split_map for the ambiguous ones."""
    b = await request.json()
    ents = b.get("entities", [])
    split_map = b.get("split_map", {})
    with _db_lock:
        conn = db()
        try:
            conn.executescript("DELETE FROM entities; DELETE FROM entity_aliases; DELETE FROM entity_map;")
            canon_id = {}
            for e in ents:
                cur = conn.execute("INSERT INTO entities (canonical, type, mentions) VALUES (?,?,?)",
                                   (e["canonical"], e.get("type"), e.get("mentions", 0)))
                canon_id[e["canonical"]] = cur.lastrowid
            # alias -> set of entity_ids
            alias_ids = {}
            for e in ents:
                eid = canon_id[e["canonical"]]
                for a in e.get("aliases", []):
                    alias_ids.setdefault(a.lower(), set()).add(eid)
                    conn.execute("INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?,?)", (a, eid))
            # auto-map mentions by unambiguous alias
            mapped = 0
            for alias_l, eids in alias_ids.items():
                if len(eids) == 1:
                    eid = next(iter(eids))
                    mapped += conn.execute(
                        "INSERT OR REPLACE INTO entity_map (mention_id, entity_id) "
                        "SELECT id, ? FROM mentions WHERE LOWER(name)=?", (eid, alias_l)).rowcount
            # apply split assignments (override)
            split_applied = 0
            for mid, canonical in split_map.items():
                eid = canon_id.get(canonical)
                if eid:
                    conn.execute("INSERT OR REPLACE INTO entity_map (mention_id, entity_id) VALUES (?,?)",
                                 (int(mid), eid))
                    split_applied += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "entities": len(ents), "auto_mapped": mapped,
                         "split_applied": split_applied})


async def edges_build(_request: Request):
    """Generate candidate edges: within each chunk, the mapped entities are
    connected by that chunk's assertions (typed by assertion type). Weight =
    number of supporting chunks; evidence = assertion ids. Directional edges
    only where the assertion type implies direction; otherwise both."""
    with _db_lock:
        conn = db()
        try:
            conn.execute("DELETE FROM edges")
            # entities present per chunk (via mapped mentions)
            rows = conn.execute("""
                SELECT m.conv_key, m.chunk_idx, em.entity_id, c.created_at
                FROM mentions m
                JOIN entity_map em ON em.mention_id = m.id
                JOIN conversations c ON c.key = m.conv_key
            """).fetchall()
            chunk_ents = defaultdict(set)
            chunk_date = {}
            for r in rows:
                k = (r["conv_key"], r["chunk_idx"])
                chunk_ents[k].add(r["entity_id"])
                chunk_date[k] = (r["created_at"] or "")[:10]

            # assertions per chunk (type + id)
            arows = conn.execute("SELECT id, conv_key, chunk_idx, type FROM assertions").fetchall()
            chunk_asserts = defaultdict(list)
            for a in arows:
                chunk_asserts[(a["conv_key"], a["chunk_idx"])].append((a["id"], a["type"]))

            agg = defaultdict(lambda: {"w": 0, "ev": [], "first": "9999", "last": "0", "rel": Counter()})
            for k, ents in chunk_ents.items():
                if len(ents) < 2:
                    continue
                asserts = chunk_asserts.get(k, [])
                if not asserts:
                    continue
                date = chunk_date.get(k, "")
                el = sorted(ents)
                for i in range(len(el)):
                    for j in range(i + 1, len(el)):
                        pair = (el[i], el[j])
                        rec = agg[pair]
                        rec["w"] += 1
                        rec["ev"].extend(a[0] for a in asserts[:3])
                        rec["rel"].update(a[1] for a in asserts)
                        if date and date < rec["first"]:
                            rec["first"] = date
                        if date and date > rec["last"]:
                            rec["last"] = date

            written = 0
            for (s, d), rec in agg.items():
                rel = rec["rel"].most_common(1)[0][0] if rec["rel"] else "related"
                conn.execute(
                    "INSERT OR REPLACE INTO edges (src_entity, dst_entity, relation, weight, evidence, first_seen, last_seen) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (s, d, rel, rec["w"], json.dumps(rec["ev"][:20]), rec["first"], rec["last"]))
                written += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "edges": written})


async def graph_edges_dump(_request: Request):
    """Dump all edges (src, dst, weight) for external community detection."""
    conn = db()
    try:
        rows = [[r["src_entity"], r["dst_entity"], r["weight"]]
                for r in conn.execute("SELECT src_entity, dst_entity, weight FROM edges")]
    finally:
        conn.close()
    return JSONResponse({"edges": rows})


# ── Edge typing ───────────────────────────────────────────────────────────────

async def edges_for_typing(request: Request):
    """Serve edges + their supporting assertion text so a model can upgrade the
    generic relation to a semantic one and stamp temporality. Params:
    min_weight (default 2), limit (default 200), offset (0), only_untyped (1)."""
    q = request.query_params
    min_w = int(q.get("min_weight", "2"))
    limit = min(int(q.get("limit", "200")), 1000)
    offset = int(q.get("offset", "0"))
    only_untyped = q.get("only_untyped", "1") != "0"
    conn = db()
    try:
        where = "WHERE weight >= ?"
        params = [min_w]
        if only_untyped:
            where += " AND (verified IS NULL OR verified = 0)"
        edge_rows = conn.execute(
            f"SELECT edge_id, src_entity, dst_entity, relation, weight, "
            f"first_seen, last_seen FROM edges {where} "
            f"ORDER BY weight DESC LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
        out = []
        for e in edge_rows:
            src = conn.execute("SELECT canonical, type FROM entities WHERE entity_id=?",
                               (e["src_entity"],)).fetchone()
            dst = conn.execute("SELECT canonical, type FROM entities WHERE entity_id=?",
                               (e["dst_entity"],)).fetchone()
            if not src or not dst:
                continue
            # chunks where BOTH entities are mapped (real co-occurrence, not
            # chunk-bag noise): intersect each entity's (conv_key, chunk_idx) set
            def _chunks(eid):
                return {(r["conv_key"], r["chunk_idx"]) for r in conn.execute(
                    "SELECT DISTINCT m.conv_key, m.chunk_idx FROM mentions m "
                    "JOIN entity_map em ON em.mention_id=m.id WHERE em.entity_id=?", (eid,))}
            shared = _chunks(e["src_entity"]) & _chunks(e["dst_entity"])
            # assertions from those shared chunks, prefer ones naming either entity
            aliases = [a["alias"].lower() for a in conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id IN (?,?)",
                (e["src_entity"], e["dst_entity"]))]
            cand = []
            for (ck, cx) in list(shared)[:60]:
                for a in conn.execute(
                        "SELECT type, statement, speaker, rationale FROM assertions "
                        "WHERE conv_key=? AND chunk_idx=?", (ck, cx)).fetchall():
                    txt = ((a["statement"] or "") + " " + (a["rationale"] or "")).lower()
                    score = sum(1 for al in aliases if al and al in txt)
                    cand.append((score, {"type": a["type"], "speaker": a["speaker"],
                                          "statement": (a["statement"] or "")[:240],
                                          "rationale": (a["rationale"] or "")[:120]}))
            cand.sort(key=lambda x: -x[0])
            ev = [c[1] for c in cand[:7]]
            out.append({
                "edge_id": e["edge_id"],
                "src": {"name": src["canonical"], "type": src["type"]},
                "dst": {"name": dst["canonical"], "type": dst["type"]},
                "relation": e["relation"], "weight": e["weight"],
                "first_seen": e["first_seen"], "last_seen": e["last_seen"],
                "evidence": ev})
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM edges {where}", params).fetchone()["n"]
    finally:
        conn.close()
    return JSONResponse({"edges": out, "returned": len(out), "matching_total": total})


async def set_edge_types(request: Request):
    """Write verified edge typings. Body: {typings:[{edge_id, relation,
    direction, temporality, as_of}]}. direction: forward(src->dst) |
    backward(dst->src) | undirected. temporality: enduring | state."""
    b = await request.json()
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with _db_lock:
        conn = db()
        try:
            for t in b.get("typings", []):
                conn.execute(
                    "UPDATE edges SET verified_relation=?, direction=?, temporality=?, "
                    "as_of=?, verified=1, verified_at=? WHERE edge_id=?",
                    (t.get("relation"), t.get("direction"), t.get("temporality"),
                     t.get("as_of"), now, t["edge_id"]))
                n += 1
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "typed": n})


async def typing_status(request: Request):
    """Live edge-typing progress as JSON. ?min_weight defaults to 2."""
    min_w = int(request.query_params.get("min_weight", "2"))
    conn = db()
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM edges WHERE weight>=?",
                             (min_w,)).fetchone()["n"]
        typed = conn.execute("SELECT COUNT(*) AS n FROM edges WHERE weight>=? AND verified=1",
                             (min_w,)).fetchone()["n"]
        rels = {r["verified_relation"]: r["n"] for r in conn.execute(
            "SELECT verified_relation, COUNT(*) AS n FROM edges WHERE verified=1 "
            "GROUP BY verified_relation ORDER BY n DESC")}
        temps = {r["temporality"]: r["n"] for r in conn.execute(
            "SELECT temporality, COUNT(*) AS n FROM edges WHERE verified=1 GROUP BY temporality")}
        recent = []
        for r in conn.execute(
                "SELECT src_entity, dst_entity, verified_relation, direction, temporality, "
                "as_of, weight, verified_at FROM edges WHERE verified=1 "
                "ORDER BY verified_at DESC LIMIT 15"):
            recent.append({
                "src": _entity_name(conn, r["src_entity"]),
                "dst": _entity_name(conn, r["dst_entity"]),
                "relation": r["verified_relation"], "direction": r["direction"],
                "temporality": r["temporality"], "as_of": r["as_of"], "weight": r["weight"]})
        # rate: typings in the last 5 minutes
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        recent5 = conn.execute(
            "SELECT COUNT(*) AS n FROM edges WHERE verified=1 AND verified_at>=?",
            (cutoff,)).fetchone()["n"]
    finally:
        conn.close()
    rate = recent5 / 300.0
    remaining = max(0, total - typed)
    eta_s = int(remaining / rate) if rate > 0 else None
    return JSONResponse({"total": total, "typed": typed, "remaining": remaining,
                         "pct": round(100 * typed / total, 1) if total else 0,
                         "rate_per_s": round(rate, 3), "eta_seconds": eta_s,
                         "relations": rels, "temporality": temps, "recent": recent})


# ── Graph feed + communities ──────────────────────────────────────────────────

async def graph_full(request: Request):
    """Full node+link dump for the force-graph visualization. Nodes carry
    type/mentions/community; links carry the typed relation + weight +
    temporality. ?min_weight (default 3) trims density. ?depth explodes each
    entity into its detail: depth>=1 adds a sample of raw mentions, depth>=2 also
    adds the assertions (decisions/problems/solutions/facts) from those chunks."""
    min_w = int(request.query_params.get("min_weight", "3"))
    depth = int(request.query_params.get("depth", "0"))
    per = min(int(request.query_params.get("per", "16")), 80)
    conn = db()
    try:
        comm_label = {r["community_id"]: r["label"]
                      for r in conn.execute("SELECT community_id, label FROM communities")}
        ent_comm = {r["entity_id"]: r["community_id"]
                    for r in conn.execute("SELECT entity_id, community_id FROM entity_community")}
        # only include nodes that actually appear on a surviving link
        links, node_ids = [], set()
        for r in conn.execute(
                "SELECT src_entity, dst_entity, relation, verified_relation, direction, "
                "temporality, as_of, weight FROM edges WHERE weight>=?", (min_w,)):
            s, d = r["src_entity"], r["dst_entity"]
            rel = r["verified_relation"] or r["relation"]
            # orient so source is the semantic subject
            if r["direction"] == "backward":
                s, d = d, s
            links.append({"source": s, "target": d, "relation": rel,
                          "weight": r["weight"], "temporality": r["temporality"],
                          "as_of": r["as_of"], "directed": r["direction"] in ("forward", "backward")})
            node_ids.add(r["src_entity"]); node_ids.add(r["dst_entity"])
        nodes = []
        for r in conn.execute("SELECT entity_id, canonical, type, mentions FROM entities"):
            if r["entity_id"] not in node_ids:
                continue
            cid = ent_comm.get(r["entity_id"])
            nodes.append({"id": r["entity_id"], "name": r["canonical"], "type": r["type"],
                          "mentions": r["mentions"], "community": cid,
                          "theme": comm_label.get(cid, "")})

        total_pieces = None
        if depth >= 1 and node_ids:
            ph = ",".join("?" * len(node_ids))
            tm = conn.execute("SELECT COUNT(*) AS n FROM entity_map").fetchone()["n"]
            total_pieces = tm
            # level 1: up to `per` raw mentions per entity (window-limited, never 100K rows)
            for r in conn.execute(f"""
                SELECT entity_id, mid, mname, kind FROM (
                    SELECT em.entity_id AS entity_id, m.id AS mid, m.name AS mname, m.kind AS kind,
                           ROW_NUMBER() OVER (PARTITION BY em.entity_id ORDER BY m.id) AS rn
                    FROM entity_map em JOIN mentions m ON m.id = em.mention_id
                    WHERE em.entity_id IN ({ph})
                ) WHERE rn <= ?
            """, (*node_ids, per)).fetchall():
                mid = f"m{r['mid']}"
                cid = ent_comm.get(r["entity_id"])
                nodes.append({"id": mid, "name": (r["mname"] or "")[:48], "type": r["kind"] or "mention",
                              "mentions": 1, "community": cid, "theme": comm_label.get(cid, ""),
                              "piece": True, "kind": "mention"})
                links.append({"source": r["entity_id"], "target": mid, "relation": "mention",
                              "weight": 1, "temporality": "enduring", "as_of": None,
                              "directed": False, "piece": True})
        if depth >= 2 and node_ids:
            ph = ",".join("?" * len(node_ids))
            total_pieces = conn.execute("SELECT COUNT(*) AS n FROM assertions").fetchone()["n"]
            # level 2: the actual knowledge — assertions from the chunks each
            # entity appears in (decisions/problems/solutions/facts/...)
            for r in conn.execute(f"""
                SELECT entity_id, aid, statement, atype FROM (
                    SELECT entity_id, aid, statement, atype,
                           ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY aid) AS rn FROM (
                        SELECT DISTINCT em.entity_id AS entity_id, a.id AS aid,
                               a.statement AS statement, a.type AS atype
                        FROM entity_map em JOIN mentions m ON m.id = em.mention_id
                        JOIN assertions a ON a.conv_key = m.conv_key AND a.chunk_idx = m.chunk_idx
                        WHERE em.entity_id IN ({ph})
                    )
                ) WHERE rn <= ?
            """, (*node_ids, per)).fetchall():
                aid = f"a{r['aid']}"
                cid = ent_comm.get(r["entity_id"])
                nodes.append({"id": aid, "name": (r["statement"] or "")[:80], "type": r["atype"] or "claim",
                              "mentions": 1, "community": cid, "theme": comm_label.get(cid, ""),
                              "piece": True, "kind": "assertion"})
                links.append({"source": r["entity_id"], "target": aid, "relation": r["atype"] or "claim",
                              "weight": 1, "temporality": "enduring", "as_of": None,
                              "directed": False, "piece": True})
    finally:
        conn.close()
    return JSONResponse({"nodes": nodes, "links": links, "depth": depth,
                         "total_pieces": total_pieces,
                         "communities": [{"id": k, "label": v} for k, v in sorted(comm_label.items())]})


async def rename_community(request: Request):
    """Rename a theme/community label (for demo-friendly tags). Body:
    {community_id, label}."""
    b = await request.json()
    with _db_lock:
        conn = db()
        try:
            conn.execute("UPDATE communities SET label=? WHERE community_id=?",
                         (b.get("label", ""), b["community_id"]))
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True})


async def graph_set_communities(request: Request):
    """Store community detection results. Body:
    {communities:[{community_id, label, entity_ids:[...]}]}."""
    b = await request.json()
    with _db_lock:
        conn = db()
        try:
            conn.executescript("DELETE FROM communities; DELETE FROM entity_community;")
            for c in b.get("communities", []):
                cid = c["community_id"]
                conn.execute("INSERT INTO communities (community_id, label, size) VALUES (?,?,?)",
                             (cid, c.get("label", ""), len(c.get("entity_ids", []))))
                for eid in c.get("entity_ids", []):
                    conn.execute("INSERT OR REPLACE INTO entity_community (entity_id, community_id) VALUES (?,?)",
                                 (eid, cid))
            conn.commit()
        finally:
            conn.close()
    return JSONResponse({"ok": True, "communities": len(b.get("communities", []))})


async def entities_list(request: Request):
    limit = min(int(request.query_params.get("limit", "300")), 2000)
    conn = db()
    try:
        rows = [dict(r) for r in conn.execute("""
            SELECT e.entity_id, e.canonical, e.type, e.mentions,
                   COUNT(DISTINCT em.mention_id) AS mapped
            FROM entities e LEFT JOIN entity_map em ON em.entity_id = e.entity_id
            GROUP BY e.entity_id ORDER BY e.mentions DESC LIMIT ?""", (limit,))]
    finally:
        conn.close()
    return JSONResponse({"entities": rows})
