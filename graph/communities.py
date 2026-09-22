"""
Tailpipe graph pipeline — Stage 4: community detection.

Runs Louvain community detection over the entity-edge graph via networkx,
labels each community by its highest-mention members, and stores the results
back to the memory core.

Communities appear in the knowledge-graph visualization as colour themes and
are navigable via the legend panel. Labels are auto-generated here but can be
renamed interactively in the visualization (POST /api/graph/rename_community).

Config env vars:
    TAILPIPE_URL   — memory core base URL (default: http://localhost:8080)
    INGEST_TOKEN   — bearer token for the memory core API

Usage:
    python -m graph.communities [--resolution 1.0] [--min-size 3] [--seed 42]

Options:
    --resolution  Louvain resolution parameter. Higher = more, smaller communities
                  (default: 1.0 — typical for knowledge-graph themes).
    --min-size    Communities with fewer members than this are not stored
                  (default: 3).
    --seed        Random seed for reproducibility (default: 42).
"""

import argparse
import json
import os
import urllib.request

import networkx as nx
from networkx.algorithms.community import louvain_communities


# ── Config from env ───────────────────────────────────────────────────────────

TAILPIPE_URL = os.environ.get("TAILPIPE_URL", "http://localhost:8080").rstrip("/")
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")


# ── Core API helpers ──────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {INGEST_TOKEN}"} if INGEST_TOKEN else {}


def _get(path: str) -> dict:
    req = urllib.request.Request(TAILPIPE_URL + path, headers=_headers())
    return json.load(urllib.request.urlopen(req, timeout=60))


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        TAILPIPE_URL + path, data=data,
        headers={**_headers(), "content-type": "application/json"},
        method="POST",
    )
    return json.load(urllib.request.urlopen(req, timeout=120))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tailpipe community detection")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain resolution parameter (default: 1.0)")
    ap.add_argument("--min-size", type=int, default=3,
                    help="Minimum community size to store (default: 3)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    # Fetch the full edge list and the entity index.
    edges = _get("/api/graph/edges")["edges"]
    ents = {e["entity_id"]: e for e in _get("/api/entities/list?limit=2000")["entities"]}

    if not edges:
        print("No edges found. Run entity resolution and edge build first.")
        return

    # Build the graph.
    G = nx.Graph()
    for src, dst, weight in edges:
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += weight
        else:
            G.add_edge(src, dst, weight=weight)

    print(f"{G.number_of_nodes()} entities, {G.number_of_edges()} edges — "
          f"running Louvain (resolution={args.resolution}, seed={args.seed})")

    comms = louvain_communities(G, weight="weight",
                                resolution=args.resolution, seed=args.seed)
    # Sort largest first so community_id 0 is always the biggest cluster.
    comms = sorted(comms, key=len, reverse=True)
    print(f"{len(comms)} communities detected\n")

    payload = []
    for cid, nodes in enumerate(comms):
        if len(nodes) < args.min_size:
            continue
        # Label: top-4 highest-mention members, joined with " / ".
        members = sorted(nodes, key=lambda n: -(ents.get(n, {}).get("mentions", 0)))
        tops = [ents[n]["canonical"] for n in members[:4] if n in ents]
        label = " / ".join(tops)
        payload.append({"community_id": cid, "label": label, "entity_ids": list(nodes)})
        top10 = ", ".join(ents[n]["canonical"] for n in members[:10] if n in ents)
        print(f"[{len(nodes):3d}] {label}")
        print(f"       {top10}")

    print()
    res = _post("/api/graph/set_communities", {"communities": payload})
    print(f"stored {len(payload)} communities: {res}")
    print(
        "\nLabels are auto-generated from top members. Rename them in the "
        "visualization or via POST /api/graph/rename_community."
    )


if __name__ == "__main__":
    main()
