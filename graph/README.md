# graph

Stage-2 through Stage-4 of the Tailpipe pipeline: resolving extracted mentions
into a canonical knowledge graph, typing the edges with semantic relations, and
grouping entities into community themes.

Run these after the extraction worker has populated the mention/assertion ledger.


## Pipeline order

```
capture  ->  extract worker  ->  [ graph pipeline ]  ->  MCP / visualization
```

Within the graph pipeline:

| Step | Script | Core endpoint(s) called |
|------|--------|------------------------|
| 1 | `resolve_entities.py` | `/api/ledger/mention_names`, `/api/ledger/mentions_by_name`, `/api/entities/build` |
| 2 | *(server-side)* | `/api/entities/build_edges` — triggered by `resolve_entities assemble` |
| 3 | `type_edges.py` | `/api/graph/edges_for_typing`, `/api/graph/set_edge_types` |
| 4 | `communities.py` | `/api/graph/edges`, `/api/entities/list`, `/api/graph/set_communities` |


## Configuration

Copy the repo's `.env.example` to `.env` and fill in the relevant variables.

| Variable              | Description |
|-----------------------|-------------|
| `TAILPIPE_URL`        | Memory core base URL (default: `http://localhost:8080`) |
| `INGEST_TOKEN`        | Bearer token that guards the memory core API |
| `EXTRACT_ENGINE`      | `anthropic` or `openai_compat` (required for entity resolution and edge typing) |
| `ANTHROPIC_API_KEY`   | Anthropic API key (when `EXTRACT_ENGINE=anthropic`) |
| `ANTHROPIC_MODEL`     | Model for resolution / typing; defaults to `claude-haiku-4-5-20251001` |
| `OPENAI_COMPAT_URL`   | Base URL of a local OpenAI-compatible server |
| `OPENAI_COMPAT_KEY`   | Key for that server (may be empty) |
| `OPENAI_COMPAT_MODEL` | Model name to request from the compat server |

Community detection (`communities.py`) requires no LLM — only `TAILPIPE_URL`
and `INGEST_TOKEN`.


## Step 1 — Entity resolution (`resolve_entities.py`)

Pulls the top-N mention names from the ledger, clusters their contexts, and
asks an LLM to decide the canonical form, entity type (person / org / project /
tool / place / concept / other), and whether a name should be skipped (noise) or
split (one name, two distinct entities).

The resolution runs in two tiers — high-frequency names first so that less
common names can be merged into already-identified entities. Each batch is saved
locally under `graph/resolution_work/` so the LLM pass is resumable.

Ambiguous names (one name resolving to multiple canonicals) are assigned to the
correct entity at the mention level using embedding cosine similarity: each
mention's context embedding is matched against the distinguishing descriptions
of the candidate canonicals.

```sh
# Full end-to-end run:
EXTRACT_ENGINE=anthropic \
ANTHROPIC_API_KEY=sk-ant-... \
TAILPIPE_URL=http://localhost:8080 \
INGEST_TOKEN=your-token \
python -u -m graph.resolve_entities run

# Or step by step:
python -m graph.resolve_entities pull           # fetch candidates, write batch files
python -m graph.resolve_entities resolve        # LLM resolution for all batches
python -m graph.resolve_entities assemble       # merge + POST to core

# Resolve a single batch (useful when re-running after a failure):
python -m graph.resolve_entities resolve --batch entities_tier1_batch_2.json

# Adjust the tier windows:
python -m graph.resolve_entities pull --tier1-end 200 --tier2-end 400
```

Intermediate files written to `graph/resolution_work/`:
- `entities_tier{1,2}_batch_N.json` — enriched candidate batches
- `resolved_entities_tier{1,2}_batch_N.json` — LLM output per batch
- `tier1_canonicals.json` — canonical list for tier-2 merge context
- `entities_assembled.json` — final merged entity set (written before POSTing)


## Step 2 — Edge build (server-side)

Triggered automatically by `resolve_entities assemble`. Can also be triggered
manually:

```sh
curl -s -XPOST http://localhost:8080/api/entities/build_edges \
  -H "Authorization: Bearer $INGEST_TOKEN"
```

This generates candidate edges: within each chunk, the mapped entities are
connected by that chunk's assertions. Edge weight = number of co-occurrence
chunks; the initial relation is the dominant assertion type.


## Step 3 — Edge typing (`type_edges.py`)

Upgrades generic edges to semantic typed relations from a controlled vocabulary,
assigns direction (which entity is the subject), and stamps temporality
(enduring fact vs point-in-time state).

```sh
python -u -m graph.type_edges \
  --min-weight 2 \
  --batch 5

# Dry run — print typings without writing back:
python -u -m graph.type_edges --dry --limit 20
```

**Relation vocabulary** (subject `<relation>` object):

| Relation | Meaning |
|---|---|
| `part-of` | subject is a component of object |
| `uses` | subject uses / depends on object (tech, tool, service) |
| `works-on` | person works on / builds / owns a project |
| `works-for` | person works for / is affiliated with an org |
| `created-by` | subject was created / authored / founded by object |
| `solved-by` | a problem is solved / addressed by a solution |
| `decided-because` | a decision was made because of a reason/factor |
| `superseded-by` | subject was replaced / deprecated in favour of object |
| `alternative-to` | competing / comparable options (symmetric) |
| `discussed-with` | people who interact together (symmetric) |
| `located-in` | subject is located in / part of a place |
| `related-to` | generic association; no stronger relation supported |

Progress is appended to `graph/edge_typing_progress.jsonl`. Live status:

```sh
curl http://localhost:8080/api/graph/typing_status \
  -H "Authorization: Bearer $INGEST_TOKEN"
```


## Step 4 — Community detection (`communities.py`)

Runs Louvain community detection over the entity-edge graph (networkx), labels
each community by its top-mention members, and stores the result. Communities
appear as colour themes in the knowledge-graph visualization.

```sh
python -m graph.communities

# Tuning:
python -m graph.communities --resolution 1.2 --min-size 5
```

Labels are auto-generated from the top-4 members. Rename any community in the
visualization (click the label in the legend) or via the API:

```sh
curl -s -XPOST http://localhost:8080/api/graph/rename_community \
  -H "Authorization: Bearer $INGEST_TOKEN" \
  -H "content-type: application/json" \
  -d '{"community_id": 0, "label": "My Label"}'
```
