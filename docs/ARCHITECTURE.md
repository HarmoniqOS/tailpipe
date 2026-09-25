# Architecture

Tailpipe turns the **exhaust of your AI conversations** into owned, queryable
memory. One always-on core, a handful of client-side tools that feed and refine
it.

```
   CAPTURE                 CORE (your box)                  USE
 ┌──────────┐         ┌───────────────────────────┐     ┌──────────────┐
 │ extension│──POST──▶│  /ingest                   │     │ any AI agent │
 │normalizers│        │    ↓                       │◀MCP─│  (search,    │
 │ cc_watcher│        │  raw archive (NDJSON)      │     │   timeline,  │
 └──────────┘         │  SQLite: FTS + vectors     │     │   what_connects)
                      │    ↓                       │     └──────────────┘
   EXTRACT            │  mention/assertion ledger  │     ┌──────────────┐
 ┌──────────┐         │    ↓                       │────▶│  /graph      │
 │  worker  │────────▶│  knowledge graph           │     │  (3D viz)    │
 └──────────┘         │  entities · edges · themes │     └──────────────┘
   GRAPH              └───────────────────────────┘
 resolve · type · communities
```

## The core (`core/`)
A single container (Starlette + FastMCP, SQLite + `sqlite-vec` + FTS5). It:
- **Ingests** schema-v1 records → keeps a full-fidelity **raw archive** (append-only
  NDJSON, canonical, kept forever) and indexes them for **hybrid search** (keyword
  FTS + local vector embeddings).
- Serves an **MCP server** (streamable HTTP) so any agent can query your memory:
  `search_memory`, `get_conversation`, `get_message`, `entity_timeline`, `entity_neighborhood`,
  `what_connects`, `list_themes`, `theme_of`, `memory_status`.
- Hosts the **knowledge-graph API** and an interactive **3D graph visualization**.
- Guards everything behind a bearer token; a privacy partition (`EXCLUDED_OWNERS`,
  plus a `sealed` tier) keeps chosen conversations out of all search/agent/graph
  surfaces while preserving the raw record.

## Capture (`capture/`, `scrapers/`)
Get conversations in, normalized to a shared **schema-v1** (native IDs, parent
pointers/branch trees, role map, a block vocabulary of text/thinking/tool_use/
tool_result, content hashes, active-path flags):
- **Browser extension** — live capture from Claude / ChatGPT via the service worker.
- **Normalizers** — turn provider exports (Claude / ChatGPT / Gemini) into schema-v1.
- **`cc_watcher`** — a service that tails your local Claude Code `.jsonl`
  transcripts and streams new turns in, so your coding work becomes memory too.

Raw transcripts are **canonical**; everything downstream is a re-runnable *view* on
top — plug in a better model later and re-extract history.

## Extract (`extract/`)
Raw chat isn't knowledge. The worker runs **rolling extraction** over each
conversation in chunks (carrying a running state of summary / open threads /
active entities) and emits a **mention & assertion ledger** with provenance:
- **mentions** — named things (people, orgs, projects, tools, places, concepts).
- **assertions** — the actual claims: decision / advice / problem / solution / fact
  / preference / lesson / open_thread / question, each with a **speaker**
  (user / assistant / both). Output is grammar-constrained JSON so it's always
  valid. Engine is pluggable (a frontier API or any local/edge server).

## Graph (`graph/`)
The ledger becomes a typed graph:
1. **Resolve** raw mention names into canonical **entities** (+ aliases), splitting
   genuine collisions by context.
2. **Build edges** — entities that co-occur in a chunk get connected, weighted by
   supporting evidence.
3. **Type edges** — a model upgrades generic co-occurrence into semantic, dated
   relations (`part-of`, `uses`, `created-by`, `solved-by`, `superseded-by`, ...)
   with a **temporality** dimension (enduring facts vs. point-in-time states).
4. **Communities** — Louvain detection surfaces the natural **themes** of your
   corpus.

## Design principles
- **Local-first.** Your transcripts, extractions, and graph live on your hardware,
  served only under your own auth. The only outbound calls are the extraction
  model requests you choose to make.
- **Raw is canonical; views are disposable.** Re-extract, re-resolve, re-type any
  time a better model shows up.
- **Any model, any vendor.** Your memory is the data plane; the reasoning model is
  swappable. Point whatever agent you like at the MCP surface.
