<h1 align="center">Tailpipe</h1>
<p align="center"><em>Build a second brain from your conversation exhaust.</em></p>

---

Every day you think out loud with AI — decisions, problems, things you figured
out, things you're still chewing on. That's your most valuable thinking, and it
comes out the **tailpipe** of your conversations as exhaust: scattered across
ChatGPT, Claude, and Gemini, locked in vendor silos, unowned and unsearchable.

**Tailpipe captures that exhaust and turns it into memory you own** — a private,
local, queryable second brain that any AI agent can plug into.

- **Capture** every conversation across providers.
- **Own it** — full-fidelity transcripts on hardware *you* control. Nothing
  leaves the house.
- **Extract** the actual knowledge — decisions, problems, solutions, lessons —
  from the raw chat.
- **Connect** it into a typed, dated knowledge graph of your entities and how
  they relate.
- **Serve** it over MCP, so any agent, any vendor, can query your whole history.
- **See** it — an interactive 3D graph of your own mind.

Not a product you rent. A system you run.

## How it works

```
  ┌── Capture ──────────────┐     ┌── Memory Core (yours) ──────────────┐
  │  browser extension      │     │                                     │
  │  Claude · ChatGPT ·      │──▶  │   raw transcripts (kept forever)    │
  │  Gemini normalizers      │     │   FTS + vector search               │
  │  local code sessions     │     │   MCP server ◀── any AI agent       │
  └─────────────────────────┘     │        │                            │
                                  │        ▼                            │
                                  │   Extract → mentions & assertions   │
                                  │        │                            │
                                  │        ▼                            │
                                  │   Knowledge graph (typed, dated)    │
                                  │   + 3D visualization                │
                                  └─────────────────────────────────────┘
```

Everything runs on your own box — a NAS, a mini-PC, a spare machine. The only
outbound calls are the ones *you* choose (e.g. a frontier model for extraction).
Your memory never leaves.

## Runs anywhere, with the models you already have

**You don't need a NAS.** The memory core is just a Docker container — run it on a
NAS, a mini-PC, a spare laptop, or a cloud VM. Wherever Docker runs, Tailpipe runs.

**Two models are involved, and you control both:**

- **Embeddings (for search)** — a small local model, **`BAAI/bge-small-en-v1.5`**
  (~130 MB), **auto-pulls on first run** via `fastembed`. It's the only model
  Tailpipe downloads for you; runs on CPU, nothing to configure.
- **Extraction (chat → knowledge)** — an LLM *you* pick. Tailpipe doesn't ship one.
  Point it at:
  - **Ollama** or any OpenAI-compatible local server (`EXTRACT_ENGINE=openai_compat`),
  - a **frontier API** like Anthropic (`EXTRACT_ENGINE=anthropic`), or
  - an edge box.

  Most people already have Ollama running or an API key — either works out of the box.

> **Coming soon:** a **purpose-trained extraction model**, dropping in the next few
> weeks — a small model tuned for exactly this pipeline, so extraction runs better
> and cheaper than a general LLM. Swap it in with one env change.

## Repository layout

| path | what |
|---|---|
| `core/` | the memory-core container — ingest, search, MCP server, graph API, 3D viz |
| `capture/extension/` | the browser extension that captures conversations (MV3) |
| `capture/normalizers/` | turn Claude / ChatGPT / Gemini exports into the shared schema |
| `extract/` | conversation → structured mentions & assertions (the knowledge layer) |
| `graph/` | entity resolution, edge typing, community detection |
| `scrapers/` | pull in other sources (local coding sessions, ...) |
| `docs/` | architecture + setup |

## Quick start

> Full setup lives in [`docs/SETUP.md`](docs/SETUP.md). The short version:

1. Copy `.env.example` → `.env`; set `INGEST_TOKEN` and your extraction engine. **Never commit `.env`.**
2. `docker compose up -d --build` — brings up the memory core + MCP.
3. Load the browser extension (`capture/extension/`) and/or run the normalizers
   on your exports to feed conversations in.
4. Run extraction + graph build (`extract/`, `graph/`) to turn raw chat into a
   knowledge graph.
5. Point any MCP-capable agent at your server and ask it about your own history.

## Privacy & the MCP surface

Tailpipe is **local-first by design**. Your transcripts, your extracted knowledge,
and your graph live on your hardware, served only under your own bearer token. The
extraction API is the one leash you keep on purpose — everything else runs on your box.

**MCP is local-network only, for now — on purpose.** Tailpipe currently serves its
MCP tools over your LAN, for local agents (Claude Desktop, Claude Code, etc.).
Exposing them to **web-based agents** — which dial in from a vendor's servers, not
your machine — is on the roadmap but **deliberately not shipped yet**. Doing it
safely requires a public endpoint, an OAuth 2.1 gateway, a read-only remote surface,
and an audit log; get it wrong and you leak your entire memory to the internet.
Local-first until that's airtight.

## License

[AGPL-3.0](LICENSE). Use it, run it, modify it — but if you offer it as a
service, your changes stay open too. Commercial licensing available separately.

---

<p align="center"><sub>Own your cognition.</sub></p>
