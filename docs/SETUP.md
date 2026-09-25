# Setup

Stand up your own Tailpipe. End to end: capture your conversations → own them →
extract the knowledge → build a graph → query it from any AI agent.

## Prerequisites
- **Docker + Docker Compose** — runs the memory core. Works on any machine (NAS,
  mini-PC, laptop, cloud VM); you do **not** need a NAS.
- **Python 3.12+** and the client deps: `pip install -r requirements.txt` (for the
  normalizers, extraction worker, and graph pipeline). If a package silently skips,
  run `python -m pip install --upgrade pip` first.
- An **extraction model you choose** — an Anthropic API key, **or** any
  OpenAI-compatible server (Ollama, llama.cpp, vLLM, an edge box). Most people use
  Ollama or a hosted API. The small search-embedding model
  (`BAAI/bge-small-en-v1.5`) auto-downloads on first run — nothing to install.

## 1. Configure
```bash
git clone <your-fork> tailpipe && cd tailpipe
cp .env.example .env
```
Edit `.env`:
- `INGEST_TOKEN` — a long random string (`python -c "import secrets; print(secrets.token_hex(24))"`). This guards your memory. **Never commit `.env`.**
- `ALLOWED_HOSTS` — add your machine's LAN address if you'll reach it from other devices (e.g. `192.168.1.50`). Localhost works out of the box.
- Extraction engine — pick one:
  - **Ollama (local, free):** `EXTRACT_ENGINE=openai_compat`,
    `OPENAI_COMPAT_URL=http://localhost:11434/v1`,
    `OPENAI_COMPAT_MODEL=qwen2.5:7b` (or any model you've pulled),
    leave `OPENAI_COMPAT_KEY` blank.
  - **Anthropic API:** `EXTRACT_ENGINE=anthropic` + `ANTHROPIC_API_KEY`.
  - **Any other OpenAI-compatible server** (llama.cpp, vLLM, edge box):
    `EXTRACT_ENGINE=openai_compat` + `OPENAI_COMPAT_URL` / `OPENAI_COMPAT_KEY` / `OPENAI_COMPAT_MODEL`.

## 2. Bring up the memory core
```bash
docker compose up -d --build
curl -s localhost:8080/health          # {"ok": true, ...}
```
This runs the ingest API, the MCP server, the graph API, and the 3D graph page —
all on your box. Your data lives in `./data` (gitignored).

## 3. Get conversations in
Two paths (use either or both):

**A. Browser extension (live capture).** Load `capture/extension/` as an unpacked
extension (Chrome: `chrome://extensions` → Developer mode → Load unpacked). Open
its options, set the ingest URL (`http://localhost:8080/ingest`) and your
`INGEST_TOKEN`. It captures new Claude / ChatGPT conversations as you have them.

**B. Export normalizers (backfill).** First export your history from each provider:
  - **Claude** (claude.ai): Settings → Account → **Export data** → you'll get an
    email with a `conversations.json`.
  - **ChatGPT**: Settings → Data controls → **Export data** → the emailed zip
    contains a `conversations.json`.
  - **Gemini**: Google **Takeout** (takeout.google.com → *My Activity / Gemini*).
    Gemini's format is messier; the parser is best-effort.

Then normalize + ingest each:
```bash
export TAILPIPE_URL=http://localhost:8080 INGEST_TOKEN=your-token
python capture/normalizers/normalize_claude_export.py  conversations.json --ingest
python capture/normalizers/normalize_chatgpt_export.py  conversations.json --ingest
```

## 4. Extract the knowledge
Turn raw chat into structured mentions & assertions:
```bash
python extract/worker.py            # reads the queue, extracts, submits to the ledger
```
Engine + server come from your `.env`. This is the step that costs API money if
you use a frontier model; it's free if you point at a local/edge server.

## 5. Build the graph
```bash
python graph/resolve_entities.py run     # raw names -> canonical entities
# build edges from the ledger:
curl -s -XPOST localhost:8080/api/entities/build_edges -H "Authorization: Bearer $INGEST_TOKEN"
python graph/type_edges.py                # upgrade edges to typed, dated relations
python graph/communities.py               # detect themes (Louvain)
```

> **First / small ingest?** Edge weight = how many conversation chunks two entities
> share, so you need repeated co-occurrence before the defaults show anything. On a
> tiny dataset the defaults (`type_edges.py --min-weight 2`, `communities.py
> --min-size 3`, and the `/graph` page starting at `3+`) can surface **nothing** and
> look broken. For a first run, drop the thresholds:
> ```bash
> python graph/type_edges.py --min-weight 1
> python graph/communities.py --min-size 2
> ```
> and use the graph page's `2+` density button. Raise them as your corpus grows.

## 6. Use it
**From any AI agent (MCP).** The server speaks MCP over streamable HTTP at
`http://<host>:8080/mcp`, authorized with `Authorization: Bearer <INGEST_TOKEN>`.

- **Claude Code:**
  ```bash
  claude mcp add --scope user --transport http tailpipe http://localhost:8080/mcp \
    --header "Authorization: Bearer <INGEST_TOKEN>"
  ```
  (or add a `tailpipe` entry to a project `.mcp.json`).
- **Claude Desktop / other MCP clients:** add an HTTP MCP server pointing at that
  URL with the same `Authorization` header.

Tools you get: `search_memory`, `get_conversation`, `get_message`, `entity_timeline`,
`entity_neighborhood`, `what_connects`, `list_themes`, `theme_of`, `memory_status`.
Then just ask: *"what did I decide about X?"*, *"what connects A and B?"*

> **MCP is served on your LAN only.** Web-based agents (that dial in from vendor
> servers) aren't supported yet — see the README for why. Point *local* agents at it.

**See your mind.** Open `http://<host>:8080/graph?token=<INGEST_TOKEN>` for the
interactive 3D knowledge graph. Drill from entities into the underlying pieces.

## Optional: capture your coding sessions
Funnel your local Claude Code transcripts into Tailpipe automatically:
```bash
export TAILPIPE_URL=http://localhost:8080 INGEST_TOKEN=your-token
python scrapers/cc_watcher.py --once     # backfill everything so far
python scrapers/cc_watcher.py            # run as a service; new turns stream in
```

## Notes
- Everything is local by default. The only outbound calls are the extraction
  model requests *you* configure.
- Back up `./data` — it becomes the most valuable thing you own.
