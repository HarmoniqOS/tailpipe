# extract

Stage-1 extraction worker for Tailpipe.

Turns raw conversations stored in the memory core into structured knowledge:
**mentions** (named entities with local context) and **assertions** (typed
claims — decisions, advice, problems, solutions, facts, preferences, lessons,
open threads, questions) with per-message provenance.

The worker claims conversations from the core's job queue in global
chronological order (life-order replay), runs rolling extraction over
fixed-size message chunks with state carried forward, validates the output,
and submits each chunk to the ledger.


## How it works

1. `POST /api/ledger/claim` — atomically claim the next pending conversation.
2. `GET /record?key=…` — fetch the conversation + messages.
3. Split active-path messages into chunks of 12; build a rolling prompt that
   carries forward a `state` object (`summary`, `open_threads`, `active_entities`).
4. Call the extraction engine (see below) with a grammar- or schema-enforced
   JSON response.
5. Validate types and speaker values; retry once with error feedback on failure.
6. `POST /api/ledger/submit` — store each chunk's mentions/assertions.
7. `POST /api/ledger/complete` — mark the conversation done (or failed).


## Configuration

Copy `.env.example` to `.env` and fill in the relevant variables.

| Variable              | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `EXTRACT_ENGINE`      | `anthropic` or `openai_compat` (required)                        |
| `ANTHROPIC_API_KEY`   | Anthropic API key (required when `EXTRACT_ENGINE=anthropic`)     |
| `ANTHROPIC_MODEL`     | Model to use; defaults to `claude-haiku-4-5-20251001`            |
| `OPENAI_COMPAT_URL`   | Base URL of a local/edge OpenAI-compatible server                |
| `OPENAI_COMPAT_KEY`   | Key for the compat server (may be empty for unauthenticated)     |
| `OPENAI_COMPAT_MODEL` | Model name to request from the compat server                     |
| `TAILPIPE_URL`        | Memory core base URL (default: `http://localhost:8080`)          |
| `INGEST_TOKEN`        | Bearer token that guards the memory core API                     |


### Engine notes

**`anthropic`** — Uses forced tool-use (`emit_extraction`) to guarantee
structured output. Any Anthropic model that supports tool-use works. Haiku is
fast and cheap for bulk extraction.

**`openai_compat`** — Uses `response_format: {type: json_object, schema: …}`
(grammar-constrained decoding). Works with llama.cpp servers, vLLM, and any
other OpenAI-compatible endpoint that supports JSON schema constraints. Point
`OPENAI_COMPAT_URL` at the server's `/v1` base path.


## Running

Seed the queue first (from the memory core):

```sh
curl -s -XPOST http://localhost:8080/api/ledger/seed \
  -H "Authorization: Bearer $INGEST_TOKEN" \
  -H "content-type: application/json" \
  -d '{"all": true}'
```

Then run the worker:

```sh
# From the repo root, with dependencies installed:
EXTRACT_ENGINE=anthropic \
ANTHROPIC_API_KEY=sk-ant-... \
TAILPIPE_URL=http://localhost:8080 \
INGEST_TOKEN=your-token \
python -u -m extract.worker

# Limit to a pilot batch:
python -u -m extract.worker --max-conversations 20
```

Progress is appended to `extract/worker_progress.jsonl` and also printed to
stdout (each line is a JSON event).

Check queue status at any time:

```sh
curl http://localhost:8080/api/ledger/status \
  -H "Authorization: Bearer $INGEST_TOKEN"
```
