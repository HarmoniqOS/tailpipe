# capture

Two paths for getting conversations into Tailpipe:

1. **Browser extension** (`extension/`) — live capture via the session APIs of Claude.ai and ChatGPT. Runs continuously in the background and syncs new conversations every 30 minutes.
2. **Export normalizers** (`normalizers/`) — one-shot conversion of saved JSON exports into schema-v1 records that can be written to disk or POSTed directly to the ingest endpoint.

---

## Browser extension

An MV3 Chrome extension (Chrome 116+). It captures conversations directly from the live session API — no export or copy-paste needed.

### Install (unpacked)

1. Open `chrome://extensions`, enable **Developer mode**.
2. Click **Load unpacked**, select the `extension/` directory.

### Configure

Click the extension icon → **Settings** (or right-click → Options).

| Setting | Default | Description |
|---|---|---|
| Ingest endpoint URL | `http://localhost:8080/ingest` | Where to POST captured records. Leave empty to queue locally only. |
| Ingest token | *(empty)* | Bearer token sent as `Authorization: Bearer <token>`. Leave empty if your server has no auth. |
| Project → owner map | `{}` | JSON object mapping a Claude project UUID or ChatGPT `gizmo_id` to an owner string. Conversations in mapped projects are attributed deterministically; everything else lands as `"unknown"`. |

The extension queues captured records in browser IndexedDB and ships them to the configured endpoint after each sync cycle. If the server is unreachable, records stay queued and retry on the next alarm (every 30 minutes). You can also trigger a sync or export the local queue as NDJSON from the popup.

### Supported providers

- **claude.ai** — fetches via `/api/organizations/{org}/chat_conversations`
- **chatgpt.com** — fetches via `/backend-api/conversations`

---

## Export normalizers

Python scripts (stdlib only) that convert raw JSON exports into schema-v1 records.

### normalize_claude_export.py

Converts a raw claude.ai conversation JSON (the session-API detail endpoint response, or a file you saved from the browser) into schema-v1.

```
python normalizers/normalize_claude_export.py <raw_export.json> [out.json]
python normalizers/normalize_claude_export.py <raw_export.json> --ingest
```

### normalize_chatgpt_export.py

Converts a raw chatgpt.com conversation JSON (from `/backend-api/conversation/{id}`) into schema-v1.

```
python normalizers/normalize_chatgpt_export.py <raw_export.json> [out.json]
python normalizers/normalize_chatgpt_export.py <raw_export.json> --ingest
```

### parse_gemini_capture.py

Exploratory decoder for Gemini `batchexecute` captures (captured via a browser console fetch hook). Decodes the `)]}'` armor and length-prefixed chunks and heuristically extracts turns from the `hNvQHb` payload. **This is a research tool, not a full normalizer** — it does not produce schema-v1 output.

```
python normalizers/parse_gemini_capture.py <gemini_capture.txt>
```

### Configuration

The normalizers read configuration from environment variables:

| Variable | Default | Description |
|---|---|---|
| `TAILPIPE_URL` | `http://localhost:8080/ingest` | Ingest endpoint for `--ingest` mode |
| `INGEST_TOKEN` | *(empty)* | Bearer token (omit if server has no auth) |

---

## Schema-v1 record shape

Both capture paths produce the same record shape accepted by `POST /ingest`:

```json
{
  "schema_version": 1,
  "conversation": {
    "source": "claude.ai",
    "native_id": "<uuid>",
    "title": "...",
    "model": "...",
    "attribution": { "owner": "unknown", "method": null, "confidence": 0.0 },
    "sync": { "ingestor": "...", "captured_via": "session-api" }
  },
  "messages": [
    {
      "native_id": "...",
      "parent_native_id": "...",
      "role": "user",
      "content": [{ "type": "text", "text": "..." }],
      "content_hash": "...",
      "on_active_path": true
    }
  ]
}
```

`attribution.owner` is set to `"unknown"` unless the conversation belongs to a project that appears in your owner map. The memory core's ingestion pipeline resolves attribution downstream.
