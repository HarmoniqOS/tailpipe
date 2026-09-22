# scrapers

Local scrapers that funnel your own machine's conversation "exhaust" into
Tailpipe. Unlike the browser-based `capture/` paths (which read web chat
sessions), these read transcripts that tools write to your local disk.

## Claude Code session watcher

[Claude Code](https://claude.com/claude-code) writes each coding session to an
append-only JSONL file:

```
~/.claude/projects/<project-dir>/<session-id>.jsonl
```

`cc_watcher.py` tails those files and streams new turns into Tailpipe's
`/ingest` endpoint, normalized to schema-v1 (the same record shape every capture
path produces). It runs as a long-lived service, or as a one-shot backfill.

Only **main sessions** are ingested — the top-level `<project>/<session>.jsonl`
files. Subagent / sidechain transcripts are skipped by design, so what lands in
memory is what you actually discussed with the agent.

### Files

| File | Purpose |
|---|---|
| `cc_watcher.py` | The service: watches transcripts, tracks cursors, POSTs to `/ingest`. |
| `claude_code_normalizer.py` | Turns a stream of Claude Code events into a schema-v1 record. Also does secret redaction. Importable/reusable. |

### Install

Standard library only — nothing to install to run in polling mode. Optionally:

```
pip install -r requirements.txt   # requests (nicer HTTP) + watchdog (event fast path)
```

Both are optional. Without `requests` the watcher uses `urllib`; without
`watchdog` it polls.

### Configure

All configuration is via environment variables (matching the rest of the repo —
see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `TAILPIPE_URL` | `http://localhost:8080` | Base URL of the memory core. `/ingest` is appended. |
| `INGEST_TOKEN` | *(empty)* | Bearer token sent as `Authorization: Bearer <token>`. Omit if your server has no auth. |
| `CLAUDE_PROJECTS_DIR` | `~/.claude/projects` | Where Claude Code writes session transcripts. |
| `TAILPIPE_STATE_DIR` | `~/.tailpipe` | Where per-file cursors are stored. |
| `POLL_SECONDS` | `5` | Poll interval in service mode. |
| `TAILPIPE_REDACT_FILE` | *(empty)* | Optional file of literal secrets to redact, one per line (`#` comments allowed). For tokens that live nowhere else on disk. |

### Run

One-shot backfill (ingest everything not yet sent, then exit — good for a first
run or a cron job):

```
python scrapers/cc_watcher.py --once
```

Long-lived service (poll forever — the reliable default):

```
python scrapers/cc_watcher.py
```

Service with the watchdog event fast path (falls back to polling if `watchdog`
isn't installed; a slow safety poll still runs underneath):

```
python scrapers/cc_watcher.py --use-watchdog
```

Force a full re-scan, ignoring stored cursors (re-sends every session; the
ingest endpoint dedups, so this is safe):

```
python scrapers/cc_watcher.py --once --reset
```

Example with explicit config:

```
TAILPIPE_URL=http://localhost:8080 \
INGEST_TOKEN=your-token \
python scrapers/cc_watcher.py
```

### How the cursor / state works

State lives in `$TAILPIPE_STATE_DIR/cc_cursors.json` (default
`~/.tailpipe/cc_cursors.json`), one entry per transcript file:

```json
{
  "/home/you/.claude/projects/my-repo/<session>.jsonl": {
    "offset": 41213,   // byte offset up to which we've parsed COMPLETE lines
    "size": 41213,     // file size seen at that offset
    "ino": 12345678,   // inode (0 on platforms that don't expose one, e.g. Windows)
    "sent_msgs": 27    // message count of the last record we POSTed
  }
}
```

* **Restart-safe.** On start the watcher resumes from each file's stored
  `offset`, so it never re-sends turns it already ingested. If a session grew
  while the watcher was down, the next pass picks up exactly the new turns.

* **Idle is cheap.** For an unchanged file the watcher does a `stat`, sees the
  size equals the stored offset, and moves on — no parsing, no HTTP.

* **Partial lines.** Claude Code appends a line at a time and the watcher may
  read the file mid-write. It only advances the cursor past a *complete* line
  (one ending in `\n`); a dangling final fragment is left untouched and re-read
  whole next pass. A half-written line is never parsed or sent.

* **New files.** New sessions appear as new `.jsonl` files; the poll loop (and
  the watchdog `on_created` handler) discovers them on the next cycle and starts
  them from offset 0. New projects (new subdirectories) are discovered too.

* **Rotation / truncation.** If a file shrinks below the stored offset, or its
  inode changes (it was replaced), the watcher re-reads it from the top. Because
  `/ingest` is newest-wins and replaces a conversation's messages on each POST,
  re-sending is harmless.

* **Whole-conversation sends.** The `/ingest` endpoint stores one conversation
  per session and *replaces* its message list on each POST. So when a session
  grows, the watcher re-parses the file to rebuild the complete turn list and
  POSTs the whole record. The cursor is an I/O optimization (skip work when
  nothing changed); correctness comes from sending the full conversation. To
  avoid redundant POSTs when only bookkeeping records were appended, a send is
  skipped if the conversational message count hasn't changed.

* **Crash-safe writes.** The state file is written to a temp file and atomically
  renamed, so a crash mid-write can't corrupt it. On failed ingest the cursor is
  not advanced, so the session is retried next pass.

### Redaction

Before any text leaves the machine, two redaction layers run:

1. **Exact match** against `TAILPIPE_REDACT_FILE` (if set) — one secret per
   line, for credentials that live nowhere else on disk.
2. **Pattern sweep** for common credential shapes: bearer headers, `sk-`/`pk-`
   API keys, GitHub tokens, AWS access keys, Slack tokens, long hex strings,
   PEM private-key blocks, and URL-embedded passwords.

Nothing sensitive is hardcoded; all secret material comes from the environment.
