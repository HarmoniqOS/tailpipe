"""
Claude Code session watcher — funnels local coding transcripts into Tailpipe.

Claude Code appends each session's events to a JSONL file at
`~/.claude/projects/<project-dir>/<session-id>.jsonl` as you work. This service
tails those files and streams new turns into Tailpipe's `/ingest` endpoint,
normalized to schema-v1 (see `claude_code_normalizer.py`).

How it stays correct across restarts, live writes, and file growth:

  * Per-file cursor. For every transcript we persist a byte offset (the point up
    to which we've fully parsed) plus the file's inode/size fingerprint, in a
    small JSON state file. On start we resume from that offset, so we never
    re-send turns we've already ingested.

  * Partial-line safety. Claude Code appends a line at a time, but we may read
    the file mid-write. We only advance the cursor past a COMPLETE line (one
    terminated by "\n"); a dangling final fragment is left for the next pass.

  * File rotation / truncation. If a file shrinks below our stored offset or its
    fingerprint changes (e.g. it was replaced), we treat it as new and re-read
    from the top — the ingest endpoint dedups by conversation, so a re-send is
    harmless.

  * Whole-conversation sends. The ingest endpoint replaces a conversation's
    messages on every POST (newest-wins by `updated_at`). So when a session
    grows we re-parse from the cursor to accumulate the FULL turn list for that
    session and POST the whole record. The cursor is purely an I/O optimization;
    correctness comes from re-sending the complete conversation.

Two modes:
    (default)  long-lived service loop — poll (or watchdog) forever.
    --once     one-shot: process everything not yet ingested, then exit
               (backfill / cron-friendly).

Configuration (all via environment; see .env.example):
    TAILPIPE_URL         base URL of the memory core   (default http://localhost:8080)
    INGEST_TOKEN         bearer token for /ingest       (default: none)
    CLAUDE_PROJECTS_DIR  where Claude Code writes        (default ~/.claude/projects)
    TAILPIPE_STATE_DIR   where cursors are stored        (default ~/.tailpipe)
    POLL_SECONDS         poll interval, service mode     (default 5)
    TAILPIPE_REDACT_FILE optional exact-match secrets list (one per line)

Dependencies: standard library only. `requests` is used if installed, otherwise
we fall back to urllib. `watchdog`, if installed, provides a lower-latency
filesystem-event fast path; polling is the reliable default and always works.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from claude_code_normalizer import Redactor, SessionAccumulator

# ── Config ──────────────────────────────────────────────────────────────────

def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(os.path.expanduser(val)) if val else default


TAILPIPE_URL = os.environ.get("TAILPIPE_URL", "http://localhost:8080").rstrip("/")
INGEST_URL = TAILPIPE_URL + "/ingest"
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")
PROJECTS_DIR = _env_path("CLAUDE_PROJECTS_DIR", Path.home() / ".claude" / "projects")
STATE_DIR = _env_path("TAILPIPE_STATE_DIR", Path.home() / ".tailpipe")
STATE_FILE = STATE_DIR / "cc_cursors.json"
REDACT_FILE = os.environ.get("TAILPIPE_REDACT_FILE", "")
try:
    POLL_SECONDS = max(1.0, float(os.environ.get("POLL_SECONDS", "5")))
except ValueError:
    POLL_SECONDS = 5.0

# How much a session may grow (in bytes) before we flush mid-file rather than
# waiting for the poll cycle. Keeps very active sessions fresh without a POST
# per line.
FLUSH_BYTES = 64 * 1024


def log(msg: str) -> None:
    print(f"[cc-watcher] {msg}", flush=True)


# ── HTTP: post a schema-v1 record to /ingest ──────────────────────────────────
# Prefer `requests` if present; fall back to urllib so the service runs with the
# standard library alone.
try:
    import requests

    def post_ingest(record: dict, timeout: float = 120.0) -> dict:
        headers = {"Content-Type": "application/json"}
        if INGEST_TOKEN:
            headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
        resp = requests.post(INGEST_URL, json=record, headers=headers, timeout=timeout)
        try:
            body = resp.json()
        except ValueError:
            body = {"error": f"non-JSON response ({resp.status_code})"}
        if not resp.ok:
            body.setdefault("error", f"HTTP {resp.status_code}")
        return body

except ImportError:
    import urllib.error
    import urllib.request

    def post_ingest(record: dict, timeout: float = 120.0) -> dict:
        headers = {"Content-Type": "application/json"}
        if INGEST_TOKEN:
            headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
        data = json.dumps(record, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(INGEST_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8"))
            except Exception:
                return {"error": f"HTTP {e.code}"}
        except urllib.error.URLError as e:
            return {"error": f"connection failed: {e.reason}"}


# ── Cursor state ──────────────────────────────────────────────────────────────
# One entry per transcript file, keyed by its path. Each entry records:
#   offset      byte offset up to which we've parsed complete lines
#   size        file size at that offset (rotation/truncation detection)
#   ino         inode number, when the platform reports one (0 on Windows)
#   sent_msgs   message count of the last record we POSTed (skip no-op re-sends)

class CursorStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log(f"warning: could not read state file {self.path}; starting fresh")
                self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write can't corrupt the state file.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, key: str) -> dict:
        return self.data.get(key, {"offset": 0, "size": 0, "ino": 0, "sent_msgs": 0})

    def set(self, key: str, entry: dict) -> None:
        self.data[key] = entry


def file_fingerprint(path: Path) -> "tuple[int, int]":
    st = path.stat()
    return (getattr(st, "st_ino", 0), st.st_size)


# ── Reading complete lines from a cursor ───────────────────────────────────────

def read_new_lines(path: Path, start_offset: int) -> "tuple[list[str], int]":
    """Read COMPLETE lines from `start_offset` to the last newline.

    Returns (lines, new_offset). A trailing fragment with no newline (a line
    still being written) is not returned and not counted in new_offset, so it's
    re-read whole on the next pass. `start_offset` is clamped to the file size so
    a shrunk/rotated file simply reads nothing here (the caller handles reset).
    """
    size = path.stat().st_size
    if start_offset > size:
        # File shrank — caller decides whether to reset. Read nothing for now.
        return [], start_offset
    with open(path, "rb") as fh:
        fh.seek(start_offset)
        chunk = fh.read()  # everything appended since last cursor
    if not chunk:
        return [], start_offset
    last_nl = chunk.rfind(b"\n")
    if last_nl == -1:
        # No complete line yet (mid-write) — leave it all for next time.
        return [], start_offset
    complete = chunk[: last_nl + 1]
    new_offset = start_offset + len(complete)
    text = complete.decode("utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return lines, new_offset


def iter_all_lines(path: Path) -> "tuple[list[str], int]":
    """Read every complete line of a file from the top (used on reset / first
    pass). Returns (lines, offset-of-last-newline)."""
    return read_new_lines(path, 0)


# ── Session discovery ──────────────────────────────────────────────────────────

def discover_transcripts() -> "list[Path]":
    """Top-level `<project>/<session>.jsonl` files only. Nested dirs (subagent
    / workflow transcripts) are excluded by design — main sessions only."""
    if not PROJECTS_DIR.exists():
        return []
    files = []
    for proj in sorted(PROJECTS_DIR.iterdir()):
        if not proj.is_dir():
            continue
        files.extend(sorted(proj.glob("*.jsonl")))
    return files


def project_slug_for(path: Path) -> str:
    return path.parent.name


# ── Core: process one transcript file ───────────────────────────────────────────

def process_file(path: Path, cursors: CursorStore, redactor: Redactor,
                 force_full: bool = False) -> bool:
    """Parse new lines from `path`, and if the session grew, POST the full
    conversation. Returns True if anything was ingested. Updates the cursor.

    We always re-parse the session from the top to build the complete message
    list (the ingest endpoint replaces messages wholesale), but only when the
    cursor tells us the file has actually grown — so a poll over an idle session
    is nearly free (a stat + a cursor compare).
    """
    key = str(path)
    entry = cursors.get(key)
    try:
        ino, size = file_fingerprint(path)
    except OSError:
        return False  # file vanished between discovery and now

    reset = force_full
    # Rotation / truncation: fingerprint changed or file is smaller than cursor.
    if not reset:
        if size < entry.get("offset", 0):
            log(f"{path.name}: shrank ({size} < {entry['offset']}) — re-reading from top")
            reset = True
        elif entry.get("ino") and ino and entry["ino"] != ino:
            log(f"{path.name}: inode changed — treating as new file")
            reset = True

    prev_offset = 0 if reset else entry.get("offset", 0)
    if not reset and size == prev_offset:
        return False  # no growth, nothing to do

    # Re-parse the whole file up to the last complete line. We rebuild the full
    # session each time because the ingest replaces all messages for the conv.
    lines, new_offset = iter_all_lines(path)
    if not lines:
        # File exists but has no complete line yet (fully mid-write, or empty).
        cursors.set(key, {"offset": new_offset, "size": size, "ino": ino,
                          "sent_msgs": entry.get("sent_msgs", 0)})
        return False

    acc = SessionAccumulator(path.stem, project_slug_for(path), redactor)
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines; keep going
        acc.add_event(rec)

    record = acc.to_record()
    sent_msgs = entry.get("sent_msgs", 0)

    if record is None:
        # Non-conversational content only (bookkeeping records). Advance cursor
        # so we don't re-scan these bytes, but don't POST.
        cursors.set(key, {"offset": new_offset, "size": size, "ino": ino,
                          "sent_msgs": sent_msgs})
        return False

    # Skip a POST if the message count didn't change since our last send AND we
    # aren't resetting — avoids re-posting an unchanged conversation on every
    # cursor bump caused by non-message records being appended.
    if not reset and acc.message_count == sent_msgs:
        cursors.set(key, {"offset": new_offset, "size": size, "ino": ino,
                          "sent_msgs": sent_msgs})
        return False

    body = post_ingest(record)
    ok = bool(body.get("ok"))
    if ok:
        skipped = body.get("skipped")
        state = "stale-skip" if skipped else "ok"
        log(f"{path.name}: {acc.message_count} msgs -> {state}")
        cursors.set(key, {"offset": new_offset, "size": size, "ino": ino,
                          "sent_msgs": acc.message_count})
        cursors.save()
        return True

    # Failure: do NOT advance sent_msgs or offset past the new data, so the next
    # pass retries this session. (We keep the old cursor entry untouched.)
    log(f"{path.name}: ingest FAILED ({body.get('error', body)}) — will retry")
    return False


# ── Modes ───────────────────────────────────────────────────────────────────

def run_once(cursors: CursorStore, redactor: Redactor) -> None:
    """Process everything not yet fully ingested, then return."""
    files = discover_transcripts()
    log(f"one-shot: {len(files)} transcript file(s) under {PROJECTS_DIR}")
    ingested = 0
    for f in files:
        if process_file(f, cursors, redactor):
            ingested += 1
    cursors.save()
    log(f"one-shot done: {ingested} session(s) ingested/updated. "
        f"redactions={dict(redactor.counts)}")


def run_service_poll(cursors: CursorStore, redactor: Redactor) -> None:
    """Reliable polling loop. Rescans the tree every POLL_SECONDS; cheap because
    idle files short-circuit on a stat+cursor compare."""
    log(f"service (poll): watching {PROJECTS_DIR} every {POLL_SECONDS:g}s "
        f"-> {INGEST_URL}")
    while True:
        try:
            for f in discover_transcripts():
                process_file(f, cursors, redactor)
        except Exception as e:  # never let one bad cycle kill the service
            log(f"cycle error: {e!r}")
        time.sleep(POLL_SECONDS)


def run_service_watchdog(cursors: CursorStore, redactor: Redactor) -> None:
    """Optional low-latency path using watchdog filesystem events. Falls back to
    polling if watchdog isn't installed. A slow poll still runs underneath as a
    safety net (events can be missed on some platforms / network mounts)."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        log("watchdog not installed — using polling")
        return run_service_poll(cursors, redactor)

    def handle(path_str: str) -> None:
        p = Path(path_str)
        if p.suffix != ".jsonl" or p.parent.parent != PROJECTS_DIR:
            return  # top-level session files only
        try:
            process_file(p, cursors, redactor)
        except Exception as e:
            log(f"event error on {p.name}: {e!r}")

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory:
                handle(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                handle(event.src_path)

    if not PROJECTS_DIR.exists():
        log(f"{PROJECTS_DIR} does not exist yet — falling back to polling")
        return run_service_poll(cursors, redactor)

    observer = Observer()
    observer.schedule(Handler(), str(PROJECTS_DIR), recursive=True)
    observer.start()
    log(f"service (watchdog): watching {PROJECTS_DIR} -> {INGEST_URL} "
        f"(safety poll every {max(POLL_SECONDS, 30):g}s)")
    safety_interval = max(POLL_SECONDS, 30)
    try:
        while True:
            time.sleep(safety_interval)
            # Safety sweep: catch files whose events we missed.
            try:
                for f in discover_transcripts():
                    process_file(f, cursors, redactor)
            except Exception as e:
                log(f"safety sweep error: {e!r}")
    finally:
        observer.stop()
        observer.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true",
                    help="process everything not yet ingested, then exit (backfill)")
    ap.add_argument("--use-watchdog", action="store_true",
                    help="use filesystem events if watchdog is installed (default: polling)")
    ap.add_argument("--reset", action="store_true",
                    help="ignore stored cursors and reprocess every session from the top")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cursors = CursorStore(STATE_FILE)
    if args.reset:
        cursors.data = {}
        log("cursors reset — reprocessing all sessions")
    redactor = Redactor(REDACT_FILE or None)
    log(f"redaction: {len(redactor.known)} exact-match secret(s) loaded")

    if not PROJECTS_DIR.exists():
        log(f"note: {PROJECTS_DIR} does not exist yet "
            f"(no Claude Code sessions here, or wrong CLAUDE_PROJECTS_DIR)")

    if args.once:
        run_once(cursors, redactor)
        return 0

    try:
        if args.use_watchdog:
            run_service_watchdog(cursors, redactor)
        else:
            run_service_poll(cursors, redactor)
    except KeyboardInterrupt:
        log("stopped.")
    finally:
        cursors.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
