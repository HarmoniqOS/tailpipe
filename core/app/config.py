"""Configuration — every environment knob in one place.

All runtime configuration comes from the environment (see `.env.example`), so
the container is the same everywhere and secrets never live in code.
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "index" / "memory.db"
RAW_DIR = DATA_DIR / "raw"
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")

# Embeddings: small CPU model, same model for corpus and queries.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ONNX, N100-friendly
EMBED_DIM = 384
EMBED_MAX_CHARS = 2000                   # embed the same projection FTS indexes
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(DATA_DIR / "index" / "fastembed"))

# Owners excluded from search/retrieval by default (privacy partition).
# "sealed" is always excluded — the tier for personal/sensitive threads that
# stay in the raw archive but are off every search, agent, and graph path.
# Owners whose conversations are kept in the raw archive but excluded from all
# search / agent / graph surfaces (privacy partition). "sealed" is always
# excluded; add others via EXCLUDED_OWNERS in .env (comma-separated).
EXCLUDED_OWNERS = {o.strip() for o in os.environ.get("EXCLUDED_OWNERS", "").split(",") if o.strip()}
EXCLUDED_OWNERS.add("sealed")

# LAN service reached by IP/hostname — loosen the default localhost-only
# DNS-rebinding guard to the addresses this box actually answers on.
# Real access control is the bearer token + LAN-only exposure.
# Set ALLOWED_HOSTS in .env to your box's LAN address(es), comma-separated.
PORT = os.environ.get("PORT", "8080")
ALLOWED_HOSTS = []
for _h in (h.strip() for h in os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")):
    if _h:
        ALLOWED_HOSTS += [_h, f"{_h}:{PORT}"]
