"""
Central configuration. Everything is env-driven so the whole tool can be
retuned (models, keyword, temperature, thresholds) without touching code.

Edit values in `.env` (copy `.env.example` -> `.env`), not here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


class Settings:
    # --- Gemini ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # Free-tier model IDs as of writing. Check https://ai.google.dev/gemini-api/docs/models
    # in Google AI Studio before deploying -- Google renames/retires free-tier models often.
    # Change these lines and nothing else needs to change.
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "gemini-2.5-flash")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gemini-2.5-flash")
    AUDIO_MODEL: str = os.getenv("AUDIO_MODEL", "gemini-2.5-flash")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    # gemini-embedding-001 defaults to 3072 dims; truncating via
    # output_dimensionality trades a little retrieval quality for a lot less
    # storage/RAM -- 768 is Google's own suggested "barely any quality loss"
    # point. Must match the `vector(N)` column size in db_schema.sql if changed.
    EMBED_DIM: int = _int("EMBED_DIM", 768)

    # --- Tool personality / output style ---
    # Kept low on purpose per spec: a tool, not a chat personality.
    TEMPERATURE: float = _float("TEMPERATURE", 0.3)

    # --- RAG behavior ---
    # The trigger keyword. Case-insensitive. Anywhere in the message/prompt.
    RAG_KEYWORD: str = os.getenv("RAG_KEYWORD", "REFERENCE BOOK:")
    DEFAULT_COLLECTION: str = os.getenv("DEFAULT_COLLECTION", "default_reference")
    TOP_K: int = _int("TOP_K", 5)
    CHUNK_SIZE_WORDS: int = _int("CHUNK_SIZE_WORDS", 220)
    CHUNK_OVERLAP_WORDS: int = _int("CHUNK_OVERLAP_WORDS", 40)
    # How many chunks are embedded + upserted together while ingesting a PDF.
    # This is the actual memory ceiling during ingest, independent of how
    # many total pages the PDF has -- a 100-page and a 5,000-page PDF both
    # only ever hold this many chunks in memory at once. Lowered from 64:
    # measured impact on peak RAM is small either way (the ~110MB library
    # import footprint dominates), but every bit of margin matters on a
    # 512MB host and the only cost is slightly more, slightly smaller
    # Postgres round-trips.
    INGEST_BATCH_SIZE: int = _int("INGEST_BATCH_SIZE", 24)
    # Cosine distance above which a retrieved chunk is considered "not actually
    # relevant" -- keeps the model from being handed noise it might paraphrase
    # into a hallucinated answer. Lower = stricter. Range is [0, 2]; 0 = identical.
    MAX_RELEVANT_DISTANCE: float = _float("MAX_RELEVANT_DISTANCE", 0.9)
    # DPI used to render a page that contains an embedded image/diagram, for
    # later /vision/page use. 150 is print quality and overkill for a model
    # reading a chart; 110 keeps a rendered page legible while meaningfully
    # shrinking both the transient render buffer during ingest and the
    # stored PNG's footprint in Postgres (matters more for slideshow-style
    # PDFs where most/every page has an image).
    PAGE_IMAGE_DPI: int = _int("PAGE_IMAGE_DPI", 110)

    # --- Persistent storage: Supabase Postgres + pgvector ---
    # Free, no card, survives container restarts/sleep/redeploys -- unlike
    # local disk on almost every free container host. Get this from
    # Supabase -> Project Settings -> Database -> Connection string (URI,
    # "Session pooler" mode is the one that works from a serverless-ish host).
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    # Connections in the asyncpg pool. Kept small on purpose -- this app is
    # meant to run as a single uvicorn worker on a small host, and each
    # pooled connection has its own (modest but nonzero) memory cost.
    DB_POOL_MAX_SIZE: int = _int("DB_POOL_MAX_SIZE", 3)

    # --- Upload handling ---
    # Uploads are streamed to disk in chunks (never fully loaded into RAM),
    # so this just guards against filling the container's ephemeral disk.
    # Default lowered to match the actual intended content -- long novels/
    # textbooks and short (~20-page) slideshows -- rather than an arbitrary
    # "giant" upload; raise it in .env if you genuinely need to ingest a
    # heavily-scanned, image-only PDF well beyond typical textbook size.
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    MAX_UPLOAD_MB: int = _int("MAX_UPLOAD_MB", 150)
    UPLOAD_STREAM_CHUNK_BYTES: int = _int("UPLOAD_STREAM_CHUNK_BYTES", 1024 * 1024)

    # --- API auth (simple shared-secret header, not OAuth -- this is a
    # personal tool, not a multi-tenant SaaS) ---
    API_KEY: str = os.getenv("API_KEY", "changeme")
    # Optional -- if unset, security.py derives one from API_KEY so this
    # works with zero extra config. Set your own random string here for
    # extra hygiene (means a compromised SESSION_SECRET alone still can't
    # forge the real API_KEY, and vice versa).
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "")
    SESSION_TTL_HOURS: int = _int("SESSION_TTL_HOURS", 24)

    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    # --- Email delivery (Gmail SMTP + app password is the free path) ---
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = _int("SMTP_PORT", 465)
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "")

    # --- Privacy ---
    # If True, only method/path/status/duration are logged -- never message
    # bodies, transcripts, file contents, or model output.
    PRIVACY_SAFE_LOGGING: bool = _bool("PRIVACY_SAFE_LOGGING", True)


settings = Settings()

Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
