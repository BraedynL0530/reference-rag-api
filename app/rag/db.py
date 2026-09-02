"""
Connection pool + schema for the durable store: Supabase Postgres with
the pgvector extension. This replaced local Chroma because free container
hosts (Render/Koyeb/SnapDeploy/etc) don't persist local disk across
sleep/restart/redeploy -- Supabase's free Postgres does, and pgvector is
included on it at no extra cost.

Set DATABASE_URL in .env to the connection string from Supabase ->
Project Settings -> Database -> Connection string (URI). Use "Session
pooler" mode -- it's built for exactly this short-lived-container,
many-cold-starts usage pattern.
"""
from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not settings.DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set in .env -- create a free Supabase "
                "project and paste its connection string (see README)."
            )
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=1,
            max_size=settings.DB_POOL_MAX_SIZE,
            init=_init_conn,
        )
    return _pool


async def init_schema() -> None:
    """
    Idempotent -- safe to call on every app startup.

    The pool's per-connection init callback (_init_conn) registers the
    pgvector type codec, which requires the `vector` type to already
    exist in the database. So the extension has to be created over a
    bare, uninitialized connection first -- creating the pool (or
    acquiring from it) before the extension exists raises
    "unknown type: public.vector".
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set in .env -- create a free Supabase "
            "project and paste its connection string (see README)."
        )
    bootstrap_conn = await asyncpg.connect(dsn=settings.DATABASE_URL)
    try:
        await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        await bootstrap_conn.close()

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS reference_chunks (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                text TEXT NOT NULL,
                page INT,
                chapter TEXT,
                section TEXT,
                source_file TEXT,
                embedding vector({settings.EMBED_DIM})
            );
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS reference_chunks_collection_idx "
            "ON reference_chunks (collection);"
        )
        # HNSW cosine index -- Supabase ships a pgvector version that supports
        # this. If yours is older, drop this line; brute-force scan over a
        # single book's worth of chunks is still fast enough to not matter.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS reference_chunks_embedding_idx "
            "ON reference_chunks USING hnsw (embedding vector_cosine_ops);"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_pages (
                collection TEXT NOT NULL,
                page INT NOT NULL,
                chapter TEXT,
                section TEXT,
                source_file TEXT,
                text TEXT,
                page_image BYTEA,
                PRIMARY KEY (collection, page)
            );
            """
        )


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
