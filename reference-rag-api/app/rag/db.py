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

IMPORTANT, learned the hard way against a real Supabase project: Supabase
commonly installs pgvector into a dedicated `extensions` schema rather
than `public` (either by default on newer projects, or because someone
enabled it via the dashboard toggle at some point). `CREATE EXTENSION IF
NOT EXISTS vector` is a no-op if it's already installed ANYWHERE in the
cluster, regardless of schema -- so blindly assuming `public` breaks with
`unknown type: public.vector`, since pgvector-python's register_vector()
hardcodes schema='public' by default. So init_schema() below discovers
the actual schema at startup instead of assuming one, and everything
downstream (the type codec registration and the DDL) uses that.
"""
from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

_pool: asyncpg.Pool | None = None
_vector_schema: str = "public"  # overwritten by init_schema() once it's discovered


async def _init_conn(conn: asyncpg.Connection) -> None:
    # NOTE: a runtime `SET search_path` here does NOT reliably survive --
    # asyncpg's pool resets session-level state on every release-back-to-
    # pool, confirmed empirically (search_path reverts to default the
    # moment a connection is released and reacquired). So this only covers
    # register_vector's own Python-side type lookup (which isn't affected
    # by that reset, since it's a client-side codec, not server session
    # state); every SQL string we write ourselves schema-qualifies `vector`
    # explicitly instead of relying on search_path -- see db.vector_type()
    # and its use in vector_store.py / this module's DDL.
    await register_vector(conn, schema=_vector_schema)


def vector_type() -> str:
    """Schema-qualified `vector` type name, safe to splice into SQL you
    write yourself (never into user-controlled input) -- e.g. for a cast
    like f"$1::{vector_type()}". Valid only after init_schema() has run."""
    return f'"{_vector_schema}".vector'


def cosine_distance_op() -> str:
    """
    Schema-qualified <=> operator, as f"embedding {cosine_distance_op()} $1::{vector_type()}".
    Needed for the same reason as vector_type() -- confirmed empirically
    that even with both operand types correctly matching, Postgres still
    fails to resolve a bare `<=>` with "operator does not exist" once
    pgvector lives outside `public` and search_path can't be relied on
    (see this module's docstring). OPERATOR(schema.<=>) sidesteps that.
    """
    return f'OPERATOR("{_vector_schema}".<=>)'


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


async def _discover_vector_schema(conn: asyncpg.Connection) -> str:
    row = await conn.fetchrow(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON e.extnamespace = n.oid "
        "WHERE e.extname = 'vector'"
    )
    return row["nspname"] if row else "public"


async def init_schema() -> None:
    """
    Idempotent -- safe to call on every app startup.

    The pool's per-connection init callback (_init_conn) registers the
    pgvector type codec, which requires the `vector` type to already
    exist in the database -- so the extension has to be created (or
    found, if Supabase already installed it elsewhere) over a bare,
    uninitialized connection first, before the pool's own connections
    try to use it.
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set in .env -- create a free Supabase "
            "project and paste its connection string (see README)."
        )
    global _vector_schema
    bootstrap_conn = await asyncpg.connect(dsn=settings.DATABASE_URL)
    try:
        await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        _vector_schema = await _discover_vector_schema(bootstrap_conn)
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
                embedding {vector_type()}({settings.EMBED_DIM})
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
            f'CREATE INDEX IF NOT EXISTS reference_chunks_embedding_idx '
            f'ON reference_chunks USING hnsw (embedding "{_vector_schema}".vector_cosine_ops);'
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
