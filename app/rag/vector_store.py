"""
Chunk storage + similarity search against reference_chunks in Postgres.
Each "reference book" (the main PDF, or any other PDF swapped in) lives
under its own `collection` string in the same table, so switching which
document RAG answers against is just a `collection` value -- same as
before, just backed by Supabase instead of local Chroma.
"""
from __future__ import annotations

from app.rag import db


async def reset_collection(collection: str) -> None:
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM reference_chunks WHERE collection = $1", collection)


async def add_chunks(collection: str, chunk_records: list[dict], embeddings: list[list[float]]) -> None:
    if not chunk_records:
        return
    pool = await db.get_pool()
    rows = [
        (
            c["id"],
            collection,
            c["text"],
            c["metadata"]["page"],
            c["metadata"]["chapter"] or None,
            c["metadata"]["section"] or None,
            c["metadata"]["source_file"],
            emb,
        )
        for c, emb in zip(chunk_records, embeddings)
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO reference_chunks (id, collection, text, page, chapter, section, source_file, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                page = EXCLUDED.page,
                chapter = EXCLUDED.chapter,
                section = EXCLUDED.section,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )


async def query(collection: str, query_embedding: list[float], top_k: int) -> list[dict]:
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT text, page, chapter, section, source_file,
                   embedding <=> $1 AS distance
            FROM reference_chunks
            WHERE collection = $2
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            query_embedding,
            collection,
            top_k,
        )
    return [
        {
            "text": r["text"],
            "metadata": {
                "page": r["page"],
                "chapter": r["chapter"] or "",
                "section": r["section"] or "",
                "source_file": r["source_file"],
            },
            "distance": float(r["distance"]),
        }
        for r in rows
    ]


async def list_collections() -> list[str]:
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT collection FROM reference_chunks ORDER BY collection")
    return [r["collection"] for r in rows]
