"""
Exact page/section index -- kept separate from the embedding-based
vector_store.py on purpose, same reasoning as before: "look at 1.1"
should return THAT text verbatim, not whatever the embedding search
thinks is closest. Now backed by reference_pages in Postgres instead of
a local JSON file, so it survives container restarts.
"""
from __future__ import annotations

from app.rag import db


async def reset_collection(collection: str) -> None:
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM reference_pages WHERE collection = $1", collection)


async def save_pages(collection: str, page_records: list[dict]) -> None:
    """
    Pure upsert -- never wipes first. Ingestion streams in batches (see
    ingest_service.py), so wiping here would erase every batch but the
    last; call reset_collection() once up front instead when a full
    re-ingest is wanted.
    """
    if not page_records:
        return
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = [
                (
                    collection,
                    r["page"],
                    r.get("chapter") or None,
                    r.get("section") or None,
                    r.get("source_file"),
                    r["text"],
                    r.get("page_image"),
                )
                for r in page_records
            ]
            await conn.executemany(
                """
                INSERT INTO reference_pages (collection, page, chapter, section, source_file, text, page_image)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (collection, page) DO UPDATE SET
                    chapter = EXCLUDED.chapter,
                    section = EXCLUDED.section,
                    source_file = EXCLUDED.source_file,
                    text = EXCLUDED.text,
                    page_image = EXCLUDED.page_image
                """,
                rows,
            )


async def find_by_page(collection: str, page_num: int) -> dict | None:
    """Includes page_image (may be None) -- this is what /vision/page reads."""
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT page, chapter, section, source_file, text, page_image FROM reference_pages "
            "WHERE collection = $1 AND page = $2",
            collection,
            page_num,
        )
    return dict(row) if row else None


async def find_by_section(collection: str, section: str) -> list[dict]:
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT page, chapter, section, source_file, text FROM reference_pages "
            "WHERE collection = $1 AND section LIKE $2 ORDER BY page",
            collection,
            f"{section}%",
        )
    return [dict(r) for r in rows]


async def find_by_chapter(collection: str, chapter_fragment: str) -> list[dict]:
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT page, chapter, section, source_file, text FROM reference_pages "
            "WHERE collection = $1 AND chapter ILIKE $2 ORDER BY page",
            collection,
            f"%{chapter_fragment}%",
        )
    return [dict(r) for r in rows]
