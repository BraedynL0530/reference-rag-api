"""
Full PDF ingest pipeline: extract -> chunk -> embed -> store, streamed.

Memory use here is bounded by settings.INGEST_BATCH_SIZE, not by how many
pages the PDF has -- a giant PDF is processed as a sequence of small
batches (extract a page, chunk it, add to the current batch; once the
batch hits INGEST_BATCH_SIZE chunks, embed + upsert it and start a new
one), so this is the piece that actually makes "allow large uploads"
true rather than aspirational.
"""
from __future__ import annotations

from app.config import settings
from app.gemini_client import embed_texts_async
from app.rag import page_index, vector_store
from app.rag.pdf_ingest import iter_page_and_chunks


async def ingest_pdf(pdf_path: str, source_file: str, collection: str, replace: bool = True) -> dict:
    """
    Streams a PDF page-by-page: chunks accumulate in memory only up to
    INGEST_BATCH_SIZE before being embedded and upserted to Postgres, and
    page records are flushed in the same batches. This is what lets an
    arbitrarily large PDF ingest in roughly constant memory.
    """
    if replace:
        await vector_store.reset_collection(collection)
        await page_index.reset_collection(collection)

    pages_processed = 0
    chunks_indexed = 0
    pages_with_visuals = 0
    chapters_seen: list[str] = []

    pending_chunks: list[dict] = []
    pending_pages: list[dict] = []

    async def flush() -> None:
        nonlocal pending_chunks, pending_pages, chunks_indexed
        if pending_chunks:
            texts = [c["text"] for c in pending_chunks]
            embeddings = await embed_texts_async(texts)
            await vector_store.add_chunks(collection, pending_chunks, embeddings)
            chunks_indexed += len(pending_chunks)
            pending_chunks = []
        if pending_pages:
            await page_index.save_pages(collection, pending_pages)
            pending_pages = []

    for page_record, chunk_records in iter_page_and_chunks(
        pdf_path=pdf_path,
        source_file=source_file,
        chunk_size_words=settings.CHUNK_SIZE_WORDS,
        overlap_words=settings.CHUNK_OVERLAP_WORDS,
    ):
        pages_processed += 1
        pending_pages.append(page_record)
        pending_chunks.extend(chunk_records)

        if page_record.get("page_image"):
            pages_with_visuals += 1
        if page_record["chapter"] and page_record["chapter"] not in chapters_seen:
            chapters_seen.append(page_record["chapter"])

        if len(pending_chunks) >= settings.INGEST_BATCH_SIZE:
            await flush()

    await flush()  # final partial batch

    return {
        "collection": collection,
        "source_file": source_file,
        "pages_processed": pages_processed,
        "chunks_indexed": chunks_indexed,
        "chapters_detected": chapters_seen,
        "pages_with_visuals": pages_with_visuals,
    }
