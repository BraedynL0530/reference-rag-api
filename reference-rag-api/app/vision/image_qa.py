"""
Vision is scoped to reference-material pages ONLY -- there is no endpoint
that accepts an arbitrary user photo. This is a deliberate scope limit,
not a technical one: an open "upload any photo and ask about it" vision
tool is trivially usable to photograph an exam/homework question and get
an answer, which isn't something this tool is meant to help with. Every
/vision/page call names a page number inside an already-ingested
collection; if that page has no embedded image (see pdf_ingest.py's
render_visual_pages), there's nothing to analyze and the call fails
cleanly rather than falling back to some other image source.
"""
from __future__ import annotations

from google.genai import types

from app.config import settings
from app.gemini_client import get_client
from app.rag import page_index
from app.utils.text_clean import strip_filler

PAGE_VISION_SYSTEM_PROMPT = (
    "You are given an image of one page from a reference document, and the "
    "page's own extracted text as extra context. Answer ONLY about what is "
    "visually shown on this page (diagrams, charts, figures, photos) plus "
    "what the extracted text says -- do not use outside knowledge. End the "
    "answer with a citation in the form (p.<page>, <chapter/section>). "
    "No first person, no greetings, no closing remarks. Output only the answer."
)


class NoVisualContentError(Exception):
    pass


class PageNotFoundError(Exception):
    pass


async def analyze_reference_page(
    collection: str,
    page: int,
    prompt: str,
) -> dict:
    record = await page_index.find_by_page(collection, page)
    if not record:
        raise PageNotFoundError(f"Page {page} was not found in collection '{collection}'.")
    if not record.get("page_image"):
        raise NoVisualContentError(
            f"Page {page} has no embedded image/diagram/figure -- nothing for vision to look at. "
            "Use /ask for text content on this page instead."
        )

    client = get_client()
    image_part = types.Part.from_bytes(data=bytes(record["page_image"]), mime_type="image/png")
    context = (
        f"[p.{record['page']} | ch.{record.get('chapter') or '?'} | "
        f"sec.{record.get('section') or '?'}]\n{record.get('text') or ''}"
    )
    text_prompt = f"PAGE TEXT CONTEXT:\n{context}\n\nQUESTION ABOUT THE PAGE'S VISUAL CONTENT:\n{prompt}"

    response = await client.aio.models.generate_content(
        model=settings.VISION_MODEL,
        contents=[image_part, text_prompt],
        config=types.GenerateContentConfig(
            system_instruction=PAGE_VISION_SYSTEM_PROMPT,
            temperature=settings.TEMPERATURE,
        ),
    )

    source = {
        "page": record["page"],
        "chapter": record.get("chapter") or None,
        "section": record.get("section") or None,
        "source_file": record.get("source_file"),
        "distance": None,
        "snippet": (record.get("text") or "")[:220],
    }
    return {
        "answer": strip_filler((response.text or "").strip()),
        "sources": [source],
    }
