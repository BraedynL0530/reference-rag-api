"""
The keyword-gated, hallucination-resistant RAG pipeline.

Flow for a message containing the trigger keyword (default
"REFERENCE BOOK:", set in .env -> RAG_KEYWORD):

  1. Look for an explicit page/section/chapter reference in the text
     ("look at 1.1", "page 42", "chapter 3"). If found, pull that exact
     text straight from the page index -- no embedding search, no room
     to hallucinate which page it meant.
  2. Otherwise, embed the question and do a top-K similarity search
     against the collection, and DROP any result whose distance is
     past MAX_RELEVANT_DISTANCE (i.e. not actually relevant) rather
     than handing the model noise to paraphrase.
  3. If nothing usable was retrieved either way, return a fixed
     "not found" string instead of calling the model at all -- the
     model can't hallucinate an answer it's never asked to produce.
  4. Otherwise, send ONLY the retrieved chunks (never the whole book)
     to Gemini with a locked system prompt that requires a citation on
     every claim and forbids outside knowledge.

Messages WITHOUT the keyword skip RAG entirely and get a plain,
non-personality answer -- this is the general-purpose chat path.
"""
from __future__ import annotations

import re

from app.config import settings
from app.gemini_client import embed_texts_async, generate_text_async
from app.rag import page_index, vector_store
from app.utils.text_clean import strip_filler

SECTION_REF_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)*)\b")
PAGE_REF_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
CHAPTER_REF_RE = re.compile(r"\bchapter\s+(\d+)\b", re.IGNORECASE)

LOCKED_SYSTEM_PROMPT = (
    "You are a strict reference-lookup tool, not a conversational assistant.\n"
    "Rules, no exceptions:\n"
    "- Answer ONLY using the CONTEXT block provided. Do not use outside knowledge "
    "and do not fill gaps with inference.\n"
    "- If the CONTEXT does not contain the answer, respond exactly: "
    '"Not found in the reference material at the retrieved pages."\n'
    "- Every claim must end with a citation in the form (p.<page>, <chapter/section>) "
    "using only page/chapter/section values that appear in the CONTEXT.\n"
    "- Never invent a page, chapter, or section number not present in the CONTEXT.\n"
    "- No first person, no greetings, no closing remarks. Output only the answer."
)

PLAIN_SYSTEM_PROMPT = (
    "Respond directly and factually. No first person, no greetings, no "
    "closing remarks or offers of further help. Output only the answer content."
)


def has_keyword(message: str) -> bool:
    return settings.RAG_KEYWORD.lower() in message.lower()


def strip_keyword(message: str) -> str:
    pattern = re.compile(re.escape(settings.RAG_KEYWORD), re.IGNORECASE)
    return pattern.sub("", message).strip()


def _extract_explicit_ref(message: str) -> tuple[str, str] | None:
    m = SECTION_REF_RE.search(message)
    if m:
        return "section", m.group(1)
    m = PAGE_REF_RE.search(message)
    if m:
        return "page", m.group(1)
    m = CHAPTER_REF_RE.search(message)
    if m:
        return "chapter", m.group(1)
    return None


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        meta = c["metadata"]
        parts.append(
            f"[p.{meta.get('page')} | ch.{meta.get('chapter') or '?'} | "
            f"sec.{meta.get('section') or '?'}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


async def _retrieve(collection: str, clean_message: str) -> list[dict]:
    ref = _extract_explicit_ref(clean_message)
    if ref:
        kind, value = ref
        if kind == "page":
            rec = await page_index.find_by_page(collection, int(value))
            if rec:
                return [{"text": rec["text"], "metadata": rec, "distance": 0.0}]
        elif kind == "section":
            recs = await page_index.find_by_section(collection, value)
            if recs:
                return [{"text": r["text"], "metadata": r, "distance": 0.0} for r in recs[: settings.TOP_K]]
        elif kind == "chapter":
            recs = await page_index.find_by_chapter(collection, value)
            if recs:
                return [{"text": r["text"], "metadata": r, "distance": 0.0} for r in recs[: settings.TOP_K]]
        # explicit ref given but not found in the index -- fall through to
        # semantic search rather than silently ignoring the user's intent

    [query_embedding] = await embed_texts_async([clean_message])
    raw = await vector_store.query(collection, query_embedding, settings.TOP_K)
    return [r for r in raw if r["distance"] <= settings.MAX_RELEVANT_DISTANCE]


def sources_from_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "page": c["metadata"].get("page"),
            "chapter": c["metadata"].get("chapter") or None,
            "section": c["metadata"].get("section") or None,
            "source_file": c["metadata"].get("source_file"),
            "distance": c.get("distance"),
            "snippet": (c["text"][:220] + "...") if len(c["text"]) > 220 else c["text"],
        }
        for c in chunks
    ]


async def answer(message: str, collection: str | None = None) -> dict:
    collection = collection or settings.DEFAULT_COLLECTION
    used_rag = has_keyword(message)

    if not used_rag:
        text = await generate_text_async(system_prompt=PLAIN_SYSTEM_PROMPT, user_content=message)
        return {"used_rag": False, "answer": strip_filler(text), "sources": []}

    clean_message = strip_keyword(message)
    retrieved = await _retrieve(collection, clean_message)

    if not retrieved:
        return {
            "used_rag": True,
            "answer": "Not found in the reference material at the retrieved pages.",
            "sources": [],
        }

    context = _format_context(retrieved)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{clean_message}"
    text = await generate_text_async(system_prompt=LOCKED_SYSTEM_PROMPT, user_content=prompt)

    return {"used_rag": True, "answer": strip_filler(text), "sources": sources_from_chunks(retrieved)}
