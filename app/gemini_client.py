"""
Single place that talks to Gemini. Uses the current unified `google-genai`
SDK (the old `google-generativeai` package is deprecated).

Model IDs are read from app.config so they can be swapped in one place as
Google renames/retires free-tier models -- check https://ai.google.dev
before deploying and update .env if these have moved on.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from app.config import settings

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


async def generate_text_async(
    system_prompt: str,
    user_content,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    user_content can be a string, or a list mixing strings with
    types.Part / types.File (image/audio) for multimodal calls.
    """
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=settings.TEMPERATURE if temperature is None else temperature,
    )
    response = await client.aio.models.generate_content(
        model=model or settings.TEXT_MODEL,
        contents=user_content,
        config=config,
    )
    return (response.text or "").strip()


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """
    Batch-embeds a list of strings with the configured embedding model,
    truncated to settings.EMBED_DIM (see config.py for why -- less
    storage/RAM for a small quality trade-off).
    """
    client = get_client()
    result = await client.aio.models.embed_content(
        model=settings.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=settings.EMBED_DIM),
    )
    return [e.values for e in result.embeddings]


async def upload_file_async(path: str, mime_type: str) -> types.File:
    client = get_client()
    return await client.aio.files.upload(
        file=path, config=types.UploadFileConfig(mime_type=mime_type)
    )


async def delete_file_async(name: str) -> None:
    client = get_client()
    try:
        await client.aio.files.delete(name=name)
    except Exception:
        pass  # best-effort cleanup; Gemini also auto-expires files after 48h
