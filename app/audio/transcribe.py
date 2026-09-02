"""
Audio -> transcript -> notes/summary.

This uses Gemini's own audio understanding rather than a local Whisper
model. Trade-off, stated plainly: the previous local-Whisper version kept
audio entirely on your own box; this version sends it to Gemini (still
covered by the same free tier, still no card). What it buys back is
significant: no ffmpeg/ctranslate2/torch in the container (the single
biggest RAM consumer on a 512MB host), and no local decode step, which
means it also handles large recordings and odd formats better -- audio
goes through Gemini's Files API rather than being loaded into this
container's memory at all.

Given notes-taking is the priority feature, this path is also just more
capable: one multimodal call produces transcript + summary + notes
together, rather than a separate transcription engine feeding a second
text-only summarization call.
"""
from __future__ import annotations

from app.config import settings
from app.gemini_client import delete_file_async, get_client, upload_file_async
from app.utils.text_clean import strip_filler

NOTES_SYSTEM_PROMPT = (
    "You are given an audio recording. Produce exactly three sections, in "
    "this order, with these exact headers and nothing before the first header:\n"
    "TRANSCRIPT:\n<a faithful transcript of everything said, speaker changes "
    "noted as 'Speaker 1:' / 'Speaker 2:' if more than one voice is present>\n"
    "SUMMARY:\n<a concise 3-6 sentence paragraph of what was discussed>\n"
    "NOTES:\n<one bullet per line, each starting with '- ', covering concrete "
    "takeaways, decisions, and action items -- skip filler and small talk>\n"
    "No first person, no greetings, no closing remarks. Base everything only "
    "on the audio content; if audio is silent or unintelligible, say so in "
    "the TRANSCRIPT section and leave SUMMARY/NOTES minimal rather than "
    "inventing content."
)


def _parse_sections(raw: str) -> dict:
    sections: dict[str, list[str]] = {"transcript": [], "summary": [], "notes": []}
    current: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("TRANSCRIPT"):
            current = "transcript"
            continue
        if upper.startswith("SUMMARY"):
            current = "summary"
            continue
        if upper.startswith("NOTES"):
            current = "notes"
            continue
        if current == "notes":
            sections["notes"].append(line.lstrip("-\u2022 ").strip())
        elif current:
            sections[current].append(line)
    return {
        "transcript": " ".join(sections["transcript"]).strip(),
        "summary": " ".join(sections["summary"]).strip(),
        "notes": [n for n in sections["notes"] if n],
    }


async def transcribe_and_summarize(audio_path: str, mime_type: str) -> dict:
    """
    Uploads via the Files API (handles large recordings without loading
    them into this process's memory), asks for transcript+summary+notes
    in one call, then deletes the uploaded copy from Gemini immediately
    rather than waiting for its default 48h auto-expiry.
    """
    client = get_client()
    uploaded = await upload_file_async(audio_path, mime_type)
    try:
        response = await client.aio.models.generate_content(
            model=settings.AUDIO_MODEL,
            contents=[uploaded, "Transcribe this audio and produce the three sections."],
            config={
                "system_instruction": NOTES_SYSTEM_PROMPT,
                "temperature": settings.TEMPERATURE,
            },
        )
    finally:
        await delete_file_async(uploaded.name)

    raw = strip_filler((response.text or "").strip())
    parsed = _parse_sections(raw)

    if not any(parsed.values()):
        # Model didn't follow the section format for some reason -- surface
        # the raw text as the summary rather than silently returning nothing.
        parsed = {"transcript": "", "summary": raw, "notes": []}

    return parsed
