"""
Belt-and-suspenders filter for tool-like output.

The system prompts (see rag/qa_engine.py, audio/transcribe.py,
vision/image_qa.py) already instruct the model not to speak in first
person or add chatty preambles. This module is the safety net for when it
does it anyway -- it strips common leading/trailing filler sentences
without touching the substantive content.
"""
from __future__ import annotations

import re

_LEADING_PATTERNS = [
    r"^(sure|okay|ok|alright|got it|certainly|of course)[,!.\s-]*",
    r"^(i'll|i will|i'm going to|i am going to|let me|i can|i'd be happy to)[^.\n]*[.\n]\s*",
    r"^(here'?s?|here is|here are)[^:\n]*:\s*",
    r"^(sure,?\s*)?(here'?s?|here is)[^.\n]*[.\n]\s*",
]

_TRAILING_PATTERNS = [
    r"\s*(let me know if[^.\n]*[.!]?)\s*$",
    r"\s*(hope this helps[^.\n]*[.!]?)\s*$",
    r"\s*(how'?s that[^.\n]*[?.!]?)\s*$",
    r"\s*(is there anything else[^.\n]*[?.!]?)\s*$",
    r"\s*(feel free to[^.\n]*[.!]?)\s*$",
]

_FIRST_PERSON_RE = re.compile(
    r"\b(i'm|i am|i'll|i will|i've|i have|i'd|i can|i think|my analysis)\b",
    re.IGNORECASE,
)


def strip_filler(text: str) -> str:
    if not text:
        return text
    out = text.strip()
    for pat in _LEADING_PATTERNS:
        out = re.sub(pat, "", out, count=1, flags=re.IGNORECASE)
    for pat in _TRAILING_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out.strip()
