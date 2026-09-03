"""
Collection resolution: an explicit `collection` name always wins (that's
how you deliberately share/reuse one across sessions, e.g. the main
"REFERENCE BOOK" the app defaults to). Otherwise, if the client sent a
session_id (the widget generates one per open tab, in sessionStorage so
a new tab = a new id), requests default to a collection scoped to that
session -- so uploading a PDF in one tab doesn't land in or overwrite
whatever another tab is using. With neither given (e.g. a bare curl
call), everything falls back to settings.DEFAULT_COLLECTION, same as
before this existed.
"""
from __future__ import annotations

import re

from app.config import settings

_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def resolve_collection(collection: str | None, session_id: str | None) -> str:
    if collection:
        return collection
    if session_id:
        safe = _SAFE_RE.sub("", session_id)[:64]
        if safe:
            return f"session_{safe}"
    return settings.DEFAULT_COLLECTION
