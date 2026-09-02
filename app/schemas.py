from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Delivery = Literal["display", "email"]


class SourceChunk(BaseModel):
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    source_file: Optional[str] = None
    distance: Optional[float] = None
    snippet: str


class AskRequest(BaseModel):
    message: str
    collection: Optional[str] = None  # explicit name always wins over session_id
    session_id: Optional[str] = None  # per-tab default collection when collection isn't given
    delivery: Delivery = "display"
    email: Optional[str] = None


class AskResponse(BaseModel):
    used_rag: bool
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    delivered: str


class IngestResponse(BaseModel):
    collection: str
    source_file: str
    pages_processed: int
    chunks_indexed: int
    chapters_detected: list[str] = Field(default_factory=list)
    pages_with_visuals: int = 0


class VisionPageRequest(BaseModel):
    """
    Vision only ever looks at a page that's already part of an ingested
    reference PDF/slideshow -- there's no path to submit an arbitrary
    photo. See app/vision/image_qa.py for why.
    """
    page: int
    prompt: str
    collection: Optional[str] = None
    session_id: Optional[str] = None
    delivery: Delivery = "display"
    email: Optional[str] = None


class VisionResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    delivered: str


class TranscribeResponse(BaseModel):
    transcript: str
    summary: str
    notes: list[str] = Field(default_factory=list)
    delivered: str


class ExportPdfRequest(BaseModel):
    title: str
    content: str
    delivery: Delivery = "display"
    email: Optional[str] = None


class BatchTask(BaseModel):
    """One unit of work inside a /batch call. Only 'ask' needs no file."""
    kind: Literal["ask"]
    message: str
    collection: Optional[str] = None
    session_id: Optional[str] = None


class BatchResponse(BaseModel):
    results: list[dict]


class ExportResponse(BaseModel):
    url: Optional[str] = None
    delivered: str
