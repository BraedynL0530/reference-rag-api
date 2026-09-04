"""
Entry point. Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000

Every request except /health requires header:  X-API-Key: <settings.API_KEY>

Design notes for running on a small free host (512MB RAM class):

- Uploads (PDF/audio/image) are streamed to a temp file in fixed-size
  chunks rather than read into memory in one go, so memory use during an
  upload is bounded by UPLOAD_STREAM_CHUNK_BYTES, not by file size --
  this is what makes "allow large uploads" actually true for a giant PDF
  or a long audio recording.
- PDF ingestion (app/rag/ingest_service.py) processes the document in
  small batches rather than building the whole book's chunk list in
  memory, and the resulting vectors are stored in Postgres (Supabase),
  not locally -- both for persistence (free hosts don't keep local disk
  across restarts) and so the container never holds an ANN index itself.
- Audio goes straight to Gemini's audio understanding via the Files API
  rather than through a local Whisper model, which was the single
  heaviest thing this app could put in a 512MB container.
- Exports build directly into memory and stream back in the same
  request rather than being written to disk and linked -- no local
  state to lose between requests. Notes export is PDF only (Google
  Slides export was dropped -- OAuth's own persistence needs didn't fit
  a stateless host well, and it added a dependency for little gain over
  a PDF).
- Vision only ever looks at a page from an already-ingested reference
  PDF/slideshow (see /vision/page and app/vision/image_qa.py) -- there
  is deliberately no endpoint that accepts an arbitrary uploaded photo,
  so this can't be used to just photograph an unrelated question and
  get an answer.
- Each request can carry a session_id (the widget generates one per
  open tab); if no explicit `collection` is given, that session's own
  collection is used by default, so two tabs ingesting different PDFs
  don't collide -- naming a collection explicitly still lets you share
  one across sessions on purpose (app/rag/sessions.py).

Concurrency note: every route is `async def`, and Gemini calls use the
async client (`client.aio`). See /batch for an explicit demonstration of
several requests resolved concurrently in one call.
"""
from __future__ import annotations

import asyncio
import hmac
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.requests import Request as StarletteRequest

from app.audio.transcribe import transcribe_and_summarize
from app.config import settings
from app.delivery.notify import deliver
from app.export.pdf_export import create_notes_pdf_bytes
from app.logging_conf import get_logger
from app.rag import db, qa_engine, vector_store
from app.rag.ingest_service import ingest_pdf
from app.rag.sessions import resolve_collection
from app.schemas import (
    AskRequest,
    AskResponse,
    BatchResponse,
    BatchTask,
    ExportPdfRequest,
    ExportResponse,
    IngestResponse,
    TranscribeResponse,
    VisionPageRequest,
    VisionResponse,
)
from app.security import create_session_token, require_api_key
from app.vision.image_qa import NoVisualContentError, PageNotFoundError, analyze_reference_page

logger = get_logger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.init_schema()  # idempotent -- creates the extension/tables on first boot
    yield
    await db.close_pool()


app = FastAPI(title="Reference RAG API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def privacy_safe_access_log(request: StarletteRequest, call_next):
    """Logs method/path/status/duration only -- never request or response bodies."""
    start = time.monotonic()
    response = await call_next(request)
    if settings.PRIVACY_SAFE_LOGGING:
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(f"{request.method} {request.url.path} {response.status_code} {duration_ms}ms")
    return response


async def _stream_upload_to_disk(file: UploadFile, dest: Path) -> int:
    """
    Writes an upload to disk in fixed-size chunks -- memory use is bounded
    by UPLOAD_STREAM_CHUNK_BYTES regardless of the file's total size.
    Raises 413 if MAX_UPLOAD_MB is exceeded partway through (deletes the
    partial file first).
    """
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(settings.UPLOAD_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"Upload exceeds MAX_UPLOAD_MB ({settings.MAX_UPLOAD_MB}MB)")
            f.write(chunk)
    return written


def _safe_filename(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-_]+", "-", title.strip()).strip("-") or "notes"
    return slug[:80]


@app.get("/health")
@app.head("/health")
async def health() -> dict:
    return {"status": "ok"}


class SessionRequest(BaseModel):
    api_key: str


@app.post("/session")
async def create_session(req: SessionRequest) -> dict:
    """
    Exchanges the real, permanent API_KEY for a short-lived session token.
    This is what the browser widget calls once per tab so it never has to
    keep the real key in localStorage indefinitely -- see app/security.py.
    """
    if not hmac.compare_digest(req.api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    token, expires_at = create_session_token()
    return {"token": token, "expires_at": expires_at}


# ---------------------------------------------------------------------------
# RAG: ingest a reference PDF (the 867-page book, or any other PDF you want
# to swap in as the active reference). Handles arbitrarily large PDFs --
# see module docstring above and app/rag/ingest_service.py.
# ---------------------------------------------------------------------------
@app.post("/ingest/reference", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest_reference(
    file: UploadFile = File(...),
    collection: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    replace: bool = Form(default=True),
) -> IngestResponse:
    collection = resolve_collection(collection, session_id)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload must be a .pdf file")

    temp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4().hex}.pdf"
    await _stream_upload_to_disk(file, temp_path)
    try:
        result = await ingest_pdf(str(temp_path), source_file=file.filename, collection=collection, replace=replace)
    finally:
        temp_path.unlink(missing_ok=True)  # original bytes aren't kept once indexed

    return IngestResponse(**result)


# ---------------------------------------------------------------------------
# Text ask -- keyword-gated RAG, otherwise plain Gemini
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask(req: AskRequest) -> AskResponse:
    collection = resolve_collection(req.collection, req.session_id)
    result = await qa_engine.answer(req.message, collection=collection)
    delivered = await deliver(result["answer"], subject="Reference answer", delivery=req.delivery, email=req.email)
    return AskResponse(used_rag=result["used_rag"], answer=result["answer"], sources=result["sources"], delivered=delivered)


# ---------------------------------------------------------------------------
# Vision -- ONLY ever looks at a page that's already part of an ingested
# reference PDF/slideshow (see app/vision/image_qa.py for why there's no
# arbitrary-photo-upload endpoint). Name a page number from a collection
# you already ingested; if that page has no embedded image, this fails
# cleanly rather than accepting some other image source.
# ---------------------------------------------------------------------------
@app.post("/vision/page", response_model=VisionResponse, dependencies=[Depends(require_api_key)])
async def vision_page(req: VisionPageRequest) -> VisionResponse:
    collection = resolve_collection(req.collection, req.session_id)
    try:
        result = await analyze_reference_page(collection, req.page, req.prompt)
    except PageNotFoundError as e:
        raise HTTPException(404, str(e))
    except NoVisualContentError as e:
        raise HTTPException(422, str(e))

    delivered = await deliver(result["answer"], subject="Page analysis", delivery=req.delivery, email=req.email)
    return VisionResponse(answer=result["answer"], sources=result["sources"], delivered=delivered)


# ---------------------------------------------------------------------------
# Audio -> transcript -> notes/summary (priority feature -- see
# app/audio/transcribe.py for the reasoning behind using Gemini directly
# instead of a local model, and for how large recordings are handled).
# ---------------------------------------------------------------------------
@app.post("/transcribe", response_model=TranscribeResponse, dependencies=[Depends(require_api_key)])
async def transcribe(
    file: UploadFile = File(...),
    delivery: str = Form(default="display"),
    email: str | None = Form(default=None),
) -> TranscribeResponse:
    suffix = Path(file.filename or "audio").suffix or ".wav"
    temp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4().hex}{suffix}"
    await _stream_upload_to_disk(file, temp_path)
    try:
        result = await transcribe_and_summarize(str(temp_path), file.content_type or "audio/mpeg")
    finally:
        temp_path.unlink(missing_ok=True)

    body = f"SUMMARY:\n{result['summary']}\n\nNOTES:\n" + "\n".join(f"- {n}" for n in result["notes"])
    delivered = await deliver(body, subject="Audio notes", delivery=delivery, email=email)
    return TranscribeResponse(
        transcript=result["transcript"], summary=result["summary"], notes=result["notes"], delivered=delivered
    )


# ---------------------------------------------------------------------------
# Batch: run several text asks concurrently in one call. This is the
# explicit demonstration of "multiple things accessible at the same time" --
# combine this with firing /transcribe and /vision as separate concurrent
# HTTP requests from the client for the full effect.
# ---------------------------------------------------------------------------
@app.post("/batch", response_model=BatchResponse, dependencies=[Depends(require_api_key)])
async def batch(tasks: list[BatchTask]) -> BatchResponse:
    async def run_one(t: BatchTask) -> dict:
        collection = resolve_collection(t.collection, t.session_id)
        result = await qa_engine.answer(t.message, collection=collection)
        return {"message": t.message, **result}

    results = await asyncio.gather(*(run_one(t) for t in tasks))
    return BatchResponse(results=list(results))


# ---------------------------------------------------------------------------
# Exports -- built in memory, streamed back directly (display) or emailed
# as an attachment (email). Nothing is written to local disk, so there's
# no persistence-between-requests dependency on the host's filesystem.
# ---------------------------------------------------------------------------
@app.post("/export/pdf", dependencies=[Depends(require_api_key)])
async def export_pdf(req: ExportPdfRequest):
    pdf_bytes = await asyncio.to_thread(create_notes_pdf_bytes, req.title, req.content)
    filename = f"{_safe_filename(req.title)}.pdf"

    if req.delivery == "email":
        delivered = await deliver(
            req.content,
            subject=req.title,
            delivery="email",
            email=req.email,
            attachment_bytes=pdf_bytes,
            attachment_filename=filename,
        )
        return ExportResponse(url=None, delivered=delivered)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/collections", dependencies=[Depends(require_api_key)])
async def collections() -> dict:
    return {"collections": await vector_store.list_collections(), "default": settings.DEFAULT_COLLECTION}
