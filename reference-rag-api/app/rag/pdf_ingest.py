"""
PDF -> page-aware, heading-aware text, streamed page-by-page.

Everything here is a generator rather than "extract the whole book into a
list first" -- the point is that ingesting a 50-page PDF and a 5,000-page
PDF use the same, constant amount of memory (bounded by
settings.INGEST_BATCH_SIZE in ingest_service.py, not by document size).
PyMuPDF itself is lazy about page content, so this only ever holds one
page's text in memory at a time until the caller consumes it.

Heading detection approach (kept deliberately simple/heuristic rather than
a heavyweight layout model):

1. Read each page with PyMuPDF (fitz), which exposes each line's font
   size and font name (bold/regular).
2. Compute the document's median body-text font size once, from a small
   sample of pages up front.
3. Any line noticeably larger than body text, or bold and short, is a
   heading *candidate*.
4. Candidates are matched against two regexes:
     - a "chapter" pattern  (e.g. "Chapter 3", "3. Introduction")
     - a "section" pattern  (e.g. "1.1", "2.3.4 Some Title")
   Whichever last matched carries forward onto every following page
   until the next heading is seen, so a page in the middle of section
   1.1 (with no heading printed on it) is still tagged "1.1".

If your PDF doesn't have clean chapter/section numbering, chapter/section
will just come back as None -- pages still get correct page numbers,
which is the part most needed for "look at page X" style requests. You
can turn heading detection off entirely with `detect_headings=False` if
it's producing junk on a particular document (footers/running heads
being mistaken for headings is the usual failure mode).
"""
from __future__ import annotations

import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF (new import name; `fitz` alias kept for readability below)

from app.config import settings

CHAPTER_RE = re.compile(r"^(chapter\s+\d+[a-z]?\b.*|\d+\.?\s+[A-Z][^.]{2,80})$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\d+\.\d+(\.\d+)*\b")


@dataclass
class PageData:
    page: int
    text: str
    chapter: str | None
    section: str | None
    headings: list[tuple[str, str]] = field(default_factory=list)
    page_image: bytes | None = None  # rendered PNG, only set if the page has embedded images


def _median_body_font_size(doc: fitz.Document, sample_pages: int = 40) -> float:
    sizes: list[float] = []
    for i, page in enumerate(doc):
        if i >= sample_pages:
            break
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    return statistics.median(sizes) if sizes else 10.0


def extract_pages_iter(
    pdf_path: str, detect_headings: bool = True, render_visual_pages: bool = True
) -> Iterator[PageData]:
    """
    Yields one PageData per page. Closes the document when exhausted.

    render_visual_pages: if a page contains at least one embedded image
    (a figure, diagram, chart, photo -- not just body text), the page is
    rendered to a PNG and attached as page_image. This is what /vision/page
    later shows to Gemini -- vision only ever looks at pages that are
    actually part of an ingested reference document, never an arbitrary
    photo someone uploads (see app/vision/image_qa.py for why). Pages with
    no embedded images get page_image=None and cost no extra storage.
    """
    doc = fitz.open(pdf_path)
    try:
        body_size = _median_body_font_size(doc)
        heading_min_size = body_size + 1.5

        current_chapter: str | None = None
        current_section: str | None = None

        for pno in range(len(doc)):
            page = doc[pno]
            d = page.get_text("dict")
            lines_out: list[str] = []
            page_headings: list[tuple[str, str]] = []

            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s["text"] for s in spans).strip()
                    if not line_text:
                        continue
                    lines_out.append(line_text)

                    if not detect_headings:
                        continue

                    max_size = max(s["size"] for s in spans)
                    is_bold = any("bold" in s.get("font", "").lower() for s in spans)
                    looks_like_heading = (
                        max_size >= heading_min_size or is_bold
                    ) and len(line_text) <= 90

                    if looks_like_heading:
                        sec_match = SECTION_RE.match(line_text)
                        chap_match = CHAPTER_RE.match(line_text)
                        if sec_match:
                            current_section = line_text[:60]
                            page_headings.append(("section", current_section))
                        elif chap_match and max_size >= heading_min_size:
                            current_chapter = line_text[:60]
                            current_section = None
                            page_headings.append(("chapter", current_chapter))

            page_image: bytes | None = None
            if render_visual_pages and page.get_images(full=True):
                # Lower DPI than print quality on purpose -- see PAGE_IMAGE_DPI
                # in config.py. The pixmap is explicitly dropped right after
                # extracting bytes rather than left for GC to get to whenever,
                # since this is the single most memory-variable step in
                # ingestion (a slideshow with a diagram on every page renders
                # one of these per page).
                pix = page.get_pixmap(dpi=settings.PAGE_IMAGE_DPI)
                page_image = pix.tobytes("png")
                pix = None

            yield PageData(
                page=pno + 1,
                text="\n".join(lines_out),
                chapter=current_chapter,
                section=current_section,
                headings=page_headings,
                page_image=page_image,
            )
    finally:
        doc.close()


def chunk_words(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size_words]))
        if i + chunk_size_words >= len(words):
            break
        i += max(1, chunk_size_words - overlap_words)
    return chunks


def iter_page_and_chunks(
    pdf_path: str,
    source_file: str,
    chunk_size_words: int,
    overlap_words: int,
    detect_headings: bool = True,
) -> Iterator[tuple[dict, list[dict]]]:
    """
    Yields (page_record, chunk_records) one page at a time. The caller
    (ingest_service.py) is responsible for batching chunk_records across
    pages up to INGEST_BATCH_SIZE before embedding+upserting -- this
    function itself never accumulates more than one page in memory.
    """
    for pg in extract_pages_iter(pdf_path, detect_headings=detect_headings):
        page_record = {
            "page": pg.page,
            "chapter": pg.chapter or "",
            "section": pg.section or "",
            "text": pg.text,
            "source_file": source_file,
            "page_image": pg.page_image,
        }
        chunk_records = [
            {
                "id": f"{source_file}_p{pg.page}_{idx}",
                "text": chunk,
                "metadata": {
                    "page": pg.page,
                    "chapter": pg.chapter or "",
                    "section": pg.section or "",
                    "source_file": source_file,
                },
            }
            for idx, chunk in enumerate(chunk_words(pg.text, chunk_size_words, overlap_words))
        ]
        yield page_record, chunk_records
