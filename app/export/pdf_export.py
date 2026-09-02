"""
Builds straight into an in-memory buffer rather than a file path -- the
container's local disk may not survive between requests on a free host,
so exports are returned directly in the response body instead of being
written, linked, and fetched in a second request.
"""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def create_notes_pdf_bytes(title: str, content: str) -> bytes:
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 0.25 * inch)]
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.12 * inch))
            continue
        story.append(Paragraph(escape(line), styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
