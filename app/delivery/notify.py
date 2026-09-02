from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage

from app.config import settings


def _send_email_blocking(
    to_addr: str,
    subject: str,
    body: str,
    attachment_bytes: bytes | None,
    attachment_filename: str | None,
) -> None:
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        raise RuntimeError(
            "SMTP not configured -- set SMTP_USER / SMTP_PASSWORD / SMTP_FROM in .env "
            "(a Gmail address + app password is the free option, see README)"
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM or settings.SMTP_USER
    msg["To"] = to_addr
    msg.set_content(body)

    if attachment_bytes:
        subtype = "pdf" if (attachment_filename or "").lower().endswith(".pdf") else "octet-stream"
        msg.add_attachment(
            attachment_bytes,
            maintype="application",
            subtype=subtype,
            filename=attachment_filename or "attachment",
        )

    with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.send_message(msg)


async def deliver(
    body_text: str,
    subject: str,
    delivery: str,
    email: str | None,
    attachment_bytes: bytes | None = None,
    attachment_filename: str | None = None,
) -> str:
    """Returns a short status string describing what happened to the result."""
    if delivery == "email":
        if not email:
            raise ValueError("delivery='email' requires an 'email' address")
        await asyncio.to_thread(
            _send_email_blocking, email, subject, body_text, attachment_bytes, attachment_filename
        )
        return f"emailed to {email}"
    return "displayed"
