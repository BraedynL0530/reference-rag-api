"""
Two ways in, both gating every endpoint except /health:

1. X-API-Key: <the real, permanent API_KEY> -- for curl/scripts/ESP32/
   anything that can keep a secret safely off a browser.
2. X-Session-Token: <short-lived token from POST /session> -- for the
   browser widget. The widget exchanges the real API_KEY for one of
   these once, then never stores the real key again -- it keeps this
   token in sessionStorage instead (clears when the tab closes) rather
   than a permanent key in localStorage. If the token leaks (XSS, a
   shared machine, a public Google Sites embed someone stumbles onto),
   it self-expires (SESSION_TTL_HOURS, default 24h) rather than handing
   out permanent access the way a raw API key sitting in localStorage
   forever would.

This is still single-secret auth, not real multi-user accounts -- the
point is bounding the blast radius of a leaked browser-side credential,
not building a login system.
"""
from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Header, HTTPException, status

from app.config import settings


def _signing_secret() -> str:
    # Falls back to a one-way derivation of API_KEY so this works with zero
    # extra config, but a real SESSION_SECRET in .env is stronger -- knowing
    # a leaked session token then reveals nothing about API_KEY either way,
    # this fallback just skips needing a second secret for personal use.
    return settings.SESSION_SECRET or hashlib.sha256(f"session:{settings.API_KEY}".encode()).hexdigest()


def create_session_token() -> tuple[str, int]:
    expires_at = int(time.time()) + settings.SESSION_TTL_HOURS * 3600
    sig = hmac.new(_signing_secret().encode(), str(expires_at).encode(), hashlib.sha256).hexdigest()
    return f"{expires_at}.{sig}", expires_at


def _verify_session_token(token: str) -> bool:
    try:
        expires_str, sig = token.split(".", 1)
        expires_at = int(expires_str)
    except (ValueError, AttributeError):
        return False
    if time.time() > expires_at:
        return False
    expected = hmac.new(_signing_secret().encode(), expires_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


async def require_api_key(
    x_api_key: str = Header(default=""),
    x_session_token: str = Header(default=""),
) -> None:
    if not settings.API_KEY or settings.API_KEY == "changeme":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: set a real API_KEY in .env before deploying.",
        )
    if x_api_key and hmac.compare_digest(x_api_key, settings.API_KEY):
        return
    if x_session_token and _verify_session_token(x_session_token):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired credentials")
