"""
Privacy-first logging.

Only method, path, status code, and duration are ever logged. Message text,
transcripts, image/audio contents, and model output are never written to
logs, on disk or otherwise. This is enforced by *not passing those values
into the logger anywhere in the codebase* -- grep for `logger.` if you add
a new endpoint and keep it that way.
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
