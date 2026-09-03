FROM python:3.11-slim

# build-essential: asyncpg/pymupdf wheels build cleanly; curl: healthcheck.
# No ffmpeg/whisper/chromadb anymore -- audio goes straight to Gemini and
# vectors live in Supabase, which is what keeps this image (and the
# running container's RAM) small enough for a 512MB free host.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Scratch space only (streamed uploads mid-request) -- nothing durable
# lives here; see UPLOAD_DIR in config.py.
RUN mkdir -p /app/data/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Pinned explicitly, not just relying on uvicorn's default: each additional
# worker is a full second process duplicating the ~110MB base import
# footprint measured for this app -- on a 512MB host, 2 workers alone would
# already eat ~220MB before serving a single request.
