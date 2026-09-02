# Reference RAG API

A self-hosted, free-tier-only API that combines:

1. **Reference-book RAG** — ingest a large (arbitrarily large) PDF, page +
   chapter/section aware, streamed so a giant PDF ingests in roughly
   constant memory. Questions are gated behind a keyword (default
   `REFERENCE BOOK:`) and answers are locked to retrieved pages only, with
   citations.
2. **Audio → transcript → notes/summary** — the priority feature. One
   Gemini call turns a recording into a transcript, summary, and bullet
   notes.
3. **Vision, scoped to reference material only.** There is no endpoint that
   accepts an arbitrary uploaded photo. `/vision/page` looks only at a page
   that's already part of an ingested PDF/slideshow and that page's own
   embedded image (a diagram, chart, figure) — nothing else. This is a
   deliberate limit: an open "photograph anything and ask about it" tool is
   trivially misusable to photograph a homework/exam question, which isn't
   something this is meant to help with.
4. **Notes export to PDF**, streamed straight back in the response.
5. **Display or email delivery** for any result.
6. **Per-session RAG isolation.** Each open tab gets its own default
   collection automatically, so ingesting a PDF in one tab doesn't collide
   with another; naming a collection explicitly still lets you share one
   on purpose.
7. A single-file **web widget** (mic + text + page-vision) you can embed in
   a Google Site or host anywhere.

## Architecture, and why it looks like this

Local disk does **not** survive a sleep/restart/redeploy on free container
hosts (Render, Koyeb, SnapDeploy, etc — this was tested and confirmed, not
assumed). So nothing durable lives on the container's own filesystem:

- **Vectors, page text, and page images live in Supabase Postgres +
  pgvector** (free, no card, survives restarts). `app/rag/db.py` /
  `vector_store.py` / `page_index.py`.
- **Audio never touches local disk for storage** — it goes straight to
  Gemini's own audio understanding via the Files API (`app/audio/
  transcribe.py`), which also means no local Whisper/ffmpeg/torch, the
  single heaviest thing this app could put in a small container.
- **PDF ingestion streams in small batches** (`INGEST_BATCH_SIZE` in
  `.env`) rather than building the whole book's chunk list in memory —
  memory use during ingest is bounded by that setting, not by how many
  pages the PDF has.
- **Uploads stream to a temp file in fixed-size chunks**, never fully
  loaded into RAM (`MAX_UPLOAD_MB` / `UPLOAD_STREAM_CHUNK_BYTES` in `.env`).
- **Exports build directly in memory and stream back in the same request**
  — nothing written to disk and linked.

Net effect: the running container only ever needs to hold FastAPI itself,
one request's streamed bytes, and a small connection pool — light enough
for a 512MB-class free host.

**Measured, not guessed** (actual RSS, running this app against a real
Postgres):

| Stage | RSS |
|---|---|
| Bare Python interpreter | ~19 MB |
| App fully imported (FastAPI, PyMuPDF, asyncpg, google-genai, reportlab) | ~110 MB |
| + DB pool created, schema initialized | ~111 MB |
| + ingesting a small test PDF | ~130 MB |

~110MB is the real fixed floor (library imports) — not something you can
tune away without dropping a dependency. The tunable knobs, all in `.env`,
in rough order of what actually moves the needle:

- `INGEST_BATCH_SIZE` (default 24) — chunks embedded+written together
  during ingest. Lower = more, smaller Postgres round-trips; higher =
  fewer round-trips but a bigger in-memory batch. Effect on peak RAM is
  modest either way since the fixed import cost dominates, but it's free
  headroom to lower on a tight host.
- `PAGE_IMAGE_DPI` (default 110) — resolution for rendering a page that
  has an embedded image (for `/vision/page`). This is the more genuinely
  variable cost, especially for slideshow-style PDFs where most/every
  page renders one; lower it further if ingesting image-heavy slides on
  a very tight host.
- `DB_POOL_MAX_SIZE` (default 3) — pooled Postgres connections. This app
  is meant to run as a single uvicorn worker (pinned explicitly in the
  Dockerfile's `CMD` — **do not raise `--workers`**, each additional
  worker is a whole second process duplicating that ~110MB floor).
- `MAX_UPLOAD_MB` (default 150) — sized for the actual intended content
  (novels/textbooks, short slideshows), not an arbitrary giant scan.
  Raise it if you genuinely need more.

## 1. Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: at minimum set GEMINI_API_KEY, DATABASE_URL, and API_KEY
```

Get a free `GEMINI_API_KEY` at https://aistudio.google.com/apikey — no
card needed. Free-tier model IDs shift over time; if a model in `.env`
starts erroring, check https://ai.google.dev/gemini-api/docs/models and
update `TEXT_MODEL` / `VISION_MODEL` / `AUDIO_MODEL` / `EMBED_MODEL` — the
only lines that should need touching.

## 2. Supabase setup (free, no card)

1. Create a project at https://supabase.com (free tier: 500MB Postgres,
   pgvector included, pauses after 7 days idle — fine for personal use,
   ping `/health` on a schedule if you want to keep it warm).
2. Project Settings → Database → Connection string → URI. Use the
   **Session pooler** connection (built for exactly this short-lived,
   many-cold-starts pattern), paste it into `.env` → `DATABASE_URL`.
3. That's it — `app/rag/db.py` creates the `vector` extension and both
   tables itself on first startup (`init_schema()`, idempotent, runs from
   `main.py`'s lifespan handler). Nothing to run manually in the Supabase
   SQL editor.

## 3. Run it

```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/health
```

## 4. Ingest a reference PDF

```bash
curl -X POST http://localhost:8000/ingest/reference \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/your-book.pdf"
```

This streams the PDF page-by-page: extracts text, detects chapter/section
headings heuristically (bigger/bold text matched against patterns like
"Chapter 3" or "1.1"), chunks each page (~220 words, 40-word overlap —
tune in `.env`), and — if a page contains an embedded image — renders
that page to a small PNG for later vision use. Everything is embedded and
upserted to Supabase in batches of `INGEST_BATCH_SIZE` chunks, so a
1,000-page PDF and a 10-page PDF both use the same, small amount of
memory. The uploaded file itself is deleted from local disk the moment
ingestion finishes.

To use a **different** PDF instead of the default, ingest it into a named
collection and pass that `collection` on later calls:

```bash
curl -X POST http://localhost:8000/ingest/reference \
  -H "X-API-Key: $API_KEY" \
  -F "file=@other.pdf" -F "collection=other_doc"
```

**Known limitation:** heading detection is a font-size/bold heuristic, not
a layout model. If a PDF's headings don't look visually different from
body text, chapter/section come back empty — page numbers still work
regardless. If it's mis-detecting running headers/footers as headings on
a specific PDF, pass `detect_headings=False` where `iter_page_and_chunks`
is called in `app/rag/ingest_service.py`.

## 5. Sessions and collections

Every request can carry a `session_id`. If you don't also pass an explicit
`collection`, the request defaults to a collection scoped to that session
— so two browser tabs (the widget generates a fresh `session_id` per tab
in `sessionStorage`) ingesting different PDFs never collide. Pass an
explicit `collection` name whenever you *want* to share one across
sessions on purpose (e.g. everyone using the same `REFERENCE_BOOK`
collection). No `session_id` and no `collection` falls back to
`DEFAULT_COLLECTION` in `.env`, same as a bare `curl` call with neither.

## 6. The keyword (text RAG)

Nothing runs RAG on `/ask` unless the message contains the keyword
(default `REFERENCE BOOK:`, set `RAG_KEYWORD` in `.env`, case-insensitive).
Without it, `/ask` just calls Gemini directly.

```
REFERENCE BOOK: look at 1.1 and explain X
```

If the message names a page ("page 42"), section ("1.1", "2.3.4"), or
"chapter 3", that exact text is pulled straight from the page index — no
embedding search, no room to guess the wrong page. Otherwise it falls back
to a top-K semantic search (`TOP_K` in `.env`) and drops any chunk whose
distance is past `MAX_RELEVANT_DISTANCE` rather than handing the model
noise. If nothing usable comes back, the API returns a fixed "not found"
string instead of calling the model at all. Every RAG answer is required
to cite `(p.X, chapter/section)` on every claim, using only values that
were actually retrieved.

## 7. Vision (reference pages only)

```bash
curl -X POST http://localhost:8000/vision/page \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"collection":"other_doc","page":42,"prompt":"what does this diagram show?"}'
```

- 404 if that page doesn't exist in the collection.
- 422 if that page exists but has no embedded image (nothing for vision to
  look at — use `/ask` for text content instead).
- Otherwise Gemini sees the page's rendered image plus its extracted text,
  under a locked prompt that requires a page/section citation and forbids
  outside knowledge.

There's no endpoint that takes an arbitrary photo upload. That's on
purpose — see the module docstring in `app/vision/image_qa.py`.

## 8. Audio → notes (priority feature)

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "X-API-Key: $API_KEY" -F "file=@recording.m4a"
```

Uploads via Gemini's Files API (handles long recordings without loading
them fully into this process), asks for transcript + summary + notes in
one call, then deletes the uploaded copy from Gemini immediately rather
than waiting for its default 48h auto-expiry.

## 9. Endpoints

All except `/health` and `/session` require header `X-API-Key: <your
API_KEY>` **or** `X-Session-Token: <token from /session>` (see §15 for why
the widget uses the latter).

| Endpoint | Method | Body |
|---|---|---|
| `/health` | GET | — (no auth) |
| `/session` | POST | JSON `{api_key}` → `{token, expires_at}` (no other auth needed — this *is* the login step) |
| `/ingest/reference` | POST | multipart: `file`, `collection?`, `session_id?`, `replace?` |
| `/ask` | POST | JSON `{message, collection?, session_id?, delivery, email?}` |
| `/vision/page` | POST | JSON `{page, prompt, collection?, session_id?, delivery?, email?}` |
| `/transcribe` | POST | multipart: `file`, `delivery?`, `email?` |
| `/batch` | POST | JSON list of `{kind:"ask", message, collection?, session_id?}` — runs concurrently |
| `/export/pdf` | POST | JSON `{title, content, delivery, email?}` |
| `/collections` | GET | — lists every collection currently in Supabase |

`delivery` is `"display"` (returned directly — a PDF streams back as the
response body) or `"email"` (sent via SMTP, see below).

## 10. Concurrency

Every route is `async`; Gemini calls use the async client
(`client.aio`); Postgres access goes through an `asyncpg` pool. Firing a
`/transcribe` request and an `/ask` or `/vision/page` request at the same
time processes both concurrently, not one after the other. `/batch` shows
this explicitly for multiple text asks in one call.

## 11. Privacy

- Audio/image/PDF uploads are deleted from local disk the moment they're
  processed — nothing lingers.
- Audio uploaded to Gemini's Files API is explicitly deleted right after
  the transcription call, rather than left to auto-expire.
- The access-log middleware logs method, path, status, and duration only
  — never message text, transcripts, or model output.
  `PRIVACY_SAFE_LOGGING=false` disables even that.
- What this **can't** control: Google's own handling of Gemini API
  requests, or Supabase's handling of stored data. Read their respective
  terms if that matters for your use case — there's no way to use either
  free tier without them seeing/storing the relevant data.

## 12. Email delivery (Gmail app password — free)

1. Turn on 2-Step Verification: https://myaccount.google.com/security
2. Create an app password: https://myaccount.google.com/apppasswords
3. In `.env`: `SMTP_USER`, `SMTP_PASSWORD` (the 16-char app password),
   `SMTP_FROM`.

## 13. Docker

```bash
cp .env.example .env   # fill in GEMINI_API_KEY, DATABASE_URL, API_KEY
docker compose up -d --build
```

`./data` is volume-mounted for scratch space during uploads only — nothing
durable is stored there anymore, so losing it on a redeploy costs nothing.

## 14. Deploying to a free host

Confirmed to work: any host that gives you a real Docker container with
outbound internet access on arbitrary ports/protocols (Render, Koyeb,
SnapDeploy, a home box behind Cloudflare Tunnel, etc). This app needs:
outbound HTTPS to Gemini, and a raw Postgres connection (not just HTTP) to
Supabase on port 5432/6543.

**Does not work: PythonAnywhere's free tier**, checked directly against
their own docs and staff forum replies — free accounts can only reach the
internet via HTTP(S) to an allowlist, and "database connections will not
work in free accounts" (their words). The free tier also has no ASGI
support at all, so FastAPI itself doesn't run there without a rewrite.
If you're evaluating a platform not listed here, the two things to verify
before deploying are: (1) does it run arbitrary Docker/ASGI apps, and (2)
does it allow outbound TCP to an arbitrary Postgres host, not just HTTP to
an allowlist.

## 15. The web widget

`frontend/widget.html` is one dependency-free file: mic recording, a
text-ask box, and a page-number field for vision — themed black / white /
dark rose-pink with a purple undertone. It generates a `session_id` once
per tab (`sessionStorage`) and sends it on every call, so each open tab
gets its own default collection automatically.

**Google Sites:** Insert → Embed → Embed code → paste the file's contents.

**Anywhere else:** open the file directly, or host it as a static page —
one HTML file, no build step.

**Auth: the real API_KEY is never stored in the browser, not even in
`localStorage`.** Raw keys sitting in `localStorage` indefinitely are a
real risk on anything embedded in a page you don't fully control (XSS, a
shared machine, or a Google Sites embed someone finds the URL to) — there's
no expiry and no way to revoke just that one exposure. Instead:

1. First use in a tab, you enter the real API_KEY once. The widget calls
   `POST /session` to exchange it for a short-lived signed token
   (`SESSION_TTL_HOURS` in `.env`, default 24h) and immediately forgets the
   real key — it's used for that one exchange call and never written
   anywhere.
2. Only the token is kept, in `sessionStorage` (clears when the tab
   closes, not shared across tabs). Every request sends
   `X-Session-Token` instead of the real key.
3. If a token leaks somehow, it self-expires rather than granting
   permanent access the way a leaked raw key would.
4. When a token expires, that tab just asks for the real key again —
   `curl`/scripts/ESP32 etc. can keep using `X-API-Key` directly, since
   they're not a browser exposure risk the same way.

This isn't a real multi-user login system — it's still one shared secret
behind it — but it bounds how much a single leaked browser-side credential
can do. For real defense-in-depth on a public host, also set
`CORS_ORIGINS` in `.env` to your actual widget's origin instead of `*`.

## Directory layout

```
app/
  main.py                 FastAPI app, all routes, upload streaming
  config.py                env-driven settings
  gemini_client.py         shared Gemini client (text + embeddings + files)
  security.py               X-API-Key auth
  schemas.py                 request/response models
  rag/
    db.py                     Postgres pool + schema (pgvector, page_image)
    pdf_ingest.py              PDF -> pages -> headings -> chunks -> page images
    embeddings.py               batched Gemini embedding calls
    vector_store.py              pgvector similarity search
    page_index.py                 exact page/section lookup + page images
    ingest_service.py               streaming batch orchestration
    qa_engine.py               keyword-gated, locked RAG answer pipeline
    sessions.py                per-tab default collection resolution
  audio/transcribe.py       Gemini Files API -> transcript/summary/notes
  vision/image_qa.py        reference-page-only vision (no photo upload path)
  export/pdf_export.py      reportlab notes PDF, built in memory
  delivery/notify.py        display vs. email (SMTP)
  utils/text_clean.py       strips first-person/filler from output
frontend/widget.html      mic + text + page-vision web widget
docker/Dockerfile
docker-compose.yml
requirements.txt
.env.example
CONTINUATION_PROMPT.md    brief for iOS/ESP32/further work (unaffected by this rework)
```
