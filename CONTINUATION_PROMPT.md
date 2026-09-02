# Continuation brief

Paste this whole file into a coding agent (Claude Code, Cursor, Aider,
whatever free/local coding agent you're using) after giving it this
repository. It has everything the agent needs to keep going without
re-deriving the existing API.

---

## Context: what already exists

A working FastAPI backend (`app/`) is already built and running (see its
own `README.md`). Do not redesign it — extend it. Summary of the
contract you're building against:

- Base URL: whatever host this is deployed to (Oracle Cloud VM, port
  8000 by default, optionally behind HTTPS via a reverse proxy).
- Auth: every request except `GET /health` needs header
  `X-API-Key: <value from .env API_KEY>`.
- `POST /ask` — JSON `{message, collection?, delivery, email?}` →
  `{used_rag, answer, sources[], delivered}`. RAG only triggers if
  `message` contains the keyword defined in `.env` `RAG_KEYWORD`
  (default `"REFERENCE BOOK:"`).
- `POST /vision` — multipart `file` (image), `prompt`, `collection?`,
  `delivery?`, `email?` → `{used_rag, answer, sources[], delivered}`.
- `POST /transcribe` — multipart `file` (audio), `delivery?`, `email?` →
  `{transcript, summary, notes[], delivered}`.
- `POST /export/pdf` — JSON `{title, content, delivery, email?}` →
  `{url?, delivered}` (url is a `/files/<name>.pdf` path on the API host
  when `delivery="display"`).
- `POST /export/slides` — JSON `{title, bullets[], delivery, email?}` →
  `{url?, delivered}` (Google Slides edit URL).
- `POST /batch` — JSON list of `{kind:"ask", message, collection?}`,
  resolved concurrently.
- Full schemas: `app/schemas.py`. Full route implementations:
  `app/main.py`.

A working reference web client already exists at `frontend/widget.html`
(vanilla HTML/JS, no build step, mic recording via `MediaRecorder`,
image capture via `<input type=file capture>`, calls the endpoints
above with `fetch`). Color theme: black `#0b0b0d` background, white
`#f2eef1` text, dark rose-pink-with-purple accent `#9b4f72`
(brighter hover state `#c17198`) — keep any new UI consistent with this
palette unless told otherwise.

---

## Task 1 — Make it feel like an app on iOS (fastest path, do this first)

Before building a native app, ship the cheap version:

1. Confirm `frontend/widget.html` is served over HTTPS (required for
   `getUserMedia` mic access and for "Add to Home Screen" to behave like
   an app).
2. Add a minimal `manifest.json` + a couple of icon sizes and register a
   trivial service worker so "Add to Home Screen" on iOS Safari gives a
   standalone, full-screen icon (no browser chrome). This is a PWA, not
   an App Store app, and needs no Apple account or review.
3. Optionally add an iOS Shortcuts recipe (documented in the README, not
   code) that lets someone record a Shortcut which POSTs to `/ask` or
   `/transcribe` with the saved API key, for a "Siri, ask the reference
   tool..." style flow.

If a real native app is still wanted after that: propose SwiftUI vs.
React Native / Expo, with the tradeoff that Expo lets you reuse the
existing JS logic in `widget.html` almost directly for the API calls.
Do not start a native build without confirming which framework is
wanted first.

## Task 2 — ESP32 companion device

Goal: a physical button/mic device that talks to the same API over
Wi-Fi, for hands-free capture (e.g. clip a mic module + button to
something, press to record, release to send to `/transcribe`).

Suggested shape (confirm before deep-diving):

- Board: ESP32 (has Wi-Fi built in; ESP32-S3 if you want built-in USB
  and more RAM headroom for audio buffering).
- Mic: I2S MEMS mic module (e.g. INMP441) — far simpler than an analog
  mic + ADC on this chip.
- Flow: button press starts recording to a local buffer (PSRAM if
  available) or streams in chunks; button release (or a max duration
  timeout) triggers an HTTP multipart POST to `/transcribe` with the
  captured audio as a WAV file; response (or just a status) can drive an
  onboard LED/buzzer for feedback since there's no screen.
- Framework: Arduino-ESP32 core or ESP-IDF directly; Arduino core is
  faster to get working, ESP-IDF gives more control over I2S DMA timing
  if the Arduino I2S library turns out to be too high-level.
- Auth: store the `X-API-Key` in the firmware (or better, in NVS flash
  set via a one-time provisioning step) rather than hardcoding it in
  source if this will ever be shared/open-sourced.
- Power: battery + charging circuit only if it needs to be portable;
  USB power is fine for a desk-mounted version.

Deliverable: firmware source, a wiring diagram (text description is
fine), and a short setup doc (Wi-Fi credentials provisioning, how to
flash it).

## Task 3 — Anything else worth doing while in here

Not required, but worth flagging to the user before doing unprompted:

- Rate limiting / IP throttling on the API if it'll sit on a public IP
  long-term (a single shared API key has no per-client limits right
  now).
- Streaming responses for `/ask` (Gemini supports streaming; the current
  implementation waits for the full response).
- A small admin view for listing/deleting ingested collections (the API
  has `GET /collections` already; there's no delete endpoint yet).
- Automated tests — there are currently none.

Confirm scope with the user before taking on any of these; the two
tasks above (PWA/Shortcuts, ESP32) are the ones actually requested.
