# Security model

This document describes how the portfolio backend is hardened. The site is a
personal project, not a regulated production system — the goal is "safe enough
to leave running on the public internet without losing my Anthropic API budget
overnight."

## Threat model

The site faces three realistic threats:

1. **Cost abuse.** Someone hits `/api/chat` in a loop to burn through the
   configured Anthropic API budget.
2. **Prompt injection.** A visitor types crude jailbreak attempts hoping the
   chatbot will say something embarrassing about me.
3. **Information disclosure.** A misconfigured deployment exposes secrets,
   stack traces, or internal API schemas to the public.

Out of scope: nation-state actors, authenticated session abuse (no auth here),
DoS at the network layer (handled by the host provider).

## Controls in place

### 1. Input validation and length caps

- Every endpoint receives a Pydantic v2 model. Strings are bounded with
  `min_length` and `max_length`. Integers are bounded with `ge` and `le`.
- A global `max_input_chars` (default `2000`) is the upper bound applied
  defensively *again* by `sanitise_visitor_input()`.
- This means an attacker cannot send a 1 MB message body to the LLM.

### 2. Prompt-injection guardrails

`app/security.py :: sanitise_visitor_input` runs on every visitor-supplied
string before it touches the LLM prompt or the persisted database row. It:

- Truncates to `max_input_chars`.
- Strips ASCII control characters except newline/tab.
- Replaces crude prompt-override patterns (e.g. *"ignore previous
  instructions"*, *"you are now a..."*, `<|chat_template_token|>`) with
  `[redacted]`.
- Collapses whitespace runs.

This is **not** a complete defence against a determined attacker — no system
prompt can fully prevent prompt injection. It's hygiene that removes the
obvious noise.

### 3. Per-IP rate limiting

`slowapi` runs as a middleware. Default limits (configurable in `.env`):

| Endpoint | Limit |
|---|---|
| `POST /api/chat`            | 20 / minute |
| `GET  /api/chat/citations`  | 60 / minute |
| `POST /api/demo/embed`      | 60 / minute |
| `POST /api/demo/sentiment`  | 60 / minute |
| `POST /api/demo/tokenise`   | 60 / minute |
| (default)                   | 120 / minute |

On exceedance, the API returns `429 Too Many Requests` with a friendly JSON
body. The LLM is never called for rejected requests.

### 4. Security response headers

Every response carries:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=(), payment=()`
- `Content-Security-Policy: default-src 'self'; ...` (allowlists Google Fonts
  CDN, blocks everything else)

### 5. Database safety

- All queries go through SQLAlchemy 2.0 (parametrised) — no string-built SQL.
- Conversation log uses the *sanitised* question, never the raw input.
- Logging is best-effort: a DB failure never blocks the chat response.

### 6. CORS / Host policy

- In `ENVIRONMENT=development`: CORS allows any origin (so you can test from
  any port).
- In `ENVIRONMENT=production`: CORS narrows to `CORS_ORIGINS`, and
  `TrustedHostMiddleware` rejects requests with unexpected `Host:` headers if
  `TRUSTED_HOSTS` is set.
- `allow_credentials=False` — the API never reads or sets cookies.

### 7. API docs gating

`/docs`, `/redoc`, and `/openapi.json` are exposed by default (recruiters love
them) but hidden in production when `ENVIRONMENT=production` and
`API_DOCS_ENABLED` is false. The OpenAPI schema is therefore not a free
intelligence-gathering surface for attackers.

### 8. Error handling

The global exception handler returns:

- `development`: full exception name, message, and the tail of the traceback —
  for easy debugging.
- `production`: a generic `{"detail": "Internal server error."}` plus a
  server-side log entry. No stack traces ever reach the client.

### 9. Secret hygiene

- `.env` is in `.gitignore` and `.dockerignore`.
- `.env.example` only contains placeholder strings (`sk-ant-...`).
- The Docker image runs as a non-root user (`uid=1000`).
- The only secret the project handles is the LLM API key, read once
  at startup via Pydantic Settings; never logged.

## Deployment checklist

Before exposing this on a public URL:

- [ ] Copy `.env.example` to `.env`, set real values.
- [ ] `ENVIRONMENT=production` to hide docs and enable strict error handling.
- [ ] If using `LLM_PROVIDER=anthropic`, double-check your Anthropic console
      spend cap is set.
- [ ] Set `CORS_ORIGINS` to your real domain.
- [ ] Set `TRUSTED_HOSTS` to your real domain.
- [ ] Verify `git log --all -- .env` returns nothing (the file was never
      committed).
- [ ] Hit your deployment with `curl -X POST .../api/chat ...` 25 times in
      ~30 seconds and confirm you get rate-limited.

## What is *not* protected

To be candid about the gaps:

- **No CAPTCHA on the chat input** — a motivated attacker who solves CAPTCHAs
  could still hit the rate limit via a botnet. The rate limit is per-IP.
- **No persistent ban list** — once a 429 fires, it's just rate-limited, not
  blocked outright.
- **Conversation log retention is indefinite** — fine for SQLite local, but a
  production deploy should set up a retention policy.
- **No CSRF protection** — moot, since there's no auth and no state-changing
  cookie-authenticated endpoints.

## Reporting

If you find a security issue, open an issue on this repository rather
than opening a public issue. I'll respond within 48 hours.
