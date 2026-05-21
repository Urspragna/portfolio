"""Security utilities: rate limiting, headers, input sanitisation.

Kept in one file so the security posture is reviewable in a single read.
"""
from __future__ import annotations

import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import get_settings

settings = get_settings()


# ────────────────────────────────────────────────────────────────────────
# Rate limiting
# ────────────────────────────────────────────────────────────────────────
# Single Limiter instance shared by all routers. Identifies callers by IP.
# Default cap applies if a route forgets its own decorator.

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_default],
)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Friendly JSON response instead of a stack trace when limit is hit."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Slow down — this is a personal portfolio.",
            "limit": str(exc.detail),
        },
    )


# ────────────────────────────────────────────────────────────────────────
# Security headers middleware
# ────────────────────────────────────────────────────────────────────────


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add baseline security headers to every response.

    Headers chosen to be safe defaults for a portfolio site that:
      - serves its own HTML/JS (no third-party scripts beyond Google Fonts CDN)
      - has a public REST API (the `/api/*` endpoints)
      - never authenticates users / never sets sensitive cookies
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=()",
        )
        # Conservative CSP: only allow scripts/styles from self + Google Fonts CDN.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'",
        )
        return response


# ────────────────────────────────────────────────────────────────────────
# Input sanitisation (basic prompt-injection guardrails)
# ────────────────────────────────────────────────────────────────────────

# Patterns that look like crude prompt-injection / jailbreak attempts.
# Not a defence against a determined attacker — just removes the noise.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (?:all )?(?:previous|prior|above) (?:instructions?|prompts?)"),
    re.compile(r"(?i)disregard (?:all )?(?:previous|prior|above)"),
    re.compile(r"(?i)you are (?:now|actually) (?:a |an )?[A-Za-z]+"),
    re.compile(r"(?i)system prompt[:\s]"),
    re.compile(r"(?i)reveal (?:your |the )?(?:system )?prompt"),
    re.compile(r"<\|.*?\|>"),  # chat-template-style tokens
]

# Strip control characters except newline/tab.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitise_visitor_input(text: str, *, max_chars: int) -> str:
    """Clean a visitor-supplied string before it reaches the LLM prompt.

    Returns a string that's been:
      - truncated to `max_chars` (defensive — Pydantic already enforces a limit)
      - stripped of control characters
      - had crude prompt-injection patterns replaced with neutral text
      - whitespace-collapsed

    This is *not* full prompt-injection defence (no system can fully prevent it).
    It's just baseline hygiene so the chatbot doesn't echo obviously hostile text
    into its own prompt or its persisted log.
    """
    if not text:
        return ""

    text = text[:max_chars]
    text = _CONTROL_CHARS.sub("", text)

    for pat in _INJECTION_PATTERNS:
        text = pat.sub("[redacted]", text)

    # Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ────────────────────────────────────────────────────────────────────────
# Wiring helper
# ────────────────────────────────────────────────────────────────────────


def install_security(app: FastAPI) -> None:
    """Wire rate limiting + security headers + 429 handler onto a FastAPI app.

    Call once during app construction.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
