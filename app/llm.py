"""LLM provider abstraction.

Three providers are wired up — pick via `LLM_PROVIDER` env var:

  • anthropic — frontier quality, requires ANTHROPIC_API_KEY (paid)
  • ollama    — local LLM via `ollama serve` (free, private)
  • echo      — deterministic stub that just echoes the retrieved context;
                lets the chatbot work *without any LLM keys at all* — perfect
                for first-time visitors and for portfolio reviewers who
                don't want to spin up an API key.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol

import httpx

from app.config import Settings, get_settings


class LLM(Protocol):
    name: str

    async def stream(self, prompt: str) -> AsyncIterator[str]:  # pragma: no cover
        ...


# ────────────────────────────────────────────────────────────────────────
# Echo (no-key deterministic stub)
# ────────────────────────────────────────────────────────────────────────


class EchoLLM:
    """Returns a structured answer assembled directly from retrieved context.

    This makes the RAG demo work *out-of-the-box* — even with no API key —
    while still showing the retrieval layer in action.
    """

    name = "echo"

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        # Extract context block + question from the rendered prompt
        ctx = ""
        question = ""
        if "Context passages:" in prompt and "Question:" in prompt:
            after = prompt.split("Context passages:", 1)[1]
            ctx_part, q_part = after.split("Question:", 1)
            ctx = ctx_part.strip().strip("-").strip()
            question = q_part.split("Answer:")[0].strip()

        intro = (
            "Based on Pragna's portfolio knowledge base, here's what's "
            f"relevant to *{question or 'your question'}*:\n\n"
        )
        yield intro

        # Yield the context as the answer body — nothing invented.
        body = ctx[:1400] if ctx else "(no relevant context found)"
        # stream in small chunks for nicer UX
        for i in range(0, len(body), 60):
            yield body[i : i + 60]

        yield (
            "\n\n— *Echo provider (no LLM key set). Add ANTHROPIC_API_KEY or run "
            "`ollama serve` to enable a real LLM response.*"
        )


# ────────────────────────────────────────────────────────────────────────
# Anthropic Claude
# ────────────────────────────────────────────────────────────────────────


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the anthropic provider")
        self.api_key = settings.anthropic_api_key
        self.model = settings.anthropic_model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 600,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        import json

                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")


# ────────────────────────────────────────────────────────────────────────
# Ollama (local LLM)
# ────────────────────────────────────────────────────────────────────────


class OllamaLLM:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json

                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "response" in evt:
                        yield evt["response"]
                    if evt.get("done"):
                        break


# ────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────


def make_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLM(settings)  # type: ignore[return-value]
    if provider == "ollama":
        return OllamaLLM(settings)  # type: ignore[return-value]
    return EchoLLM()  # type: ignore[return-value]
