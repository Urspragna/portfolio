"""LLM provider abstraction: anthropic, ollama, or echo."""
from __future__ import annotations

from typing import AsyncIterator, Protocol

import httpx

from app.config import Settings, get_settings


class LLM(Protocol):
    name: str

    async def stream(self, prompt: str) -> AsyncIterator[str]:  # pragma: no cover
        ...


# Echo provider (no API key)

class EchoLLM:
    """Echo back the retrieved context. Works without any API key."""

    name = "echo"

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        # Extract context and question from the prompt
        ctx = ""
        question = ""
        if "Context passages:" in prompt and "Question:" in prompt:
            after = prompt.split("Context passages:", 1)[1]
            ctx_part, q_part = after.split("Question:", 1)
            ctx = ctx_part.strip().strip("-").strip()
            question = q_part.split("Answer:")[0].strip()
        # FIXME: this parsing is a bit fragile, should use regex or structured format

        intro = f"Based on my portfolio: here's what's relevant to {question or 'your question'}:\n\n"
        yield intro

        body = ctx[:1400] if ctx else "(no relevant context found)"
        # Stream in small chunks for better UX
        for i in range(0, len(body), 60):
            yield body[i : i + 60]

        yield "\n\n— Echo provider (no LLM key). Set ANTHROPIC_API_KEY or run `ollama serve` for real responses."


# Anthropic Claude

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
                        # TODO: log these parse errors somewhere
                        continue
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")


# Ollama (local LLM)

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


# Factory

def make_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLM(settings)  # type: ignore[return-value]
    if provider == "ollama":
        return OllamaLLM(settings)  # type: ignore[return-value]
    return EchoLLM()  # type: ignore[return-value]
