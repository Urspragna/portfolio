"""Provider abstraction: hosted, local, or echo."""
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

        body = ctx[:1400] if ctx else "(no relevant context found)"
        # Stream in small chunks for better UX
        for i in range(0, len(body), 60):
            yield body[i : i + 60]


# Hosted provider

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


# Groq

class GroqLLM:
    name = "groq"

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for the groq provider")
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self._model_checked = False

    # Groq retires models regularly; if the configured one is gone, use the
    # closest model it still serves instead of failing every chat.
    PREFERRED = ("llama-3.3-70b", "gpt-oss-120b", "llama-4", "70b", "llama")

    async def _resolve_model(self, client: httpx.AsyncClient, headers: dict) -> None:
        if self._model_checked:
            return
        try:
            resp = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
            resp.raise_for_status()
            ids = [m["id"] for m in resp.json().get("data", [])]
            if ids and self.model not in ids:
                pick = next((i for p in self.PREFERRED for i in ids if p in i), ids[0])
                print(f"[llm] groq model {self.model} unavailable; using {pick}")
                self.model = pick
            self._model_checked = True
        except Exception as exc:  # keep the configured model; chat call reports errors
            print(f"[llm] could not list groq models ({type(exc).__name__})")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        import json
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            await self._resolve_model(client, headers)
        payload = {
            "model": self.model,
            "max_tokens": 600,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
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
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = evt.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text


# Local provider

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
    if provider == "groq" and settings.groq_api_key:
        return GroqLLM(settings)  # type: ignore[return-value]
    if provider == "ollama":
        return OllamaLLM(settings)  # type: ignore[return-value]
    return EchoLLM()  # type: ignore[return-value]
