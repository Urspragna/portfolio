"""POST /api/chat — RAG chatbot (streaming SSE)."""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db import Conversation, SessionLocal
from app.rag import RAGIndex, build_prompt

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=8, description="Number of chunks to retrieve")
    # TODO: add optional session_id for tracking conversations


async def _try_log_conversation(**fields) -> None:
    """Best-effort logging; never raises."""
    try:
        async with SessionLocal() as s:
            s.add(Conversation(**fields))
            await s.commit()
    except Exception as exc:  # pragma: no cover
        # Silently fail on logging — shouldn't break chat
        print(f"[chat] log skipped: {type(exc).__name__}: {exc}")


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
) -> StreamingResponse:
    """Stream a RAG-grounded answer (SSE)."""
    index: RAGIndex = request.app.state.rag_index
    llm = request.app.state.llm

    hits = index.retrieve(body.message, k=body.k)
    citations = [
        {"title": h.chunk.title, "score": round(h.score, 4)} for h in hits
    ]

    prompt = build_prompt(
        body.message, hits, max_chars=request.app.state.settings.max_context_chars
    )

    async def event_stream() -> AsyncIterator[bytes]:
        t0 = time.perf_counter()
        # Citations first, so UI shows retrieval before LLM starts
        yield _sse("citations", {"citations": citations})

        chunks: list[str] = []
        async for tok in llm.stream(prompt):
            chunks.append(tok)
            yield _sse("token", {"text": tok})

        latency_ms = int((time.perf_counter() - t0) * 1000)
        full_answer = "".join(chunks)

        # Log the turn (anonymously, best-effort)
        # FIXME: should maybe truncate really long answers before storing
        await _try_log_conversation(
            question=body.message,
            answer=full_answer[:8000],
            citations=json.dumps(citations),
            latency_ms=latency_ms,
            provider=llm.name,
        )

        yield _sse("done", {"latency_ms": latency_ms, "provider": llm.name})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> bytes:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


@router.get("/chat/citations")
async def preview_citations(query: str, request: Request, k: int = 4) -> dict:
    """Retrieval only (no LLM)."""
    index: RAGIndex = request.app.state.rag_index
    hits = index.retrieve(query, k=k)
    return {
        "query": query,
        "k": k,
        "hits": [
            {
                "title": h.chunk.title,
                "score": round(h.score, 4),
                "preview": h.chunk.text[:240] + ("…" if len(h.chunk.text) > 240 else ""),
            }
            for h in hits
        ],
    }
