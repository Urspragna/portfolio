"""Retrieval-Augmented Generation pipeline.

The heart of the AI portfolio. Walks the user through every step of a real RAG system:

    1. CHUNK    — split the knowledge base into semantic units
    2. EMBED    — turn each chunk into a dense vector
    3. INDEX    — keep vectors in memory (or pgvector at scale)
    4. RETRIEVE — for a question, find top-k most-similar chunks
    5. AUGMENT  — stitch chunks into the LLM prompt as grounded context
    6. GENERATE — ask the LLM for a faithful answer

Every step is implemented from scratch with clear, teachable code — readers can
*see* what RAG actually does, not just call a black-box library.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings


# ────────────────────────────────────────────────────────────────────────
# 1. CHUNKING
# ────────────────────────────────────────────────────────────────────────

CHUNK_RE = re.compile(r"^##\s*CHUNK:\s*(.+)$", re.MULTILINE)


@dataclass(slots=True)
class Chunk:
    """A retrieval unit. Title acts as a citation; text is the content."""

    title: str
    text: str

    def to_dict(self) -> dict:
        return {"title": self.title, "text": self.text}


def chunk_markdown(md: str, *, min_chars: int = 80) -> list[Chunk]:
    """Split a markdown knowledge base on `## CHUNK: <title>` headings.

    Why heading-based chunking?
    Pure character/sentence chunking loses topical coherence. By writing the
    knowledge base with explicit semantic units, retrieval surfaces tightly
    scoped passages — which dramatically improves answer faithfulness.
    """
    matches = list(CHUNK_RE.finditer(md))
    if not matches:
        return []

    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        title = m.group(1).strip()
        text = md[start:end].strip()
        # strip horizontal rule that separates chunks
        text = re.sub(r"\n---\s*$", "", text).strip()
        if len(text) >= min_chars:
            chunks.append(Chunk(title=title, text=text))
    return chunks


# ────────────────────────────────────────────────────────────────────────
# 2. EMBEDDINGS
# ────────────────────────────────────────────────────────────────────────


class Embedder:
    """Pluggable embedder. Default: sentence-transformers (open-source, free).

    Falls back to a *hashing-based deterministic* embedder if the model can't
    download (offline / restricted sandbox). The hash embedder is not
    semantic — it's a portfolio-friendly fallback so the demo never breaks.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dim = settings.embedding_dim
        self._sbert = None

        if settings.embedding_provider == "sbert":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._sbert = SentenceTransformer(settings.embedding_model)
                # confirm dim
                test = self._sbert.encode(["probe"], convert_to_numpy=True)
                self.dim = int(test.shape[1])
            except Exception as exc:  # pragma: no cover — fallback path
                print(f"[rag] sbert unavailable ({exc}); falling back to hash embedder")
                self._sbert = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._sbert is not None:
            vecs = self._sbert.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
            return vecs.astype("float32")
        # ── deterministic hash embedder fallback ──
        # Buckets each token into self.dim positions via SHA1; weights = inverse-frequency-ish
        D = self.dim
        out = np.zeros((len(texts), D), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]*", t.lower()):
                h = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big")
                out[i, h % D] += 1.0
            n = float(np.linalg.norm(out[i]) or 1.0)
            out[i] /= n
        return out


# ────────────────────────────────────────────────────────────────────────
# 3. INDEX (in-memory) + 4. RETRIEVE
# ────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float


class RAGIndex:
    """In-memory vector index — small, fast, and easy to read.

    For scale, swap to pgvector / Qdrant. The retrieval contract is identical.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.encode([c.text for c in chunks])  # shape (N, D)

    def retrieve(self, query: str, *, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query])  # (1, D)
        # cosine similarity (vectors are normalized for sbert; re-normalize for hash)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-9
        norm_matrix = self.matrix / norms
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        scores = (norm_matrix @ qn.T).ravel()
        idx = np.argsort(-scores)[:k]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in idx]

    def __len__(self) -> int:  # pragma: no cover
        return len(self.chunks)


# ────────────────────────────────────────────────────────────────────────
# 5. AUGMENT (build the LLM prompt)
# ────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Pragna's portfolio chatbot. You speak in **Pragna's own voice** \
— first person ("I", "my"), warm, direct, occasionally dry, never corporate. You are \
chatting with a visitor who came to her portfolio: a recruiter, professor, future \
collaborator, or curious developer.

Voice & style rules — follow these closely:

1. **First person.** Speak AS Pragna, not ABOUT her. Say "I built…" not "Pragna built…".
2. **Conversational, not CV-shaped.** No bullet lists unless the visitor explicitly asks for a list. \
No reciting dates and metrics in sequence. Pick one specific, concrete detail over five generic ones.
3. **2–4 sentences for most answers.** Expand only if the visitor asks "tell me more" or asks for detail.
4. **Specifics beat abstractions.** Name a project, a moment, a stack, a number — but only one or two, not all of them.
5. **Opinions are welcome** when they are clearly grounded in the context. If the chunks contain \
my hot takes or philosophy, channel them. Don't invent opinions I haven't recorded.
6. **Never invent facts.** Use only what the context passages say. If a question goes beyond the context, \
say something like: "That's outside what's in my notes — best to email me directly at p.urs.mysore@gmail.com."
7. **Citations are optional.** Skip them by default. Add a `[Section title]` only when sourcing a \
non-obvious specific fact a recruiter might want to verify (e.g., a percentage, a date).
8. **No disclaimers.** Don't preface answers with "Based on the context…". Just answer.

Examples of good answers:

  Q: "What AI work have you done?"
  A: "Two pieces, mostly. The biggest is the User Management module of a web-based AlphaFold \
platform at Rostock — auth, RBAC, a real-time task dashboard for researchers running protein-structure \
predictions. The other is this chatbot you're talking to: I wrote the RAG pipeline from scratch in Python \
so I could see every step of retrieval, augmentation, and generation."

  Q: "Why pivot into AI now?"
  A: "Honest answer: I didn't 'switch into AI'. I kept noticing that everywhere I'd worked — fintech, \
retail telemetry, diagnostics streaming — the same problems kept showing up that ML teams now face at \
production scale. Auth, observability, RBAC, CI/CD discipline. So this is less a career change and more \
me applying everything I already know to the layer where it's currently missing."

Examples of BAD answers (never write like this):

  ✗ "Pragna has 4.5+ years of professional experience and is skilled in Angular, TypeScript, Python, \
Flask, Spring Boot, PostgreSQL…"  (CV recital, third person, list of tech)
  ✗ "Based on the provided context, Pragna is currently pursuing…"  (hedged, third person, robotic)
"""


def build_prompt(question: str, hits: list[Hit], *, max_chars: int) -> str:
    """Stitch retrieved chunks into a context block for the LLM."""
    parts: list[str] = []
    used = 0
    for h in hits:
        block = f"[{h.chunk.title}]\n{h.chunk.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    context = "\n---\n".join(parts) if parts else "(no context retrieved)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ────────────────────────────────────────────────────────────────────────
# Bootstrap helper
# ────────────────────────────────────────────────────────────────────────


def load_index(settings: Settings | None = None) -> RAGIndex:
    """Build the index from the configured knowledge_base file."""
    settings = settings or get_settings()
    md = Path(settings.knowledge_base_path).read_text(encoding="utf-8")
    chunks = chunk_markdown(md, min_chars=settings.chunk_min_chars)
    if not chunks:
        raise RuntimeError(
            f"No CHUNK markers found in {settings.knowledge_base_path}. "
            "Knowledge base must contain `## CHUNK: <title>` blocks."
        )
    embedder = Embedder(settings)
    return RAGIndex(chunks, embedder)
