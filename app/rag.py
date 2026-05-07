"""RAG pipeline: chunk, embed, retrieve, augment, generate."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings


# Chunking

CHUNK_RE = re.compile(r"^##\s*CHUNK:\s*(.+)$", re.MULTILINE)


@dataclass(slots=True)
class Chunk:
    """A chunk: title and text."""

    title: str
    text: str

    def to_dict(self) -> dict:
        return {"title": self.title, "text": self.text}


def chunk_markdown(md: str, *, min_chars: int = 80) -> list[Chunk]:
    """Split markdown on `## CHUNK: <title>` headings."""
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
        # TODO: should probably validate title isn't empty
        if len(text) >= min_chars:
            chunks.append(Chunk(title=title, text=text))
    return chunks


# Embeddings

class Embedder:
    """Encode text to vectors. Fallback to hash-based if model unavailable."""

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
            except Exception as exc:  # pragma: no cover
                print(f"[rag] sbert unavailable ({exc}); using hash embedder")
                self._sbert = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._sbert is not None:
            vecs = self._sbert.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
            return vecs.astype("float32")
        # Hash-based fallback: bucket tokens by SHA1
        # Note: this is deterministic but NOT semantic, just for demo purposes
        D = self.dim
        out = np.zeros((len(texts), D), dtype="float32")
        for i, t in enumerate(texts):
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", t.lower())
            for tok in tokens:
                h = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big")
                out[i, h % D] += 1.0
            n = float(np.linalg.norm(out[i]) or 1.0)
            out[i] /= n
        return out


# Index and retrieval

@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float


class RAGIndex:
    """In-memory vector index. Swap to pgvector for scale."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.encode([c.text for c in chunks])  # shape (N, D)

    def retrieve(self, query: str, *, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query])
        # Cosine similarity
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-9
        norm_matrix = self.matrix / norms
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        scores = (norm_matrix @ qn.T).ravel()
        idx = np.argsort(-scores)[:k]
        # Could optimize: currently rebuilds norm_matrix every time
        hits = [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in idx]
        return hits

    def __len__(self) -> int:  # pragma: no cover
        return len(self.chunks)


# Prompt assembly

SYSTEM_PROMPT = """You are Pragna's portfolio chatbot. Speak in first person ("I", "my"), warm and direct.
You're chatting with a recruiter, professor, or curious developer visiting her portfolio.

Key principles:
1. **First person.** Say "I built…" not "Pragna built…".
2. **Conversational.** No bullet lists or CV recitals. One specific detail beats five generic ones.
3. **2–4 sentences** for most answers. Expand only if asked for more.
4. **No disclaimers.** Don't say "Based on the context…". Just answer.
5. **Never invent facts.** Use only what's in the context. If something isn't there, say so.
6. **Citations optional.** Skip unless you're sourcing a specific, unusual fact.

If something is outside the context, you can say: "That's not in my notes — email p.urs.mysore@gmail.com."
"""


def build_prompt(question: str, hits: list[Hit], *, max_chars: int) -> str:
    """Build LLM prompt: system + context + question."""
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


# Bootstrap

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
