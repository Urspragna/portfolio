# Pragna's Portfolio

A portfolio website with a working backend. FastAPI + RAG chatbot (grounded on CV) + 4 live ML demos (embeddings, retrieval, sentiment, tokenisation). No frameworks — hand-written frontend, from-scratch RAG.

Portfolio at `/` · AI Lab demos at `/lab`

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| API framework | **FastAPI** + Uvicorn | Async, type-safe, auto OpenAPI docs at `/docs` |
| Validation | **Pydantic v2** + `pydantic-settings` | Fastest validation lib in Python; canonical config pattern |
| ORM | **SQLAlchemy 2.0** (async) | Modern Mapped[] typing, native async sessions |
| Database (dev) | **SQLite** via `aiosqlite` | Zero setup; ships file-based |
| Database (prod) | **PostgreSQL 16 + pgvector** | Production-grade vector DB via Docker |
| LLM | **Anthropic Claude** (default) / **Ollama** (local) / **echo** (no-key) | Bring-your-own-key or run free locally |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` | Open-source 384-dim vectors; hash fallback |
| Sentiment | **HF Transformers** DistilBERT-SST2 | Industry-standard NLP demo; lexicon fallback |
| HTTP client | **httpx** (async) | Streaming SSE support |
| Frontend | Hand-written HTML + CSS + vanilla JS | No Node toolchain needed; everything ships in two HTML files |
| Container | Multi-stage **Dockerfile** + **docker-compose** | One-command production stack |

---

## Project layout

```
Portfolio/
├── index.html                  ← portfolio homepage (with floating chat widget)
├── lab.html                    ← AI Learning Lab — 4 live demos
├── Pragna_Master_CV.pdf        ← résumé linked from the site
│
├── app/                        ← FastAPI backend
│   ├── __init__.py
│   ├── main.py                 ← app entry, lifecycle, static mount
│   ├── config.py               ← Pydantic Settings v2
│   ├── db.py                   ← SQLAlchemy 2.0 async + models
│   ├── rag.py                  ← chunking · embedding · retrieval · prompt
│   ├── llm.py                  ← provider abstraction (Claude / Ollama / echo)
│   └── routers/
│       ├── chat.py             ← POST /api/chat (streaming SSE)
│       └── demos.py            ← /api/demo/{embed,sentiment,tokenise}
│
├── data/
│   └── knowledge_base.md       ← CV chunked into RAG retrieval units
│
├── requirements.txt            ← pip install -r
├── pyproject.toml              ← uv / hatch metadata
├── Dockerfile                  ← multi-stage build
├── docker-compose.yml          ← FastAPI + Postgres+pgvector
├── .env.example                ← copy → .env to override defaults
├── .dockerignore
└── .gitignore
```

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit `http://localhost:8000` (portfolio + chat) or `http://localhost:8000/lab` (demos).

SQLite + RAG index build on startup. Default chatbot is **echo** (no API key needed).

### Use a real LLM

**Claude** (recommended):
```bash
cp .env.example .env
# set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

**Ollama** (local, free):
```bash
ollama pull llama3.2 && ollama serve
# .env: LLM_PROVIDER=ollama
uvicorn app.main:app --reload
```

---

## Production (Docker + Postgres)

```bash
cp .env.example .env && docker compose up --build
```

Brings up FastAPI on `:8000` and Postgres+pgvector on `:5432`. To use pgvector for storage instead of in-memory vectors, edit `app/rag.py` to write/read from a `vector(384)` column.

---

## Deploy

| Platform | Note |
|---|---|
| **Hugging Face Spaces** | New Space → Docker → set `ANTHROPIC_API_KEY` secret |
| **Railway** | `railway init` + PostgreSQL plugin |
| **Fly.io** | `fly launch` + `fly postgres create` |
| **Render** | Web Service + managed Postgres |

---

## How it works

From-scratch RAG pipeline in `app/rag.py` (no LangChain, no LlamaIndex). The flow:

1. Embed query → 384-dim vector (sentence-transformers)
2. Cosine similarity search vs. CV chunks
3. Top-k retrieval (default k=4)
4. Build prompt with context + question
5. Stream response via LLM

The **AI Lab** at `/lab` has 4 interactive demos:
- **RAG** — query, retrieval, LLM answer
- **Embeddings** — vector similarity + preview
- **Sentiment** — DistilBERT classifier (+ lexicon fallback)
- **Tokenisation** — visualize token split

---

## Next steps

1. **Swap to pgvector** — store vectors in Postgres instead of memory
2. **Reranking** — add cross-encoder between retrieval + LLM
3. **Hybrid search** — combine BM25 + dense retrieval
4. **Eval suite** — write tests for retrieval quality
5. **Function calling** — let LLM call `/api/demo/*` endpoints
6. **Fine-tune** — label conversations, train on custom data
7. **Observability** — wire langfuse or helicone for logging

---

## Personalization

Edit `data/knowledge_base.md` (add/remove `## CHUNK:` blocks), restart server. Chunker rebuilds on each boot. For zero-downtime updates, add a `POST /api/admin/reindex` endpoint.

---

## Tests

Add `tests/test_rag.py` with pytest-asyncio. Useful tests:
- `chunk_markdown` splits fixtures correctly
- `RAGIndex.retrieve` finds expected chunks
- `/api/health` returns 200 with chunk count
- `/api/chat/citations` returns results with scores

---

## Design choices

- **No framework (frontend)** — hand-written HTML/CSS/JS, readable in any editor
- **No migrations** — add Alembic when schema grows
- **No auth** — open by design, add rate-limiting before public deploy
- **In-memory vectors** — faster than DB round-trips for ~15 chunks, swap at scale

---

---

Built by **Pragna Urs Mysore Gopal** · Rostock, Germany · [LinkedIn](https://linkedin.com/in/pragna-urs) · [p.urs.mysore@gmail.com](mailto:p.urs.mysore@gmail.com)
