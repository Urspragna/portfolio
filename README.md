# Pragna's AI Portfolio

A portfolio website that's also a working AI engineering project. The frontend is a dark, terminal-inspired single-page site; the backend is a FastAPI app that runs **RAG over Pragna's CV** and ships **four live ML demos** — embeddings, RAG retrieval, sentiment, and tokenisation. Built with the canonical Python AI stack of 2026.

> Two sites in one: the **portfolio** at `/` showcases who Pragna is, while the **AI Lab** at `/lab` shows what she can *build*. The whole project doubles as a teaching repo — every step in the RAG pipeline is implemented from scratch in clear, readable Python.

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

## Quickstart (local, 60 seconds)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
uvicorn app.main:app --reload

# 3. Open
#    http://localhost:8000        ← portfolio (with chat widget)
#    http://localhost:8000/lab    ← AI Learning Lab
#    http://localhost:8000/docs   ← interactive API docs (Swagger UI)
```

That's it. SQLite is created in the project root; the RAG index is built on startup; the chatbot uses the **echo** provider (no LLM key needed) and answers from retrieved context only.

### Use a real LLM

**Option A — Anthropic Claude** (recommended quality):

```bash
cp .env.example .env
# edit .env:
#   LLM_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

**Option B — Ollama** (local, free, private):

```bash
# https://ollama.com — install once, then:
ollama pull llama3.2
ollama serve

# .env:
#   LLM_PROVIDER=ollama
uvicorn app.main:app --reload
```

---

## Production stack (Docker + Postgres + pgvector)

```bash
cp .env.example .env  # set ANTHROPIC_API_KEY here
docker compose up --build
```

This brings up:
- **api** — the FastAPI app on `:8000`
- **db** — Postgres 16 with the pgvector extension on `:5432`

Switch the embedder to use pgvector by editing `app/rag.py` to upsert chunk vectors into a `chunks` table with a `vector(384)` column — that swap is the only code change needed to scale.

---

## Deploy options

| Platform | Steps | Best for |
|---|---|---|
| **Hugging Face Spaces** (free) | New Space → Docker → push this repo → set `ANTHROPIC_API_KEY` secret | AI portfolios — the audience already lives on HF |
| **Railway** ($5/mo) | `railway init` → connect repo → add PostgreSQL plugin → deploy | One-click prod database |
| **Fly.io** (free tier) | `fly launch` → `fly postgres create` → `fly deploy` | Edge regions, generous free tier |
| **Render** (free tier) | New Web Service → Docker → connect repo → add managed Postgres | Easy GitHub-driven deploys |
| **Local** | `uvicorn app.main:app` | Demos, dev, no internet required |

For Hugging Face Spaces, the included `Dockerfile` runs as a non-root user on port 7860 if you set `EXPOSE 7860` and pass `--port 7860` — adjust the `CMD` line if needed.

---

## How the AI works (the part Pragna built to learn)

The `/api/chat` endpoint is **a from-scratch RAG pipeline** — no LangChain, no LlamaIndex. Reading `app/rag.py` end-to-end takes about 5 minutes and walks you through the entire concept.

```
Question
   │
   ▼
[1] Embed query              ── sentence-transformers → 384-dim vector
   │
   ▼
[2] Cosine similarity         ── vs. all CV chunk vectors
   │
   ▼
[3] Top-k retrieval           ── default k=4 → most relevant CV passages
   │
   ▼
[4] Prompt assembly           ── system prompt + retrieved context + question
   │
   ▼
[5] LLM stream                ── Claude / Ollama / echo emits tokens
   │
   ▼
SSE stream → frontend → typewriter UI
```

Every step is observable from the **AI Lab** at `/lab`:

1. **RAG demo** — type a query, watch retrieval happen, then see the LLM stream a grounded answer.
2. **Embeddings demo** — paste two strings, see their cosine similarity and the first 8 components of each vector.
3. **Sentiment demo** — DistilBERT classifier with a lexicon fallback so it works even on machines that can't download HF models.
4. **Tokenisation demo** — visualise word vs. byte-pair-encoding-style tokens.

---

## What to learn next (suggested AI study path)

The repo is intentionally a launchpad. Once it's running, the natural next experiments are:

1. **Swap to pgvector**: keep the chunk vectors in Postgres instead of memory. Edit `app/rag.py` to write vectors into a `vector(384)` column at startup and replace the `np.matmul` retrieval with `ORDER BY embedding <-> query_vec LIMIT k`. *Production-shaped RAG.*
2. **Reranking**: add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and LLM. Compare answer faithfulness with/without.
3. **Hybrid search**: layer BM25 (`rank_bm25` library) on top of dense retrieval and combine scores. Most production RAG is hybrid.
4. **Eval harness**: write a small `pytest` that asks 20 known questions and checks the cited chunks are correct. *This is the most valuable AI engineering skill there is.*
5. **Function calling**: extend the chatbot to call `/api/demo/embed` or `/api/demo/sentiment` mid-conversation. Teach the LLM to *use tools*.
6. **Fine-tuning**: take the conversation logs in the `conversations` table, label them, and fine-tune a small open model. *Real MLOps.*
7. **Observability**: wire `langfuse` or `helicone` to log every RAG turn — input, retrieved chunks, prompt, output, latency. *Necessary for any production AI app.*

---

## Featured personalisation

Edits ripple instantly. To update what the chatbot knows:

```bash
# 1. Edit data/knowledge_base.md — add/remove/edit any `## CHUNK: <title>` block
# 2. Restart the server
uvicorn app.main:app --reload
```

The chunker re-runs on every boot. For zero-downtime updates, add a `POST /api/admin/reindex` endpoint that calls `load_index()` and swaps `app.state.rag_index`.

---

## Tests (next addition)

A starter test suite lives in `pyproject.toml` config — add `tests/test_rag.py` with `pytest-asyncio`. Useful first tests:
- `chunk_markdown` correctly splits a fixture file
- `RAGIndex.retrieve` returns the obvious chunk for an obvious question
- `/api/health` returns 200 and reports >0 chunks
- `/api/chat/citations?query=...` returns at least one hit with score > 0

---

## What's intentionally minimal

- **No frontend framework.** Everything is hand-written HTML/CSS/JS so the source is readable in any text editor.
- **No CMS / no database migrations yet.** Start simple. Add Alembic when the schema grows.
- **No auth on the chatbot.** Open by design — visitors should be able to ask questions instantly. Add rate-limiting (`slowapi`) before public deploy.
- **No vector DB at first.** `numpy.matmul` against ~15 chunks is faster than any Postgres round-trip. Swap to pgvector when chunk count > a few hundred.

---

## How this was built

This project was developed with **Claude as a pair programmer** — used the way a senior engineer uses a fast, well-read junior: directing the work, reviewing every output, debugging when it got things wrong (Python 3.14 + greenlet was a fun one), and reading the resulting code until I understood every line.

Architecture decisions, content, voice, debugging, and the knowledge base are mine. Code scaffolding and library suggestions were accelerated by Claude. I can walk through any file in this repo and explain why it's there.

Using AI productively as an engineer is itself a 2026 skill — shipping this is also how I learned to do that well.

---

Built with ♥ by **Pragna Urs Mysore Gopal** · Rostock, Germany · [LinkedIn](https://linkedin.com/in/pragna-urs) · [p.urs.mysore@gmail.com](mailto:p.urs.mysore@gmail.com)
