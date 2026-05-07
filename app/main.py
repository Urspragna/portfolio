"""FastAPI app entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.db import init_db
from app.llm import make_llm
from app.rag import Embedder, load_index
from app.routers.chat import router as chat_router
from app.routers.demos import router as demos_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Init DB, load RAG index, set up LLM."""
    print(f"[{settings.app_name}] starting up — env={settings.environment}")

    # DB init is best-effort; if it fails, chat still works.
    db_ok = True
    try:
        await init_db()
    except Exception as exc:  # pragma: no cover
        db_ok = False
        print(
            f"[db] init failed ({type(exc).__name__}: {exc}). Chat works; logging disabled."
        )

    # Build RAG index once at startup
    index = load_index(settings)
    embedder = Embedder(settings)
    llm = make_llm(settings)

    print(
        f"[rag] indexed {len(index)} chunks, embedding_dim={index.embedder.dim}, "
        f"llm={llm.name}, db={'on' if db_ok else 'off'}"
    )

    app.state.rag_index = index
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.settings = settings
    app.state.db_ok = db_ok

    yield

    print(f"[{settings.app_name}] shutting down")


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "AI-powered backend for Pragna's portfolio.\n\n"
        "• `/api/chat` — RAG chatbot grounded on her CV\n"
        "• `/api/demo/embed` — embedding similarity\n"
        "• `/api/demo/sentiment` — sentiment analysis\n"
        "• `/api/demo/tokenise` — tokenisation visualiser\n"
    ),
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/api/health", tags=["meta"])
async def health() -> dict:
    """Health check with status."""
    return {
        "ok": True,
        "version": __version__,
        "llm": app.state.llm.name,
        "embedding_dim": app.state.rag_index.embedder.dim,
        "chunks": len(app.state.rag_index),
        "environment": settings.environment,
        "db": "on" if getattr(app.state, "db_ok", False) else "off",
    }


# Routers
app.include_router(chat_router)
app.include_router(demos_router)


# Static frontend (serves HTML/assets; /api/* routes take precedence)
PUBLIC = Path(settings.public_dir)
INDEX_FILE = PUBLIC / "index.html"
LAB_FILE = PUBLIC / "lab.html"

print(f"[startup] PUBLIC={PUBLIC}, exists={PUBLIC.exists()}")
print(f"[startup] INDEX_FILE={INDEX_FILE}, exists={INDEX_FILE.exists()}")

if INDEX_FILE.exists():
    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(INDEX_FILE)

    @app.get("/lab", include_in_schema=False)
    async def lab() -> FileResponse:
        if not LAB_FILE.exists():
            return JSONResponse({"detail": "lab.html not found"}, status_code=404)
        return FileResponse(LAB_FILE)

    # Serve other static files
    app.mount("/", StaticFiles(directory=str(PUBLIC), html=True), name="static")
