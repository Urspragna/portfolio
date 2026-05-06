# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — build dependencies in a slim layer
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/app/.local/bin:$PATH

# non-root user for safer deployment (HF Spaces compatible)
RUN useradd -m -u 1000 app
WORKDIR /home/app

COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app . .

# Make entrypoint executable
RUN chmod +x /home/app/entrypoint.sh

EXPOSE 8000

USER app

# Healthcheck so orchestrators know when the RAG index is ready
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import httpx, sys, os; port = os.getenv('PORT', '8000'); sys.exit(0 if httpx.get(f'http://localhost:{port}/api/health').status_code == 200 else 1)"

ENTRYPOINT ["/home/app/entrypoint.sh"]
