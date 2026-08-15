# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# LLM Hallucination Tester — Production Dockerfile
#
# Build:  docker build -t hallucination-tester .
# Run:    docker run -p 8080:8080 --env-file .env hallucination-tester
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build tools required for some native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first to maximise Docker layer cache hits
COPY requirements.txt .

# Install into a prefix we can copy cleanly into the runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Minimal runtime image ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install runtime-only system deps (curl for health-check probes, ca-certs for HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Security: run as a non-root user ─────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Ensure Python can find our package
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cloud Run requires PORT 8080
EXPOSE 8080

# Switch to non-root user
USER appuser

# ── Health check (Cloud Run also probes /health via HTTP) ────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# ── Production server: gunicorn driving uvicorn workers ─────────────────────
#
# -k uvicorn.workers.UvicornWorker  — async ASGI workers
# --workers 2                        — 2 × async workers per container (tune to CPU)
# --bind 0.0.0.0:8080               — Cloud Run standard port
# --timeout 120                      — LLM calls can take 30-60 s
# --access-logfile -                 — route access logs to stdout (Cloud Logging)
# --error-logfile -                  — route error logs to stdout
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:${PORT:-8080} --timeout 120 --access-logfile - --error-logfile - --log-level info"]
