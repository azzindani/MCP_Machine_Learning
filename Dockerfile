# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# mcp-machine-learning — production container for all 3 ML MCP servers
# (ml-basic, ml-medium, ml-advanced). One root `uv sync` covers every
# sub-server (they share one dependency set); each installs a console script
# (ml-basic/ml-medium/ml-advanced) already on PATH after sync.
#
# One image, N containers: select which sub-server a given container runs via
# SERVER_SCRIPT. See docker-compose.yml for the one-service-per-sub-server
# layout (each with its own port).
#
# Build:  docker build -t mcp-machine-learning:latest .
# Run ml_basic:
#   docker run --rm -p 8820:8820 -e SERVER_SCRIPT=ml-basic \
#     -e ML_BASIC_TRANSPORT=http -e ML_BASIC_HOST=0.0.0.0 \
#     mcp-machine-learning:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12-slim

FROM python:${PYTHON_VERSION} AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock ./
COPY shared ./shared
COPY servers ./servers
RUN uv sync --frozen

FROM python:${PYTHON_VERSION} AS runtime
RUN groupadd -r app && useradd -r -g app app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/shared /app/shared
COPY --from=builder /app/servers /app/servers
COPY pyproject.toml ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "\
import os, urllib.request; \
prefix = os.environ.get('SERVER_SCRIPT', 'ml-basic').upper().replace('-', '_'); \
port = os.environ[f'{prefix}_PORT']; \
urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)" || exit 1

ENTRYPOINT ["sh", "-c", "exec \"$SERVER_SCRIPT\""]
