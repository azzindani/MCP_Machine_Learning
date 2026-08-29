# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# mcp-machine-learning — production container, ONE process for all 3 tiers.
#
# unified_server.py mounts basic/medium/advanced as separate MCP endpoints
# (/basic/mcp, /medium/mcp, /advanced/mcp) inside one Starlette app on one
# port, so pandas/numpy/scikit-learn/xgboost load once instead of three
# times — was previously 3 containers (~520 MiB idle combined), now 1
# (~240 MiB idle). Each tier's own /health, /version, /mcp routes (defined
# via @mcp.custom_route in its own server.py) come along for free under its
# mount prefix. Per-tier stdio/individual-HTTP servers (servers/ml_*/server.py)
# are untouched — still usable directly for local LM Studio installs.
#
# Build:  docker build -t mcp-machine-learning:latest .
# Run:    docker run --rm -p 8820:8820 -e ML_TRANSPORT=http mcp-machine-learning:latest
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.14-slim

FROM python:${PYTHON_VERSION} AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock ./
COPY shared ./shared
COPY servers ./servers
RUN uv sync --frozen

FROM python:${PYTHON_VERSION} AS runtime
RUN groupadd -r app && useradd -r -g app app \
    && mkdir -p /home/app && chown app:app /home/app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/shared /app/shared
COPY --from=builder /app/servers /app/servers
COPY pyproject.toml unified_server.py ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    ML_HOST=0.0.0.0 \
    ML_PORT=8820

USER app
EXPOSE 8820

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"ML_PORT\"]}/health', timeout=3)" || exit 1

ENTRYPOINT ["python", "unified_server.py"]
