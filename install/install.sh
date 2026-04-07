#!/usr/bin/env sh
# install.sh — MCP Machine Learning server installer for Linux / macOS
# Requires: Python 3.12+, uv, git
set -e

REPO_URL="https://github.com/azzindani/MCP_Machine_Learning.git"
INSTALL_DIR="${HOME}/.mcp_servers/MCP_Machine_Learning"
VRAM_GB=0

echo "=== MCP Machine Learning Installer ==="
echo ""

# --- Python version check ---
PY=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PY_MAJOR=$(echo "$PY" | cut -d. -f1)
PY_MINOR=$(echo "$PY" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
    echo "ERROR: Python 3.12+ required. Found Python ${PY}."
    echo "Install from https://www.python.org/downloads/ or use pyenv."
    exit 1
fi
echo "Python ${PY} OK"

# --- uv check ---
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi
echo "uv $(uv --version) OK"

# --- VRAM detection (nvidia-smi) ---
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -gt 0 ] 2>/dev/null; then
        VRAM_GB=$((VRAM_MB / 1024))
        echo "GPU VRAM: ${VRAM_GB} GB"
    fi
fi

# --- Constrained mode ---
CONSTRAINED_MODE=0
if [ "$VRAM_GB" -gt 0 ] && [ "$VRAM_GB" -le 8 ]; then
    CONSTRAINED_MODE=1
    echo "Constrained mode enabled (VRAM <= 8 GB)"
fi

# --- Clone or update ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating existing installation at ${INSTALL_DIR}..."
    cd "$INSTALL_DIR"
    git fetch --quiet origin
    git reset --hard origin/main --quiet
else
    echo "Cloning to ${INSTALL_DIR}..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# --- Sync dependencies ---
echo "Installing dependencies (uv sync)..."
uv sync --quiet
echo "Dependencies installed."

# --- Write MCP config ---
echo ""
echo "=== MCP Client Configuration ==="
echo ""
python3 install/mcp_config_writer.py --constrained "$CONSTRAINED_MODE"

echo ""
echo "Installation complete!"
echo ""
echo "Set MCP_CONSTRAINED_MODE=1 in your MCP client env block if you have <= 8 GB VRAM."
