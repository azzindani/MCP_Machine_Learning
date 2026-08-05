#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MCP_Machine_Learning — remote run + tunnel (Google Colab / any fresh Linux
# VM, no Docker needed). Starts each tier directly (uv run <console-script>)
# and opens one Cloudflare Quick Tunnel per tier — same idea as
# azzindani/Folio's launch.sh, and this repo's own launch_tunnel.sh (which
# does the same thing via Docker). Use this one when Docker isn't available.
#
# Usage:
#   REPO_DIR=/content/MCP_Machine_Learning ./remote_launch.sh
#   ./remote_launch.sh stop
#
# NOT for production. Quick Tunnels are unauthenticated at the transport
# level — set ML_API_KEY or ML_TOKENS_FILE before launching so /mcp still
# requires a bearer token even while it's publicly reachable.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MCP_Machine_Learning}"
LOG_DIR="/tmp/ml-remote"
mkdir -p "$LOG_DIR"

# name:port:console-script triples — one per tier.
SERVERS=(
  "basic:8820:ml-basic"
  "medium:8821:ml-medium"
  "advanced:8822:ml-advanced"
)

if [ "${1:-}" = "stop" ]; then
  pkill -f "cloudflared tunnel --url http://localhost" 2>/dev/null || true
  for entry in "${SERVERS[@]}"; do
    script="${entry##*:}"
    pkill -f "$script --transport http" 2>/dev/null || true
  done
  echo "stopped"
  exit 0
fi

if ! command -v cloudflared &>/dev/null; then
  echo "[remote_launch] installing cloudflared..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi
export PATH="${HOME}/.local/bin:${PATH}"

pkill -f "cloudflared tunnel --url http://localhost" 2>/dev/null || true
for entry in "${SERVERS[@]}"; do
  script="${entry##*:}"
  pkill -f "$script --transport http" 2>/dev/null || true
done
sleep 1

cd "$REPO_DIR"
echo "[remote_launch] starting tiers..."
for entry in "${SERVERS[@]}"; do
  name="${entry%%:*}"
  rest="${entry#*:}"
  port="${rest%%:*}"
  script="${rest#*:}"
  nohup uv run "$script" --transport http --host 0.0.0.0 --port "$port" > "$LOG_DIR/${name}.server.log" 2>&1 &
done

echo "[remote_launch] waiting for tiers to come up..."
for entry in "${SERVERS[@]}"; do
  rest="${entry#*:}"
  port="${rest%%:*}"
  for i in $(seq 1 30); do
    curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1 && break
    sleep 1
  done
done

echo "[remote_launch] starting cloudflared quick tunnels..."
for entry in "${SERVERS[@]}"; do
  name="${entry%%:*}"
  rest="${entry#*:}"
  port="${rest%%:*}"
  log="$LOG_DIR/${name}.tunnel.log"
  : > "$log"
  nohup cloudflared tunnel --url "http://localhost:${port}" > "$log" 2>&1 &
done

echo "[remote_launch] waiting up to 30s per tunnel for a public URL..."
declare -A URLS
for entry in "${SERVERS[@]}"; do
  name="${entry%%:*}"
  log="$LOG_DIR/${name}.tunnel.log"
  url=""
  for i in $(seq 1 30); do
    url=$(grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1 || true)
    [ -n "$url" ] && break
    sleep 1
  done
  URLS[$name]="${url:-<not found, check $log>}"
done

echo ""
echo "  remote endpoints:"
for entry in "${SERVERS[@]}"; do
  name="${entry%%:*}"
  echo "    ${name}  ->  ${URLS[$name]}/mcp"
done
echo ""
echo "  stop:  ./remote_launch.sh stop"
