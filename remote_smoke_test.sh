#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MCP_Machine_Learning — remote smoke test.
#
# NOT part of pytest / CI (see CLAUDE.md §20 "Remote smoke tests"). This
# script is the separate, manual/on-demand check that actually exercises the
# deployed HTTP endpoint: real auth enforcement + a real handwritten-prompt-
# style tool call on a real generated dataset, against the real public domain.
#
# Tools here read datasets by server-side file_path (not upload), so this
# script docker-cp's a small generated CSV into the running container first —
# only works run on the same host as the deployment (self-hosted, by design).
#
# Usage:
#   ./remote_smoke_test.sh                      # reads ML_API_KEY from .env
#   ML_API_KEY=sk-... ./remote_smoke_test.sh     # or pass it directly
#   DOMAIN=http://localhost:8820 ./remote_smoke_test.sh   # test a different target
#   CONTAINER=mcp-ml ./remote_smoke_test.sh      # override container name
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DOMAIN="${DOMAIN:-https://ml.casava.space}"
CONTAINER="${CONTAINER:-mcp-ml}"
if [ -f .env ]; then
  set -a; source .env; set +a
fi
KEY="${ML_API_KEY:?Set ML_API_KEY (env var or .env file) before running}"
DATASET_PATH="/tmp/remote-smoke-test/dataset.csv"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }

echo "Target: $DOMAIN"
echo
echo "== seed a real dataset into the container =="
TMP_CSV=$(mktemp)
python3 -c "
import random
random.seed(42)
print('f1,f2,label')
for _ in range(150):
    a = random.gauss(0, 1); b = random.gauss(0, 1)
    print(f'{a:.4f},{b:.4f},{1 if (a + b) > 0 else 0}')
" > "$TMP_CSV"
docker exec "$CONTAINER" mkdir -p /tmp/remote-smoke-test
docker cp "$TMP_CSV" "$CONTAINER:$DATASET_PATH"
rm -f "$TMP_CSV"
# docker cp preserves the source file's mode/owner (root:root, 600 here since
# mktemp made it) — the container runs as non-root `app`, so it can't read
# the file until we hand it over.
docker exec -u root "$CONTAINER" chown app:app "$DATASET_PATH"
pass "150-row synthetic dataset copied to $CONTAINER:$DATASET_PATH"

echo
echo "== auth enforcement =="

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$DOMAIN/basic/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}')
[ "$code" = "401" ] && pass "no token -> 401" || fail "no token -> expected 401, got $code"

SID=$(curl -s -i -X POST "$DOMAIN/basic/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H "Authorization: Bearer $KEY" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
  | grep -i mcp-session-id | tr -d '\r' | awk '{print $2}')
[ -n "$SID" ] && pass "valid token -> session established" || fail "valid token -> no session id returned"

curl -s -X POST "$DOMAIN/basic/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H "Authorization: Bearer $KEY" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"notifications/initialized"}' > /dev/null

echo
echo '== prompt: "train a random forest classifier on this dataset" -> train_classifier =='
RESULT=$(curl -s -X POST "$DOMAIN/basic/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H "Authorization: Bearer $KEY" -H "mcp-session-id: $SID" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"train_classifier\",\"arguments\":{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\",\"model\":\"rf\"}}}")
echo "$RESULT" | grep -q '"success":true' && pass "train_classifier trained a real model on real data" || fail "unexpected result: $RESULT"

echo
echo "ALL CHECKS PASSED against $DOMAIN"
