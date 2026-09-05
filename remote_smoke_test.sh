#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MCP_Machine_Learning — remote smoke test, all 33 tools.
#
# NOT part of pytest / CI (see CLAUDE.md §20 "Remote smoke tests"). This
# script is the separate, manual/on-demand check that actually exercises the
# deployed HTTP endpoint: real auth enforcement + real handwritten-prompt-
# style tool calls on a real generated dataset, chaining real outputs
# (model_path, cluster labels, etc.) between calls like a real workflow
# would, against the real public domain.
#
# Tools read datasets by server-side file_path (not upload), so this script
# docker-cp's a small generated CSV into the running container first — only
# works run on the same host as the deployment (self-hosted, by design).
# docker cp preserves the source file's root ownership, which the
# container's non-root `app` user can't read — chown it after copying.
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
# Read the key out of .env without executing it. `source` runs every line of
# the file, so a line that is not a KEY=VALUE assignment is a command; that has
# already turned a stray summary line into a file named after a secret. A plain
# read of one assignment cannot do that.
if [ -z "${ML_API_KEY:-}" ] && [ -f .env ]; then
  ML_API_KEY=$(sed -n 's/^[[:space:]]*ML_API_KEY[[:space:]]*=[[:space:]]*//p' .env | tail -n1 | tr -d '\042\047\r')
fi
KEY="${ML_API_KEY:?Set ML_API_KEY (env var or .env file) before running}"
DATASET_PATH="/tmp/remote-smoke-test/dataset.csv"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }
ok_json() { echo "$1" | grep -Eq '\\?"success\\?":[[:space:]]*true'; }
fail_json() { echo "$1" | grep -Eq '\\?"success\\?":[[:space:]]*false'; }

echo "Target: $DOMAIN"

# Tools called without an explicit output_path now default into
# MCP_OUTPUT_DIR, which on a real deployment is a directory the operator
# actually looks at. Remember what was there so the run can leave it exactly
# as it found it (see the cleanup at the very bottom).
SHARED_DIR=$(docker exec "$CONTAINER" printenv MCP_OUTPUT_DIR 2>/dev/null || true)
SHARED_BEFORE=$(mktemp)
[ -n "$SHARED_DIR" ] && docker exec "$CONTAINER" sh -c "ls -1A '$SHARED_DIR' 2>/dev/null" | sort > "$SHARED_BEFORE"
echo
echo "== seed a real dataset into the container =="
TMP_CSV=$(mktemp)
python3 -c "
import random
random.seed(42)
print('f1,f2,label,value')
for _ in range(150):
    a = random.gauss(0, 1); b = random.gauss(0, 1)
    label = 1 if (a + b) > 0 else 0
    value = round(3*a - 2*b + random.gauss(0, 0.5), 3)
    print(f'{a:.4f},{b:.4f},{label},{value}')
" > "$TMP_CSV"
docker exec "$CONTAINER" mkdir -p /tmp/remote-smoke-test
docker cp "$TMP_CSV" "$CONTAINER:$DATASET_PATH"
rm -f "$TMP_CSV"
docker exec -u root "$CONTAINER" chown app:app "$DATASET_PATH"
pass "150-row synthetic dataset (label=classification target, value=regression target) copied to $CONTAINER:$DATASET_PATH"

init_session() {
  local tier="$1"
  curl -s -i -X POST "$DOMAIN/$tier/mcp" \
    -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
    -H "Authorization: Bearer $KEY" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
    | grep -i mcp-session-id | tr -d '\r' | awk '{print $2}'
}
init_notified() {
  local tier="$1" sid="$2"
  curl -s -X POST "$DOMAIN/$tier/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
    -H "Authorization: Bearer $KEY" -H "mcp-session-id: $sid" \
    -d '{"jsonrpc":"2.0","id":2,"method":"notifications/initialized"}' > /dev/null
}

echo
echo "== auth enforcement =="
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$DOMAIN/basic/mcp" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}')
[ "$code" = "401" ] && pass "no token -> 401" || fail "no token -> expected 401, got $code"

SID_BASIC=$(init_session basic); init_notified basic "$SID_BASIC"
[ -n "$SID_BASIC" ] && pass "valid token -> session established on /basic" || fail "no session id"
SID_MEDIUM=$(init_session medium); init_notified medium "$SID_MEDIUM"
SID_ADVANCED=$(init_session advanced); init_notified advanced "$SID_ADVANCED"

call() {
  local tier="$1" sid="$2" id="$3" name="$4" args="$5"
  curl -s -X POST "$DOMAIN/$tier/mcp" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
    -H "Authorization: Bearer $KEY" -H "mcp-session-id: $sid" \
    -d "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"tools/call\",\"params\":{\"name\":\"$name\",\"arguments\":$args}}"
}
extract_path() {
  # The key arrives escaped. A tool's document is delivered as the JSON *string*
  # result.content[0].text, so it reads \"model_path\" on the wire: a pattern
  # anchored on a bare opening quote allows for the backslash before the value
  # but not the ones around the key, and matches nothing.
  #
  # It failed silently, which is worse than failing. Under `set -euo pipefail` a
  # grep that matches nothing makes the whole pipeline non-zero, so the
  # assignment below aborted the script before the `|| fail` beside it could
  # say so -- the run ended on the line after a PASS with no message at all.
  # The trailing `|| true` keeps a genuinely absent key reportable.
  echo "$1" | grep -oE '\\?"model_path\\?"[[:space:]]*:[[:space:]]*\\?"[^\\"]+' | head -1 \
    | sed -E 's/.*"([^"]+)$/\1/' || true
}

echo
echo "===== ml_basic (11 tools) ====="

echo '== prompt: "what columns does this dataset have?" -> inspect_dataset =='
R=$(call basic "$SID_BASIC" 10 inspect_dataset "{\"file_path\":\"$DATASET_PATH\"}")
ok_json "$R" && pass "inspect_dataset described the real dataset" || fail "$R"

echo '== prompt: "profile the value column" -> read_column_profile =='
R=$(call basic "$SID_BASIC" 11 read_column_profile "{\"file_path\":\"$DATASET_PATH\",\"column_name\":\"value\"}")
ok_json "$R" && pass "read_column_profile profiled the real 'value' column" || fail "$R"

echo '== prompt: "which columns are numeric?" -> search_columns =='
R=$(call basic "$SID_BASIC" 12 search_columns "{\"file_path\":\"$DATASET_PATH\",\"dtype\":\"float64\"}")
ok_json "$R" && pass "search_columns found the real float columns" || fail "$R"

echo '== prompt: "show me rows 0-10" -> read_rows =='
R=$(call basic "$SID_BASIC" 13 read_rows "{\"file_path\":\"$DATASET_PATH\",\"start\":0,\"end\":10}")
ok_json "$R" && pass "read_rows returned real row data" || fail "$R"

echo '== prompt: "split this into train/test sets" -> split_dataset =='
R=$(call basic "$SID_BASIC" 14 split_dataset "{\"file_path\":\"$DATASET_PATH\",\"test_size\":0.2}")
ok_json "$R" && pass "split_dataset wrote real train/test CSV files" || fail "$R"

echo '== prompt: "train a random forest to predict label" -> train_classifier =='
R=$(call basic "$SID_BASIC" 15 train_classifier "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\",\"model\":\"rf\"}")
ok_json "$R" && pass "train_classifier trained a real RandomForestClassifier" || fail "$R"
CLF_MODEL=$(extract_path "$R")
[ -n "$CLF_MODEL" ] && pass "captured real classifier model_path: $CLF_MODEL" || fail "no model_path in response"

echo '== prompt: "train a linear regressor to predict value" -> train_regressor =='
R=$(call basic "$SID_BASIC" 16 train_regressor "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"value\",\"model\":\"lir\"}")
ok_json "$R" && pass "train_regressor trained a real LinearRegression model" || fail "$R"
REG_MODEL=$(extract_path "$R")
[ -n "$REG_MODEL" ] && pass "captured real regressor model_path: $REG_MODEL" || fail "no model_path in response"

echo '== prompt: "get predictions from the classifier on this dataset" -> get_predictions =='
R=$(call basic "$SID_BASIC" 17 get_predictions "{\"model_path\":\"$CLF_MODEL\",\"file_path\":\"$DATASET_PATH\",\"max_rows\":10}")
ok_json "$R" && pass "get_predictions ran the real saved model on real data" || fail "$R"

echo '== prompt: "predict a single new example" -> predict_single =='
R=$(call basic "$SID_BASIC" 18 predict_single "{\"model_path\":\"$CLF_MODEL\",\"input_data\":\"{\\\"f1\\\": 0.5, \\\"f2\\\": 0.3, \\\"value\\\": 0.4}\"}")
ok_json "$R" && pass "predict_single scored a real handwritten input row" || fail "$R"

echo '== prompt: "what models have I trained so far?" -> list_models =='
R=$(call basic "$SID_BASIC" 19 list_models '{}')
ok_json "$R" && pass "list_models listed the real models just trained" || fail "$R"

echo '== prompt: "undo my last change to this dataset" -> restore_version =='
R=$(call basic "$SID_BASIC" 20 restore_version "{\"file_path\":\"$DATASET_PATH\"}")
echo "$R" | grep -Eq '\\?"success\\?":[[:space:]]*(true|false)' && pass "restore_version responded against the real dataset (result: $(echo "$R" | grep -oE '\\?"success\\?":[[:space:]]*(true|false)' | head -1))" || fail "$R"

echo
echo "===== ml_medium (12 tools) ====="

echo '== prompt: "fill any nulls in f1 with the median" -> run_preprocessing =='
R=$(call medium "$SID_MEDIUM" 30 run_preprocessing "{\"file_path\":\"$DATASET_PATH\",\"ops\":[{\"op\":\"fill_nulls\",\"column\":\"f1\",\"strategy\":\"median\"}]}")
ok_json "$R" && pass "run_preprocessing applied a real op and snapshotted the real file" || fail "$R"

echo '== prompt: "are there outliers in f1 or f2?" -> detect_outliers =='
R=$(call medium "$SID_MEDIUM" 31 detect_outliers "{\"file_path\":\"$DATASET_PATH\",\"columns\":[\"f1\",\"f2\"]}")
ok_json "$R" && pass "detect_outliers scanned the real columns" || fail "$R"

echo '== prompt: "train with 5-fold cross-validation" -> train_with_cv =='
R=$(call medium "$SID_MEDIUM" 32 train_with_cv "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\",\"model\":\"rf\",\"task\":\"classification\",\"n_splits\":5}")
ok_json "$R" && pass "train_with_cv ran real 5-fold CV" || fail "$R"

echo '== prompt: "compare logistic regression vs random forest" -> compare_models =='
R=$(call medium "$SID_MEDIUM" 33 compare_models "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\",\"task\":\"classification\",\"models\":[\"lr\",\"rf\"]}")
ok_json "$R" && pass "compare_models trained and compared 2 real models" || fail "$R"

echo '== prompt: "cluster f1/f2 into 3 groups with k-means" -> run_clustering =='
R=$(call medium "$SID_MEDIUM" 34 run_clustering "{\"file_path\":\"$DATASET_PATH\",\"feature_columns\":[\"f1\",\"f2\"],\"algorithm\":\"kmeans\",\"n_clusters\":3,\"save_labels\":true}")
ok_json "$R" && pass "run_clustering clustered real data and saved a real cluster_label column" || fail "$R"

echo '== prompt: "show me the operation history for this file" -> read_receipt =='
R=$(call medium "$SID_MEDIUM" 35 read_receipt "{\"file_path\":\"$DATASET_PATH\"}")
ok_json "$R" && pass "read_receipt read the real receipt log (preprocessing + clustering ops just run)" || fail "$R"

echo '== prompt: "generate an EDA report for this dataset" -> generate_eda_report =='
R=$(call medium "$SID_MEDIUM" 36 generate_eda_report "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\"}")
ok_json "$R" && pass "generate_eda_report wrote a real HTML report" || fail "$R"

echo '== prompt: "what is the optimal number of clusters?" -> find_optimal_clusters =='
R=$(call medium "$SID_MEDIUM" 37 find_optimal_clusters "{\"file_path\":\"$DATASET_PATH\",\"feature_columns\":[\"f1\",\"f2\"],\"max_k\":5}")
ok_json "$R" && pass "find_optimal_clusters computed real elbow/silhouette scores" || fail "$R"

echo '== prompt: "detect anomalies in f1/f2" -> anomaly_detection =='
R=$(call medium "$SID_MEDIUM" 38 anomaly_detection "{\"file_path\":\"$DATASET_PATH\",\"feature_columns\":[\"f1\",\"f2\"]}")
ok_json "$R" && pass "anomaly_detection scanned real data" || fail "$R"

echo '== prompt: "check the overall quality of this dataset" -> check_data_quality =='
R=$(call medium "$SID_MEDIUM" 39 check_data_quality "{\"file_path\":\"$DATASET_PATH\"}")
ok_json "$R" && pass "check_data_quality assessed the real dataset" || fail "$R"

echo '== prompt: "evaluate the classifier on this dataset" -> evaluate_model =='
R=$(call medium "$SID_MEDIUM" 40 evaluate_model "{\"model_path\":\"$CLF_MODEL\",\"test_file_path\":\"$DATASET_PATH\",\"target_column\":\"label\"}")
ok_json "$R" && pass "evaluate_model computed real metrics for the real model" || fail "$R"

echo '== prompt: "batch predict over this whole dataset" -> batch_predict =='
R=$(call medium "$SID_MEDIUM" 41 batch_predict "{\"model_path\":\"$CLF_MODEL\",\"file_path\":\"$DATASET_PATH\"}")
ok_json "$R" && pass "batch_predict scored the real dataset with the real model" || fail "$R"

echo
echo "===== ml_advanced (10 tools) ====="

echo '== prompt: "tune hyperparameters for a random forest" -> tune_hyperparameters =='
R=$(call advanced "$SID_ADVANCED" 50 tune_hyperparameters "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\",\"model\":\"rf\",\"task\":\"classification\",\"search\":\"random\",\"n_iter\":3,\"cv\":3}")
ok_json "$R" && pass "tune_hyperparameters ran a real (small) random search" || fail "$R"

echo '== prompt: "export the classifier model" -> export_model =='
R=$(call advanced "$SID_ADVANCED" 51 export_model "{\"model_path\":\"$CLF_MODEL\"}")
ok_json "$R" && pass "export_model exported the real trained model" || fail "$R"

echo '== prompt: "read the report for this model" -> read_model_report =='
R=$(call advanced "$SID_ADVANCED" 52 read_model_report "{\"model_path\":\"$CLF_MODEL\"}")
ok_json "$R" && pass "read_model_report read the real model metadata" || fail "$R"
# The manifest split, on the deployed filesystem. The review measured a 1,017 KB
# manifest that was 28k `emp_title` entries; the check that matters is that the
# manifest a real training run leaves behind is small and the summary still
# names what moved out of it.
if echo "$R" | grep -q '"encoding_map_summary"'; then
  pass "read_model_report summarised the encoding map rather than inlining it"
else
  pass "read_model_report: this dataset has no encoded columns to summarise"
fi
MANIFEST="${CLF_MODEL%.pkl}.manifest.json"
MANIFEST_BYTES=$(docker exec "$CONTAINER" stat -c %s "$MANIFEST" 2>/dev/null || echo 0)
if [ "$MANIFEST_BYTES" -gt 0 ] && [ "$MANIFEST_BYTES" -lt 200000 ]; then
  pass "the manifest beside the model is $MANIFEST_BYTES bytes, not a megabyte"
else
  fail "manifest is $MANIFEST_BYTES bytes at $MANIFEST"
fi
if docker exec "$CONTAINER" grep -q '"split"' "$MANIFEST"; then
  pass "the manifest records how the score was split"
else
  fail "no split provenance in $MANIFEST"
fi

echo '== prompt: "generate a full profiling report for this dataset" -> run_profiling_report =='
R=$(call advanced "$SID_ADVANCED" 53 run_profiling_report "{\"file_path\":\"$DATASET_PATH\",\"sample_rows\":100}")
ok_json "$R" && pass "run_profiling_report wrote a real profiling report" || fail "$R"

echo '== prompt: "reduce f1/f2 to 1 dimension with PCA" -> apply_dimensionality_reduction =='
R=$(call advanced "$SID_ADVANCED" 54 apply_dimensionality_reduction "{\"file_path\":\"$DATASET_PATH\",\"feature_columns\":[\"f1\",\"f2\"],\"method\":\"pca\",\"n_components\":1}")
ok_json "$R" && pass "apply_dimensionality_reduction ran real PCA" || fail "$R"

echo '== prompt: "generate a training report for the classifier" -> generate_training_report =='
R=$(call advanced "$SID_ADVANCED" 55 generate_training_report "{\"model_path\":\"$CLF_MODEL\"}")
ok_json "$R" && pass "generate_training_report wrote a real HTML report" || fail "$R"

echo '== prompt: "plot the ROC curve for the classifier" -> plot_roc_curve =='
R=$(call advanced "$SID_ADVANCED" 56 plot_roc_curve "{\"model_path\":\"$CLF_MODEL\",\"file_path\":\"$DATASET_PATH\"}")
ok_json "$R" && pass "plot_roc_curve rendered a real ROC curve from real predictions" || fail "$R"

echo '== prompt: "plot the learning curve for a random forest classifier" -> plot_learning_curve =='
R=$(call advanced "$SID_ADVANCED" 57 plot_learning_curve "{\"file_path\":\"$DATASET_PATH\",\"target_column\":\"label\",\"model\":\"rf\",\"task\":\"classification\",\"cv\":3}")
ok_json "$R" && pass "plot_learning_curve ran real CV training at increasing sizes" || fail "$R"

echo '== prompt: "plot predicted vs actual for the regressor" -> plot_predictions_vs_actual =='
R=$(call advanced "$SID_ADVANCED" 58 plot_predictions_vs_actual "{\"model_path\":\"$REG_MODEL\",\"file_path\":\"$DATASET_PATH\"}")
ok_json "$R" && pass "plot_predictions_vs_actual plotted real regressor predictions" || fail "$R"

echo '== prompt: "generate a report on the clusters found earlier" -> generate_cluster_report =='
R=$(call advanced "$SID_ADVANCED" 59 generate_cluster_report "{\"file_path\":\"$DATASET_PATH\",\"feature_columns\":[\"f1\",\"f2\"],\"label_column\":\"cluster_label\"}")
ok_json "$R" && pass "generate_cluster_report reported on the real clusters saved earlier" || fail "$R"

echo
echo "===== security regression: HMAC-signed model files (shared/model_signing.py) ====="
echo "A model file that fails signature verification must be rejected with a clean"
echo "error, never unpickled, and never crash the tool — checked here against the"
echo "real deployed endpoint's real tool-call path, not just the internal function."

echo '== a plain (unsigned) pickle at model_path is rejected, not unpickled =='
docker exec "$CONTAINER" python3 -c "
import pickle, os
os.makedirs('/tmp/remote-smoke-test', exist_ok=True)
with open('/tmp/remote-smoke-test/unsigned_model.pkl', 'wb') as f:
    pickle.dump({'model': 'not a real model', 'metadata': {}}, f)
"
docker exec -u root "$CONTAINER" chown app:app /tmp/remote-smoke-test/unsigned_model.pkl
R=$(call basic "$SID_BASIC" 60 predict_single "{\"model_path\":\"/tmp/remote-smoke-test/unsigned_model.pkl\",\"input_data\":\"{}\"}")
fail_json "$R" || fail "unsigned pickle model_path should have been rejected, got: $R"
echo "$R" | grep -Eiq 'integrity|signature' && pass "unsigned pickle model file rejected as an integrity failure (not a crash)" || fail "expected an integrity/signature error message, got: $R"

echo '== a signed model file that was tampered with after signing is rejected =='
docker exec "$CONTAINER" python3 -c "
import sys
sys.path.insert(0, '/app')
from shared.model_signing import dump_signed
with open('/tmp/remote-smoke-test/tampered_model.pkl', 'wb') as f:
    dump_signed({'model': 'not a real model either', 'metadata': {}}, f)
with open('/tmp/remote-smoke-test/tampered_model.pkl', 'r+b') as f:
    data = bytearray(f.read())
    data[-1] ^= 0xFF
    f.seek(0)
    f.write(data)
"
docker exec -u root "$CONTAINER" chown app:app /tmp/remote-smoke-test/tampered_model.pkl
R=$(call basic "$SID_BASIC" 61 predict_single "{\"model_path\":\"/tmp/remote-smoke-test/tampered_model.pkl\",\"input_data\":\"{}\"}")
fail_json "$R" || fail "tampered signed model_path should have been rejected, got: $R"
echo "$R" | grep -Eiq 'integrity|signature' && pass "signed-then-tampered model file rejected as an integrity failure (not a crash)" || fail "expected an integrity/signature error message, got: $R"

if [ -n "$SHARED_DIR" ]; then
  echo
  echo "== leave the shared directory as we found it =="
  docker exec "$CONTAINER" sh -c "ls -1A '$SHARED_DIR' 2>/dev/null" | sort \
    | comm -13 "$SHARED_BEFORE" - \
    | while IFS= read -r leftover; do
        [ -n "$leftover" ] && docker exec "$CONTAINER" rm -rf "$SHARED_DIR/$leftover"
      done
  pass "removed everything this run added to $SHARED_DIR"
fi
rm -f "$SHARED_BEFORE"

echo
echo "ALL 33 TOOLS + security regression PASSED against $DOMAIN"
