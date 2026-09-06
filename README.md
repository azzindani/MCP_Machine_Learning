# MCP Machine Learning

A self-hosted MCP server that gives local LLMs structured access to the full supervised + unsupervised machine learning pipeline. No cloud APIs, no API keys — everything runs on your machine.

**Release [`v0.1.2`](https://github.com/azzindani/MCP_Machine_Learning/releases/tag/v0.1.2)** — source only. No wheel and no container image are published: install from the tag with the bundled installer, or build the image yourself from the `Dockerfile` in this repo.

## Features

- **33 tools** across 3 tiers: basic (11), medium (12), advanced (10)
- **LOCATE → INSPECT → PATCH → VERIFY** workflow for surgical ML operations
- **Automatic version control** — every write is snapshotted and fully restorable
- **Operation receipt logging** — full audit trail of all modifications
- **Constrained mode** — reduces row/result limits for lower-memory machines
- **Interactive HTML reports** — EDA, training metrics, clustering, ROC curves, learning curves, profiling
- **Full ML pipeline** — inspect → preprocess → train → evaluate → tune → export
- **7 classification algorithms** — LR, SVM, RF, DTC, KNN, NB, XGBoost
- **7 regression algorithms** — Linear, Polynomial, Lasso, Ridge, DT, RF, XGBoost
- **3 clustering algorithms** — K-Means, Mean-Shift, DBSCAN
- **Dimensionality reduction** — PCA and ICA
- **Hyperparameter tuning** — GridSearch and RandomSearch
- **Leakage detection on every tool that takes a target** — `train_classifier`, `train_regressor`, `train_with_cv`, `compare_models`, `check_data_quality` and `evaluate_model` all name features that may already contain the outcome, with the evidence: how well a feature separates the classes alone, whether its *missingness* tracks the target, and whether it is named like a post-outcome field. Suspects, never verdicts — nothing is dropped and no training is refused
- **Split provenance in the manifest** — test size, seed, stratification, CV folds and whether the split was time-ordered, because a 0.9628 from a random split of time-ordered rows is not the same number as one from a forward-chained split
- **Split manifests** — a 28,000-category encoding map moves to a `.encoding_map.json` sidecar so the manifest stays kilobytes; the `.pkl` keeps the full metadata, so a model shipped alone still predicts
- **Quality score with a breakdown** — `{completeness, validity, uniqueness, drift}`, from a scorer byte-identical with MCP_Data_Analyst's, so the same file cannot score two ways across the two servers
- **Dark / light / device theme** — all HTML outputs accept `theme: "dark" | "light" | "device"`
- **Mobile-responsive HTML** — viewport meta + CSS breakpoints on every report
- **Modular architecture** — each engine split into focused sub-modules, all under 1 000 lines
- **Handover protocol** — every tool response includes `context` + `handover` for automatic tool-call chaining
- **Encoding-robust CSV loading** — UTF-8 → UTF-8-BOM → CP1252 → Latin-1 fallback chain with bad-line recovery
- **DA server compatible** — designed to work side-by-side with [MCP_Data_Analyst](https://github.com/azzindani/MCP_Data_Analyst) via shared workspace and handover conventions

## Important: File Path Only

> **Do not attach files via the LM Studio attachment button.**
>
> LM Studio will RAG-chunk any attached file and send fragments to the model — the MCP tools will never see the actual data. This MCP works exclusively through **absolute file paths**.
>
> Always tell the model where the file lives on disk:
> ```
> Train a classifier on C:\Users\you\data\churn.csv
> ```
> The model will pass that path directly to the MCP tools. Attachment-based workflows are not supported and will silently produce wrong results.

## Quick Install (LM Studio)

> **Tested on Windows 11** with LM Studio 0.4.x and uv 0.5+.

### Requirements

- **Git** — `git --version`
- **Python 3.14 or higher** — `python --version`
- **uv** — `uv --version` ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **LM Studio** with a model that supports tool calling (Gemma 4, Qwen 3.5, etc.)

### Platform Support

| Platform | Status |
|---|---|
| Windows | Tested — real-world verified (Windows 11) |
| macOS | Untested — CI/CD pipeline passes |
| Linux | Untested — CI/CD pipeline passes |

> Real-world usage has only been verified on Windows. macOS and Linux are supported by design and pass the automated CI pipeline, but have not been tested by hand. Reports from non-Windows users are welcome.

### First Run

The first launch clones the repo and installs dependencies (~2–5 minutes). Subsequent launches are instant.

> **Pre-install recommended:** To avoid the 60-second LM Studio connection timeout on first launch, run this once in PowerShell before connecting:
> ```powershell
> $d = Join-Path $env:USERPROFILE '.mcp_servers\MCP_Machine_Learning'
> $g = Join-Path $d '.git'
> if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; git clone https://github.com/azzindani/MCP_Machine_Learning.git $d --quiet }
> Set-Location "$d\servers\ml_basic"; uv sync
> Set-Location "$d\servers\ml_medium"; uv sync
> Set-Location "$d\servers\ml_advanced"; uv sync
> ```
> If you skip this step and LM Studio times out, press **Restart** in the MCP Servers panel — it will reconnect and complete the install immediately.

### Steps

1. Open LM Studio → **Developer** tab (`</>` icon) or you can find via **Integrations**
2. Find **mcp.json** or **Edit mcp.json** → click to open
3. Paste this config:

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "powershell",
      "args": [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Machine_Learning'; $g = Join-Path $d '.git'; if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; git clone https://github.com/azzindani/MCP_Machine_Learning.git $d --quiet } else { Set-Location $d; git fetch origin --quiet; git reset --hard FETCH_HEAD --quiet }; Set-Location (Join-Path $d 'servers\\ml_basic'); uv sync --quiet; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    },
    "ml-medium": {
      "command": "powershell",
      "args": [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Machine_Learning'; $g = Join-Path $d '.git'; if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; git clone https://github.com/azzindani/MCP_Machine_Learning.git $d --quiet } else { Set-Location $d; git fetch origin --quiet; git reset --hard FETCH_HEAD --quiet }; Set-Location (Join-Path $d 'servers\\ml_medium'); uv sync --quiet; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    },
    "ml-advanced": {
      "command": "powershell",
      "args": [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Machine_Learning'; $g = Join-Path $d '.git'; if (!(Test-Path $g)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d }; git clone https://github.com/azzindani/MCP_Machine_Learning.git $d --quiet } else { Set-Location $d; git fetch origin --quiet; git reset --hard FETCH_HEAD --quiet }; Set-Location (Join-Path $d 'servers\\ml_advanced'); uv sync --quiet; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

4. Wait for the blue dot next to each server
5. Start chatting — the model will see all 33 tools

> **Low-memory machines:** Set `MCP_CONSTRAINED_MODE` to `"1"` in all `env` blocks and omit `ml-advanced` if needed. See [Configuration](#configuration) for details.

### macOS / Linux

Replace the `"command"` and `"args"` in each entry with the bash equivalent:

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "bash",
      "args": [
        "-c",
        "d=\"$HOME/.mcp_servers/MCP_Machine_Learning\"; if [ ! -d \"$d/.git\" ]; then rm -rf \"$d\"; git clone https://github.com/azzindani/MCP_Machine_Learning.git \"$d\" --quiet; else cd \"$d\" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; fi; cd \"$d/servers/ml_basic\"; uv sync --quiet; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    },
    "ml-medium": {
      "command": "bash",
      "args": [
        "-c",
        "d=\"$HOME/.mcp_servers/MCP_Machine_Learning\"; if [ ! -d \"$d/.git\" ]; then rm -rf \"$d\"; git clone https://github.com/azzindani/MCP_Machine_Learning.git \"$d\" --quiet; else cd \"$d\" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; fi; cd \"$d/servers/ml_medium\"; uv sync --quiet; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    },
    "ml-advanced": {
      "command": "bash",
      "args": [
        "-c",
        "d=\"$HOME/.mcp_servers/MCP_Machine_Learning\"; if [ ! -d \"$d/.git\" ]; then rm -rf \"$d\"; git clone https://github.com/azzindani/MCP_Machine_Learning.git \"$d\" --quiet; else cd \"$d\" && git fetch origin --quiet && git reset --hard FETCH_HEAD --quiet; fi; cd \"$d/servers/ml_advanced\"; uv sync --quiet; uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

## Available Tools

### Tier 1 — ml-basic (11 tools)

| Tool | Purpose |
|---|---|
| `inspect_dataset` | Schema, row count, dtypes, null summary |
| `read_column_profile` | Stats for one column: mean, std, nulls, unique, top values |
| `search_columns` | Find columns by criteria: has_nulls, name_contains, and `dtype` — one of `numeric` `categorical` `bool` `datetime`, or a concrete pandas name (`float64`, `object`, …) which is widened to its group |
| `read_rows` | Bounded row slice |
| `train_classifier` | Train classifier: `lr svm rf dtc knn nb xgb` — AUC-ROC, class_weight, train_score |
| `train_regressor` | Train regressor: `lir pr lar rr dtr rfr xgb` |
| `get_predictions` | Run predictions on a CSV — supports `return_proba=True` for probabilities |
| `restore_version` | Rollback model or dataset to any previous snapshot |
| `predict_single` | Predict one JSON record — no CSV needed |
| `list_models` | List all saved `.pkl` models with metadata |
| `split_dataset` | Split CSV into train/test files |

### Tier 2 — ml-medium (12 tools)

| Tool | Purpose |
|---|---|
| `run_preprocessing` | **15-op pipeline**: fill_nulls, drop_outliers, label_encode, onehot_encode, scale, drop_duplicates, drop_column, rename_column, convert_dtype, bin_numeric, add_date_parts, log_transform, drop_null_rows, clip_column |
| `detect_outliers` | IQR and std-dev outlier report per column |
| `train_with_cv` | K-fold cross-validation training — per-fold scores + mean ± std |
| `compare_models` | Train multiple algorithms on the same split, return ranked table |
| `run_clustering` | K-Means / Mean-Shift / DBSCAN — returns silhouette score |
| `read_receipt` | Read operation history for a file |
| `generate_eda_report` | Interactive HTML EDA with quality score + 8-alert panel |
| `find_optimal_clusters` | Elbow + silhouette chart to find best K |
| `anomaly_detection` | Isolation Forest or LOF anomaly detection |
| `check_data_quality` | JSON quality score 0–100 with typed alerts (model-readable). Pass `target_column` to add the leakage check |
| `evaluate_model` | Score a saved model on an external labeled test CSV, and flag features that may already contain the outcome |
| `batch_predict` | Predict all rows and save predictions CSV — no row limit |

> Earlier versions of this table listed `filter_rows` and `merge_datasets`.
> Both exist in the engine but neither is registered as an MCP tool, so a
> caller who read this table and tried them burned a loop iteration on a tool
> that was never there. Filtering and merging belong to
> [MCP_Data_Analyst](https://github.com/azzindani/MCP_Data_Analyst)
> (`filter_dataset`, `merge_datasets`), which shares this workspace.

### Tier 3 — ml-advanced (10 tools)

| Tool | Purpose |
|---|---|
| `tune_hyperparameters` | GridSearch or RandomSearch hyperparameter tuning |
| `export_model` | Export `.pkl` with metadata manifest — the training record is carried through, never replaced |
| `read_model_report` | Feature importance, confusion matrix, metrics from saved model. `skip_encoding_map` defaults to **True**; pass `top_n` for a bounded slice of the map |
| `run_profiling_report` | Interactive Plotly HTML profile: distributions, correlations, summary stats |
| `apply_dimensionality_reduction` | PCA or ICA — returns reduced dataset path |
| `generate_training_report` | Full HTML training report: metrics, confusion matrix, feature importance |
| `plot_roc_curve` | Interactive ROC curve with AUC for classifiers |
| `plot_learning_curve` | Train vs validation score by training size |
| `plot_predictions_vs_actual` | Scatter: predicted vs actual for regressors |
| `generate_cluster_report` | HTML cluster report with PCA scatter + cluster profiles |

All chart-producing tools accept `theme: "dark" | "light" | "device"`, `output_path`, and `open_after`.

### Theme options (all HTML outputs)

| Value | Behaviour |
|---|---|
| `"dark"` | GitHub-style dark palette, Plotly dark template (default) |
| `"light"` | Light palette, Plotly white template |
| `"device"` | Auto-detects system `prefers-color-scheme`, switches at runtime via JS |

## Preprocessing Operations (`run_preprocessing`)

Pass an `ops` array to apply a pipeline in one call:

```json
[
  {"op": "fill_nulls",     "column": "revenue",    "strategy": "median"},
  {"op": "fill_nulls",     "column": "region",     "strategy": "mode"},
  {"op": "drop_outliers",  "column": "age",        "method": "iqr"},
  {"op": "label_encode",   "column": "gender"},
  {"op": "onehot_encode",  "column": "region"},
  {"op": "scale",          "columns": ["age", "salary"], "method": "standard"},
  {"op": "drop_duplicates","subset": ["customer_id"]},
  {"op": "drop_column",    "column": "id"},
  {"op": "rename_column",  "from": "rev",  "to": "revenue_usd"},
  {"op": "convert_dtype",  "column": "date", "to": "datetime"},
  {"op": "bin_numeric",    "column": "age",  "bins": 5, "new_column": "age_group"},
  {"op": "add_date_parts", "column": "date", "parts": ["year", "month", "dayofweek"]},
  {"op": "log_transform",  "column": "revenue", "base": "natural"},
  {"op": "drop_null_rows", "column": "critical_field"},
  {"op": "clip_column",    "column": "age", "lower": 0, "upper": 120}
]
```

## Supported Algorithms

### Classification

| Code | Algorithm |
|---|---|
| `lr` | Logistic Regression |
| `svm` | Support Vector Machine (RBF kernel) |
| `rf` | Random Forest Classifier |
| `dtc` | Decision Tree Classifier |
| `knn` | K-Nearest Neighbors |
| `nb` | Gaussian Naive Bayes |
| `xgb` | XGBoost |

### Regression

| Code | Algorithm |
|---|---|
| `lir` | Linear Regression |
| `pr` | Polynomial Regression |
| `lar` | Lasso Regression |
| `rr` | Ridge Regression |
| `dtr` | Decision Tree Regressor |
| `rfr` | Random Forest Regressor |
| `xgb` | XGBoost |

### Clustering

| Code | Algorithm |
|---|---|
| `kmeans` | K-Means |
| `meanshift` | Mean-Shift |
| `dbscan` | DBSCAN |

## EDA Quality Alerts

`generate_eda_report` runs 8 automated checks and returns a **quality score 0–100**:

| Alert Type | Severity | Trigger |
|---|---|---|
| `constant_column` | High | Single unique value |
| `high_missing` | High | > 20% null values |
| `zero_inflated` | Medium | > 50% zeros in numeric column |
| `high_cardinality` | Medium | > 50% unique ratio with > 20 uniques |
| `class_imbalance` | High | Dominant class > 90% of target |
| `extreme_skewness` | Medium | \|skewness\| > 2 |
| `multicollinearity` | High | Pearson \|r\| > 0.9 between two features |
| `duplicate_rows` | Medium | Duplicate rows detected |

Each alert includes a **recommendation** — actionable next steps to fix the issue.

## Usage Examples

### Inspect a dataset

```
Inspect the file C:\data\churn.csv and tell me about its schema
```

### Find problem columns

```
Search for columns in C:\data\churn.csv that have null values
```

### Train a classifier

```
Train a random forest on C:\data\churn.csv to predict the churned column
```

### Full classification workflow

```
Inspect C:\data\churn.csv, preprocess it to fill nulls and encode categoricals,
then train a random forest and show me the training report
```

### Regression with tuning

```
Train a random forest regressor on C:\data\housing.csv to predict price,
then tune it with grid search and plot predictions vs actual
```

### Clustering + EDA

```
Run EDA on C:\data\customers.csv, find the optimal number of clusters,
then cluster with K-Means and generate a cluster report
```

### Undo a change

```
Restore C:\data\churn.csv to the previous version
```

### Full four-tool pattern

```
1. inspect_dataset("churn.csv")         → schema, row count, target candidates
2. read_column_profile("churned")       → class balance, null count
3. train_classifier("rf", "churned")    → accuracy 0.89, f1 0.87
4. generate_training_report(model_path) → confusion matrix, feature importance
```

## Inter-server compatibility (MCP_Data_Analyst)

This server is designed to work side-by-side with [MCP_Data_Analyst](https://github.com/azzindani/MCP_Data_Analyst). Both servers share the same:

- **Handover protocol** — every tool success response includes a `context` block (what was just done + artifact paths) and a `handover` block (suggested next tools + carry-forward parameters). The LLM uses these to chain tool calls automatically without needing you to repeat paths or state.
- **Workspace alias resolution** — paths like `workspace:my_project/cleaned.csv` resolve to the same on-disk locations in both servers.
- **CSV loading** — encoding-robust loader (`shared.file_utils.read_csv`) handles UTF-8, UTF-8-BOM, CP1252, and Latin-1 files with bad-line recovery, matching the DA server's behavior exactly.

A typical cross-server workflow:

```
DA: generate_eda_report("churn.csv")        → EDA HTML + handover suggests train_classifier
ML: train_classifier("churn.csv", "rf")    → model + handover suggests generate_training_report
ML: generate_training_report(model_path)   → confusion matrix, feature importance
```

## Configuration

### Constrained Mode

For lower-memory machines, set `MCP_CONSTRAINED_MODE=1` in the `env` section of `mcp.json`. This reduces:

| Resource | Standard | Constrained |
|---|---|---|
| Rows returned per call | 100 | 20 |
| Search results | 50 | 10 |
| Columns returned | 50 | 20 |
| CV folds | 5 | 3 |
| Max models in compare | 7 | 3 |

### Recommended Loading by Available Memory

| Available RAM | Recommended load | Total tools |
|---|---|---|
| 4–8 GB | ml-basic only | 11 |
| 8–16 GB | ml-basic + ml-medium | 25 |
| 16 GB+ | all three tiers | 35 |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MCP_CONSTRAINED_MODE` | `0` | Set to `1` for low-memory machines |
| `MCP_OUTPUT_DIR` | `~/Downloads` | Where models, reports, plots and prediction CSVs land by default |
| `MCP_PUBLIC_BASE_URL` | _(unset)_ | Public URL serving `MCP_OUTPUT_DIR`; adds `public_url` to results |
| `MCP_FETCH_URLS` | `0` | `1` lets any `file_path` argument be an `http(s)` URL |
| `MCP_FETCH_ALLOW_PRIVATE` | `0` | `1` permits fetching hosts on private/loopback addresses |
| `MCP_MAX_FETCH_MB` | `100` | Size cap for a fetched URL |

### Hybrid local + remote file handling

The same engine serves a local stdio install and a self-hosted HTTP endpoint,
but over HTTP the caller shares no filesystem with the server: a server-local
output path means nothing to it, and it may hold a link rather than a path.
Three opt-in variables close that gap without changing local behaviour — all
are unset by default, so a local install stays offline and writes to
`~/Downloads` exactly as before:

- **`MCP_OUTPUT_DIR`** — bind-mount a directory here and every trained model,
  HTML report, plot and prediction CSV defaults into it, so outputs land
  somewhere you can actually see. It outranks the input file's own directory;
  an explicit `output_path` still wins.
- **`MCP_PUBLIC_BASE_URL`** — the public URL that serves that directory. Every
  produced file then comes back with a `public_url` you can open or pass on.
- **`MCP_FETCH_URLS=1`** — `train_classifier("https://…/train.csv", …)`, and
  every other path argument, accepts a link; it is downloaded into
  `MCP_OUTPUT_DIR/inbox` before the tool runs. Hosts resolving to
  loopback/link-local/private addresses are refused, redirects included,
  unless `MCP_FETCH_ALLOW_PRIVATE=1`.

Report and export tools also accept `return_content=True`, which embeds the
file's bytes as `content_base64` for callers that have neither a shared
filesystem nor access to the public URL.

## Deployment

| Mode | Best for | Transport | Auth |
|---|---|---|---|
| **Local stdio** (default, above) | LM Studio / Claude Code on your machine | stdio | none |
| **Local Docker / HTTP** | Testing, or one other machine on your LAN | HTTP | optional |
| **VPS Docker** | Remote MCP clients (claude.ai, hosted harnesses) | HTTP | **required** |

Each tier keeps its own stdio server (`servers/ml_*/server.py`) for local LM
Studio "add one tier" installs. For Docker/remote deployment all 3 tiers run
as separate MCP endpoints inside **one process** (`unified_server.py`) on
**one port** — pandas/numpy/scikit-learn/xgboost load once instead of three
times (~240 MiB vs ~520 MiB idle), and all three still share one bearer-token
set.

### HTTP transport (no Docker)

```bash
uv run python unified_server.py --port 8820
curl http://localhost:8820/health            # {"status":"ok","version":"0.1.2","tiers":[...]}
curl http://localhost:8820/basic/health      # per-tier health
```

### Docker

```bash
docker compose up -d --build
curl http://localhost:8820/health            # aggregate
curl http://localhost:8820/basic/mcp         # basic tier
curl http://localhost:8820/medium/mcp        # medium tier
curl http://localhost:8820/advanced/mcp      # advanced tier
```

With auth (**required** for any publicly reachable deploy — this is how the
production `ml.casava.space` endpoint runs):

```bash
echo "ML_API_KEY=$(openssl rand -hex 24)" > .env   # gitignored, auto-loaded by docker-compose.yml
docker compose up -d --build
```

For multiple named clients instead of one shared key (Folio-style):

```bash
cp tokens.example.json tokens.json   # edit: replace placeholders with `openssl rand -hex 32`
ML_TOKENS_FILE=/path/to/tokens.json docker compose up -d --build
```

`/<tier>/mcp` requires `Authorization: Bearer <token>` once any of
`ML_TOKENS_FILE` / `ML_TOKENS` / `ML_API_KEY` is set; `/health` and
`/version` (aggregate and per-tier) stay unauthenticated.

### Deployment environment variables

| Variable | Default | Description |
|---|---|---|
| `ML_HOST` | `0.0.0.0` | Bind address for the unified server |
| `ML_PORT` | `8820` | Port for the unified server (all 3 tiers) |
| `ML_TOKENS_FILE` | unset | JSON file of named bearer tokens (`{"name": "token"}`) — highest priority, shared across all 3 tiers |
| `ML_TOKENS` | unset | Inline `"name:token,name2:token2"` |
| `ML_API_KEY` | unset | Single shared bearer token |

### Remote testing (Cloudflare Quick Tunnel)

Same idea as `azzindani/Folio`'s `launch.sh`: bring the Docker deployment up
and expose it at an ephemeral `*.trycloudflare.com` URL — no VPS, no DNS, no
account — so all 3 tiers are reachable from any MCP-compatible harness for a
quick remote smoke test.

```bash
./launch_tunnel.sh          # docker compose up -d --build, then tunnel
./launch_tunnel.sh stop     # tear the tunnel down (container keeps running)
```

Not for production: Quick Tunnels are unauthenticated at the transport layer.
Set `ML_API_KEY` or `ML_TOKENS_FILE` before tunneling so `/<tier>/mcp` still
requires a bearer token even while it's publicly reachable.

### Remote smoke test (`remote_smoke_test.sh`)

Run in CI against a container (the `e2e` job) and by hand against the
deployment. `pytest` itself stays offline. Exercises a running HTTP endpoint: auth enforcement plus a real
handwritten-prompt-style call for **all 33 tools** across all 3 tiers
(basic/medium/advanced), against a real generated dataset, chaining real
outputs (`model_path`, cluster labels, etc.) between calls the way an actual
workflow would — e.g. `train_classifier` → `plot_roc_curve` on the resulting
model.

```bash
./remote_smoke_test.sh                      # reads ML_API_KEY from .env, targets ml.casava.space
DOMAIN=http://localhost:8820 ./remote_smoke_test.sh   # test a different target
CONTAINER=mcp-ml ./remote_smoke_test.sh      # override container name
```

## Uninstall

**Step 1:** Remove from LM Studio
1. Open LM Studio → Developer tab (`</>`)
2. Delete `ml-basic`, `ml-medium`, `ml-advanced` from MCP Servers
3. Restart LM Studio

**Step 2:** Delete installed files

```cmd
rmdir /s /q %USERPROFILE%\.mcp_servers\MCP_Machine_Learning
```

Or run the uninstall script:

```bash
# Windows
%USERPROFILE%\.mcp_servers\MCP_Machine_Learning\install\install.bat

# macOS / Linux
~/.mcp_servers/MCP_Machine_Learning/install/install.sh
```

## Architecture

```
MCP_Machine_Learning/
├── servers/
│   ├── ml_basic/
│   │   ├── server.py          ← thin MCP wrapper (zero domain logic)
│   │   ├── engine.py          ← public API re-exports
│   │   ├── _basic_helpers.py  ← shared helpers
│   │   ├── _basic_train.py    ← train_classifier + train_regressor
│   │   ├── _basic_predict.py  ← get_predictions + predict_single
│   │   └── pyproject.toml
│   ├── ml_medium/
│   │   ├── server.py
│   │   ├── engine.py          ← public API re-exports
│   │   ├── _medium_helpers.py ← shared helpers
│   │   ├── _medium_preprocess.py ← run_preprocessing ops
│   │   ├── _medium_train.py   ← train_with_cv + compare_models
│   │   ├── _medium_cluster.py ← run_clustering + find_optimal_clusters
│   │   ├── _medium_data.py    ← filter_rows + merge_datasets + batch_predict
│   │   ├── _medium_eda.py     ← generate_eda_report + check_data_quality
│   │   └── pyproject.toml
│   └── ml_advanced/
│       ├── server.py
│       ├── engine.py          ← public API re-exports
│       ├── _adv_helpers.py    ← tuning + export + model report helpers
│       ├── _adv_viz.py        ← all HTML chart and report generation
│       └── pyproject.toml
├── shared/
│   ├── version_control.py     ← snapshot() and restore()
│   ├── patch_validator.py     ← validate op arrays
│   ├── file_utils.py          ← path resolution, atomic writes, encoding-robust read_csv
│   ├── platform_utils.py      ← constrained mode, row limits
│   ├── progress.py            ← ok/fail/info/warn helpers
│   ├── receipt.py             ← operation receipt logging
│   ├── handover.py            ← make_context() / make_handover() for tool-chaining protocol
│   ├── workspace_utils.py     ← workspace alias resolution, pipeline save/load
│   ├── ml_utils.py            ← shared ML helper functions (encoding, preprocessing)
│   ├── registry.py            ← canonical algorithm key sets (classifiers, regressors)
│   ├── project_utils.py       ← backward-compatible shim for workspace_utils
│   ├── html_layout.py         ← output path helpers, Plotly layout base
│   └── html_theme.py          ← CSS vars, Plotly templates, responsive HTML helpers
├── install/
│   ├── install.sh             ← macOS/Linux installer
│   ├── install.bat            ← Windows installer
│   └── mcp_config_writer.py   ← writes LM Studio / Claude Desktop / Cursor config
└── tests/
    ├── fixtures/              ← classification, regression, clustering, large CSVs
    ├── conftest.py
    ├── test_ml_basic.py
    ├── test_ml_medium.py
    └── test_ml_advanced.py
```

## Development

### Local Testing

```bash
# Install dependencies
uv sync

# Run all tests (Windows)
set PYTHONPATH=. && uv run python -m pytest tests/ -q --tb=short

# Run all tests (Linux/macOS)
PYTHONPATH=. uv run python -m pytest tests/ -q --tb=short

# Lint
uv run ruff check .

# Format check
uv run ruff format --check .

# Type check
uv run pyright servers/ shared/
```

### Run a single tier server locally

```bash
cd servers/ml_basic
uv sync
uv run python server.py
```

## License

MIT
