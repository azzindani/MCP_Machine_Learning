# MCP Machine Learning

A self-hosted MCP server that gives local LLMs structured access to the full supervised + unsupervised machine learning pipeline. No cloud APIs, no API keys — everything runs on your machine.

## Features

- **35 tools** across 3 tiers: basic (11), medium (14), advanced (10)
- **LOCATE → INSPECT → PATCH → VERIFY** workflow for surgical ML operations
- **Automatic version control** — every write is snapshotted and fully restorable
- **Operation receipt logging** — full audit trail of all modifications
- **Constrained mode** — reduces row/result limits for lower-memory machines
- **Interactive HTML reports** — EDA, training metrics, clustering, ROC curves, learning curves
- **Full ML pipeline** — inspect → preprocess → train → evaluate → tune
- **7 classification algorithms** — LR, SVM, RF, DTC, KNN, NB, XGBoost
- **7 regression algorithms** — Linear, Polynomial, Lasso, Ridge, DT, RF, XGBoost
- **3 clustering algorithms** — K-Means, Mean-Shift, DBSCAN
- **Dimensionality reduction** — PCA and ICA
- **Hyperparameter tuning** — GridSearch and RandomSearch
- **Light / dark theme** — all HTML outputs accept `theme: "dark" | "light"`
- **100% local execution** — your data never leaves your machine

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
- **Python 3.12 or higher** — `python --version`
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

The first launch clones the repo and installs dependencies (~200 MB including XGBoost). Subsequent launches are instant.

> **Pre-install recommended:** To avoid the 60-second LM Studio connection timeout on first launch, run this once in PowerShell before connecting:
> ```powershell
> $d = Join-Path $env:USERPROFILE '.mcp_servers\MCP_Machine_Learning'
> git clone https://github.com/azzindani/MCP_Machine_Learning.git $d
> Set-Location $d; uv sync
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
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Machine_Learning'; if (!(Test-Path $d)) { git clone https://github.com/azzindani/MCP_Machine_Learning.git $d } else { Set-Location $d; git pull --quiet }; Set-Location (Join-Path $d 'servers\\ml_basic'); uv run python server.py"
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
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Machine_Learning'; if (!(Test-Path $d)) { git clone https://github.com/azzindani/MCP_Machine_Learning.git $d } else { Set-Location $d; git pull --quiet }; Set-Location (Join-Path $d 'servers\\ml_medium'); uv run python server.py"
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
        "$d = Join-Path $env:USERPROFILE '.mcp_servers\\MCP_Machine_Learning'; if (!(Test-Path $d)) { git clone https://github.com/azzindani/MCP_Machine_Learning.git $d } else { Set-Location $d; git pull --quiet }; Set-Location (Join-Path $d 'servers\\ml_advanced'); uv run python server.py"
      ],
      "env": { "MCP_CONSTRAINED_MODE": "0" },
      "timeout": 600000
    }
  }
}
```

4. Wait for the blue dot next to each server
5. Start chatting — the model will see all 35 tools

> **Low-memory machines:** Set `MCP_CONSTRAINED_MODE` to `"1"` in all `env` blocks and omit `ml-advanced` if needed. See [Configuration](#configuration) for details.

### macOS / Linux

Replace `powershell` / `args` with:

```json
{
  "command": "bash",
  "args": [
    "-c",
    "d=\"$HOME/.mcp_servers/MCP_Machine_Learning\"; if [ ! -d \"$d\" ]; then git clone https://github.com/azzindani/MCP_Machine_Learning.git \"$d\"; else cd \"$d\" && git pull --quiet; fi; cd \"$d/servers/ml_basic\"; uv run python server.py"
  ]
}
```

Repeat for `ml_medium` and `ml_advanced`, adjusting the server directory in the path.

---

## Available Tools

### Tier 1 — ml-basic (11 tools)

| # | Tool | Category | Description |
|---|---|---|---|
| 1 | `inspect_dataset` | LOCATE | Schema, row count, dtypes, null summary |
| 2 | `read_column_profile` | INSPECT | Stats for one column (mean, std, nulls, unique) |
| 3 | `search_columns` | LOCATE | Find columns matching a condition |
| 4 | `read_rows` | INSPECT | Bounded row slice |
| 5 | `train_classifier` | PATCH | Train classifier: `lr svm rf dtc knn nb xgb` — AUC-ROC + class_weight + train_score |
| 6 | `train_regressor` | PATCH | Train regressor: `lir pr lar rr dtr rfr xgb` |
| 7 | `get_predictions` | VERIFY | Run predictions — supports `return_proba=True` for probabilities |
| 8 | `restore_version` | CONTROL | Rollback model or dataset to previous snapshot |
| 9 | `predict_single` | VERIFY | Predict one JSON record — no CSV needed |
| 10 | `list_models` | LOCATE | List all saved `.pkl` models with metadata |
| 11 | `split_dataset` | PATCH | Split CSV into train/test files |

### Tier 2 — ml-medium (14 tools)

| # | Tool | Category | Description |
|---|---|---|---|
| 1 | `run_preprocessing` | PATCH | 14-op pipeline: encode, scale, fill, bin, log, date parts, drop nulls, clip |
| 2 | `detect_outliers` | INSPECT | IQR and std-dev outlier report per column |
| 3 | `train_with_cv` | PATCH | K-fold cross-validation training |
| 4 | `compare_models` | PATCH | Train multiple algorithms, return sorted table |
| 5 | `run_clustering` | PATCH | K-Means / Mean-Shift / DBSCAN — returns silhouette score |
| 6 | `read_receipt` | CONTROL | Read operation history for a file |
| 7 | `generate_eda_report` | ANALYZE | Interactive HTML EDA with quality score + alerts |
| 8 | `filter_rows` | PATCH | Filter rows by column condition, save CSV |
| 9 | `merge_datasets` | PATCH | Merge two CSVs on a key column |
| 10 | `find_optimal_clusters` | ANALYZE | Elbow + silhouette chart to find best K |
| 11 | `anomaly_detection` | INSPECT | Isolation Forest or LOF anomaly detection |
| 12 | `check_data_quality` | INSPECT | JSON quality score 0-100 with typed alerts (model-readable) |
| 13 | `evaluate_model` | VERIFY | Score saved model on external labeled test CSV |
| 14 | `batch_predict` | PATCH | Predict all rows, save predictions CSV — no row limit |

### Tier 3 — ml-advanced (10 tools)

| # | Tool | Category | Description |
|---|---|---|---|
| 1 | `tune_hyperparameters` | OPTIMIZE | GridSearch or RandomSearch tuning |
| 2 | `export_model` | EXPORT | Pickle export with metadata manifest |
| 3 | `read_model_report` | VERIFY | Feature importance, confusion matrix, metrics |
| 4 | `run_profiling_report` | ANALYZE | ydata-profiling HTML report for dataset |
| 5 | `apply_dimensionality_reduction` | TRANSFORM | PCA or ICA, return reduced dataset path |
| 6 | `generate_training_report` | ANALYZE | Full HTML training report: metrics + charts |
| 7 | `plot_roc_curve` | ANALYZE | Interactive ROC curve with AUC |
| 8 | `plot_learning_curve` | ANALYZE | Train vs validation score by training size |
| 9 | `plot_predictions_vs_actual` | ANALYZE | Scatter: predicted vs actual (regression) |
| 10 | `generate_cluster_report` | ANALYZE | HTML cluster report with PCA scatter + profile |

---

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

---

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

---

## HTML Reports

All visualization tools save standalone interactive HTML files that open automatically in your browser. No server required — files work fully offline.

| Tool | Output |
|---|---|
| `generate_eda_report` | Dataset overview, quality score, alerts, distributions, correlations |
| `generate_training_report` | Confusion matrix, feature importance, classification report |
| `run_profiling_report` | Deep ydata-profiling report (or Plotly fallback) |
| `plot_roc_curve` | ROC curve with AUC for classifiers |
| `plot_learning_curve` | Train vs validation score by training size |
| `plot_predictions_vs_actual` | Predicted vs actual scatter for regressors |
| `find_optimal_clusters` | Elbow curve + silhouette score chart |
| `generate_cluster_report` | PCA scatter, cluster profiles, size chart |

All reports support `theme="light"` (default) or `theme="dark"`.

---

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

---

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

---

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
| ydata-profiling mode | full | minimal |

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

---

## Version Control & Audit

Every write operation:
1. Takes a snapshot to `.mcp_versions/` before overwriting
2. Returns a `"backup"` key with the snapshot path
3. Appends an entry to the file's `.mcp_receipt.json` audit log

To view history: `read_receipt("mydata.csv")`  
To rollback: `restore_version("mydata.csv", timestamp="2026-04-06T10-30-00Z")`

---

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

---

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
│   ├── file_utils.py          ← path resolution, atomic writes
│   ├── platform_utils.py      ← constrained mode, row limits
│   ├── progress.py            ← ok/fail/info/warn helpers
│   ├── receipt.py             ← operation receipt logging
│   └── html_theme.py          ← CSS vars, Plotly templates
├── install/
│   ├── install.sh             ← macOS/Linux installer
│   ├── install.bat            ← Windows installer
│   └── mcp_config_writer.py   ← writes LM Studio / Claude Desktop / Cursor config
├── tests/
│   ├── fixtures/              ← classification, regression, clustering, large CSVs
│   ├── conftest.py
│   ├── test_ml_basic.py
│   ├── test_ml_medium.py
│   └── test_ml_advanced.py
├── pyproject.toml             ← root workspace
├── uv.lock
└── .python-version            ← 3.12
```

---

## Development

### Local Testing

```bash
# Install dependencies
uv sync

# Run all tests (Windows)
set PYTHONPATH=. && set MCP_CONSTRAINED_MODE=1 && python -m pytest tests/ -v

# Run all tests (Linux/macOS)
PYTHONPATH=. MCP_CONSTRAINED_MODE=1 uv run python -m pytest tests/ -q

# Lint
uvx ruff check servers/ shared/ tests/ --exclude "**/.venv/**"

# Type check
uv run pyright servers/ shared/
```

### Run a single tier server locally

```bash
cd servers/ml_basic
uv sync
uv run python server.py
```

---

## License

MIT
