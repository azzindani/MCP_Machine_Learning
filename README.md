# MCP Machine Learning

A self-hosted **Model Context Protocol** server that gives local LLMs structured access to the full supervised + unsupervised machine learning pipeline — without any cloud dependency, API key, or subscription.

All computation runs on your CPU/GPU using scikit-learn, XGBoost, pandas, and numpy. Your data never leaves your machine.

---

## Why MCP Machine Learning?

| Capability | This project |
|---|---|
| 100% local execution | ✓ Works offline after first install |
| Full ML pipeline | ✓ Inspect → Preprocess → Train → Evaluate → Tune |
| Interactive HTML reports | ✓ EDA, Training, Clustering, ROC, Learning Curves |
| Version control | ✓ Automatic snapshots before every write |
| Audit log | ✓ Per-file operation receipt |
| Constrained mode | ✓ Optimised for 8 GB VRAM / 9B models |
| No cloud APIs | ✓ No OpenAI, SageMaker, Vertex, Azure ML |

---

## Three-Tier Architecture

The server suite is split into three focused tiers. Load only what you need — each tier has a strict tool-count budget to stay within LLM context limits.

| Tier | Server | Tools | Purpose |
|---|---|---|---|
| 1 | `ml-basic` | 11 | Dataset inspection + single-model training + prediction |
| 2 | `ml-medium` | 14 | Preprocessing pipelines + CV + clustering + EDA + batch predict |
| 3 | `ml-advanced` | 10 | Tuning + export + evaluation charts + profiling |

**Recommended loading by hardware:**

| VRAM | Recommended load | Total tools |
|---|---|---|
| 4–6 GB | ml-basic only | 11 |
| 8 GB | ml-basic + ml-medium | 25 |
| 12–16 GB | all three tiers | 35 |
| 24 GB+ | all three tiers | 35 |

---

## Tool Reference

### Tier 1 — ml-basic

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

### Tier 2 — ml-medium

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

### Tier 3 — ml-advanced

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

## Installation

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Quick Start (Local Clone)

```bash
git clone https://github.com/azzindani/MCP_Machine_Learning.git
cd MCP_Machine_Learning
uv sync
```

---

## MCP Configuration

### Option A — Local Clone (recommended)

Clone the repo first, then point your MCP client to the local directory. Replace `/path/to/MCP_Machine_Learning` with your actual clone path.

<details>
<summary><strong>All Three Servers</strong></summary>

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-basic"],
      "env": { "PYTHONPATH": "/path/to/MCP_Machine_Learning" }
    },
    "ml-medium": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-medium"],
      "env": { "PYTHONPATH": "/path/to/MCP_Machine_Learning" }
    },
    "ml-advanced": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-advanced"],
      "env": { "PYTHONPATH": "/path/to/MCP_Machine_Learning" }
    }
  }
}
```

</details>

<details>
<summary><strong>ml-basic only (4–6 GB VRAM)</strong></summary>

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-basic"],
      "env": { "PYTHONPATH": "/path/to/MCP_Machine_Learning" }
    }
  }
}
```

</details>

<details>
<summary><strong>ml-basic + ml-medium (8 GB VRAM)</strong></summary>

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-basic"],
      "env": { "PYTHONPATH": "/path/to/MCP_Machine_Learning" }
    },
    "ml-medium": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-medium"],
      "env": { "PYTHONPATH": "/path/to/MCP_Machine_Learning" }
    }
  }
}
```

</details>

### Option B — Pull from Repository (no clone needed)

Run directly from the GitHub repository. uv downloads and caches the code automatically.

<details>
<summary><strong>All Three Servers</strong></summary>

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/azzindani/MCP_Machine_Learning.git#subdirectory=servers/ml_basic",
        "ml-basic"
      ]
    },
    "ml-medium": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/azzindani/MCP_Machine_Learning.git#subdirectory=servers/ml_medium",
        "ml-medium"
      ]
    },
    "ml-advanced": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/azzindani/MCP_Machine_Learning.git#subdirectory=servers/ml_advanced",
        "ml-advanced"
      ]
    }
  }
}
```

</details>

<details>
<summary><strong>ml-basic only</strong></summary>

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/azzindani/MCP_Machine_Learning.git#subdirectory=servers/ml_basic",
        "ml-basic"
      ]
    }
  }
}
```

</details>

<details>
<summary><strong>ml-basic + ml-medium</strong></summary>

```json
{
  "mcpServers": {
    "ml-basic": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/azzindani/MCP_Machine_Learning.git#subdirectory=servers/ml_basic",
        "ml-basic"
      ]
    },
    "ml-medium": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/azzindani/MCP_Machine_Learning.git#subdirectory=servers/ml_medium",
        "ml-medium"
      ]
    }
  }
}
```

</details>

### Where to add this configuration

| Client | Config file location |
|---|---|
| **LM Studio** | `%APPDATA%\LM-Studio\mcp-config.json` (Win) / `~/.lmstudio/mcp-config.json` (Mac/Linux) |
| **Claude Desktop** | `%APPDATA%\Claude\claude_desktop_config.json` (Win) / `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) |
| **Cursor** | `~/.cursor/mcp.json` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |

### Constrained Mode (8 GB VRAM)

Add `MCP_CONSTRAINED_MODE` to the `env` block of any server config:

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/path/to/MCP_Machine_Learning", "ml-basic"],
  "env": {
    "PYTHONPATH": "/path/to/MCP_Machine_Learning",
    "MCP_CONSTRAINED_MODE": "1"
  }
}
```

In constrained mode, row limits, search results, CV folds, and model comparison counts are automatically reduced to fit within 8 GB VRAM.

---

## Example Workflows

### Classification — End to End

```
User: "Train a random forest on my churn.csv to predict the 'churned' column"

1. inspect_dataset("churn.csv")
   → 5,000 rows × 18 cols, target candidates: ["churned"]

2. read_column_profile("churn.csv", "churned")
   → bool dtype, 482 True / 4518 False, balance_ratio: 0.107

3. run_preprocessing("churn.csv", [
     {"op": "fill_nulls", "column": "tenure", "strategy": "median"},
     {"op": "label_encode", "column": "contract_type"}
   ])
   → saved churn_preprocessed.csv

4. train_classifier("churn_preprocessed.csv", "churned", "rf")
   → accuracy: 0.89, f1_weighted: 0.87
   → model_path: .mcp_models/churn_rf_2026-04-06T10-30-00Z.pkl

5. generate_training_report(".mcp_models/churn_rf_....pkl")
   → Opens browser: confusion matrix, feature importance chart
```

### Regression — With Hyperparameter Tuning

```
1. inspect_dataset("housing.csv")
2. read_column_profile("housing.csv", "price")
3. train_regressor("housing.csv", "price", "rfr")
   → R²: 0.81, RMSE: 42300

4. tune_hyperparameters("housing.csv", "price", "rfr", "regression",
                        search="grid", cv=5)
   → best_params: {n_estimators: 100, max_depth: 10}
   → best_score: 0.87

5. plot_predictions_vs_actual(".mcp_models/housing_rfr_tuned.pkl", "housing.csv")
   → Opens browser: scatter plot with R²
```

### EDA + Clustering

```
1. generate_eda_report("customers.csv", target_column="segment", theme="dark")
   → Opens browser: quality score 78/100, 3 alerts, histograms, correlations

2. find_optimal_clusters("customers.csv", ["age", "spend", "tenure"], max_k=10)
   → best_k: 4 (highest silhouette score 0.42)
   → Opens browser: elbow + silhouette chart

3. run_clustering("customers.csv", ["age", "spend", "tenure"],
                  "kmeans", n_clusters=4, save_labels=True)
   → cluster_label column added to customers.csv

4. generate_cluster_report("customers.csv", ["age", "spend", "tenure"],
                           "cluster_label", theme="dark")
   → Opens browser: PCA scatter, cluster profiles, size chart
```

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

## Version Control & Audit

Every write operation:
1. Takes a snapshot to `.mcp_versions/` before overwriting
2. Returns a `"backup"` key with the snapshot path
3. Appends an entry to the file's `.mcp_receipt.json` audit log

To view history: `read_receipt("mydata.csv")`
To rollback: `restore_version("mydata.csv", timestamp="2026-04-06T10-30-00Z")`

---

## The Four-Tool Pattern

All ML workflows follow this pattern:

```
LOCATE  →  inspect_dataset()           # What does my data look like?
INSPECT →  read_column_profile()       # What are the details of this column?
PATCH   →  train_classifier()          # Train, preprocess, or transform
VERIFY  →  generate_training_report()  # Was the result good?
```

This keeps each LLM tool call focused on a single operation, minimising token use and maximising traceability.

---

## Development

```bash
# Run tests
PYTHONPATH=. MCP_CONSTRAINED_MODE=1 uv run python -m pytest tests/ -q

# Type check
uv run pyright servers/ shared/

# Lint
uv run ruff check .
```

---

## Hardware Sovereignty

This project is built on the principle that **your data belongs to you**:

- All ML computation is local (scikit-learn, XGBoost, numpy)
- No data is sent to any cloud service
- No API keys, no subscriptions, no usage limits
- Works on air-gapped machines after first install
- Model weights are stored on your disk under `.mcp_models/`

---

## License

MIT License — see [LICENSE](LICENSE) for details.
