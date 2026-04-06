# CLAUDE.md — MCP Machine Learning Server Development Guide

This document is the authoritative development guide for AI coding agents working on
this repository. Read it completely before writing any code. All rules here are
binding. Where this file conflicts with STANDARDS.md, this file takes precedence.

**Reference:** [STANDARDS.md](./STANDARDS.md) — General-purpose MCP server standards  
**Reference:** [ML_Lookup.ipynb](./ML_Lookup.ipynb) — ML pipeline reference implementations

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [Repository Structure](#3-repository-structure)
4. [Architecture Principles](#4-architecture-principles)
5. [Three-Tier ML Server Design](#5-three-tier-ml-server-design)
6. [Tool Catalogue — ml_basic (Tier 1)](#6-tool-catalogue--ml_basic-tier-1)
7. [Tool Catalogue — ml_medium (Tier 2)](#7-tool-catalogue--ml_medium-tier-2)
8. [Tool Catalogue — ml_advanced (Tier 3)](#8-tool-catalogue--ml_advanced-tier-3)
9. [ML Pipeline Implementation Rules](#9-ml-pipeline-implementation-rules)
10. [Supported Algorithms Reference](#10-supported-algorithms-reference)
11. [Return Value Contracts for ML Tools](#11-return-value-contracts-for-ml-tools)
12. [Error Handling — ML-Specific Patterns](#12-error-handling--ml-specific-patterns)
13. [Model Persistence and Versioning](#13-model-persistence-and-versioning)
14. [Hardware and Resource Constraints](#14-hardware-and-resource-constraints)
15. [Shared Module Contracts](#15-shared-module-contracts)
16. [Testing Standards for ML](#16-testing-standards-for-ml)
17. [What the AI Must Never Do](#17-what-the-ai-must-never-do)
18. [Progress Tracker](#18-progress-tracker)

---

## 1. Project Overview

This is a **self-hosted, local-first MCP (Model Context Protocol) server project**
for machine learning. It exposes ML operations as structured tools that a local
language model calls with JSON arguments and receives JSON results from.

The ML pipeline this project covers — derived from ML_Lookup.ipynb:

- **Data inspection** — schema discovery, column profiling, row sampling
- **Data preprocessing** — null handling, outlier detection, type conversion, encoding, scaling
- **Supervised learning** — classification and regression (7+ algorithm families)
- **Unsupervised learning** — clustering (K-Means, Mean-Shift, DBSCAN) and dimensionality reduction (PCA, ICA)
- **Model evaluation** — accuracy, F1, confusion matrix, MSE, R², cross-validation
- **Hyperparameter tuning** — GridSearchCV, RandomizedSearchCV
- **Model persistence** — pickle serialization, version snapshots

All execution is **100% local**. No data leaves the user's machine. No cloud APIs.
No API keys. No subscriptions. The tools run on the user's CPU/GPU using scikit-learn,
XGBoost, pandas, and numpy.

---

## 2. Goals and Non-Goals

### Goals

- Provide a local LLM with clean, surgical ML tools that stay within a 10,000-token
  context window on 8 GB VRAM hardware
- Support the full supervised + unsupervised ML pipeline through three focused servers
- Make every tool testable without starting an MCP server process
- Snapshot model artifacts before any overwrite
- Follow the LOCATE → INSPECT → PATCH → VERIFY four-tool pattern for all ML workflows

### Non-Goals

- **Not** a cloud ML wrapper (no SageMaker, Vertex AI, Azure ML, OpenAI fine-tune)
- **Not** a deep learning training server (PyTorch training is out of scope for these tiers)
- **Not** a data visualization server (charts are paths to saved files, never raw arrays)
- **Not** a streaming/real-time inference server
- **Not** a replacement for MLflow UI (tracking is local and file-based)

---

## 3. Repository Structure

```
MCP_Machine_Learning/
│
├── shared/                             # Shared across ALL servers — never duplicate
│   ├── __init__.py
│   ├── version_control.py              # snapshot() / restore_version()
│   ├── patch_validator.py              # validate op arrays before applying
│   ├── file_utils.py                   # resolve_path(), atomic writes, JSON helpers
│   ├── platform_utils.py               # is_constrained_mode(), get_max_rows(), etc.
│   ├── progress.py                     # ok() / fail() / info() / warn() / undo()
│   └── receipt.py                      # append_receipt() / read_receipt_log()
│
├── servers/
│   ├── ml_basic/                       # Tier 1 — dataset inspection + single-model training
│   │   ├── __init__.py
│   │   ├── server.py                   # FastMCP tool definitions (thin wrappers only)
│   │   ├── engine.py                   # Pure ML logic (zero MCP imports)
│   │   └── pyproject.toml
│   │
│   ├── ml_medium/                      # Tier 2 — preprocessing pipeline + CV + multi-model
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── engine.py
│   │   └── pyproject.toml
│   │
│   └── ml_advanced/                    # Tier 3 — tuning, export, evaluation reports
│       ├── __init__.py
│       ├── server.py
│       ├── engine.py
│       └── pyproject.toml
│
├── tests/
│   ├── fixtures/
│   │   ├── classification_simple.csv   # clean binary classification dataset
│   │   ├── classification_messy.csv    # nulls, mixed types, class imbalance
│   │   ├── regression_simple.csv       # clean continuous target dataset
│   │   ├── regression_messy.csv        # outliers, skewed distributions
│   │   ├── clustering_simple.csv       # 2D points with clear cluster structure
│   │   └── large_10k.csv               # 10,000 rows for truncation tests
│   ├── conftest.py
│   ├── test_ml_basic.py
│   ├── test_ml_medium.py
│   └── test_ml_advanced.py
│
├── install/
│   ├── install.sh                      # Linux/macOS — POSIX sh compatible
│   ├── install.bat                     # Windows CMD
│   └── mcp_config_writer.py            # writes to LM Studio / Claude Desktop / Cursor
│
├── pyproject.toml                      # root workspace
├── uv.lock
├── .python-version                     # 3.11
├── .gitattributes
├── CLAUDE.md                           # this file
├── STANDARDS.md                        # general MCP server standards
└── README.md
```

### File ownership rules

- `engine.py` owns all ML logic: data loading, preprocessing, model training,
  evaluation, serialization. Zero MCP imports.
- `server.py` owns all MCP protocol concerns: `@mcp.tool()` decorators, `FastMCP`
  setup, `main()` entry point. Zero domain logic — every tool body is one line
  calling `engine.py`.
- `shared/` is imported by both. Never duplicate shared logic in engine files.


---

## 4. Architecture Principles

These are the non-negotiable rules. Every line of code written for this project must
comply with all of them.

### 4.1 Self-Hosted Execution Principle

Every tool must complete its primary operation with the machine **disconnected from
the internet**. No scikit-learn tool calls an external API. No XGBoost tool sends
data to a cloud endpoint. Model weight downloads (first-run only) are the only
permitted network operations, and they must be cached locally and documented.

**The test:** Disconnect from the network. Run the tool. It must succeed.

### 4.2 Engine / Server Split

```
server.py   →   @mcp.tool() wrappers only, one-liners calling engine.py
engine.py   →   all ML logic, zero MCP imports
```

If a tool function in `server.py` has more than **two lines**, the excess logic
belongs in `engine.py`. No exceptions.

```python
# server.py — correct
@mcp.tool()
def train_classifier(file_path: str, target_column: str, model: str,
                     test_size: float = 0.2, dry_run: bool = False) -> dict:
    """Train classifier on CSV. model: lr svm rf dtc knn nb xgb."""
    return engine.train_classifier(file_path, target_column, model, test_size, dry_run)
```

### 4.3 The Four-Tool Pattern for ML

All ML workflows follow this exact loop:

```
LOCATE   →  inspect_dataset()        # schema, row count, column names, dtypes
INSPECT  →  read_column_profile()    # detailed stats for one column
PATCH    →  train_classifier()       # train model, return metrics + model_path
VERIFY   →  read_model_report()      # confusion matrix, feature importance
```

Never combine LOCATE + INSPECT in one call. Never combine training + evaluation +
export in one call. Each step is a separate tool.

### 4.4 Surgical Read Protocol

- `inspect_dataset` returns column names + dtypes + row count — never actual row data
- `read_column_profile` returns stats for **one column** — never all columns at once
- `search_columns` returns column names matching a condition — never the column data
- `read_rows` returns a bounded slice — hard limit enforced in engine via `get_max_rows()`
- Model weights are **never returned** — return `model_path` (string) and a metrics summary

### 4.5 Snapshot Before Write

Every tool that writes to disk calls `shared.version_control.snapshot()` before
writing. This includes:
- Tools that save trained models (overwriting an existing `model.pkl`)
- Tools that modify dataset files in place
- Tools that write preprocessed datasets

The `"backup"` key must appear in every write tool's success response.

### 4.6 Token Budget Discipline

The target hardware is 8 GB VRAM with a 9B model. Effective context: ~10,000–12,000
tokens. Tool schemas + history consume ~2,000–3,000 tokens minimum.

Rules:
- Tool docstrings ≤ 80 characters (enforced by CI)
- Read tool responses ≤ 500 tokens
- Write confirmations ≤ 150 tokens
- Never return raw DataFrames, weight arrays, or confusion matrix arrays as nested lists
- Return `token_estimate = len(str(response)) // 4` in every response

### 4.7 Hardware Mode Flag

Read `MCP_CONSTRAINED_MODE` env var via `shared.platform_utils` helpers. Never
hardcode row/result limits:

```python
from shared.platform_utils import get_max_rows, get_max_results, is_constrained_mode

# In engine functions:
max_rows = get_max_rows()        # 20 in constrained, 100 in standard
max_results = get_max_results()  # 10 in constrained, 50 in standard
```


---

## 5. Three-Tier ML Server Design

### Tier 1 — ml_basic

**Purpose:** Dataset inspection + single-model training + prediction.
A user doing basic classification or regression should never need tier 2 or 3.

**Tool count target: 8 tools**

| # | Tool name | Category | Description |
|---|---|---|---|
| 1 | `inspect_dataset` | LOCATE | Schema, row count, dtypes, null summary |
| 2 | `read_column_profile` | INSPECT | Stats for one column (mean, std, nulls, unique) |
| 3 | `search_columns` | LOCATE | Find columns matching a condition |
| 4 | `read_rows` | INSPECT | Bounded row slice (respects get_max_rows) |
| 5 | `train_classifier` | PATCH | Train classification model, return metrics + path |
| 6 | `train_regressor` | PATCH | Train regression model, return metrics + path |
| 7 | `get_predictions` | VERIFY | Run predictions with saved model on new data |
| 8 | `restore_version` | CONTROL | Rollback model or dataset to previous snapshot |

**Tier 1 must stand alone.** No cross-tier imports.

---

### Tier 2 — ml_medium

**Purpose:** Preprocessing pipelines + cross-validation training + model comparison +
clustering. Loaded alongside tier 1; combined total must not exceed 15 tools.

**Tool count target: 6 tools**

| # | Tool name | Category | Description |
|---|---|---|---|
| 1 | `run_preprocessing` | PATCH | Encode + scale + fill nulls pipeline |
| 2 | `detect_outliers` | INSPECT | IQR and std-dev outlier report per column |
| 3 | `train_with_cv` | PATCH | Train with K-fold cross-validation |
| 4 | `compare_models` | PATCH | Train multiple algorithms, return sorted metrics |
| 5 | `run_clustering` | PATCH | K-Means / Mean-Shift / DBSCAN clustering |
| 6 | `read_receipt` | CONTROL | Read operation history for a file |

---

### Tier 3 — ml_advanced

**Purpose:** Hyperparameter tuning, model export, evaluation reports. Load standalone
in dedicated sessions — these tools require significant compute and context.

**Tool count target: 5 tools**

| # | Tool name | Category | Description |
|---|---|---|---|
| 1 | `tune_hyperparameters` | OPTIMIZE | GridSearch or RandomSearch tuning |
| 2 | `export_model` | EXPORT | Pickle export with metadata manifest |
| 3 | `read_model_report` | VERIFY | Feature importance, confusion matrix, metrics |
| 4 | `run_profiling_report` | ANALYZE | ydata-profiling HTML report for dataset |
| 5 | `apply_dimensionality_reduction` | TRANSFORM | PCA or ICA, return reduced dataset path |


---

## 6. Tool Catalogue — ml_basic (Tier 1)

### 6.1 inspect_dataset

```python
@mcp.tool()
def inspect_dataset(file_path: str) -> dict:
    """Inspect dataset schema, row count, dtypes, null summary."""
```

**Engine behaviour:**
- Load CSV with pandas (no full data in memory beyond header scan for large files)
- Return: `columns` (list of `{name, dtype, null_count, null_pct}`), `row_count`,
  `file_size_kb`, candidate target columns (columns with ≤ 20 unique values or dtype bool)
- Enforce: never return actual row data
- Constrained mode: column list capped at `get_max_results()` with truncation flag

**Return skeleton:**
```python
{
    "success": True,
    "op": "inspect_dataset",
    "file": "customer_churn.csv",
    "row_count": 5000,
    "column_count": 18,
    "columns": [
        {"name": "churned", "dtype": "bool", "null_count": 0, "null_pct": 0.0},
        ...
    ],
    "target_candidates": ["churned", "active"],
    "progress": [...],
    "token_estimate": 210,
    "truncated": False
}
```

---

### 6.2 read_column_profile

```python
@mcp.tool()
def read_column_profile(file_path: str, column_name: str) -> dict:
    """Profile one column. Returns stats, null count, top values."""
```

**Engine behaviour:**
- Numeric: mean, std, min, max, median, 25th/75th percentile, null_count, skewness
- Categorical: unique_count, top_values (up to 10), null_count, mode
- Boolean: true_count, false_count, null_count, balance_ratio
- Never return raw value arrays — always aggregated stats

---

### 6.3 search_columns

```python
@mcp.tool()
def search_columns(
    file_path: str,
    has_nulls: bool = False,
    dtype: str = "",          # "numeric", "categorical", "bool", "datetime"
    name_contains: str = "",
    max_results: int = 20,
) -> dict:
    """Search columns by condition. Returns names only, no data."""
```

**Engine behaviour:**
- Returns list of column names matching the given filters
- Never returns column data — addresses only

---

### 6.4 read_rows

```python
@mcp.tool()
def read_rows(file_path: str, start: int, end: int) -> dict:
    """Read bounded row slice. Max rows enforced by hardware mode."""
```

**Engine behaviour:**
- Hard limit: `min(end - start, get_max_rows())`
- If requested range exceeds limit, return limit rows + `"truncated": True`
- Return rows as list of dicts (one dict per row)

---

### 6.5 train_classifier

```python
@mcp.tool()
def train_classifier(
    file_path: str,
    target_column: str,
    model: str,               # "lr" "svm" "rf" "dtc" "knn" "nb" "xgb"
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train classifier on CSV. model: lr svm rf dtc knn nb xgb."""
```

**Engine behaviour:**
1. Resolve path, validate file exists and is `.csv`
2. Check available RAM via `psutil` — fail fast if insufficient
3. Snapshot any existing model file before overwriting
4. Load data, auto-encode categoricals (LabelEncoder), drop rows with null target
5. Split: `train_test_split(test_size=test_size, random_state=random_state, stratify=y)`
6. Train selected algorithm (see section 10 for parameters)
7. Evaluate: accuracy, F1 (weighted), confusion matrix (as counts dict, not raw array)
8. Save model via pickle to `.mcp_models/{file_stem}_{model}_{timestamp}.pkl`
9. Append receipt, return success dict

**`dry_run=True`:** Skip steps 6–9, return what would be done.

---

### 6.6 train_regressor

```python
@mcp.tool()
def train_regressor(
    file_path: str,
    target_column: str,
    model: str,               # "lir" "pr" "lar" "rr" "dtr" "rfr" "xgb"
    degree: int = 5,          # polynomial degree (only used when model="pr")
    alpha: float = 0.01,      # regularization (lasso/ridge only)
    n_estimators: int = 10,   # tree ensemble size
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train regressor on CSV. model: lir pr lar rr dtr rfr xgb."""
```

**Engine behaviour:** Same pipeline as classifier. Metrics: MSE, RMSE, R² (no confusion matrix).

---

### 6.7 get_predictions

```python
@mcp.tool()
def get_predictions(
    model_path: str,
    file_path: str,
    max_rows: int = 20,
) -> dict:
    """Run predictions with saved model. Returns bounded prediction list."""
```

**Engine behaviour:**
- Load model from pickle
- Load data, apply same encoding as training (stored in model metadata)
- Return predictions capped at `min(max_rows, get_max_rows())`
- Never return the full prediction array — always bounded

---

### 6.8 restore_version

```python
@mcp.tool()
def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file/model to previous snapshot. Empty timestamp = latest."""
```

**Engine behaviour:**
- Lists available snapshots if `timestamp` is empty (returns list, no restoration)
- Restores from `.mcp_versions/` backup matching timestamp
- Delegates to `shared.version_control.restore_version()`


---

## 7. Tool Catalogue — ml_medium (Tier 2)

### 7.1 run_preprocessing

```python
@mcp.tool()
def run_preprocessing(
    file_path: str,
    ops: list[dict],
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Apply preprocessing pipeline ops to dataset. Snapshot before write."""
```

**Supported ops (patch protocol format):**

```python
[
    {"op": "fill_nulls",     "column": "revenue",   "strategy": "median"},
    {"op": "fill_nulls",     "column": "region",    "strategy": "mode"},
    {"op": "drop_outliers",  "column": "age",       "method": "iqr",  "replace": False},
    {"op": "label_encode",   "column": "gender"},
    {"op": "onehot_encode",  "column": "region"},
    {"op": "scale",          "columns": ["age", "salary"], "method": "standard"},
    {"op": "scale",          "columns": ["price"],          "method": "minmax"},
    {"op": "drop_duplicates","subset": ["customer_id"]},
    {"op": "drop_column",    "column": "id"},
    {"op": "rename_column",  "from": "rev",         "to": "revenue_usd"},
    {"op": "convert_dtype",  "column": "date",      "to": "datetime"},
]
```

Allowed `fill_nulls` strategies: `mean`, `median`, `mode`, `ffill`, `bfill`, `zero`  
Allowed `scale` methods: `standard` (StandardScaler), `minmax` (MinMaxScaler)  
Max ops per batch: 50. Validate entire array before applying any operation.

---

### 7.2 detect_outliers

```python
@mcp.tool()
def detect_outliers(
    file_path: str,
    columns: list[str],
    method: str = "iqr",      # "iqr" or "std"
    th1: float = 0.25,        # IQR lower quantile
    th3: float = 0.75,        # IQR upper quantile
) -> dict:
    """Detect outliers in numeric columns. method: iqr std."""
```

**Engine behaviour:**
- IQR method: lower = Q1 - 1.5×IQR, upper = Q3 + 1.5×IQR (using `th1`/`th3` quantiles)
- Std method: mean ± 3σ thresholds
- Returns per-column report: `outlier_count`, `lower_bound`, `upper_bound`, `sample_outliers`
  (up to 5 example values)
- Never returns the full outlier row list — counts and bounds only

---

### 7.3 train_with_cv

```python
@mcp.tool()
def train_with_cv(
    file_path: str,
    target_column: str,
    model: str,
    task: str,                # "classification" or "regression"
    n_splits: int = 5,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train with K-fold cross-validation. Returns per-fold and mean scores."""
```

**Engine behaviour:**
- Classification: StratifiedKFold with `n_splits`; reports accuracy and F1 per fold
- Regression: KFold with `n_splits`; reports R² and RMSE per fold
- Returns mean ± std across folds, plus per-fold scores array
- Saves best-fold model to `.mcp_models/`

---

### 7.4 compare_models

```python
@mcp.tool()
def compare_models(
    file_path: str,
    target_column: str,
    task: str,                # "classification" or "regression"
    models: list[str],        # e.g. ["lr", "rf", "xgb"]
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train multiple models, return sorted comparison table."""
```

**Engine behaviour:**
- Train each model sequentially on the same train/test split
- Classification ranking: F1 weighted (primary), accuracy (secondary)
- Regression ranking: R² (primary), RMSE (secondary)
- Return sorted results table (list of dicts, one per model)
- Max models per call: 7 (all supported algorithm families)
- Save **only the best model** to disk — do not save all variants

---

### 7.5 run_clustering

```python
@mcp.tool()
def run_clustering(
    file_path: str,
    feature_columns: list[str],
    algorithm: str,           # "kmeans" "meanshift" "dbscan"
    n_clusters: int = 3,      # kmeans only
    eps: float = 3.0,         # dbscan only
    min_samples: int = 5,     # dbscan only
    reduce_dims: str = "",    # "pca" or "ica" or "" (none)
    n_components: int = 2,
    save_labels: bool = False,
    dry_run: bool = False,
) -> dict:
    """Cluster dataset. algorithm: kmeans meanshift dbscan."""
```

**Engine behaviour:**
- Scale features with StandardScaler before clustering
- Optionally reduce dimensions with PCA or FastICA before clustering
- Return: cluster label counts, inertia (K-Means), n_clusters found (Mean-Shift/DBSCAN)
- If `save_labels=True`, append cluster label column to dataset and snapshot

---

### 7.6 read_receipt

```python
@mcp.tool()
def read_receipt(file_path: str) -> dict:
    """Read operation history for a file. Returns log entries."""
```

Delegates to `shared.receipt.read_receipt_log()`.


---

## 8. Tool Catalogue — ml_advanced (Tier 3)

### 8.1 tune_hyperparameters

```python
@mcp.tool()
def tune_hyperparameters(
    file_path: str,
    target_column: str,
    model: str,
    task: str,                # "classification" or "regression"
    search: str = "grid",     # "grid" or "random"
    param_grid: str = "",     # JSON string of param grid, or "" for defaults
    cv: int = 5,
    n_iter: int = 10,         # random search only
    dry_run: bool = False,
) -> dict:
    """Tune hyperparameters via grid or random search. search: grid random."""
```

**Engine behaviour:**
- `param_grid` is a JSON string parsed inside engine (not a `dict` parameter — see STANDARDS §10)
- Default param grids per model are defined as constants in `engine.py`
- Return: `best_score`, `best_params`, top-5 results table (sorted by score)
- Save best model as snapshot
- Cap `cv_results_` to top 20 rows — never return full grid results

**Default param grids (constants in engine.py):**
```python
DEFAULT_PARAMS = {
    "svm":  {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "rf":   {"n_estimators": [10, 50, 100], "max_depth": [None, 5, 10]},
    "xgb":  {"max_depth": [3, 5, 7], "eta": [0.1, 0.3], "n_estimators": [50, 100]},
    "knn":  {"n_neighbors": [3, 5, 7, 11]},
    "lr":   {"C": [0.01, 0.1, 1, 10]},
}
```

---

### 8.2 export_model

```python
@mcp.tool()
def export_model(
    model_path: str,
    output_dir: str = "",
    format: str = "pickle",   # "pickle" only for v1; "onnx" reserved for future
    dry_run: bool = False,
) -> dict:
    """Export trained model with metadata manifest. format: pickle."""
```

**Engine behaviour:**
- Copy model `.pkl` to `output_dir` (default: same directory)
- Write companion `{model_name}.manifest.json` with: model type, training date,
  feature columns, target column, training metrics, Python + library versions
- Snapshot before overwrite if output file already exists

**Manifest schema:**
```json
{
    "model_type": "RandomForestClassifier",
    "task": "classification",
    "trained_on": "customer_churn.csv",
    "training_date": "2026-04-06T10:30:00Z",
    "feature_columns": ["age", "tenure", "monthly_charges"],
    "target_column": "churned",
    "metrics": {"accuracy": 0.89, "f1_weighted": 0.87},
    "python_version": "3.11.x",
    "sklearn_version": "1.4.x",
    "xgboost_version": "2.x.x"
}
```

---

### 8.3 read_model_report

```python
@mcp.tool()
def read_model_report(model_path: str) -> dict:
    """Read model metrics, feature importance, confusion matrix summary."""
```

**Engine behaviour:**
- Load model and its companion manifest
- Classification: return confusion matrix as `{"TP": n, "FP": n, "FN": n, "TN": n}`
  for binary, or per-class counts for multiclass (max 10 classes shown)
- Return feature importances as top-10 list `[{feature, importance}]` (tree models only)
- Return full classification report as text (sklearn format, bounded to 500 chars)
- Never return raw weight matrices or full prediction arrays

---

### 8.4 run_profiling_report

```python
@mcp.tool()
def run_profiling_report(
    file_path: str,
    output_path: str = "",
    sample_rows: int = 0,     # 0 = use all rows; >0 = sample for speed
    dry_run: bool = False,
) -> dict:
    """Generate ydata-profiling HTML report for dataset."""
```

**Engine behaviour:**
- Uses `ydata_profiling.ProfileReport` with `minimal=True` for constrained mode
- Saves HTML to `output_path` (default: same dir as CSV, `.html` extension)
- Returns: output path, file size, key stats summary (row count, column count,
  missing cells pct, duplicate rows pct)
- Never return raw profile data — return the path

---

### 8.5 apply_dimensionality_reduction

```python
@mcp.tool()
def apply_dimensionality_reduction(
    file_path: str,
    feature_columns: list[str],
    method: str,              # "pca" or "ica"
    n_components: int = 2,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Reduce dimensions with PCA or ICA. Saves reduced dataset."""
```

**Engine behaviour:**
- Scale features with StandardScaler before reduction
- PCA: use `sklearn.decomposition.PCA`; return `explained_variance_ratio_` per component
- ICA: use `sklearn.decomposition.FastICA`
- Save reduced dataset as new CSV (original columns replaced with `component_1`, `component_2`, etc.)
- Return: output path, variance explained (PCA only), n_components used


---

## 9. ML Pipeline Implementation Rules

### 9.1 Data Loading

Always use `pandas.read_csv()`. Do not use `polars` for this project — the ML
libraries expect pandas DataFrames or numpy arrays.

```python
df = pd.read_csv(path, low_memory=False)
```

After loading, immediately call `check_memory()` to verify available RAM before
any heavy computation:

```python
required_gb = (df.memory_usage(deep=True).sum() / 1e9) * 3  # 3× for transforms
mem_check = check_memory(required_gb)
if mem_check:
    return mem_check  # early return with error dict
```

### 9.2 Automatic Preprocessing Before Training

When `train_classifier` or `train_regressor` is called, the engine **automatically**
applies these steps in order (without the user needing to call `run_preprocessing`):

1. Drop rows where `target_column` is null
2. LabelEncode all object/category dtype columns (excluding target for regression)
3. Fill remaining numeric nulls with column median
4. Store encoding mapping in model metadata for use in `get_predictions`

Do not scale features automatically in the basic tier — let the user call
`run_preprocessing` with `scale` op if needed. Exception: KNN and SVM always
scale internally within the engine before fitting, then inverse-transform is not
needed since we only return predictions.

### 9.3 Train/Test Split

Always use `sklearn.model_selection.train_test_split`:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=test_size,          # default 0.2
    random_state=random_state,    # default 42
    stratify=y if task == "classification" else None,
)
```

Always pass `stratify=y` for classification to preserve class balance across splits.

### 9.4 Minimum Dataset Size Guard

Before training, validate dataset has enough rows:

```python
MIN_ROWS_CLASSIFIER = 20
MIN_ROWS_REGRESSOR = 10

if len(df) < MIN_ROWS_CLASSIFIER:
    return {
        "success": False,
        "error": f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_CLASSIFIER}.",
        "hint": "Provide a dataset with more samples before training.",
    }
```

Also validate target column has at least 2 unique values for classification.

### 9.5 Model Saving Convention

```
.mcp_models/
    {file_stem}_{model}_{YYYY-MM-DDTHH-MM-SSZ}.pkl
    {file_stem}_{model}_{YYYY-MM-DDTHH-MM-SSZ}.manifest.json
```

Save both files atomically. The `.manifest.json` companion is required alongside
every `.pkl`. Never save a model without its manifest.

Use `shared.version_control.snapshot()` before overwriting an existing model file.

### 9.6 Score Formatting

Return scores as floats in [0, 1] range. Do not format as percentage strings
(the ML_Lookup.ipynb pattern of `str(round(score * 100, 2)) + '%'` is for human
display, not for MCP tool returns).

```python
# Correct — return raw float
{"accuracy": 0.89, "f1_weighted": 0.87}

# Wrong — string percentage
{"accuracy": "89.0%"}
```

### 9.7 Confusion Matrix Format

Return confusion matrices as a summary dict, never as a raw 2D array:

```python
# Binary classification
{"TP": 120, "FP": 15, "FN": 10, "TN": 355}

# Multiclass (up to 10 classes)
{"class_0": {"precision": 0.91, "recall": 0.88, "f1": 0.89, "support": 200}, ...}
```

### 9.8 XGBoost Usage Pattern

Follow the ML_Lookup.ipynb patterns exactly:

```python
import xgboost as xgb

# Classification
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest  = xgb.DMatrix(x_test,  label=y_test)
params = {
    "max_depth": 3, "eta": 0.3, "silent": 1,
    "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
    "num_class": n_classes,
}
model = xgb.train(params, dtrain, num_boost_round=num_round)
preds = model.predict(dtest)

# Multiclass: argmax over probability columns
if n_classes > 2:
    y_predicted = np.asarray([np.argmax(line) for line in preds])
else:
    y_predicted = (preds > 0.5).astype(int)
```


---

## 10. Supported Algorithms Reference

This table maps the short `model` parameter string to the scikit-learn / XGBoost
class. Use these exact classes and default parameters unless overridden by the user.

### Classification Algorithms

| model string | Class | Default key params |
|---|---|---|
| `"lr"` | `LogisticRegression` | `random_state=42, max_iter=200` |
| `"svm"` | `SVC` | `kernel='rbf', gamma='auto', random_state=42` |
| `"rf"` | `RandomForestClassifier` | `n_estimators=100, random_state=42` |
| `"dtc"` | `DecisionTreeClassifier` | `random_state=42` |
| `"knn"` | `KNeighborsClassifier` | `n_neighbors=5, metric='minkowski', p=2` |
| `"nb"` | `GaussianNB` | *(no key params)* |
| `"xgb"` | `xgb.train()` | `max_depth=3, eta=0.3, num_round=10` |

### Regression Algorithms

| model string | Class | Default key params |
|---|---|---|
| `"lir"` | `LinearRegression` | *(no key params)* |
| `"pr"` | `LinearRegression` + `PolynomialFeatures` | `degree=5` |
| `"lar"` | `Lasso` | `alpha=0.01, max_iter=200, tol=0.1` |
| `"rr"` | `Ridge` | `alpha=0.01, max_iter=100, tol=0.1` |
| `"dtr"` | `DecisionTreeRegressor` | `random_state=42` |
| `"rfr"` | `RandomForestRegressor` | `n_estimators=10, random_state=42` |
| `"xgb"` | `xgb.train()` | `objective='reg:squarederror', num_round=5` |

### Clustering Algorithms

| algorithm string | Class | Default key params |
|---|---|---|
| `"kmeans"` | `KMeans` | `n_clusters=3, max_iter=100, random_state=42` |
| `"meanshift"` | `MeanShift` | *(bandwidth auto-estimated)* |
| `"dbscan"` | `DBSCAN` | `eps=3.0, min_samples=5` |

### Dimensionality Reduction

| method string | Class | Default key params |
|---|---|---|
| `"pca"` | `PCA` | `n_components=2` |
| `"ica"` | `FastICA` | `n_components=2` |

### Hyperparameter Tuning

| search string | Class | Key params |
|---|---|---|
| `"grid"` | `GridSearchCV` | `cv=5, return_train_score=False` |
| `"random"` | `RandomizedSearchCV` | `cv=5, n_iter=10, random_state=42` |

### Evaluation Metrics — by Task

**Classification:** `accuracy_score`, `f1_score(average='weighted')`,
`classification_report`, `confusion_matrix`

**Regression:** `mean_squared_error`, `r2_score`
(RMSE = `np.sqrt(mean_squared_error(...))`)

**Cross-validation:** `cross_val_score` with `scoring='accuracy'` (classification)
or `scoring='r2'` (regression)


---

## 11. Return Value Contracts for ML Tools

### 11.1 Required Fields in Every Response

Every tool returns a `dict`. No exceptions.

| Field | Type | When required |
|---|---|---|
| `"success"` | `bool` | Always — first key |
| `"op"` | `str` | On success — names the operation |
| `"error"` | `str` | On failure — human-readable reason |
| `"hint"` | `str` | On failure — actionable recovery step |
| `"backup"` | `str` | After any write — path to snapshot |
| `"progress"` | `list[dict]` | Always — step-by-step log |
| `"token_estimate"` | `int` | Always — `len(str(response)) // 4` |
| `"truncated"` | `bool` | On bounded reads — explicit, never absent |
| `"dry_run"` | `bool` | When `dry_run=True` — confirms simulation |

### 11.2 Train Tool Success Response

```python
{
    "success": True,
    "op": "train_classifier",
    "model": "rf",
    "model_class": "RandomForestClassifier",
    "task": "classification",
    "target_column": "churned",
    "feature_columns": ["age", "tenure", "monthly_charges"],
    "row_count": 5000,
    "train_size": 4000,
    "test_size": 1000,
    "metrics": {
        "accuracy": 0.89,
        "f1_weighted": 0.87,
        "confusion_matrix": {"TP": 198, "FP": 22, "FN": 18, "TN": 762},
    },
    "model_path": ".mcp_models/customer_churn_rf_2026-04-06T10-30-00Z.pkl",
    "backup": ".mcp_versions/customer_churn_rf_prev.pkl.bak",
    "progress": [
        {"icon": "✔", "msg": "Loaded customer_churn.csv", "detail": "5,000 rows × 18 cols"},
        {"icon": "✔", "msg": "Encoded 3 categorical columns", "detail": "LabelEncoder"},
        {"icon": "✔", "msg": "Split dataset", "detail": "4,000 train / 1,000 test (stratified)"},
        {"icon": "✔", "msg": "Trained RandomForestClassifier", "detail": "n_estimators=100"},
        {"icon": "✔", "msg": "Evaluated model", "detail": "accuracy=0.89, f1=0.87"},
        {"icon": "✔", "msg": "Saved model", "detail": ".mcp_models/customer_churn_rf_...pkl"},
    ],
    "token_estimate": 185
}
```

### 11.3 Regression Success Response

```python
{
    "success": True,
    "op": "train_regressor",
    "model": "rfr",
    "metrics": {
        "mse": 142500.0,
        "rmse": 377.5,
        "r2": 0.84,
    },
    "model_path": "...",
    "backup": "...",
    "progress": [...],
    "token_estimate": 140
}
```

### 11.4 Compare Models Response

```python
{
    "success": True,
    "op": "compare_models",
    "task": "classification",
    "results": [
        {"model": "xgb", "accuracy": 0.91, "f1_weighted": 0.90, "rank": 1},
        {"model": "rf",  "accuracy": 0.89, "f1_weighted": 0.87, "rank": 2},
        {"model": "lr",  "accuracy": 0.81, "f1_weighted": 0.80, "rank": 3},
    ],
    "best_model": "xgb",
    "best_model_path": "...",
    "backup": "...",
    "progress": [...],
    "token_estimate": 165
}
```

### 11.5 Preprocessing Response

```python
{
    "success": True,
    "op": "run_preprocessing",
    "applied": 4,
    "ops_summary": [
        {"op": "fill_nulls", "column": "revenue", "filled": 23},
        {"op": "label_encode", "column": "gender", "classes": ["F", "M"]},
        {"op": "scale", "columns": ["age", "salary"], "method": "standard"},
        {"op": "drop_duplicates", "removed": 12},
    ],
    "output_path": "customer_churn_preprocessed.csv",
    "backup": "...",
    "progress": [...],
    "token_estimate": 175
}
```


---

## 12. Error Handling — ML-Specific Patterns

All exceptions are caught in `engine.py` and returned as error dicts. Never raise
to the MCP layer. Use the standard patterns below.

### 12.1 File Errors

```python
f"File not found: {file_path}"
# hint: "Check that file_path is absolute and the CSV file exists."

f"Expected .csv file, got .{ext}"
# hint: "Provide a CSV file path. Use inspect_dataset() to verify the file."
```

### 12.2 Column Errors

```python
f"Column '{name}' not found. Available: {', '.join(columns[:10])}"
# hint: "Use inspect_dataset() to list all column names."

f"Target column '{name}' has only 1 unique value — cannot train classifier."
# hint: "Choose a column with at least 2 distinct class values."

f"Target column '{name}' has {n} unique values. For regression use train_regressor()."
# hint: "Use task='regression' or choose a binary/categorical target column."
```

### 12.3 Resource Errors

```python
f"Insufficient RAM: need ~{required_gb:.1f} GB, available ~{available_gb:.1f} GB."
# hint: "Use read_rows() to sample a subset, or increase system RAM."

f"GPU not available. Set device='cpu' to run on CPU."
# hint: "XGBoost will fall back to CPU automatically if tree_method is not set to 'gpu_hist'."
```

### 12.4 Data Quality Errors

```python
f"Dataset has {rows} rows but need at least {min_rows} to train reliably."
# hint: "Provide a dataset with more samples before training."

f"Column '{name}' is non-numeric. Encode it first with run_preprocessing()."
# hint: "Use op 'label_encode' or 'onehot_encode' in run_preprocessing() first."

f"All values in column '{name}' are null — cannot use as feature."
# hint: "Drop this column or fill nulls with run_preprocessing() fill_nulls op."
```

### 12.5 Model Errors

```python
f"Model file not found: {model_path}"
# hint: "Use train_classifier() or train_regressor() to train a model first."

f"Model type mismatch: loaded '{loaded_task}' model, expected '{task}'."
# hint: "Check model_path points to the correct model type."

f"Unknown algorithm: '{model}'. Allowed: {', '.join(ALLOWED_CLASSIFIERS)}"
# hint: "Use one of: lr svm rf dtc knn nb xgb"
```

### 12.6 Preprocessing Op Errors

```python
f"Unknown op: '{op}'. Allowed: {', '.join(ALLOWED_OPS)}"
# hint: "Check the op name spelling. Use run_preprocessing() docstring for valid ops."

f"Op '{op}' missing required field: '{field}'"
# hint: f"Add '{field}' key to the op dict."

f"Strategy '{strategy}' not valid for fill_nulls. Allowed: mean median mode ffill bfill zero"
```

### 12.7 Error Dict Template

```python
def _error(error: str, hint: str, backup: str | None = None) -> dict:
    base = {"success": False, "error": error, "hint": hint}
    if backup:
        base["backup"] = backup
    base["token_estimate"] = len(str(base)) // 4
    return base
```


---

## 13. Model Persistence and Versioning

### 13.1 Model Storage Layout

```
{dataset_dir}/
├── customer_churn.csv
├── customer_churn.csv.mcp_state.json       # companion state
├── customer_churn.csv.mcp_receipt.json     # operation receipt log
│
├── .mcp_models/
│   ├── customer_churn_rf_2026-04-06T10-30-00Z.pkl
│   ├── customer_churn_rf_2026-04-06T10-30-00Z.manifest.json
│   └── customer_churn_xgb_2026-04-06T11-00-00Z.pkl
│
└── .mcp_versions/
    ├── customer_churn_2026-04-06T09-00-00Z.csv.bak
    └── customer_churn_rf_2026-04-05T14-00-00Z.pkl.bak
```

### 13.2 Pickle Saving Pattern

```python
import pickle, tempfile, shutil
from pathlib import Path

def _save_model(model, path: Path, metadata: dict) -> None:
    payload = {"model": model, "metadata": metadata}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl",
                                    dir=path.parent) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    # Write manifest alongside
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(metadata, indent=2))
```

### 13.3 Model Metadata (stored in pkl payload AND manifest.json)

```python
metadata = {
    "model_type": type(model).__name__,
    "task": task,                          # "classification" or "regression"
    "trained_on": Path(file_path).name,
    "training_date": datetime.now(timezone.utc).isoformat(),
    "feature_columns": feature_columns,
    "target_column": target_column,
    "encoding_map": encoding_map,          # {column_name: {original: encoded}}
    "scaler": scaler_obj_or_None,          # embedded scaler if used
    "metrics": metrics_dict,
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
}
```

### 13.4 Loading Models for Prediction

```python
def _load_model(model_path: str) -> tuple[object, dict]:
    path = resolve_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["metadata"]
```

---

## 14. Hardware and Resource Constraints

### 14.1 RAM Check Before Heavy Operations

Always check before loading large datasets or training memory-intensive models:

```python
import psutil

def check_memory(required_gb: float) -> dict | None:
    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < required_gb:
        return {
            "success": False,
            "error": f"Need ~{required_gb:.1f} GB RAM, only {available_gb:.1f} GB available.",
            "hint": "Use read_rows() with a row limit or increase available memory.",
            "token_estimate": 60,
        }
    return None
```

### 14.2 Constrained Mode Limits

Set `MCP_CONSTRAINED_MODE=1` on machines with ≤ 8 GB VRAM. The installer sets
this automatically.

| Resource | Standard | Constrained (≤8 GB VRAM) |
|---|---|---|
| Max rows returned per call | 100 | 20 |
| Max search results | 50 | 10 |
| Max JSON depth | 5 levels | 3 levels |
| Max columns returned | 50 | 20 |
| ydata-profiling mode | full | minimal=True |
| compare_models max | 7 | 3 |
| tune_hyperparameters cv | 5 | 3 |
| tune_hyperparameters n_iter | 10 | 5 |

### 14.3 ML Hardware Reference

| VRAM | Recommended model | Max simultaneous tools | Tier recommendation |
|---|---|---|---|
| 4–6 GB | 3–7B models | 6 | ml_basic only |
| 8 GB | 9B models (Q3/Q4) | 12 | ml_basic + ml_medium |
| 12–16 GB | 14B models | 16 | all three tiers |
| 24 GB+ | 32B models | 20 | all three tiers |

Do not load ml_advanced alongside ml_basic + ml_medium on 8 GB VRAM.


---

## 15. Shared Module Contracts

Every shared module must be implemented exactly as specified. Do not modify
interfaces — add new functions instead.

### shared/version_control.py

```python
def snapshot(file_path: str) -> str:
    """Snapshot file to .mcp_versions/. Returns backup path. Raises if source missing."""

def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore from snapshot. Empty timestamp = list available snapshots."""

def list_snapshots(file_path: str) -> list[dict]:
    """List available snapshots for file. Returns [{timestamp, path, size_kb}]."""
```

### shared/platform_utils.py

```python
def is_constrained_mode() -> bool:
    return os.environ.get("MCP_CONSTRAINED_MODE", "0") == "1"

def get_max_rows() -> int:       return 20 if is_constrained_mode() else 100
def get_max_results() -> int:    return 10 if is_constrained_mode() else 50
def get_max_depth() -> int:      return 3  if is_constrained_mode() else 5
def get_max_columns() -> int:    return 20 if is_constrained_mode() else 50
def get_cv_folds() -> int:       return 3  if is_constrained_mode() else 5
def get_max_models() -> int:     return 3  if is_constrained_mode() else 7
```

### shared/progress.py

```python
def ok(msg: str, detail: str = "")   -> dict: return {"icon": "✔", "msg": msg, "detail": detail}
def fail(msg: str, detail: str = "") -> dict: return {"icon": "✘", "msg": msg, "detail": detail}
def info(msg: str, detail: str = "") -> dict: return {"icon": "→", "msg": msg, "detail": detail}
def warn(msg: str, detail: str = "") -> dict: return {"icon": "⚠", "msg": msg, "detail": detail}
def undo(msg: str, detail: str = "") -> dict: return {"icon": "↩", "msg": msg, "detail": detail}
```

Always use `Path(x).name` in `msg` — never full absolute paths.

### shared/receipt.py

```python
def append_receipt(file_path: str, tool: str, args: dict,
                   result: str, backup: str | None) -> None:
    """Append one record to receipt log. Never raises."""

def read_receipt_log(file_path: str) -> list[dict]:
    """Read full receipt log. Returns [] if no log exists."""
```

### shared/file_utils.py

```python
def resolve_path(file_path: str) -> Path:
    """Resolve to absolute path. Never use raw string paths in engine."""

def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
```

### shared/patch_validator.py

```python
ALLOWED_PREPROCESSING_OPS = {
    "fill_nulls", "drop_outliers", "label_encode", "onehot_encode",
    "scale", "drop_duplicates", "drop_column", "rename_column", "convert_dtype",
}

def validate_ops(ops: list[dict], allowed: set[str]) -> tuple[bool, str]:
    """Validate op array. Returns (True, '') or (False, error_message)."""
```

---

## 16. Testing Standards for ML

### 16.1 Test Engine Directly

```python
# tests/test_ml_basic.py
from servers.ml_basic.engine import (
    inspect_dataset, train_classifier, train_regressor,
    get_predictions, restore_version
)
```

Never start an MCP server process in tests.

### 16.2 Required Fixtures

| Fixture | Rows | Features | Purpose |
|---|---|---|---|
| `classification_simple.csv` | 200 | 5 numeric | Happy path classification |
| `classification_messy.csv` | 150 | 8 mixed | Nulls, imbalance, categoricals |
| `regression_simple.csv` | 200 | 5 numeric | Happy path regression |
| `regression_messy.csv` | 150 | 6 mixed | Outliers, skewed target |
| `clustering_simple.csv` | 300 | 2 numeric | Clear cluster structure |
| `large_10k.csv` | 10,000 | 10 mixed | Truncation and memory tests |

### 16.3 Required Tests Per Tool

**Every tool must have:**
1. `test_{tool}_success` — happy path, `"success": True`
2. `test_{tool}_file_not_found` — error dict with hint
3. `test_{tool}_token_estimate_present` — `"token_estimate"` key in response
4. `test_{tool}_progress_present` — `"progress"` array in success response

**Every write tool must additionally have:**
5. `test_{tool}_snapshot_created` — `.mcp_versions/` has new `.bak` file
6. `test_{tool}_backup_in_response` — `"backup"` key present in success response
7. `test_{tool}_dry_run` — `dry_run=True` returns without modifying any file
8. `test_{tool}_constrained_mode` — set `MCP_CONSTRAINED_MODE=1`, verify limits enforced

**train_classifier and train_regressor additionally:**
9. `test_train_insufficient_rows` — < MIN_ROWS returns error dict
10. `test_train_single_class_target` — 1 unique class value returns error dict
11. `test_train_all_algorithms` — parametrize over all model strings
12. `test_train_model_saved` — `.mcp_models/` has new `.pkl` and `.manifest.json`

### 16.4 Coverage Requirements

| Module | Minimum coverage |
|---|---|
| `shared/` | 100% |
| `servers/ml_basic/engine.py` | ≥ 90% |
| `servers/ml_medium/engine.py` | ≥ 90% |
| `servers/ml_advanced/engine.py` | ≥ 85% |
| All error paths documented in §12 | Must be tested |

### 16.5 CI Configuration

```yaml
# .github/workflows/test.yml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-22.04, windows-latest, macos-13]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright servers/ shared/
      - run: uv run pytest tests/ --cov=servers --cov=shared --cov-fail-under=90
      - run: python verify_tool_docstrings.py
    env:
      MCP_CONSTRAINED_MODE: "1"
```


---

## 17. What the AI Must Never Do

These are absolute prohibitions. Any generated code that violates them is a defect
and must be corrected immediately.

### Protocol Violations

1. **Never print to stdout in any engine or server module.**
   `print()` corrupts the MCP stdio channel. Use `logger.debug()` to stderr only.

2. **Never return a plain string, list, None, or boolean from a tool.**
   Every tool returns a `dict` with `"success"` as the first key.

3. **Never put domain logic in server.py.**
   Tool bodies in `server.py` are single-line calls to `engine.py`. If you write
   more than two lines in a `@mcp.tool()` function body, you are doing it wrong.

4. **Never import MCP modules in engine.py.**
   `from mcp import ...` and `from fastmcp import ...` are forbidden in engine files.

### Data Safety Violations

5. **Never write to any file without calling `snapshot()` first.**
   This includes model saves, dataset modifications, and output file writes.
   No exceptions for "small changes".

6. **Never swallow exceptions silently.**
   Every `except` block must return an error dict with `"success": False`, `"error"`,
   and `"hint"`. Never use bare `except: pass`.

7. **Never return raw DataFrames, model weight arrays, or full prediction arrays.**
   Return paths, metrics, and summaries. Bounded lists of row dicts are permitted
   within `get_max_rows()` limits.

8. **Never return a raw numpy confusion matrix array.**
   Convert to a named dict (`{"TP": n, ...}` for binary or per-class dicts for
   multiclass) before returning.

### Architecture Violations

9. **Never exceed 10 tools in a single server.**
   ml_basic: 8, ml_medium: 6, ml_advanced: 5. Exceeding this requires a new tier.

10. **Never hardcode row/result limits as magic numbers.**
    Always call `get_max_rows()`, `get_max_results()`, `get_max_columns()` from
    `shared/platform_utils.py`.

11. **Never use string concatenation for file paths.**
    Always use `pathlib.Path / operator` or `resolve_path()`.

12. **Never combine LOCATE + INSPECT or INSPECT + PATCH in one tool.**
    The four-tool pattern separation is mandatory. One operation per tool.

### ML-Specific Violations

13. **Never call a cloud ML API as the primary execution engine.**
    No boto3 for SageMaker, no google-cloud-aiplatform, no Azure ML SDK calls
    for model training. Use scikit-learn and XGBoost locally.

14. **Never save a model without its `.manifest.json` companion.**
    Every `.pkl` file must have a corresponding `.manifest.json` written atomically.

15. **Never use `Optional[T]`, `Union[T, S]`, `Any`, or `dict` without type
    parameters in tool function signatures.**
    Use `T = None`, split tools, or use `str` with enum values in docstring.

16. **Never write a tool docstring longer than 80 characters.**
    CI will fail. Run `verify_tool_docstrings.py` before committing.

17. **Never train a model without first validating minimum row count and
    target column cardinality.**
    Guard clauses must come before any pandas or sklearn operations.

18. **Never return the full `cv_results_` dict from GridSearchCV.**
    Cap to top 20 rows sorted by score. The full dict can contain thousands of
    entries and will overflow the context window.

---

## 18. Progress Tracker

Track implementation progress here. Update checkboxes as work completes.

### Phase 0 — Shared Infrastructure
- [ ] `shared/__init__.py`
- [ ] `shared/version_control.py` — snapshot / restore / list
- [ ] `shared/patch_validator.py` — validate_ops + ALLOWED_PREPROCESSING_OPS
- [ ] `shared/file_utils.py` — resolve_path / atomic_write_json
- [ ] `shared/platform_utils.py` — all constrained mode helpers
- [ ] `shared/progress.py` — ok / fail / info / warn / undo
- [ ] `shared/receipt.py` — append_receipt / read_receipt_log
- [ ] Unit tests for all shared modules (100% coverage)

### Phase 1 — ml_basic (Tier 1)
- [ ] `servers/ml_basic/__init__.py`
- [ ] `servers/ml_basic/pyproject.toml`
- [ ] `servers/ml_basic/engine.py` — inspect_dataset
- [ ] `servers/ml_basic/engine.py` — read_column_profile
- [ ] `servers/ml_basic/engine.py` — search_columns
- [ ] `servers/ml_basic/engine.py` — read_rows
- [ ] `servers/ml_basic/engine.py` — train_classifier (all 7 algorithms)
- [ ] `servers/ml_basic/engine.py` — train_regressor (all 7 algorithms)
- [ ] `servers/ml_basic/engine.py` — get_predictions
- [ ] `servers/ml_basic/engine.py` — restore_version
- [ ] `servers/ml_basic/server.py` — all 8 @mcp.tool() wrappers
- [ ] `tests/fixtures/` — all 6 fixture CSVs
- [ ] `tests/test_ml_basic.py` — all required tests (§16.3)
- [ ] `uv run pytest tests/test_ml_basic.py` — all pass
- [ ] `uv run pyright servers/ml_basic/` — no errors
- [ ] `verify_tool_docstrings.py` — all ≤ 80 chars
- [ ] Manual test in LM Studio (9B model) — four-tool loop works

### Phase 2 — ml_medium (Tier 2)
- [ ] `servers/ml_medium/__init__.py`
- [ ] `servers/ml_medium/pyproject.toml`
- [ ] `servers/ml_medium/engine.py` — run_preprocessing + all ops
- [ ] `servers/ml_medium/engine.py` — detect_outliers (IQR + std)
- [ ] `servers/ml_medium/engine.py` — train_with_cv
- [ ] `servers/ml_medium/engine.py` — compare_models
- [ ] `servers/ml_medium/engine.py` — run_clustering (3 algorithms)
- [ ] `servers/ml_medium/engine.py` — read_receipt
- [ ] `servers/ml_medium/server.py` — all 6 @mcp.tool() wrappers
- [ ] `tests/test_ml_medium.py` — all required tests
- [ ] `uv run pytest tests/test_ml_medium.py` — all pass
- [ ] Manual test: ml_basic + ml_medium loaded together (≤ 14 tools total)

### Phase 3 — ml_advanced (Tier 3)
- [ ] `servers/ml_advanced/__init__.py`
- [ ] `servers/ml_advanced/pyproject.toml`
- [ ] `servers/ml_advanced/engine.py` — tune_hyperparameters (grid + random)
- [ ] `servers/ml_advanced/engine.py` — export_model + manifest
- [ ] `servers/ml_advanced/engine.py` — read_model_report
- [ ] `servers/ml_advanced/engine.py` — run_profiling_report
- [ ] `servers/ml_advanced/engine.py` — apply_dimensionality_reduction
- [ ] `servers/ml_advanced/server.py` — all 5 @mcp.tool() wrappers
- [ ] `tests/test_ml_advanced.py` — all required tests
- [ ] `uv run pytest tests/test_ml_advanced.py` — all pass

### Phase 4 — Installation and Distribution
- [ ] `install/install.sh` — Python 3.11 check, uv sync, VRAM detection, client config
- [ ] `install/install.bat` — Windows equivalent
- [ ] `install/mcp_config_writer.py` — LM Studio / Claude Desktop / Cursor / Windsurf
- [ ] `root pyproject.toml` — workspace with all three server members
- [ ] `uv.lock` committed
- [ ] `.python-version` = `3.11`
- [ ] `.gitattributes` — `* text=auto eol=lf`
- [ ] Test install on clean machine / VM

### Phase 5 — CI/CD
- [ ] `.github/workflows/test.yml` — matrix: ubuntu + windows + macos
- [ ] `verify_tool_docstrings.py` — ≤ 80 char enforcement
- [ ] All CI checks passing on all three platforms
- [ ] `MCP_CONSTRAINED_MODE=1` enforced in CI environment

### Phase 6 — Documentation
- [ ] `README.md` — full required sections (§29 of STANDARDS.md)
- [ ] Hardware sovereignty statement in README
- [ ] Tool reference table in README
- [ ] Usage examples showing four-tool loop for classification and regression

