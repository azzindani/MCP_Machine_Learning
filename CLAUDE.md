# CLAUDE.md — MCP Machine Learning Server Development Guide

This document is the authoritative development guide for AI coding agents working on
this repository. Read it completely before writing any code. All rules here are binding.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [Repository Structure](#3-repository-structure)
4. [Architecture Principles](#4-architecture-principles)
5. [MCP Primitives — Tools, Resources, and Prompts](#5-mcp-primitives--tools-resources-and-prompts)
6. [Tool Annotations](#6-tool-annotations)
7. [Security Considerations](#7-security-considerations)
8. [Engine Sub-Module Pattern](#8-engine-sub-module-pattern)
9. [Three-Tier ML Server Design](#9-three-tier-ml-server-design)
10. [Tool Catalogue — ml_basic (Tier 1)](#10-tool-catalogue--ml_basic-tier-1)
11. [Tool Catalogue — ml_medium (Tier 2)](#11-tool-catalogue--ml_medium-tier-2)
12. [Tool Catalogue — ml_advanced (Tier 3)](#12-tool-catalogue--ml_advanced-tier-3)
13. [ML Pipeline Implementation Rules](#13-ml-pipeline-implementation-rules)
14. [Supported Algorithms Reference](#14-supported-algorithms-reference)
15. [Return Value Contracts for ML Tools](#15-return-value-contracts-for-ml-tools)
16. [Error Handling — ML-Specific Patterns](#16-error-handling--ml-specific-patterns)
17. [Model Persistence and Versioning](#17-model-persistence-and-versioning)
18. [Hardware and Resource Constraints](#18-hardware-and-resource-constraints)
19. [Shared Module Contracts](#19-shared-module-contracts)
20. [Testing Standards for ML](#20-testing-standards-for-ml)
21. [What the AI Must Never Do](#21-what-the-ai-must-never-do)

---

## 1. Project Overview

This is a **self-hosted, local-first MCP (Model Context Protocol) server project**
for machine learning. It exposes ML operations as structured tools that a local
language model calls with JSON arguments and receives JSON results from.

The ML pipeline this project covers:

- **Data inspection** — schema discovery, column profiling, row sampling
- **Data preprocessing** — null handling, outlier detection, type conversion, encoding, scaling
- **Supervised learning** — classification and regression (7+ algorithm families)
- **Unsupervised learning** — clustering (K-Means, Mean-Shift, DBSCAN) and dimensionality reduction (PCA, ICA)
- **Model evaluation** — accuracy, F1, confusion matrix, MSE, R², cross-validation
- **Hyperparameter tuning** — GridSearchCV, RandomizedSearchCV
- **Model persistence** — pickle serialization, version snapshots
- **HTML reports** — EDA, training metrics, ROC, learning curves, cluster visualization

All execution is **100% local**. No data leaves the user's machine. No cloud APIs.
No API keys. No subscriptions. The tools run on the user's CPU using scikit-learn,
XGBoost, pandas, and numpy.

---

## 2. Goals and Non-Goals

### Goals

- Provide a local LLM with clean, surgical ML tools that stay within a 10,000-token
  context window on memory-constrained hardware
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
│   ├── receipt.py                      # append_receipt() / read_receipt_log()
│   └── html_theme.py                   # CSS vars, Plotly templates, responsive HTML helpers
│
├── servers/
│   ├── ml_basic/                       # Tier 1 — dataset inspection + single-model training
│   │   ├── __init__.py
│   │   ├── server.py                   # FastMCP tool definitions (thin wrappers only)
│   │   ├── engine.py                   # Public API re-exports from sub-modules
│   │   ├── _basic_helpers.py           # Shared helpers, path resolution, data loading
│   │   ├── _basic_train.py             # train_classifier + train_regressor
│   │   ├── _basic_predict.py           # get_predictions + predict_single
│   │   └── pyproject.toml
│   │
│   ├── ml_medium/                      # Tier 2 — preprocessing pipeline + CV + multi-model
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── engine.py
│   │   ├── _medium_helpers.py          # Shared helpers
│   │   ├── _medium_preprocess.py       # run_preprocessing ops
│   │   ├── _medium_train.py            # train_with_cv + compare_models
│   │   ├── _medium_cluster.py          # run_clustering + find_optimal_clusters
│   │   ├── _medium_data.py             # filter_rows + merge_datasets + batch_predict
│   │   ├── _medium_eda.py              # generate_eda_report + check_data_quality
│   │   └── pyproject.toml
│   │
│   └── ml_advanced/                    # Tier 3 — tuning, export, evaluation reports
│       ├── __init__.py
│       ├── server.py
│       ├── engine.py
│       ├── _adv_helpers.py             # tune / export / read_model_report / profiling / reduction
│       ├── _adv_viz.py                 # all HTML chart and report generation
│       └── pyproject.toml
│
├── tests/
│   ├── fixtures/
│   │   ├── classification_simple.csv
│   │   ├── classification_messy.csv
│   │   ├── regression_simple.csv
│   │   ├── regression_messy.csv
│   │   ├── clustering_simple.csv
│   │   └── large_10k.csv
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_ml_basic.py
│   ├── test_ml_medium.py
│   └── test_ml_advanced.py
│
├── install/
│   ├── install.sh                      # macOS/Linux installer
│   ├── install.bat                     # Windows installer
│   └── mcp_config_writer.py            # writes LM Studio / Claude Desktop / Cursor config
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
├── pyproject.toml                      # root workspace
├── pyrightconfig.json
├── uv.lock
├── .python-version                     # 3.12
├── .gitattributes
├── verify_tool_docstrings.py
└── CLAUDE.md                           # this file
```

### File ownership rules

- `engine.py` owns all ML logic via re-exports from sub-modules. Zero MCP imports.
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
data to a cloud endpoint.

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
writing. The `"backup"` key must appear in every write tool's success response.

### 4.6 Token Budget Discipline

Rules:
- Tool docstrings ≤ 80 characters (enforced by `verify_tool_docstrings.py`)
- Read tool responses ≤ 500 tokens
- Write confirmations ≤ 150 tokens
- Never return raw DataFrames, weight arrays, or confusion matrix arrays as nested lists
- Return `token_estimate = len(str(response)) // 4` in every response

### 4.7 Hardware Mode Flag

Read `MCP_CONSTRAINED_MODE` env var via `shared.platform_utils` helpers. Never
hardcode row/result limits:

```python
from shared.platform_utils import get_max_rows, get_max_results, is_constrained_mode

max_rows = get_max_rows()        # 20 in constrained, 100 in standard
max_results = get_max_results()  # 10 in constrained, 50 in standard
```

---

## 5. MCP Primitives — Tools, Resources, and Prompts

### Tools

**Tools** are called by the model with arguments and return structured JSON. Use
tools for all operations that read, transform, write, or execute something.

### Resources

**Resources** expose stable, re-readable context without a tool call. Use resources
for reference data the model needs repeatedly (e.g. supported algorithm lists).
Resources are read-only and stateless.

### Prompts

**Prompts** are reusable workflow templates. Use sparingly — only when you need
a structured starting workflow. Most servers do not need prompts.

### Rule of thumb

```
Model needs to call it to do work          → Tool
Model needs to reference it for context    → Resource
User needs a starting workflow template    → Prompt
```

---

## 6. Tool Annotations

Always set annotations on every `@mcp.tool()` decorator:

```python
@mcp.tool(
    annotations={
        "readOnlyHint": True,       # does not modify any state
        "destructiveHint": False,   # does not destroy data
        "idempotentHint": True,     # safe to call multiple times
        "openWorldHint": False,     # does not interact with external services
    }
)
```

### Annotation rules by ML tool type

| Tool type | readOnlyHint | destructiveHint | idempotentHint | openWorldHint |
|---|---|---|---|---|
| inspect / read / search / profile | True | False | True | False |
| train / preprocess / cluster (with snapshot) | False | False | False | False |
| drop column / delete rows | False | True | False | False |
| export / generate report | False | False | True | False |

---

## 7. Security Considerations

### Path Traversal Prevention

All file paths from tool parameters must be validated through `resolve_path()`
before any I/O. The check blocks null bytes and bare filesystem roots while
allowing any absolute path the user explicitly provides — including paths on
different drives (e.g. `D:\` on Windows):

```python
def resolve_path(file_path: str, allowed_extensions: tuple[str, ...] = ()) -> Path:
    raw = str(file_path)
    if "\x00" in raw:
        raise ValueError(f"Invalid path (null byte): {file_path}")
    path = Path(raw).resolve()
    if path.parent == path:   # True only for roots like '/' or 'C:\'
        raise ValueError(f"Path resolves to filesystem root: {file_path}")
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Extension {path.suffix!r} not allowed. Expected: {allowed_extensions}"
        )
    return path
```

Never use raw `file_path` strings in `open()`, `pd.read_csv()`, or subprocess
calls without first calling `resolve_path()`.

### No eval() / exec()

For any tool that evaluates expressions, never use `eval()` or `exec()`. Parse
with AST and an allowlist.

### Subprocess Safety

Use argument lists with `shell=False` and `timeout`:

```python
subprocess.run(
    ["some_cmd", str(validated_path)],
    shell=False, capture_output=True, timeout=300,
)
```

### Sensitive Data in Responses

Never include full system paths, credentials, or environment variables in tool
responses. Always use `Path(x).name` (filename only) in progress messages.

---

## 8. Engine Sub-Module Pattern

When `engine.py` grows beyond ~400–500 lines, split into focused sub-modules.
The engine entry point becomes a thin router.

### engine.py as thin router

```python
# engine.py — thin router, zero MCP imports
from ._adv_helpers import tune_hyperparameters, export_model, read_model_report
from ._adv_viz    import generate_training_report, plot_roc_curve

__all__ = ["tune_hyperparameters", "export_model", ...]
```

Tests still import from `engine.py` — sub-module structure is invisible to tests.

### Lazy imports for heavy ML libraries

Sub-modules that depend on large libraries should import inside the function:

```python
def run_profiling_report(file_path: str, ...) -> dict:
    from ydata_profiling import ProfileReport   # lazy — only when called
    import pandas as pd
    ...
```

---

## 9. Three-Tier ML Server Design

| Tier | Server | Tools | Purpose |
|---|---|---|---|
| 1 | `ml-basic` | 11 | Dataset inspection + single-model training + prediction |
| 2 | `ml-medium` | 14 | Preprocessing + CV + clustering + EDA + batch predict |
| 3 | `ml-advanced` | 10 | Tuning + export + evaluation charts + profiling |

**Tier 1 must stand alone.** No cross-tier imports.

---

## 10. Tool Catalogue — ml_basic (Tier 1)

| # | Tool | Category | Signature summary |
|---|---|---|---|
| 1 | `inspect_dataset` | LOCATE | `(file_path)` → schema, row count, dtypes, null summary |
| 2 | `read_column_profile` | INSPECT | `(file_path, column_name)` → stats for one column |
| 3 | `search_columns` | LOCATE | `(file_path, has_nulls, dtype, name_contains, max_results)` → names only |
| 4 | `read_rows` | INSPECT | `(file_path, start, end)` → bounded row slice |
| 5 | `train_classifier` | PATCH | `(file_path, target_column, model, test_size, random_state, class_weight, return_train_score, dry_run)` |
| 6 | `train_regressor` | PATCH | `(file_path, target_column, model, degree, alpha, n_estimators, test_size, random_state, dry_run)` |
| 7 | `get_predictions` | VERIFY | `(model_path, file_path, max_rows, return_proba)` → bounded predictions |
| 8 | `restore_version` | CONTROL | `(file_path, timestamp)` → rollback or list snapshots |
| 9 | `predict_single` | VERIFY | `(model_path, input_data)` → predict one JSON record |
| 10 | `list_models` | LOCATE | `(directory)` → list all `.pkl` models with metadata |
| 11 | `split_dataset` | PATCH | `(file_path, test_size, stratify_column, output_dir, random_state)` → saves train/test CSVs |

### Key implementation rules for Tier 1

- `inspect_dataset`: never return actual row data; cap column list at `get_max_results()`
- `read_column_profile`: numeric → mean/std/min/max/median/percentiles/skewness; categorical → top_values (≤10); bool → balance_ratio
- `search_columns`: return column names only, never column data
- `read_rows`: hard limit `min(end - start, get_max_rows())`; return `"truncated": True` when capped
- `train_classifier` / `train_regressor`: auto-encode categoricals (LabelEncoder), fill numeric nulls with median, stratify split for classification, save model + manifest
- `get_predictions`: cap at `min(max_rows, get_max_rows())`; apply stored encoding from metadata
- `predict_single`: parse `input_data` as JSON string, apply same encoding as training
- `list_models`: scan `directory` (default: home dir `.mcp_models/`), return metadata list
- `split_dataset`: snapshot before writing; support `stratify_column` for classification splits

---

## 11. Tool Catalogue — ml_medium (Tier 2)

| # | Tool | Category | Signature summary |
|---|---|---|---|
| 1 | `run_preprocessing` | PATCH | `(file_path, ops, output_path, dry_run)` → apply op pipeline |
| 2 | `detect_outliers` | INSPECT | `(file_path, columns, method, th1, th3)` → per-column outlier report |
| 3 | `train_with_cv` | PATCH | `(file_path, target_column, model, task, n_splits, random_state, dry_run)` |
| 4 | `compare_models` | PATCH | `(file_path, target_column, task, models, test_size, random_state, dry_run)` |
| 5 | `run_clustering` | PATCH | `(file_path, feature_columns, algorithm, n_clusters, eps, min_samples, reduce_dims, n_components, save_labels, dry_run)` |
| 6 | `read_receipt` | CONTROL | `(file_path)` → operation history log |
| 7 | `generate_eda_report` | ANALYZE | `(file_path, target_column, theme, output_path, open_browser, dry_run)` → HTML EDA |
| 8 | `filter_rows` | PATCH | `(file_path, column, operator, value, output_path, dry_run)` → filtered CSV |
| 9 | `merge_datasets` | PATCH | `(file_path_1, file_path_2, on, how, output_path, dry_run)` → merged CSV |
| 10 | `find_optimal_clusters` | ANALYZE | `(file_path, feature_columns, max_k, theme, output_path, open_browser)` → elbow chart |
| 11 | `anomaly_detection` | INSPECT | `(file_path, feature_columns, method, contamination, save_labels, dry_run)` |
| 12 | `check_data_quality` | INSPECT | `(file_path)` → quality score 0–100 with typed alerts |
| 13 | `evaluate_model` | VERIFY | `(model_path, test_file_path, target_column)` → metrics on external test set |
| 14 | `batch_predict` | PATCH | `(model_path, file_path, output_path, dry_run)` → all rows, saves CSV |

### Key implementation rules for Tier 2

- `run_preprocessing`: validate all ops before applying any; snapshot before write; max 50 ops per call
- `detect_outliers`: IQR (th1/th3 quantiles ± 1.5×IQR) or std (mean ± 3σ); return counts and bounds, not full row list; up to 5 sample outlier values
- `train_with_cv`: StratifiedKFold for classification, KFold for regression; save best-fold model; return per-fold scores + mean ± std
- `compare_models`: same train/test split for all; rank by F1 weighted (classification) or R² (regression); save only best model; max 7 models (3 in constrained)
- `run_clustering`: StandardScaler before fitting; optional PCA/ICA reduction before clustering; `save_labels=True` snapshots before appending column
- `generate_eda_report`: 8 quality checks (constant, high_missing, zero_inflated, high_cardinality, class_imbalance, extreme_skewness, multicollinearity, duplicate_rows); each alert includes recommendation
- `anomaly_detection`: Isolation Forest or LOF; `save_labels=True` appends `_anomaly` column with snapshot
- `batch_predict`: no row limit; applies stored encoding from model metadata

---

## 12. Tool Catalogue — ml_advanced (Tier 3)

| # | Tool | Category | Signature summary |
|---|---|---|---|
| 1 | `tune_hyperparameters` | OPTIMIZE | `(file_path, target_column, model, task, search, param_grid, cv, n_iter, dry_run)` |
| 2 | `export_model` | EXPORT | `(model_path, output_dir, format, dry_run)` → copy + manifest |
| 3 | `read_model_report` | VERIFY | `(model_path)` → feature importance, confusion matrix, metrics |
| 4 | `run_profiling_report` | ANALYZE | `(file_path, output_path, sample_rows, dry_run)` → ydata-profiling HTML |
| 5 | `apply_dimensionality_reduction` | TRANSFORM | `(file_path, feature_columns, method, n_components, output_path, dry_run)` |
| 6 | `generate_training_report` | ANALYZE | `(model_path, theme, output_path, open_browser, dry_run)` → full HTML |
| 7 | `plot_roc_curve` | ANALYZE | `(model_path, file_path, theme, output_path, open_browser, dry_run)` → HTML |
| 8 | `plot_learning_curve` | ANALYZE | `(file_path, target_column, model, task, cv, theme, output_path, open_browser, dry_run)` |
| 9 | `plot_predictions_vs_actual` | ANALYZE | `(model_path, file_path, theme, output_path, open_browser, dry_run)` → HTML |
| 10 | `generate_cluster_report` | ANALYZE | `(file_path, feature_columns, label_column, theme, output_path, open_browser, dry_run)` |

### Key implementation rules for Tier 3

- `tune_hyperparameters`: `param_grid` is a JSON string parsed inside engine; default grids defined as constants; cap `cv_results_` to top 20 rows; save best model as snapshot
- `export_model`: write `{model_name}.manifest.json` alongside `.pkl`; snapshot before overwrite
- `read_model_report`: confusion matrix as named dict (`{"TP": n, ...}` binary; per-class for multiclass up to 10 classes); feature importances as top-10 list; classification report bounded to 500 chars
- `run_profiling_report`: `minimal=True` in constrained mode; return path + key stats summary, never raw profile data
- `apply_dimensionality_reduction`: StandardScaler before reduction; PCA returns `explained_variance_ratio_`; ICA uses FastICA; save reduced dataset as new CSV
- All HTML tools: support `theme="light"` or `theme="dark"`; `open_browser=True` opens result after save

---

## 13. ML Pipeline Implementation Rules

### 13.1 Data Loading

Always use `pandas.read_csv()`. Do not use `polars`.

```python
df = pd.read_csv(path, low_memory=False)
```

After loading, check available RAM before heavy computation:

```python
required_gb = (df.memory_usage(deep=True).sum() / 1e9) * 3  # 3× for transforms
mem_check = check_memory(required_gb)
if mem_check:
    return mem_check  # early return with error dict
```

### 13.2 Automatic Preprocessing Before Training

When `train_classifier` or `train_regressor` is called, the engine **automatically**:

1. Drops rows where `target_column` is null
2. LabelEncodes all object/category dtype columns
3. Fills remaining numeric nulls with column median
4. Stores encoding mapping in model metadata for use in `get_predictions`

Do not scale features automatically in the basic tier. Exception: KNN and SVM scale
internally before fitting.

### 13.3 Train/Test Split

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y if task == "classification" else None,
)
```

### 13.4 Minimum Dataset Size Guard

```python
MIN_ROWS_CLASSIFIER = 20
MIN_ROWS_REGRESSOR = 10
```

Validate before training. Also validate target column has at least 2 unique values
for classification.

### 13.5 Model Saving Convention

```
.mcp_models/{file_stem}_{model}_{YYYY-MM-DDTHH-MM-SSZ}.pkl
.mcp_models/{file_stem}_{model}_{YYYY-MM-DDTHH-MM-SSZ}.manifest.json
```

Save both files atomically. The `.manifest.json` companion is required alongside
every `.pkl`. Never save a model without its manifest.

### 13.6 Score Formatting

Return scores as floats in [0, 1] range. Do not format as percentage strings.

```python
# Correct
{"accuracy": 0.89, "f1_weighted": 0.87}

# Wrong
{"accuracy": "89.0%"}
```

### 13.7 Confusion Matrix Format

```python
# Binary classification
{"TP": 120, "FP": 15, "FN": 10, "TN": 355}

# Multiclass (up to 10 classes)
{"class_0": {"precision": 0.91, "recall": 0.88, "f1": 0.89, "support": 200}, ...}
```

### 13.8 XGBoost Usage Pattern

```python
import xgboost as xgb

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest  = xgb.DMatrix(x_test,  label=y_test)
params = {
    "max_depth": 3, "eta": 0.3, "silent": 1,
    "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
    "num_class": n_classes,
}
model = xgb.train(params, dtrain, num_boost_round=num_round)
preds = model.predict(dtest)

if n_classes > 2:
    y_predicted = np.asarray([np.argmax(line) for line in preds])
else:
    y_predicted = (preds > 0.5).astype(int)
```

---

## 14. Supported Algorithms Reference

### Classification

| model string | Class | Default key params |
|---|---|---|
| `"lr"` | `LogisticRegression` | `random_state=42, max_iter=200` |
| `"svm"` | `SVC` | `kernel='rbf', gamma='auto', random_state=42` |
| `"rf"` | `RandomForestClassifier` | `n_estimators=100, random_state=42` |
| `"dtc"` | `DecisionTreeClassifier` | `random_state=42` |
| `"knn"` | `KNeighborsClassifier` | `n_neighbors=5, metric='minkowski', p=2` |
| `"nb"` | `GaussianNB` | *(no key params)* |
| `"xgb"` | `xgb.train()` | `max_depth=3, eta=0.3, num_round=10` |

### Regression

| model string | Class | Default key params |
|---|---|---|
| `"lir"` | `LinearRegression` | *(no key params)* |
| `"pr"` | `LinearRegression` + `PolynomialFeatures` | `degree=5` |
| `"lar"` | `Lasso` | `alpha=0.01, max_iter=200, tol=0.1` |
| `"rr"` | `Ridge` | `alpha=0.01, max_iter=100, tol=0.1` |
| `"dtr"` | `DecisionTreeRegressor` | `random_state=42` |
| `"rfr"` | `RandomForestRegressor` | `n_estimators=10, random_state=42` |
| `"xgb"` | `xgb.train()` | `objective='reg:squarederror', num_round=5` |

### Clustering

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

### Default Tuning Param Grids (constants in engine)

```python
DEFAULT_PARAMS = {
    "svm":  {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "rf":   {"n_estimators": [10, 50, 100], "max_depth": [None, 5, 10]},
    "xgb":  {"max_depth": [3, 5, 7], "eta": [0.1, 0.3], "n_estimators": [50, 100]},
    "knn":  {"n_neighbors": [3, 5, 7, 11]},
    "lr":   {"C": [0.01, 0.1, 1, 10]},
}
```

### Evaluation Metrics

**Classification:** `accuracy_score`, `f1_score(average='weighted')`, `classification_report`, `confusion_matrix`

**Regression:** `mean_squared_error`, `r2_score` (RMSE = `np.sqrt(mean_squared_error(...))`)

---

## 15. Return Value Contracts for ML Tools

### Required Fields in Every Response

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

### Train Tool Success Response

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
        {"icon": "✔", "msg": "Trained RandomForestClassifier", "detail": "n_estimators=100"},
        {"icon": "✔", "msg": "Saved model", "detail": ".mcp_models/customer_churn_rf_...pkl"},
    ],
    "token_estimate": 185
}
```

---

## 16. Error Handling — ML-Specific Patterns

All exceptions are caught in `engine.py` and returned as error dicts. Never raise
to the MCP layer.

```python
def _error(error: str, hint: str, backup: str | None = None) -> dict:
    base = {"success": False, "error": error, "hint": hint}
    if backup:
        base["backup"] = backup
    base["token_estimate"] = len(str(base)) // 4
    return base
```

### Standard error messages

```python
# File errors
f"File not found: {file_path}"
# hint: "Check that file_path is absolute and the CSV file exists."

# Column errors
f"Column '{name}' not found. Available: {', '.join(columns[:10])}"
# hint: "Use inspect_dataset() to list all column names."

# Data quality
f"Dataset has {rows} rows but need at least {min_rows} to train reliably."
# hint: "Provide a dataset with more samples before training."

# Model errors
f"Model file not found: {model_path}"
# hint: "Use train_classifier() or train_regressor() to train a model first."

# Algorithm errors
f"Unknown algorithm: '{model}'. Allowed: {', '.join(ALLOWED_CLASSIFIERS)}"
# hint: "Use one of: lr svm rf dtc knn nb xgb"

# Resource errors
f"Insufficient RAM: need ~{required_gb:.1f} GB, available ~{available_gb:.1f} GB."
# hint: "Use read_rows() to sample a subset, or increase available memory."
```

Never use bare `except: pass`. Every `except` block must return an error dict.

---

## 17. Model Persistence and Versioning

### Storage Layout

```
{dataset_dir}/
├── dataset.csv
├── dataset.csv.mcp_state.json
├── dataset.csv.mcp_receipt.json
├── .mcp_models/
│   ├── dataset_rf_2026-04-06T10-30-00Z.pkl
│   └── dataset_rf_2026-04-06T10-30-00Z.manifest.json
└── .mcp_versions/
    └── dataset_2026-04-06T09-00-00Z.csv.bak
```

### Pickle Saving Pattern

```python
import pickle, tempfile, shutil

def _save_model(model, path: Path, metadata: dict) -> None:
    payload = {"model": model, "metadata": metadata}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl",
                                    dir=path.parent) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2))
```

### Model Metadata (stored in pkl payload AND manifest.json)

```python
metadata = {
    "model_type": type(model).__name__,
    "task": task,                          # "classification" or "regression"
    "trained_on": Path(file_path).name,
    "training_date": datetime.now(timezone.utc).isoformat(),
    "feature_columns": feature_columns,
    "target_column": target_column,
    "encoding_map": encoding_map,
    "scaler": scaler_obj_or_None,
    "metrics": metrics_dict,
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
}
```

---

## 18. Hardware and Resource Constraints

### RAM Check Before Heavy Operations

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

### Constrained Mode Limits

| Resource | Standard | Constrained (`MCP_CONSTRAINED_MODE=1`) |
|---|---|---|
| Max rows returned per call | 100 | 20 |
| Max search results | 50 | 10 |
| Max JSON depth | 5 levels | 3 levels |
| Max columns returned | 50 | 20 |
| ydata-profiling mode | full | minimal=True |
| compare_models max | 7 | 3 |
| tune_hyperparameters cv | 5 | 3 |
| tune_hyperparameters n_iter | 10 | 5 |

### Recommended Loading by Available RAM

| Available RAM | Recommended load | Total tools |
|---|---|---|
| 4–8 GB | ml-basic only | 11 |
| 8–16 GB | ml-basic + ml-medium | 25 |
| 16 GB+ | all three tiers | 35 |

---

## 19. Shared Module Contracts

Every shared module must be implemented exactly as specified. Add new functions
rather than modifying existing interfaces.

### shared/version_control.py

```python
def snapshot(file_path: str) -> str:
    """Snapshot file to .mcp_versions/. Returns backup path. Raises if source missing."""

def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore from snapshot. Empty timestamp = list available snapshots."""

def list_snapshots(file_path: str) -> list[dict]:
    """List available snapshots. Returns [{timestamp, path, size_kb}]."""
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
def resolve_path(file_path: str, allowed_extensions: tuple[str, ...] = ()) -> Path:
    """Resolve path, enforce home-dir boundary, validate extension."""

def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
```

### shared/patch_validator.py

```python
ALLOWED_PREPROCESSING_OPS = {
    "fill_nulls", "drop_outliers", "label_encode", "onehot_encode",
    "scale", "drop_duplicates", "drop_column", "rename_column", "convert_dtype",
    "bin_numeric", "add_date_parts", "log_transform", "drop_null_rows", "clip_column",
}

def validate_ops(ops: list[dict], allowed: set[str]) -> tuple[bool, str]:
    """Validate op array. Returns (True, '') or (False, error_message)."""
```

---

## 20. Testing Standards for ML

### Test Engine Directly

```python
from servers.ml_basic.engine import (
    inspect_dataset, train_classifier, train_regressor,
    get_predictions, restore_version
)
```

Never start an MCP server process in tests.

### Required Tests Per Tool

Every tool must have:
1. `test_{tool}_success` — happy path, `"success": True`
2. `test_{tool}_file_not_found` — error dict with hint
3. `test_{tool}_token_estimate_present` — `"token_estimate"` key in response
4. `test_{tool}_progress_present` — `"progress"` array in success response

Every write tool additionally:
5. `test_{tool}_snapshot_created` — `.mcp_versions/` has new `.bak` file
6. `test_{tool}_backup_in_response` — `"backup"` key in success response
7. `test_{tool}_dry_run` — `dry_run=True` returns without modifying any file
8. `test_{tool}_constrained_mode` — set `MCP_CONSTRAINED_MODE=1`, verify limits

`train_classifier` and `train_regressor` additionally:
9. `test_train_insufficient_rows` — `< MIN_ROWS` returns error dict
10. `test_train_single_class_target` — 1 unique class returns error dict
11. `test_train_all_algorithms` — parametrize over all model strings
12. `test_train_model_saved` — `.mcp_models/` has new `.pkl` and `.manifest.json`

### Coverage Requirements

| Module | Minimum coverage |
|---|---|
| `shared/` | 100% |
| `servers/ml_basic/engine.py` | ≥ 90% |
| `servers/ml_medium/engine.py` | ≥ 90% |
| `servers/ml_advanced/engine.py` | ≥ 85% |

### CI Configuration

```yaml
# .github/workflows/ci.yml
env:
  MCP_CONSTRAINED_MODE: "1"
  PYTHONPATH: "."
steps:
  - run: uv sync --frozen
  - run: uv run ruff check .
  - run: uv run ruff format --check .
  - run: uv run pyright servers/ shared/
  - run: uv run pytest tests/ --cov=servers --cov=shared --cov-fail-under=90
  - run: python verify_tool_docstrings.py
```

---

## 21. What the AI Must Never Do

### Protocol Violations

1. **Never print to stdout in any engine or server module.**
   Use `logger.debug()` to stderr only.

2. **Never return a plain string, list, None, or boolean from a tool.**
   Every tool returns a `dict` with `"success"` as the first key.

3. **Never put domain logic in server.py.**
   Tool bodies in `server.py` are single-line calls to `engine.py`.

4. **Never import MCP modules in engine.py.**
   `from mcp import ...` and `from fastmcp import ...` are forbidden in engine files.

### Data Safety Violations

5. **Never write to any file without calling `snapshot()` first.**
   No exceptions for "small changes".

6. **Never swallow exceptions silently.**
   Every `except` block must return `{"success": False, "error": ..., "hint": ...}`.

7. **Never return raw DataFrames, model weight arrays, or full prediction arrays.**
   Return paths, metrics, and summaries. Bounded lists of row dicts are permitted
   within `get_max_rows()` limits.

8. **Never return a raw numpy confusion matrix array.**
   Convert to a named dict before returning.

### Architecture Violations

9. **Never hardcode row/result limits as magic numbers.**
   Always call `get_max_rows()`, `get_max_results()`, `get_max_columns()`.

10. **Never use string concatenation for file paths.**
    Always use `pathlib.Path / operator` or `resolve_path()`.

11. **Never combine LOCATE + INSPECT or INSPECT + PATCH in one tool.**
    The four-tool pattern separation is mandatory.

### ML-Specific Violations

12. **Never call a cloud ML API as the primary execution engine.**
    Use scikit-learn and XGBoost locally.

13. **Never save a model without its `.manifest.json` companion.**
    Every `.pkl` file must have a corresponding `.manifest.json` written atomically.

14. **Never use `Optional[T]`, `Union[T, S]`, `Any`, or bare `dict` without type
    parameters in tool function signatures.**

15. **Never write a tool docstring longer than 80 characters.**
    `verify_tool_docstrings.py` enforces this.

16. **Never train a model without validating minimum row count and target column
    cardinality first.**

17. **Never return the full `cv_results_` dict from GridSearchCV.**
    Cap to top 20 rows sorted by score.

18. **Never use `eval()` or `exec()` on any user-provided input.**

19. **Never pass user-provided strings into subprocess calls with `shell=True`.**

20. **Never use raw `file_path` strings without calling `resolve_path()` first.**

21. **Never import heavy libraries (sklearn, xgboost, ydata_profiling) at module
    level in sub-modules.** Use lazy imports inside functions.
