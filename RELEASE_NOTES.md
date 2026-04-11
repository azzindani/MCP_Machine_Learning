# Release Notes

## v0.1.0 — 2026-04-11

Initial public release of **MCP Machine Learning** — a self-hosted, local-first MCP server that gives local LLMs structured access to the full supervised and unsupervised machine learning pipeline. No cloud APIs. No API keys. Everything runs on your machine.

---

### Highlights

- **35 tools** across three focused tiers — works with any tool-calling model (Gemma 4, Qwen 3.5, etc.)
- **100% local execution** — no data leaves the machine, works offline after first install
- **Full ML pipeline** — inspect → preprocess → train → evaluate → tune, end to end
- **Interactive HTML reports** — all visualization outputs are standalone offline files
- **Automatic version control** — every write is snapshotted and fully restorable
- **Constrained mode** — reduces row and result limits for lower-memory machines
- **Multi-platform CI** — passes automated tests on Windows, macOS, and Linux

---

### Tools

#### Tier 1 — ml-basic (11 tools)

Dataset inspection and single-model training. Designed to stand alone without loading any other tier.

| Tool | Purpose |
|---|---|
| `inspect_dataset` | Schema, row count, dtypes, null summary |
| `read_column_profile` | Stats for one column — mean, std, nulls, top values |
| `search_columns` | Find columns by dtype, null status, or name pattern |
| `read_rows` | Bounded row slice with constrained-mode enforcement |
| `train_classifier` | Train classifier: `lr svm rf dtc knn nb xgb` |
| `train_regressor` | Train regressor: `lir pr lar rr dtr rfr xgb` |
| `get_predictions` | Run predictions with probability support (`return_proba=True`) |
| `restore_version` | Rollback any model or dataset to a previous snapshot |
| `predict_single` | Predict one JSON record without a CSV file |
| `list_models` | List all saved `.pkl` models with metadata |
| `split_dataset` | Split CSV into train/test files with optional stratification |

#### Tier 2 — ml-medium (14 tools)

Preprocessing pipelines, cross-validation, model comparison, clustering, EDA, and batch operations.

| Tool | Purpose |
|---|---|
| `run_preprocessing` | 15-op pipeline: encode, scale, fill, bin, log, date parts, drop nulls, clip |
| `detect_outliers` | IQR and std-dev outlier report per column |
| `train_with_cv` | K-fold cross-validation — per-fold scores + mean ± std |
| `compare_models` | Train multiple algorithms, return sorted comparison table |
| `run_clustering` | K-Means / Mean-Shift / DBSCAN with optional PCA/ICA pre-reduction |
| `read_receipt` | Read the full operation history log for any file |
| `generate_eda_report` | Interactive HTML EDA — quality score 0–100 + 8 alert types |
| `filter_rows` | Filter rows by column condition, save result CSV |
| `merge_datasets` | Merge two CSVs on a key column (left/right/inner/outer) |
| `find_optimal_clusters` | Elbow + silhouette chart to determine best K |
| `anomaly_detection` | Isolation Forest or LOF anomaly scoring |
| `check_data_quality` | Machine-readable quality score with typed alerts |
| `evaluate_model` | Score a saved model on an external labeled test CSV |
| `batch_predict` | Predict all rows with no row limit, save to CSV |

#### Tier 3 — ml-advanced (10 tools)

Hyperparameter tuning, model export, and full evaluation visualization suite.

| Tool | Purpose |
|---|---|
| `tune_hyperparameters` | GridSearch or RandomSearch with default or custom param grids |
| `export_model` | Pickle export with companion metadata manifest |
| `read_model_report` | Feature importance, confusion matrix, classification report |
| `run_profiling_report` | ydata-profiling HTML report (minimal mode for low-memory machines) |
| `apply_dimensionality_reduction` | PCA or ICA — saves reduced dataset CSV |
| `generate_training_report` | Full HTML report: confusion matrix + feature importance charts |
| `plot_roc_curve` | Interactive ROC curve with AUC |
| `plot_learning_curve` | Train vs validation score by training size |
| `plot_predictions_vs_actual` | Scatter of predicted vs actual for regression models |
| `generate_cluster_report` | HTML cluster report: PCA scatter + per-cluster profiles |

---

### Supported Algorithms

**Classification:** Logistic Regression · SVM (RBF) · Random Forest · Decision Tree · KNN · Naive Bayes · XGBoost

**Regression:** Linear · Polynomial · Lasso · Ridge · Decision Tree · Random Forest · XGBoost

**Clustering:** K-Means · Mean-Shift · DBSCAN

**Dimensionality Reduction:** PCA · ICA (FastICA)

**Hyperparameter Tuning:** GridSearchCV · RandomizedSearchCV

---

### Preprocessing Operations

`run_preprocessing` supports 15 operations in a single pipeline call:

`fill_nulls` · `drop_outliers` · `label_encode` · `onehot_encode` · `scale` · `drop_duplicates` · `drop_column` · `rename_column` · `convert_dtype` · `bin_numeric` · `add_date_parts` · `log_transform` · `drop_null_rows` · `clip_column`

---

### HTML Reports

All report tools produce standalone, interactive HTML files that work fully offline.

- Light and dark themes (`theme="light"` / `theme="dark"`)
- Responsive layout — readable on any screen size
- Auto-opens in browser after generation
- Saved to `~/Downloads/` by default

| Report | Tool |
|---|---|
| EDA with quality score + 8 alert types | `generate_eda_report` |
| Training metrics, confusion matrix, feature importance | `generate_training_report` |
| Deep ydata-profiling dataset profile | `run_profiling_report` |
| ROC curve with AUC | `plot_roc_curve` |
| Train vs validation learning curve | `plot_learning_curve` |
| Predicted vs actual scatter (regression) | `plot_predictions_vs_actual` |
| Elbow + silhouette chart | `find_optimal_clusters` |
| PCA scatter + per-cluster profiles | `generate_cluster_report` |

---

### Version Control & Audit

- Every write operation snapshots the target file to `.mcp_versions/` before overwriting
- Every operation is appended to a `.mcp_receipt.json` audit trail per file
- `restore_version` rolls back to any snapshot by timestamp or to the latest
- The `"backup"` key is always present in write tool responses

---

### Constrained Mode

Set `MCP_CONSTRAINED_MODE=1` in the server `env` block to reduce resource usage on lower-memory machines:

| Resource | Standard | Constrained |
|---|---|---|
| Rows per call | 100 | 20 |
| Search results | 50 | 10 |
| Columns returned | 50 | 20 |
| CV folds | 5 | 3 |
| Models in compare | 7 | 3 |
| ydata-profiling | full | minimal |

---

### Quality & Testing

- **253 tests** across all three tiers — happy path, error paths, dry-run, snapshot, constrained mode, and all algorithm variants
- **Ruff** lint and format enforced in CI
- **Pyright** type checking with zero errors
- **Tool docstring length** enforced at ≤ 80 characters (`verify_tool_docstrings.py`)
- **Multi-platform CI** — matrix: Ubuntu, Windows, macOS
- **Performance** — tested with datasets up to 250,000+ rows

---

### Bug Fixes

- **File paths on any drive now accepted** — `resolve_path()` previously rejected any path outside the user's home directory (e.g. `D:\MCP_Test\`). The check now blocks only null bytes and bare filesystem roots (`C:\`, `/`), allowing files on any drive or directory.
- **WebGL scatter plot fallback** — scatter plots now fall back gracefully when WebGL is unavailable in the browser.
- **Bar chart sort order** — bar charts now consistently show the highest values first.
- **HTML report text wrapping** — long column names and values no longer overflow report layout.
- **Large dataset performance** — resolved bottlenecks that caused timeouts on datasets with 250,000+ rows.

---

### Known Limitations

- Real-world testing verified on **Windows 11** only; macOS and Linux pass CI but have not been hand-tested
- `ml-advanced` should not be loaded simultaneously with `ml-basic + ml-medium` on machines with less than 16 GB RAM
- ONNX export reserved for a future release — only `pickle` format is supported in v0.1.0
- Deep learning (PyTorch/TensorFlow training) is out of scope for this release

---

### Installation

See [README.md](README.md) for the full quick-install guide.

**Requirements:** Python 3.12+, uv, Git, LM Studio (or any MCP-compatible client)
