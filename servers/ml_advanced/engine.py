"""ml_advanced engine — Tier 3 ML logic. Zero MCP imports."""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost as xgb
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shared.file_utils import resolve_path
from shared.platform_utils import get_cv_folds, is_constrained_mode
from shared.progress import info, ok, warn
from shared.receipt import append_receipt
from shared.version_control import snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CLASSIFIERS = {"lr", "svm", "rf", "dtc", "knn", "nb", "xgb"}
ALLOWED_REGRESSORS = {"lir", "pr", "lar", "rr", "dtr", "rfr", "xgb"}
MODELS_DIR = ".mcp_models"

DEFAULT_PARAMS: dict[str, dict] = {
    "svm": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "rf": {"n_estimators": [10, 50, 100], "max_depth": [None, 5, 10]},
    "xgb": {"max_depth": [3, 5, 7], "eta": [0.1, 0.3], "n_estimators": [50, 100]},
    "knn": {"n_neighbors": [3, 5, 7, 11]},
    "lr": {"C": [0.01, 0.1, 1, 10]},
    "dtc": {"max_depth": [None, 3, 5, 10]},
    "rfr": {"n_estimators": [10, 50, 100], "max_depth": [None, 5, 10]},
    "dtr": {"max_depth": [None, 3, 5, 10]},
    "lar": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "rr": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "lir": {},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(error: str, hint: str, backup: str | None = None) -> dict:
    base: dict = {"success": False, "error": error, "hint": hint}
    if backup:
        base["backup"] = backup
    base["token_estimate"] = len(str(base)) // 4
    return base


def _check_memory(required_gb: float) -> dict | None:
    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < required_gb:
        return {
            "success": False,
            "error": f"Insufficient RAM: need ~{required_gb:.1f} GB, available ~{available_gb:.1f} GB.",
            "hint": "Use read_rows() with a row limit or increase available memory.",
            "token_estimate": 60,
        }
    return None


def _auto_preprocess(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict, list[str]]:
    df = df.dropna(subset=[target_column]).copy()
    encoding_map: dict = {}
    encoded_cols: list[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("nan").astype(str))
            encoding_map[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            encoded_cols.append(col)
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df, encoding_map, encoded_cols


def _build_estimator(model: str, task: str) -> object:
    """Build a sklearn-compatible estimator for GridSearch/RandomSearch."""
    if model == "lr":
        return LogisticRegression(random_state=42, max_iter=200)
    if model == "svm":
        return SVC(kernel="rbf", gamma="auto", random_state=42)
    if model == "rf":
        return RandomForestClassifier(random_state=42)
    if model == "dtc":
        return DecisionTreeClassifier(random_state=42)
    if model == "knn":
        return KNeighborsClassifier()
    if model == "nb":
        return GaussianNB()
    if model == "lir":
        return LinearRegression()
    if model == "lar":
        return Lasso(max_iter=200, tol=0.1)
    if model == "rr":
        return Ridge(max_iter=100, tol=0.1)
    if model == "dtr":
        return DecisionTreeRegressor(random_state=42)
    if model == "rfr":
        return RandomForestRegressor(random_state=42)
    raise ValueError(f"Cannot build estimator for model '{model}'. XGBoost tuning uses defaults.")


def _save_model(model_obj: object, path: Path, metadata: dict) -> None:
    payload = {"model": model_obj, "metadata": metadata}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=path.parent) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(metadata, indent=2, default=str))


def _load_model(model_path: str) -> tuple[object, dict]:
    path = resolve_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload.get("model"), payload.get("metadata", {})


# ---------------------------------------------------------------------------
# Tool: tune_hyperparameters
# ---------------------------------------------------------------------------

MIN_ROWS_TUNE = 20


def tune_hyperparameters(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    search: str = "grid",
    param_grid: str = "",
    cv: int = 5,
    n_iter: int = 10,
    dry_run: bool = False,
) -> dict:
    """Tune hyperparameters via grid or random search. search: grid random."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    if task not in ("classification", "regression"):
        return _error(f"Unknown task: '{task}'.", "Use 'classification' or 'regression'.")

    allowed = ALLOWED_CLASSIFIERS if task == "classification" else ALLOWED_REGRESSORS
    if model not in allowed:
        return _error(
            f"Unknown algorithm: '{model}'. Allowed: {', '.join(sorted(allowed))}",
            f"Use one of: {' '.join(sorted(allowed))}",
        )

    if search not in ("grid", "random"):
        return _error(f"Unknown search: '{search}'.", "Use 'grid' or 'random'.")

    # Parse param_grid — JSON string or use defaults
    if param_grid:
        try:
            pg: dict = json.loads(param_grid)
        except json.JSONDecodeError as exc:
            return _error(f"Invalid param_grid JSON: {exc}", "Provide a valid JSON string for param_grid.")
    else:
        pg = DEFAULT_PARAMS.get(model, {})

    if not pg:
        return _error(
            f"No param grid available for model '{model}'.",
            "Provide a custom param_grid JSON string.",
        )

    # XGBoost not supported through sklearn GridSearch in this implementation
    if model == "xgb":
        return _error(
            "XGBoost tuning is not supported via GridSearch in this tier.",
            "Use 'rf', 'svm', 'lr', 'knn', 'dtc', 'rfr', 'dtr', 'lar', or 'rr' for tuning.",
        )

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if target_column not in df.columns:
        return _error(
            f"Column '{target_column}' not found.",
            "Use inspect_dataset() to list all column names.",
        )

    if len(df) < MIN_ROWS_TUNE:
        return _error(
            f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_TUNE}.",
            "Provide a dataset with more samples before tuning.",
        )

    # Constrained-mode limits
    cv = min(cv, get_cv_folds())
    if is_constrained_mode():
        n_iter = min(n_iter, 5)

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "tune_hyperparameters",
            "dry_run": True,
            "model": model,
            "task": task,
            "search": search,
            "cv": cv,
            "param_grid": pg,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    df, encoding_map, _ = _auto_preprocess(df, target_column)
    x = df.drop(columns=[target_column]).values
    y = df[target_column].values

    estimator = _build_estimator(model, task)
    scoring = "f1_weighted" if task == "classification" else "r2"

    if search == "grid":
        searcher = GridSearchCV(estimator, pg, cv=cv, scoring=scoring, return_train_score=False)
    else:
        searcher = RandomizedSearchCV(
            estimator, pg, cv=cv, n_iter=n_iter, scoring=scoring, random_state=42, return_train_score=False
        )

    progress.append(info(f"Running {search} search", f"cv={cv}"))
    searcher.fit(x, y)
    progress.append(ok("Search complete", f"best_score={searcher.best_score_:.4f}"))

    # Cap cv_results_ to top 20
    results_df = pd.DataFrame(searcher.cv_results_)
    results_df = results_df.sort_values("mean_test_score", ascending=False).head(20)
    top_results = results_df[["mean_test_score", "std_test_score", "params"]].to_dict("records")

    # Save best model
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    models_dir = path.parent / MODELS_DIR
    models_dir.mkdir(exist_ok=True)
    mp = models_dir / f"{path.stem}_{model}_tuned_{ts}.pkl"

    backup = ""
    if mp.exists():
        try:
            backup = snapshot(str(mp))
        except Exception:
            pass

    metadata = {
        "model_type": type(searcher.best_estimator_).__name__,
        "task": task,
        "trained_on": path.name,
        "training_date": datetime.now(UTC).isoformat(),
        "feature_columns": list(df.drop(columns=[target_column]).columns),
        "target_column": target_column,
        "encoding_map": encoding_map,
        "best_params": searcher.best_params_,
        "best_score": float(searcher.best_score_),
        "search_type": search,
        "metrics": {"best_score": float(searcher.best_score_)},
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
    }
    _save_model(searcher.best_estimator_, mp, metadata)
    progress.append(ok("Saved best model", mp.name))

    append_receipt(
        str(path), "tune_hyperparameters", {"model": model, "task": task, "search": search}, "success", backup
    )

    resp = {
        "success": True,
        "op": "tune_hyperparameters",
        "model": model,
        "task": task,
        "search": search,
        "best_score": round(float(searcher.best_score_), 4),
        "best_params": searcher.best_params_,
        "top_results": top_results,
        "model_path": str(mp),
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: export_model
# ---------------------------------------------------------------------------


def export_model(
    model_path: str,
    output_dir: str = "",
    format: str = "pickle",
    dry_run: bool = False,
) -> dict:
    """Export trained model with metadata manifest. format: pickle."""
    progress: list[dict] = []

    if format != "pickle":
        return _error(f"Format '{format}' not supported. Only 'pickle' is supported.", "Use format='pickle'.")

    try:
        src_path = resolve_path(model_path)
    except ValueError as exc:
        return _error(str(exc), "Check that model_path is inside your home directory.")
    if not src_path.exists():
        return _error(
            f"Model file not found: {model_path}", "Use train_classifier() or train_regressor() to train a model first."
        )  # noqa: E501
    if src_path.suffix.lower() != ".pkl":
        return _error(f"Expected .pkl file, got {src_path.suffix!r}", "Provide a path to a .pkl model file.")

    out_dir = Path(output_dir) if output_dir else src_path.parent
    try:
        out_dir_resolved = resolve_path(str(out_dir))
    except ValueError:
        out_dir_resolved = out_dir

    dst_path = out_dir_resolved / src_path.name
    manifest_dst = dst_path.with_suffix(".manifest.json")

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "export_model",
            "dry_run": True,
            "source": str(src_path),
            "destination": str(dst_path),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    backup = ""
    if dst_path.exists() and dst_path != src_path:
        try:
            backup = snapshot(str(dst_path))
            progress.append(ok("Snapshot created", Path(backup).name))
        except Exception as exc:
            progress.append(warn("Snapshot failed", str(exc)))

    # Load model to extract/build manifest
    try:
        model_obj, metadata = _load_model(str(src_path))
    except Exception as exc:
        return _error(f"Failed to load model: {exc}", "Check model_path points to a valid .pkl file.")

    out_dir_resolved.mkdir(parents=True, exist_ok=True)

    if dst_path != src_path:
        shutil.copy2(src_path, dst_path)

    # Build/update manifest
    manifest_data = {
        "model_type": metadata.get("model_type", "unknown"),
        "task": metadata.get("task", "unknown"),
        "trained_on": metadata.get("trained_on", "unknown"),
        "training_date": metadata.get("training_date", ""),
        "feature_columns": metadata.get("feature_columns", []),
        "target_column": metadata.get("target_column", ""),
        "metrics": metadata.get("metrics", {}),
        "python_version": metadata.get("python_version", sys.version),
        "sklearn_version": metadata.get("sklearn_version", sklearn.__version__),
        "xgboost_version": xgb.__version__,
    }
    manifest_dst.write_text(json.dumps(manifest_data, indent=2, default=str))
    progress.append(ok("Exported model", dst_path.name))
    progress.append(ok("Wrote manifest", manifest_dst.name))

    resp = {
        "success": True,
        "op": "export_model",
        "model_path": str(dst_path),
        "manifest_path": str(manifest_dst),
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: read_model_report
# ---------------------------------------------------------------------------


def read_model_report(model_path: str) -> dict:
    """Read model metrics, feature importance, confusion matrix summary."""
    progress: list[dict] = []
    try:
        path = resolve_path(model_path)
    except ValueError as exc:
        return _error(str(exc), "Check that model_path is inside your home directory.")
    if not path.exists():
        return _error(
            f"Model file not found: {model_path}", "Use train_classifier() or train_regressor() to train a model first."
        )  # noqa: E501

    try:
        model_obj, metadata = _load_model(str(path))
    except Exception as exc:
        return _error(f"Failed to load model: {exc}", "Check model_path points to a valid .pkl file.")

    progress.append(ok("Loaded model", path.name))

    task = metadata.get("task", "unknown")
    model_type = metadata.get("model_type", "unknown")
    metrics = metadata.get("metrics", {})
    feature_columns = metadata.get("feature_columns", [])

    # Feature importance (tree models)
    feature_importance: list[dict] = []
    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        fi_pairs = sorted(zip(feature_columns, importances.tolist()), key=lambda x: x[1], reverse=True)[:10]
        feature_importance = [{"feature": f, "importance": round(i, 4)} for f, i in fi_pairs]

    # Confusion matrix from metadata if stored
    confusion = metrics.get("confusion_matrix", {})

    # Classification report from metadata
    clf_report = metadata.get("classification_report", "")
    if clf_report and len(clf_report) > 500:
        clf_report = clf_report[:500]

    manifest_path = path.with_suffix(".manifest.json")
    manifest_data: dict = {}
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text())
        except Exception:
            pass

    resp: dict = {
        "success": True,
        "op": "read_model_report",
        "model_path": str(path),
        "model_type": model_type,
        "task": task,
        "trained_on": metadata.get("trained_on", ""),
        "training_date": metadata.get("training_date", ""),
        "feature_columns": feature_columns[:20],
        "target_column": metadata.get("target_column", ""),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": feature_importance,
        "classification_report": clf_report,
        "manifest": manifest_data,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: run_profiling_report
# ---------------------------------------------------------------------------


def run_profiling_report(
    file_path: str,
    output_path: str = "",
    sample_rows: int = 0,
    dry_run: bool = False,
) -> dict:
    """Generate ydata-profiling HTML report for dataset."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    out_path = Path(output_path) if output_path else path.with_suffix(".html")
    try:
        out_path_resolved = resolve_path(str(out_path))
    except ValueError:
        out_path_resolved = out_path

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "run_profiling_report",
            "dry_run": True,
            "output_path": str(out_path_resolved),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if sample_rows > 0 and sample_rows < len(df):
        df = df.sample(sample_rows, random_state=42)
        progress.append(info("Sampled dataset", f"{sample_rows} rows"))

    # Try ydata-profiling first; fall back to built-in Plotly report
    try:
        from ydata_profiling import ProfileReport

        minimal = is_constrained_mode()
        profile = ProfileReport(df, minimal=minimal, title=path.name)
        out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        profile.to_file(str(out_path_resolved))
        progress.append(ok("Saved ydata-profiling report", out_path_resolved.name))
        ydata_used = True
    except ImportError:
        ydata_used = False
        progress.append(info("ydata-profiling not installed", "Using built-in Plotly report"))

    if not ydata_used:
        # Built-in Plotly-based profiling report
        import plotly.graph_objects as go

        from shared.html_theme import (
            build_html_report,
            data_table_html,
            get_theme,
            metrics_cards_html,
            plotly_div,
        )

        t = get_theme("light")
        sections: list[dict] = []
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Overview cards
        missing_total = int(df.isnull().sum().sum())
        total_cells = len(df) * len(df.columns)
        miss_pct = round(missing_total / total_cells * 100, 2) if total_cells else 0.0
        dup_rows_n = int(df.duplicated().sum())
        sections.append(
            {
                "id": "overview",
                "heading": "Overview",
                "html": metrics_cards_html(
                    {
                        "rows": f"{len(df):,}",
                        "columns": len(df.columns),
                        "numeric_cols": len(numeric_cols),
                        "missing_cells": f"{miss_pct}%",
                        "duplicate_rows": dup_rows_n,
                    }
                ),
            }
        )

        # Distributions
        if numeric_cols:
            from plotly.subplots import make_subplots

            show = numeric_cols[:12]
            cols_n = min(3, len(show))
            rows_n = (len(show) + cols_n - 1) // cols_n
            fig = make_subplots(rows=rows_n, cols=cols_n, subplot_titles=show)
            for i, col in enumerate(show):
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, showlegend=False, marker_color=t["accent"]),
                    row=i // cols_n + 1,
                    col=i % cols_n + 1,
                )
            fig.update_layout(
                title="Distributions",
                template=t["plotly_template"],
                paper_bgcolor=t["paper_color"],
                plot_bgcolor=t["bg_color"],
                font_color=t["text_color"],
                height=280 * rows_n,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            sections.append(
                {"id": "dist", "heading": "Distributions", "html": plotly_div(fig, height=280 * rows_n + 60)}
            )

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_c = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                )
            )
            fig_c.update_layout(
                title="Correlation",
                template=t["plotly_template"],
                paper_bgcolor=t["paper_color"],
                plot_bgcolor=t["bg_color"],
                font_color=t["text_color"],
                height=480,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            sections.append({"id": "corr", "heading": "Correlation Heatmap", "html": plotly_div(fig_c, 520)})

        if numeric_cols:
            desc = df[numeric_cols[:12]].describe().round(3).transpose()
            desc.index.name = "column"
            rows_data = [{"column": idx, **row.to_dict()} for idx, row in desc.iterrows()]
            sections.append({"id": "stats", "heading": "Summary Statistics", "html": data_table_html(rows_data)})

        plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        from datetime import datetime as _dt

        html = build_html_report(
            title=f"Profile Report — {path.name}",
            subtitle=f"Generated {_dt.now().strftime('%Y-%m-%d %H:%M')} · {len(df):,} rows",
            sections=sections,
            theme="light",
            open_browser=False,
            output_path="",
        )
        html = html.replace("</head>", f"  {plotly_cdn}\n</head>")
        out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        out_path_resolved.write_text(html, encoding="utf-8")
        progress.append(ok("Saved Plotly profile report", out_path_resolved.name))

    from shared.html_theme import _open_file

    _open_file(out_path_resolved)

    file_size_kb = out_path_resolved.stat().st_size // 1024 if out_path_resolved.exists() else 0
    missing_pct_r = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
    dup_rows_r = int(df.duplicated().sum())
    dup_pct_r = round(dup_rows_r / len(df) * 100, 2) if len(df) > 0 else 0.0

    resp = {
        "success": True,
        "op": "run_profiling_report",
        "output_path": str(out_path_resolved),
        "file_size_kb": file_size_kb,
        "engine_used": "ydata-profiling" if ydata_used else "plotly",
        "summary": {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_cells_pct": missing_pct_r,
            "duplicate_rows": dup_rows_r,
            "duplicate_rows_pct": dup_pct_r,
        },
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: apply_dimensionality_reduction
# ---------------------------------------------------------------------------


def apply_dimensionality_reduction(
    file_path: str,
    feature_columns: list[str],
    method: str,
    n_components: int = 2,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Reduce dimensions with PCA or ICA. Saves reduced dataset."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    if method not in ("pca", "ica"):
        return _error(f"Unknown method: '{method}'.", "Use 'pca' or 'ica'.")

    out_path_str = output_path or str(path.parent / f"{path.stem}_{method}_reduced.csv")
    try:
        out_path = resolve_path(out_path_str)
    except ValueError:
        out_path = Path(out_path_str)

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return _error(
            f"Columns not found: {', '.join(missing[:5])}",
            "Use inspect_dataset() to list valid column names.",
        )

    nc = min(n_components, len(feature_columns))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "apply_dimensionality_reduction",
            "dry_run": True,
            "method": method,
            "n_components": nc,
            "output_path": str(out_path),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    x = df[feature_columns].values.astype(float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    progress.append(ok("Scaled features", "StandardScaler"))

    variance_explained: list[float] = []
    if method == "pca":
        reducer = PCA(n_components=nc)
        x_reduced = reducer.fit_transform(x_scaled)
        variance_explained = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        progress.append(ok("PCA complete", f"variance={sum(variance_explained):.3f}"))
    else:
        reducer = FastICA(n_components=nc, random_state=42)
        x_reduced = reducer.fit_transform(x_scaled)
        progress.append(ok("ICA complete", f"{nc} components"))

    # Build output DataFrame: drop feature_columns, add component columns
    df_out = df.drop(columns=feature_columns).copy()
    for i in range(nc):
        df_out[f"component_{i + 1}"] = x_reduced[:, i]

    backup = ""
    if out_path.exists():
        try:
            backup = snapshot(str(out_path))
            progress.append(ok("Snapshot created", Path(backup).name))
        except Exception as exc:
            progress.append(warn("Snapshot failed", str(exc)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    progress.append(ok("Saved reduced dataset", out_path.name))

    append_receipt(
        str(path),
        "apply_dimensionality_reduction",
        {"method": method, "n_components": nc, "feature_columns": feature_columns},
        "success",
        backup,
    )

    resp = {
        "success": True,
        "op": "apply_dimensionality_reduction",
        "method": method,
        "n_components": nc,
        "feature_columns": feature_columns,
        "output_path": str(out_path),
        "variance_explained": variance_explained,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: generate_training_report
# ---------------------------------------------------------------------------


def generate_training_report(
    model_path: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate HTML report with metrics, confusion matrix, feature importance."""
    import plotly.express as px
    import plotly.graph_objects as go

    from shared.html_theme import (
        _open_file,
        build_html_report,
        data_table_html,
        get_theme,
        metrics_cards_html,
        plotly_div,
    )

    progress: list[dict] = []
    try:
        mp = resolve_path(model_path)
    except ValueError as exc:
        return _error(str(exc), "Check that model_path is inside your home directory.")
    if not mp.exists():
        return _error(
            f"Model file not found: {model_path}", "Train a model first with train_classifier() or train_regressor()."
        )  # noqa: E501
    if mp.suffix.lower() != ".pkl":
        return _error(f"Expected .pkl file, got {mp.suffix!r}", "Provide a path to a .pkl model file.")

    out_path_str = output_path or str(mp.parent / f"{mp.stem}_report.html")
    try:
        out_path = resolve_path(out_path_str)
    except ValueError:
        out_path = Path(out_path_str)

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "generate_training_report",
            "dry_run": True,
            "output_path": str(out_path),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    try:
        model_obj, metadata = _load_model(str(mp))
    except Exception as exc:
        return _error(f"Failed to load model: {exc}", "Check model_path points to a valid .pkl file.")
    progress.append(ok("Loaded model", mp.name))

    t = get_theme(theme)
    sections: list[dict] = []

    # --- Model overview ---
    task = metadata.get("task", "unknown")
    model_type = metadata.get("model_type", "unknown")
    metrics = metadata.get("metrics", {})
    feature_columns = metadata.get("feature_columns", [])
    target_column = metadata.get("target_column", "")
    training_date = metadata.get("training_date", "")
    trained_on = metadata.get("trained_on", "")

    overview_cards: dict = {
        "model": model_type,
        "task": task,
        "trained_on": trained_on,
        "features": len(feature_columns),
        "target": target_column,
    }
    # Add top metrics to overview
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "classification_report") and not isinstance(v, dict):
            overview_cards[k] = v

    sections.append(
        {
            "id": "overview",
            "heading": "Model Overview",
            "html": metrics_cards_html(overview_cards),
        }
    )
    progress.append(ok("Overview section", model_type))

    # --- Metrics section ---
    clean_metrics = {
        k: v
        for k, v in metrics.items()
        if k not in ("confusion_matrix", "classification_report") and not isinstance(v, dict)
    }
    if clean_metrics:
        metric_rows = [
            {"metric": k.replace("_", " ").title(), "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in clean_metrics.items()
        ]
        sections.append(
            {
                "id": "metrics",
                "heading": "Performance Metrics",
                "html": data_table_html(metric_rows),
            }
        )

    # --- Confusion matrix heatmap ---
    cm_data = metrics.get("confusion_matrix", {})
    if cm_data:
        if set(cm_data.keys()) <= {"TP", "FP", "FN", "TN"}:
            # Binary
            tp = cm_data.get("TP", 0)
            fp = cm_data.get("FP", 0)
            fn = cm_data.get("FN", 0)
            tn = cm_data.get("TN", 0)
            z = [[tn, fp], [fn, tp]]
            labels = ["Negative", "Positive"]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=[[str(v) for v in row] for row in z],
                    texttemplate="%{text}",
                    showscale=True,
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                template=t["plotly_template"],
                paper_bgcolor=t["paper_color"],
                plot_bgcolor=t["bg_color"],
                font_color=t["text_color"],
                height=380,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            sections.append(
                {
                    "id": "confusion",
                    "heading": "Confusion Matrix",
                    "html": plotly_div(fig_cm, height=420),
                }
            )
            progress.append(ok("Confusion matrix chart", "binary classification"))

    # --- Feature importance ---
    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        fi_pairs = sorted(zip(feature_columns, importances.tolist()), key=lambda x: x[1], reverse=True)[:20]
        fi_features = [p[0] for p in fi_pairs]
        fi_values = [p[1] for p in fi_pairs]

        fig_fi = px.bar(
            x=fi_values,
            y=fi_features,
            orientation="h",
            title="Feature Importance (Top 20)",
            labels={"x": "Importance", "y": "Feature"},
            template=t["plotly_template"],
            color=fi_values,
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            paper_bgcolor=t["paper_color"],
            plot_bgcolor=t["bg_color"],
            font_color=t["text_color"],
            height=max(300, len(fi_pairs) * 28 + 80),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append(
            {
                "id": "features",
                "heading": "Feature Importance",
                "html": plotly_div(fig_fi, height=max(300, len(fi_pairs) * 28 + 120)),
            }
        )
        progress.append(ok("Feature importance chart", f"{len(fi_pairs)} features"))

    # --- Classification report table ---
    clf_report = metadata.get("classification_report", "")
    if clf_report:
        sections.append(
            {
                "id": "clf_report",
                "heading": "Classification Report",
                "html": f"<pre style='font-size:0.85rem;overflow-x:auto;'>{clf_report[:2000]}</pre>",
            }
        )

    # --- Build report ---
    subtitle = f"Model: {model_type} · Task: {task} · Trained: {training_date[:10] if training_date else 'unknown'}"

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    html = build_html_report(
        title=f"Training Report — {mp.stem}",
        subtitle=subtitle,
        sections=sections,
        theme=theme,
        open_browser=False,
        output_path="",
    )
    html = html.replace("</head>", f"  {plotly_cdn}\n</head>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    progress.append(ok("Saved HTML report", out_path.name))

    if open_browser:
        _open_file(out_path)

    file_size_kb = out_path.stat().st_size // 1024

    resp = {
        "success": True,
        "op": "generate_training_report",
        "model_path": str(mp),
        "output_path": str(out_path),
        "file_size_kb": file_size_kb,
        "sections_generated": len(sections),
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    model_path: str,
    file_path: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Plot ROC curve for a classifier model. Saves interactive HTML."""
    import pickle

    import pandas as pd

    from shared.file_utils import resolve_path
    from shared.html_theme import get_theme, save_chart
    from shared.progress import info, ok

    progress = []
    try:
        mp = resolve_path(model_path)
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file paths.", "token_estimate": 30}

    if not mp.exists():
        return {
            "success": False,
            "error": f"Model not found: {model_path}",
            "hint": "Train a classifier first.",
            "token_estimate": 30,
        }
    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    if dry_run:
        return {
            "success": True,
            "op": "plot_roc_curve",
            "dry_run": True,
            "model_path": str(mp),
            "file_path": str(dp),
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.graph_objects as go
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "classification")
        if task != "classification":
            return {
                "success": False,
                "error": "ROC curve is only for classifiers.",
                "hint": "Use plot_predictions_vs_actual for regression.",
                "token_estimate": 30,
            }

        feature_columns = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", "")
        encoding_map = metadata.get("encoding_map", {})

        df = pd.read_csv(dp, low_memory=False)
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        # Encode
        for col, mapping in encoding_map.items():
            if col in df.columns and col != target_column:
                df[col] = df[col].map(mapping).fillna(df[col])

        available = [c for c in feature_columns if c in df.columns]
        if not available:
            return {
                "success": False,
                "error": "No feature columns found in dataset.",
                "hint": "Use the same dataset used for training.",
                "token_estimate": 30,
            }

        X = df[available].select_dtypes(include="number").fillna(0)
        y_true = df[target_column] if target_column in df.columns else None

        if y_true is None:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not found.",
                "hint": "Provide the same dataset used for training.",
                "token_estimate": 30,
            }

        # Encode target if needed
        if y_true.dtype == object or str(y_true.dtype) in ("string", "StringDtype"):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_true = pd.Series(le.fit_transform(y_true))

        classes = sorted(y_true.unique())
        n_classes = len(classes)

        # Get probabilities
        has_proba = hasattr(model, "predict_proba")
        if not has_proba:
            # Try XGBoost Booster
            try:
                import xgboost as xgb

                if isinstance(model, xgb.Booster):
                    dmat = xgb.DMatrix(X)
                    raw_preds = model.predict(dmat)
                    if raw_preds.ndim == 1:
                        y_prob = np.column_stack([1 - raw_preds, raw_preds])
                    else:
                        y_prob = raw_preds
                    has_proba = True
            except Exception:
                pass

        if not has_proba:
            return {
                "success": False,
                "error": "Model does not support probability estimates for ROC curve.",
                "hint": "Use a classifier that supports predict_proba (lr, rf, xgb, nb, dtc).",
                "token_estimate": 40,
            }

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)

        t = get_theme(theme)
        fig = go.Figure()

        auc_scores = {}
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            auc_scores["binary"] = round(roc_auc, 4)
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})", line=dict(color=t["accent"], width=2)
                )
            )
        else:
            y_bin = label_binarize(y_true, classes=classes)
            colors = [t["accent"], t["success"], t["warning"], t["danger"]]
            for i, cls in enumerate(classes[:10]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores[str(cls)] = round(roc_auc, 4)
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"Class {cls} (AUC={roc_auc:.3f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )

        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.5)", line=dict(color=t["grid_color"], dash="dash")
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1),
            template=t["plotly_template"],
        )

        out_path_str = output_path or str(dp.parent / f"{dp.stem}_roc_curve.html")
        out_abs, out_name = save_chart(
            fig, out_path_str, theme=theme, open_browser=open_browser, title=f"ROC Curve — {mp.stem}"
        )
        progress.append(ok("Saved ROC curve", out_name))

        resp = {
            "success": True,
            "op": "plot_roc_curve",
            "output_path": out_abs,
            "auc_scores": auc_scores,
            "n_classes": n_classes,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check model and data compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# plot_learning_curve
# ---------------------------------------------------------------------------


def plot_learning_curve(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    cv: int = 5,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Plot learning curve (train vs val score vs training size). HTML output."""
    import pandas as pd

    from shared.file_utils import resolve_path
    from shared.html_theme import get_theme, save_chart
    from shared.progress import info, ok

    progress = []
    try:
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file path.", "token_estimate": 30}

    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    if dry_run:
        return {
            "success": True,
            "op": "plot_learning_curve",
            "dry_run": True,
            "file_path": str(dp),
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.graph_objects as go
        from sklearn.model_selection import learning_curve
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv(dp, low_memory=False)
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not found.",
                "hint": "Use inspect_dataset() to list column names.",
                "token_estimate": 30,
            }

        df = df.dropna(subset=[target_column])
        y = df[target_column]
        # Encode categoricals
        for col in df.select_dtypes(include=["object", "string"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        X = df.drop(columns=[target_column]).select_dtypes(include="number").fillna(0)

        # Build estimator
        CLASSIFIERS = {
            "lr": ("sklearn.linear_model", "LogisticRegression", {"random_state": 42, "max_iter": 200}),
            "rf": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50, "random_state": 42}),
            "dtc": ("sklearn.tree", "DecisionTreeClassifier", {"random_state": 42}),
            "knn": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
            "svm": ("sklearn.svm", "SVC", {"kernel": "rbf", "gamma": "auto", "random_state": 42}),
        }
        REGRESSORS = {
            "lir": ("sklearn.linear_model", "LinearRegression", {}),
            "rfr": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 50, "random_state": 42}),
            "dtr": ("sklearn.tree", "DecisionTreeRegressor", {"random_state": 42}),
        }
        model_map = CLASSIFIERS if task == "classification" else REGRESSORS
        if model not in model_map:
            allowed = ", ".join(model_map.keys())
            return {
                "success": False,
                "error": f"Unknown model '{model}'. Allowed: {allowed}",
                "hint": "Check model string.",
                "token_estimate": 30,
            }

        mod_name, cls_name, kwargs = model_map[model]
        import importlib

        cls = getattr(importlib.import_module(mod_name), cls_name)
        estimator = cls(**kwargs)

        scoring = "accuracy" if task == "classification" else "r2"
        train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
        )
        progress.append(ok("Computed learning curves", f"{len(train_sizes_abs)} points"))

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        t = get_theme(theme)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=train_sizes_abs.tolist(),
                y=train_mean.tolist(),
                mode="lines+markers",
                name="Train score",
                line=dict(color=t["accent"], width=2),
                error_y=dict(type="data", array=train_std.tolist(), visible=True, color=t["accent"]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=train_sizes_abs.tolist(),
                y=val_mean.tolist(),
                mode="lines+markers",
                name="Validation score",
                line=dict(color=t["success"], width=2),
                error_y=dict(type="data", array=val_std.tolist(), visible=True, color=t["success"]),
            )
        )

        fig.update_layout(
            title=f"Learning Curve — {model} ({task})",
            xaxis_title="Training Examples",
            yaxis_title=scoring.upper(),
            template=t["plotly_template"],
        )

        out_path_str = output_path or str(dp.parent / f"{dp.stem}_{model}_learning_curve.html")
        out_abs, out_name = save_chart(
            fig, out_path_str, theme=theme, open_browser=open_browser, title=f"Learning Curve — {model}"
        )
        progress.append(ok("Saved learning curve", out_name))

        resp = {
            "success": True,
            "op": "plot_learning_curve",
            "output_path": out_abs,
            "final_train_score": round(float(train_mean[-1]), 4),
            "final_val_score": round(float(val_mean[-1]), 4),
            "scoring": scoring,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check data and model compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# plot_predictions_vs_actual
# ---------------------------------------------------------------------------


def plot_predictions_vs_actual(
    model_path: str,
    file_path: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Scatter plot of predicted vs actual values for regression. HTML."""
    import pickle

    import pandas as pd

    from shared.file_utils import resolve_path
    from shared.html_theme import get_theme, save_chart
    from shared.progress import info, ok

    progress = []
    try:
        mp = resolve_path(model_path)
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file paths.", "token_estimate": 30}

    if not mp.exists():
        return {
            "success": False,
            "error": f"Model not found: {model_path}",
            "hint": "Train a regressor first.",
            "token_estimate": 30,
        }
    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    if dry_run:
        return {
            "success": True,
            "op": "plot_predictions_vs_actual",
            "dry_run": True,
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.graph_objects as go
        from sklearn.metrics import mean_squared_error, r2_score

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "regression")
        if task != "regression":
            return {
                "success": False,
                "error": "This chart is only for regression models.",
                "hint": "Use plot_roc_curve for classifiers.",
                "token_estimate": 30,
            }

        feature_columns = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", "")
        encoding_map = metadata.get("encoding_map", {})

        df = pd.read_csv(dp, low_memory=False)
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        for col, mapping in encoding_map.items():
            if col in df.columns and col != target_column:
                df[col] = df[col].map(mapping).fillna(df[col])

        available = [c for c in feature_columns if c in df.columns]
        X = df[available].select_dtypes(include="number").fillna(0)
        y_true = df[target_column].values if target_column in df.columns else None

        if y_true is None:
            return {
                "success": False,
                "error": f"Target '{target_column}' not found.",
                "hint": "Provide the same dataset used for training.",
                "token_estimate": 30,
            }

        try:
            import xgboost as xgb

            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(X)
                y_pred = model.predict(dmat)
            else:
                y_pred = model.predict(X)
        except Exception:
            y_pred = model.predict(X)

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        t = get_theme(theme)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_true.tolist(),
                y=y_pred.tolist(),
                mode="markers",
                name="Predictions",
                marker=dict(color=t["accent"], opacity=0.6, size=6),
            )
        )
        # Perfect prediction line
        mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        fig.add_trace(
            go.Scatter(
                x=[mn, mx],
                y=[mn, mx],
                mode="lines",
                name="Perfect fit",
                line=dict(color=t["danger"], dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Predictions vs Actual — R²={r2:.3f}",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            template=t["plotly_template"],
        )

        out_path_str = output_path or str(dp.parent / f"{dp.stem}_pred_vs_actual.html")
        out_abs, out_name = save_chart(
            fig, out_path_str, theme=theme, open_browser=open_browser, title="Predictions vs Actual"
        )
        progress.append(ok("Saved chart", out_name))

        resp = {
            "success": True,
            "op": "plot_predictions_vs_actual",
            "output_path": out_abs,
            "metrics": {"mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
            "n_points": len(y_true),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check model and data compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# generate_cluster_report
# ---------------------------------------------------------------------------


def generate_cluster_report(
    file_path: str,
    feature_columns: list[str],
    label_column: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate HTML cluster visualization report with scatter and profile."""
    import pandas as pd

    from shared.file_utils import resolve_path
    from shared.html_theme import (
        _open_file,
        build_html_report,
        data_table_html,
        get_theme,
        metrics_cards_html,
        plotly_div,
    )
    from shared.progress import info, ok

    progress = []
    try:
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file path.", "token_estimate": 30}

    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    if dry_run:
        return {
            "success": True,
            "op": "generate_cluster_report",
            "dry_run": True,
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(dp, low_memory=False)
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        if label_column not in df.columns:
            return {
                "success": False,
                "error": f"Label column '{label_column}' not found.",
                "hint": "Use run_clustering(save_labels=True) first.",
                "token_estimate": 30,
            }

        missing_feats = [c for c in feature_columns if c not in df.columns]
        if missing_feats:
            return {
                "success": False,
                "error": f"Feature columns not found: {', '.join(missing_feats[:5])}",
                "hint": "Use inspect_dataset() to list column names.",
                "token_estimate": 30,
            }

        X = df[feature_columns].select_dtypes(include="number").fillna(0)
        labels = df[label_column].astype(str)
        n_clusters = labels.nunique()
        t = get_theme(theme)
        sections = []

        # --- Summary cards ---
        label_counts = labels.value_counts().to_dict()
        summary = {
            "n_clusters": n_clusters,
            "n_samples": len(df),
            "n_features": len(X.columns),
        }
        sections.append(
            {
                "id": "summary",
                "heading": "Summary",
                "html": metrics_cards_html(summary),
            }
        )

        # --- Cluster size table ---
        size_rows = [
            {"cluster": str(k), "count": int(v), "pct": f"{v / len(df) * 100:.1f}%"} for k, v in label_counts.items()
        ]
        sections.append(
            {
                "id": "cluster_sizes",
                "heading": "Cluster Sizes",
                "html": data_table_html(size_rows),
            }
        )

        # --- PCA scatter (2D) ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(2, X_scaled.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)
            explained = [round(float(v), 3) for v in pca.explained_variance_ratio_]

            scatter_df = pd.DataFrame(
                {
                    "PC1": coords[:, 0],
                    "PC2": coords[:, 1],
                    "cluster": labels.values,
                }
            )
            fig_scatter = px.scatter(
                scatter_df,
                x="PC1",
                y="PC2",
                color="cluster",
                title=f"PCA Scatter — {explained[0] * 100:.1f}%/{explained[1] * 100:.1f}% variance",
                template=t["plotly_template"],
            )
            sections.append(
                {
                    "id": "scatter",
                    "heading": "PCA Cluster Scatter",
                    "html": plotly_div(fig_scatter, height=450),
                }
            )
            progress.append(ok("PCA scatter", f"var explained: {sum(explained) * 100:.1f}%"))

        # --- Feature means per cluster ---
        df_feat = X.copy()
        df_feat["_cluster"] = labels.values
        cluster_profile = df_feat.groupby("_cluster").mean().round(3)
        profile_rows = []
        for idx, row in cluster_profile.iterrows():
            r = {"cluster": str(idx)}
            r.update({col: round(float(val), 3) for col, val in row.items()})
            profile_rows.append(r)

        sections.append(
            {
                "id": "feature_profile",
                "heading": "Feature Means by Cluster",
                "html": data_table_html(profile_rows),
            }
        )

        # --- Bar chart: cluster sizes ---
        fig_bar = go.Figure(
            go.Bar(
                x=list(label_counts.keys()),
                y=list(label_counts.values()),
                marker_color=t["accent"],
            )
        )
        fig_bar.update_layout(
            title="Cluster Sizes",
            xaxis_title="Cluster",
            yaxis_title="Count",
            template=t["plotly_template"],
        )
        sections.append(
            {
                "id": "size_chart",
                "heading": "Cluster Size Chart",
                "html": plotly_div(fig_bar, height=300),
            }
        )

        out_path_str = output_path or str(dp.parent / f"{dp.stem}_cluster_report.html")
        plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        html = build_html_report(
            title=f"Cluster Report — {dp.name}",
            subtitle=f"Clusters: {n_clusters} · Samples: {len(df)} · Features: {len(feature_columns)}",
            sections=sections,
            theme=theme,
            open_browser=False,
            output_path="",
        )
        html = html.replace("</head>", f"  {plotly_cdn}\n</head>")

        from pathlib import Path

        out = Path(out_path_str).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        progress.append(ok("Saved cluster report", out.name))

        if open_browser:
            _open_file(out)

        resp = {
            "success": True,
            "op": "generate_cluster_report",
            "output_path": str(out),
            "n_clusters": n_clusters,
            "n_samples": len(df),
            "sections_generated": len(sections),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {"success": False, "error": str(exc), "hint": "Check data and label column.", "token_estimate": 30}
