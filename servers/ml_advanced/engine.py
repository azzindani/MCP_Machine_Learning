"""ml_advanced engine — Tier 3 ML logic. Zero MCP imports."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from shared.file_utils import atomic_write_json, atomic_write_text, get_output_dir, resolve_path
from shared.platform_utils import get_cv_folds, get_n_iter, is_constrained_mode
from shared.progress import info, ok, warn
from shared.receipt import append_receipt
from shared.version_control import snapshot

from ._adv_helpers import (
    ALLOWED_CLASSIFIERS,
    ALLOWED_REGRESSORS,
    DEFAULT_PARAMS,
    _auto_preprocess,
    _build_estimator,
    _error,
    _load_model,
    _save_model,
    get_output_path,
)
from ._adv_viz import (
    generate_cluster_report,
    plot_learning_curve,
    plot_predictions_vs_actual,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)

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
    n_iter = min(n_iter, get_n_iter())

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
    import os as _os

    _override = _os.environ.get("MCP_OUTPUT_DIR")
    models_dir = Path(_override) if _override else path.parent / ".mcp_models"
    models_dir.mkdir(parents=True, exist_ok=True)
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
    atomic_write_json(manifest_dst, manifest_data)
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
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate Plotly HTML profile report for a dataset."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    out_path_resolved = get_output_path(output_path, path, "profile", "html")

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

    import plotly.graph_objects as go

    from shared.html_theme import (
        apply_fig_theme,
        build_html_report,
        calc_chart_height,
        data_table_html,
        get_theme,
        metrics_cards_html,
        plotly_div,
        plotly_template,
    )

    t = get_theme("dark")
    tmpl = plotly_template("dark")
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
        prof_dist_h = calc_chart_height(rows_n, mode="subplot")
        fig.update_layout(
            title="Distributions",
            template=tmpl,
            height=prof_dist_h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig, "dark")
        sections.append(
            {"id": "dist", "heading": "Distributions", "html": plotly_div(fig, height=prof_dist_h, theme="dark")}
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
        prof_corr_h = calc_chart_height(len(numeric_cols), mode="heatmap", extra_base=60)
        fig_c.update_layout(
            title="Correlation",
            template=tmpl,
            height=prof_corr_h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig_c, "dark")
        sections.append(
            {
                "id": "corr",
                "heading": "Correlation Heatmap",
                "html": plotly_div(fig_c, height=prof_corr_h, theme="dark"),
            }
        )

    if numeric_cols:
        desc = df[numeric_cols[:12]].describe().round(3).transpose()
        desc.index.name = "column"
        rows_data = [{"column": idx, **row.to_dict()} for idx, row in desc.iterrows()]
        sections.append({"id": "stats", "heading": "Summary Statistics", "html": data_table_html(rows_data)})

    html = build_html_report(
        title=f"Profile Report — {path.name}",
        subtitle="",
        sections=sections,
        theme="dark",
        open_after=False,
        output_path="",
        sidebar_title="Profile Report",
        sidebar_meta=f"{path.name}<br>{len(df):,} rows",
    )
    out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path_resolved, html)
    progress.append(ok("Saved profile report", out_path_resolved.name))

    if open_after:
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
        "output_name": out_path_resolved.name,
        "file_size_kb": file_size_kb,
        "engine_used": "plotly",
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

    out_path_str = output_path or str(get_output_dir() / f"{path.stem}_{method}_reduced.csv")
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
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate HTML report with metrics, confusion matrix, feature importance."""
    import plotly.express as px
    import plotly.graph_objects as go

    from shared.html_theme import (
        _open_file,
        apply_fig_theme,
        build_html_report,
        calc_chart_height,
        data_table_html,
        metrics_cards_html,
        plotly_div,
        plotly_template,
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

    out_path = get_output_path(output_path, mp, "report", "html")

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

    tmpl = plotly_template(theme)
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
                template=tmpl,
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            apply_fig_theme(fig_cm, theme)
            sections.append(
                {
                    "id": "confusion",
                    "heading": "Confusion Matrix",
                    "html": plotly_div(fig_cm, height=400, theme=theme),
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
            template=tmpl,
            color=fi_values,
            color_continuous_scale="Blues",
        )
        fi_h = calc_chart_height(len(fi_pairs), mode="bar")
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            height=fi_h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig_fi, theme)
        sections.append(
            {
                "id": "features",
                "heading": "Feature Importance",
                "html": plotly_div(fig_fi, height=fi_h, theme=theme),
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
    trained_str = training_date[:10] if training_date else "unknown"
    html = build_html_report(
        title=f"Training Report — {mp.stem}",
        subtitle="",
        sections=sections,
        theme=theme,
        open_after=False,
        output_path="",
        sidebar_title="Training Report",
        sidebar_meta=f"{model_type}<br>Task: {task}<br>Trained: {trained_str}",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, html)
    progress.append(ok("Saved HTML report", out_path.name))

    if open_after:
        _open_file(out_path)

    file_size_kb = out_path.stat().st_size // 1024

    resp = {
        "success": True,
        "op": "generate_training_report",
        "model_path": str(mp),
        "output_path": str(out_path),
        "output_name": out_path.name,
        "file_size_kb": file_size_kb,
        "sections_generated": len(sections),
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


__all__ = [
    # core tools
    "apply_dimensionality_reduction",
    "export_model",
    "generate_training_report",
    "read_model_report",
    "run_profiling_report",
    "tune_hyperparameters",
    # viz sub-module re-exports
    "generate_cluster_report",
    "plot_learning_curve",
    "plot_predictions_vs_actual",
    "plot_roc_curve",
]
