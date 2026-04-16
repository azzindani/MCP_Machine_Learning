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
from shared.handover import make_handover
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

    results_df = pd.DataFrame(searcher.cv_results_)
    results_df = results_df.sort_values("mean_test_score", ascending=False).head(20)
    top_results = results_df[["mean_test_score", "std_test_score", "params"]].to_dict("records")

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
    resp["handover"] = make_handover(
        "PATCH",
        ["evaluate_model", "generate_training_report", "read_model_report"],
        {"model_path": str(mp), "file_path": file_path},
    )
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

    try:
        model_obj, metadata = _load_model(str(src_path))
    except Exception as exc:
        return _error(f"Failed to load model: {exc}", "Check model_path points to a valid .pkl file.")

    out_dir_resolved.mkdir(parents=True, exist_ok=True)

    if dst_path != src_path:
        shutil.copy2(src_path, dst_path)

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

    feature_importance: list[dict] = []
    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        fi_pairs = sorted(zip(feature_columns, importances.tolist()), key=lambda x: x[1], reverse=True)[:10]
        feature_importance = [{"feature": f, "importance": round(i, 4)} for f, i in fi_pairs]

    confusion = metrics.get("confusion_matrix", {})

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
    resp["handover"] = make_handover(
        "VERIFY",
        ["generate_training_report", "evaluate_model", "plot_roc_curve"],
        {"model_path": model_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
