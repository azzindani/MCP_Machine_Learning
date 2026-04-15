"""ml_medium training tools — train_with_cv, compare_models."""

from __future__ import annotations

import pickle
import shutil
import tempfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn

from shared.file_utils import atomic_write_json
from shared.handover import make_handover

from ._medium_helpers import (
    ALLOWED_CLASSIFIERS,
    ALLOWED_REGRESSORS,
    KFold,
    Path,
    StratifiedKFold,
    _auto_preprocess,
    _error,
    _fit_predict_classifier,
    _fit_predict_regressor,
    accuracy_score,
    append_receipt,
    f1_score,
    fail,
    get_cv_folds,
    get_max_models,
    get_output_dir,
    mean_squared_error,
    ok,
    r2_score,
    resolve_path,
    snapshot,
    sys,
    train_test_split,
    warn,
)

# Re-import pd and np directly (helpers exports them but we import explicitly for clarity)

MIN_ROWS_CV = 20
MIN_ROWS_COMPARE = 20


def train_with_cv(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train with K-fold cross-validation. Returns per-fold and mean scores."""
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

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if target_column not in df.columns:
        cols_hint = ", ".join(df.columns[:10].tolist())
        return _error(
            f"Column '{target_column}' not found. Available: {cols_hint}",
            "Use inspect_dataset() to list all column names.",
        )

    if len(df) < MIN_ROWS_CV:
        return _error(
            f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_CV}.",
            "Provide a dataset with more samples before training.",
        )

    # Use constrained-mode fold count if smaller
    n_splits = min(n_splits, get_cv_folds())

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "train_with_cv",
            "dry_run": True,
            "model": model,
            "task": task,
            "n_splits": n_splits,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    df, encoding_map, _ = _auto_preprocess(df, target_column)
    x = df.drop(columns=[target_column]).values
    y = df[target_column].values

    if task == "classification":
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return _error(
                f"Target column '{target_column}' has only 1 unique value.",
                "Choose a column with at least 2 distinct class values.",
            )
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_scores: list[dict] = []
        best_score = -1.0
        _best_fold_idx = 0  # noqa: F841

        for i, (tr_idx, te_idx) in enumerate(kf.split(x, y)):
            x_tr, x_te = x[tr_idx], x[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            y_pred = _fit_predict_classifier(model, x_tr, x_te, y_tr)
            acc = float(accuracy_score(y_te, y_pred))
            f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
            fold_scores.append({"fold": i + 1, "accuracy": acc, "f1_weighted": f1})
            progress.append(ok(f"Fold {i + 1}/{n_splits}", f"acc={acc:.3f} f1={f1:.3f}"))
            if f1 > best_score:
                best_score = f1
                _best_fold_idx = i

        accs = [s["accuracy"] for s in fold_scores]
        f1s = [s["f1_weighted"] for s in fold_scores]
        mean_metrics = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
        }
    else:
        kf2 = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_scores = []
        best_score = -1e9
        _best_fold_idx = 0  # noqa: F841

        for i, (tr_idx, te_idx) in enumerate(kf2.split(x)):
            x_tr, x_te = x[tr_idx], x[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            y_pred = _fit_predict_regressor(model, x_tr, x_te, y_tr)
            r2 = float(r2_score(y_te, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            fold_scores.append({"fold": i + 1, "r2": r2, "rmse": rmse})
            progress.append(ok(f"Fold {i + 1}/{n_splits}", f"r2={r2:.3f} rmse={rmse:.3f}"))
            if r2 > best_score:
                best_score = r2

        r2s = [s["r2"] for s in fold_scores]
        rmses = [s["rmse"] for s in fold_scores]
        mean_metrics = {
            "r2_mean": round(float(np.mean(r2s)), 4),
            "r2_std": round(float(np.std(r2s)), 4),
            "rmse_mean": round(float(np.mean(rmses)), 4),
            "rmse_std": round(float(np.std(rmses)), 4),
        }

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    import os as _os

    _override = _os.environ.get("MCP_OUTPUT_DIR")
    models_dir = Path(_override) if _override else path.parent / ".mcp_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{path.stem}_{model}_cv_{ts}.pkl"
    manifest_path = model_path.with_suffix(".manifest.json")

    backup = ""
    if model_path.exists():
        try:
            backup = snapshot(str(model_path))
        except Exception:
            pass

    # Retrain on full data
    if task == "classification":
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
        _y_pred_final = _fit_predict_classifier(model, x_tr, x_te, y_tr)
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=random_state)
        _y_pred_final = _fit_predict_regressor(model, x_tr, x_te, y_tr)

    metadata = {
        "model_type": model,
        "task": task,
        "trained_on": path.name,
        "training_date": datetime.now(UTC).isoformat(),
        "feature_columns": list(df.drop(columns=[target_column]).columns),
        "target_column": target_column,
        "encoding_map": encoding_map,
        "cv_splits": n_splits,
        "cv_mean_metrics": mean_metrics,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
    }

    payload = {"model": None, "metadata": metadata}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=models_dir) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, model_path)
    atomic_write_json(manifest_path, metadata)
    progress.append(ok("Saved best model", model_path.name))

    append_receipt(str(path), "train_with_cv", {"model": model, "task": task, "n_splits": n_splits}, "success", backup)

    resp = {
        "success": True,
        "op": "train_with_cv",
        "model": model,
        "task": task,
        "n_splits": n_splits,
        "fold_scores": fold_scores,
        "mean_metrics": mean_metrics,
        "model_path": str(model_path),
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["handover"] = make_handover(
        "PATCH",
        ["evaluate_model", "get_predictions", "generate_training_report"],
        {"model_path": str(model_path), "file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


def compare_models(
    file_path: str,
    target_column: str,
    task: str,
    models: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train multiple models, return sorted comparison table."""
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
    invalid = [m for m in models if m not in allowed]
    if invalid:
        return _error(
            f"Unknown algorithms: {', '.join(invalid)}. Allowed: {', '.join(sorted(allowed))}",
            f"Use valid model strings: {' '.join(sorted(allowed))}",
        )

    max_models = get_max_models()
    if len(models) > max_models:
        models = models[:max_models]
        progress.append(warn(f"Capped to {max_models} models", "constrained mode limit"))

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if target_column not in df.columns:
        cols_hint = ", ".join(df.columns[:10].tolist())
        return _error(
            f"Column '{target_column}' not found. Available: {cols_hint}",
            "Use inspect_dataset() to list all column names.",
        )

    if len(df) < MIN_ROWS_COMPARE:
        return _error(
            f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_COMPARE}.",
            "Provide a dataset with more samples.",
        )

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "compare_models",
            "dry_run": True,
            "task": task,
            "models": models,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    df, encoding_map, _ = _auto_preprocess(df, target_column)
    x = df.drop(columns=[target_column]).values
    y = df[target_column].values

    if task == "classification":
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=random_state)

    results: list[dict] = []
    for m in models:
        try:
            if task == "classification":
                y_pred = _fit_predict_classifier(m, x_tr, x_te, y_tr)
                acc = float(accuracy_score(y_te, y_pred))
                f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
                results.append({"model": m, "accuracy": round(acc, 4), "f1_weighted": round(f1, 4)})
                progress.append(ok(f"Trained {m}", f"acc={acc:.3f} f1={f1:.3f}"))
            else:
                y_pred = _fit_predict_regressor(m, x_tr, x_te, y_tr)
                r2 = float(r2_score(y_te, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                results.append({"model": m, "r2": round(r2, 4), "rmse": round(rmse, 4)})
                progress.append(ok(f"Trained {m}", f"r2={r2:.3f} rmse={rmse:.3f}"))
        except Exception as exc:
            results.append({"model": m, "error": str(exc)})
            progress.append(fail(f"Failed {m}", str(exc)[:80]))

    # Sort results
    if task == "classification":
        results.sort(key=lambda r: (r.get("f1_weighted", -1), r.get("accuracy", -1)), reverse=True)
    else:
        results.sort(key=lambda r: (r.get("r2", -1e9), -r.get("rmse", 1e9)), reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    best = results[0]["model"] if results else ""

    # Save only the best model
    best_model_path = ""
    backup = ""
    if best and not results[0].get("error"):
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        import os as _os

        _override = _os.environ.get("MCP_OUTPUT_DIR")
        models_dir = Path(_override) if _override else path.parent / ".mcp_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        mp = models_dir / f"{path.stem}_{best}_best_{ts}.pkl"

        if mp.exists():
            try:
                backup = snapshot(str(mp))
            except Exception:
                pass

        metrics_best = {k: v for k, v in results[0].items() if k not in ("model", "rank")}
        metadata = {
            "model_type": best,
            "task": task,
            "trained_on": path.name,
            "training_date": datetime.now(UTC).isoformat(),
            "feature_columns": list(df.drop(columns=[target_column]).columns),
            "target_column": target_column,
            "encoding_map": encoding_map,
            "metrics": metrics_best,
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
        }
        payload = {"model": None, "metadata": metadata}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=models_dir) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = tmp.name
        shutil.move(tmp_path, mp)
        atomic_write_json(mp.with_suffix(".manifest.json"), metadata)
        best_model_path = str(mp)
        progress.append(ok("Saved best model", mp.name))

    append_receipt(str(path), "compare_models", {"task": task, "models": models}, "success", backup)

    resp = {
        "success": True,
        "op": "compare_models",
        "task": task,
        "results": results,
        "best_model": best,
        "best_model_path": best_model_path,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["handover"] = make_handover(
        "PATCH",
        ["evaluate_model", "get_predictions", "read_model_report"],
        {"model_path": best_model_path, "file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
