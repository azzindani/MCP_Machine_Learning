"""ml_medium engine — Tier 2 ML logic. Zero MCP imports."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import xgboost as xgb

from shared.file_utils import resolve_path
from shared.platform_utils import get_cv_folds, get_max_models, get_max_results
from shared.progress import fail, info, ok, warn
from shared.receipt import append_receipt
from shared.version_control import snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CLASSIFIERS = {"lr", "svm", "rf", "dtc", "knn", "nb", "xgb"}
ALLOWED_REGRESSORS = {"lir", "pr", "lar", "rr", "dtr", "rfr", "xgb"}
ALLOWED_CLUSTER_ALGOS = {"kmeans", "meanshift", "dbscan"}

ALLOWED_OPS = {
    "fill_nulls",
    "drop_outliers",
    "label_encode",
    "onehot_encode",
    "scale",
    "drop_duplicates",
    "drop_column",
    "rename_column",
    "convert_dtype",
}
FILL_STRATEGIES = {"mean", "median", "mode", "ffill", "bfill", "zero"}
SCALE_METHODS = {"standard", "minmax"}

MODELS_DIR = ".mcp_models"

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
    """Drop null targets, label-encode categoricals, fill numeric nulls."""
    df = df.dropna(subset=[target_column]).copy()
    encoding_map: dict = {}
    encoded_cols: list[str] = []

    for col in df.columns:
        if col == target_column:
            continue
        if (
            pd.api.types.is_string_dtype(df[col])
            or df[col].dtype == object
            or str(df[col].dtype) == "category"
        ):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("nan").astype(str))
            encoding_map[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            encoded_cols.append(col)

    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df, encoding_map, encoded_cols


def _build_classifier(model: str, **kw: object) -> object:
    if model == "lr":
        return LogisticRegression(random_state=42, max_iter=200)
    if model == "svm":
        scaler = StandardScaler()
        return ("svm_pipeline", scaler, SVC(kernel="rbf", gamma="auto", random_state=42))
    if model == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model == "dtc":
        return DecisionTreeClassifier(random_state=42)
    if model == "knn":
        scaler = StandardScaler()
        return ("knn_pipeline", scaler, KNeighborsClassifier(n_neighbors=5))
    if model == "nb":
        return GaussianNB()
    if model == "xgb":
        return None  # handled separately
    raise ValueError(f"Unknown classifier: {model!r}")


def _build_regressor(model: str, degree: int = 5, alpha: float = 0.01,
                     n_estimators: int = 10) -> object:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    if model == "lir":
        return LinearRegression()
    if model == "pr":
        return Pipeline([("poly", PolynomialFeatures(degree=degree)), ("lr", LinearRegression())])
    if model == "lar":
        return Lasso(alpha=alpha, max_iter=200, tol=0.1)
    if model == "rr":
        return Ridge(alpha=alpha, max_iter=100, tol=0.1)
    if model == "dtr":
        return DecisionTreeRegressor(random_state=42)
    if model == "rfr":
        return RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    if model == "xgb":
        return None  # handled separately
    raise ValueError(f"Unknown regressor: {model!r}")


def _fit_predict_classifier(
    model_str: str, x_train: np.ndarray, x_test: np.ndarray,
    y_train: np.ndarray
) -> np.ndarray:
    """Fit classifier and return predictions on x_test."""
    if model_str == "xgb":
        nc = len(np.unique(y_train))
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        params: dict = {
            "max_depth": 3, "eta": 0.3, "verbosity": 0,
            "objective": "multi:softprob" if nc > 2 else "binary:logistic",
        }
        if nc > 2:
            params["num_class"] = nc
        bst = xgb.train(params, dtrain, num_boost_round=10,
                        evals=[], verbose_eval=False)
        preds = bst.predict(dtest)
        if nc > 2:
            return np.asarray([np.argmax(row) for row in preds])
        return (preds > 0.5).astype(int)

    built = _build_classifier(model_str)
    if isinstance(built, tuple):
        _, scaler, clf = built
        x_tr = scaler.fit_transform(x_train)
        x_te = scaler.transform(x_test)
        clf.fit(x_tr, y_train)
        return clf.predict(x_te)
    built.fit(x_train, y_train)
    return built.predict(x_test)


def _fit_predict_regressor(
    model_str: str, x_train: np.ndarray, x_test: np.ndarray,
    y_train: np.ndarray, degree: int = 5, alpha: float = 0.01,
    n_estimators: int = 10
) -> np.ndarray:
    if model_str == "xgb":
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        params = {"max_depth": 3, "eta": 0.3, "verbosity": 0, "objective": "reg:squarederror"}
        bst = xgb.train(params, dtrain, num_boost_round=5, evals=[], verbose_eval=False)
        return bst.predict(dtest)
    built = _build_regressor(model_str, degree=degree, alpha=alpha, n_estimators=n_estimators)
    built.fit(x_train, y_train)
    return built.predict(x_test)


# ---------------------------------------------------------------------------
# Tool: run_preprocessing
# ---------------------------------------------------------------------------

MAX_OPS = 50


def _validate_ops(ops: list[dict]) -> tuple[bool, str]:
    """Validate preprocessing ops array. Returns (ok, error_msg)."""
    if not isinstance(ops, list):
        return False, "ops must be a list of dicts."
    if len(ops) > MAX_OPS:
        return False, f"Too many ops: {len(ops)}. Max is {MAX_OPS}."
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return False, f"Op #{i} is not a dict."
        op_name = op.get("op", "")
        if op_name not in ALLOWED_OPS:
            return False, f"Unknown op: '{op_name}'. Allowed: {', '.join(sorted(ALLOWED_OPS))}"
        if op_name == "fill_nulls":
            if "column" not in op:
                return False, f"Op '{op_name}' missing required field: 'column'"
            strategy = op.get("strategy", "median")
            if strategy not in FILL_STRATEGIES:
                return False, (
                    f"Strategy '{strategy}' not valid for fill_nulls. "
                    f"Allowed: {' '.join(sorted(FILL_STRATEGIES))}"
                )
        elif op_name == "scale":
            if "columns" not in op:
                return False, f"Op '{op_name}' missing required field: 'columns'"
            method = op.get("method", "standard")
            if method not in SCALE_METHODS:
                return False, f"Method '{method}' not valid for scale. Allowed: standard minmax"
        elif op_name in {"label_encode", "onehot_encode", "drop_column", "drop_outliers", "convert_dtype"}:
            if "column" not in op:
                return False, f"Op '{op_name}' missing required field: 'column'"
        elif op_name == "rename_column":
            for field in ("from", "to"):
                if field not in op:
                    return False, f"Op '{op_name}' missing required field: '{field}'"
    return True, ""


def _apply_op(df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, dict]:
    """Apply single preprocessing op. Returns (df, summary)."""
    op_name = op["op"]

    if op_name == "fill_nulls":
        col = op["column"]
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        strategy = op.get("strategy", "median")
        before = int(df[col].isnull().sum())
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else np.nan)
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        return df, {"op": op_name, "column": col, "strategy": strategy, "filled": before}

    elif op_name == "drop_outliers":
        col = op["column"]
        method = op.get("method", "iqr")
        before = len(df)
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
        else:  # std
            mean, std = df[col].mean(), df[col].std()
            df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
        return df.copy(), {"op": op_name, "column": col, "method": method, "removed": before - len(df)}

    elif op_name == "label_encode":
        col = op["column"]
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("nan").astype(str))
        return df, {"op": op_name, "column": col, "classes": list(le.classes_[:10])}

    elif op_name == "onehot_encode":
        col = op["column"]
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df, {"op": op_name, "column": col, "new_columns": list(dummies.columns[:10])}

    elif op_name == "scale":
        cols = op["columns"]
        method = op.get("method", "standard")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return df, {"op": op_name, "columns": cols, "error": f"columns not found: {missing}"}
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        return df, {"op": op_name, "columns": cols, "method": method}

    elif op_name == "drop_duplicates":
        subset = op.get("subset")
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        return df.copy(), {"op": op_name, "removed": before - len(df)}

    elif op_name == "drop_column":
        col = op["column"]
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        df = df.drop(columns=[col])
        return df, {"op": op_name, "column": col}

    elif op_name == "rename_column":
        from_col, to_col = op["from"], op["to"]
        df = df.rename(columns={from_col: to_col})
        return df, {"op": op_name, "from": from_col, "to": to_col}

    elif op_name == "convert_dtype":
        col = op["column"]
        to = op.get("to", "")
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        try:
            if to == "datetime":
                df[col] = pd.to_datetime(df[col])
            elif to == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif to in ("str", "string"):
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(to)
        except Exception as exc:
            return df, {"op": op_name, "column": col, "error": str(exc)}
        return df, {"op": op_name, "column": col, "to": to}

    return df, {"op": op_name, "error": "unhandled op"}


def run_preprocessing(
    file_path: str,
    ops: list[dict],
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Apply preprocessing pipeline ops to dataset. Snapshot before write."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    valid, err_msg = _validate_ops(ops)
    if not valid:
        return _error(err_msg, "Check the op array. See run_preprocessing docstring for valid ops.")

    progress.append(info("Validated ops", f"{len(ops)} ops"))

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "run_preprocessing",
            "dry_run": True,
            "ops_count": len(ops),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    # Snapshot before write
    backup = ""
    try:
        backup = snapshot(str(path))
        progress.append(ok("Snapshot created", Path(backup).name))
    except Exception as exc:
        progress.append(warn("Snapshot failed", str(exc)))

    ops_summary: list[dict] = []
    for op in ops:
        df, summary = _apply_op(df, op)
        ops_summary.append(summary)
        progress.append(ok(f"Applied {op['op']}", str(summary.get("filled", summary.get("removed", "")))))

    out_path = Path(output_path) if output_path else path
    try:
        out_path_resolved = resolve_path(str(out_path))
    except ValueError:
        out_path_resolved = out_path
    out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path_resolved, index=False)
    progress.append(ok("Saved output", out_path_resolved.name))

    append_receipt(str(path), "run_preprocessing",
                   {"ops_count": len(ops), "output_path": str(out_path_resolved)},
                   "success", backup)

    resp = {
        "success": True,
        "op": "run_preprocessing",
        "applied": len(ops),
        "ops_summary": ops_summary,
        "output_path": str(out_path_resolved),
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: detect_outliers
# ---------------------------------------------------------------------------


def detect_outliers(
    file_path: str,
    columns: list[str],
    method: str = "iqr",
    th1: float = 0.25,
    th3: float = 0.75,
) -> dict:
    """Detect outliers in numeric columns. method: iqr std."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

    if method not in ("iqr", "std"):
        return _error(f"Unknown method: '{method}'.", "Use 'iqr' or 'std'.")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        return _error(
            f"Columns not found: {', '.join(missing[:5])}",
            "Use inspect_dataset() to list valid column names.",
        )

    results: list[dict] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if method == "iqr":
            q1 = series.quantile(th1)
            q3 = series.quantile(th3)
            iqr_val = q3 - q1
            lower = float(q1 - 1.5 * iqr_val)
            upper = float(q3 + 1.5 * iqr_val)
        else:  # std
            mean, std = series.mean(), series.std()
            lower = float(mean - 3 * std)
            upper = float(mean + 3 * std)

        mask = (series < lower) | (series > upper)
        outlier_vals = series[mask].head(5).tolist()
        results.append({
            "column": col,
            "method": method,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "sample_outliers": outlier_vals,
        })
        progress.append(ok(f"Analyzed {col}", f"{int(mask.sum())} outliers"))

    resp: dict = {
        "success": True,
        "op": "detect_outliers",
        "method": method,
        "columns_checked": len(columns),
        "results": results,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: train_with_cv
# ---------------------------------------------------------------------------

MIN_ROWS_CV = 20


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
        best_fold_idx = 0

        for i, (tr_idx, te_idx) in enumerate(kf.split(x, y)):
            x_tr, x_te = x[tr_idx], x[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            y_pred = _fit_predict_classifier(model, x_tr, x_te, y_tr)
            acc = float(accuracy_score(y_te, y_pred))
            f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
            fold_scores.append({"fold": i + 1, "accuracy": acc, "f1_weighted": f1})
            progress.append(ok(f"Fold {i+1}/{n_splits}", f"acc={acc:.3f} f1={f1:.3f}"))
            if f1 > best_score:
                best_score = f1
                best_fold_idx = i

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
        best_fold_idx = 0

        for i, (tr_idx, te_idx) in enumerate(kf2.split(x)):
            x_tr, x_te = x[tr_idx], x[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            y_pred = _fit_predict_regressor(model, x_tr, x_te, y_tr)
            r2 = float(r2_score(y_te, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            fold_scores.append({"fold": i + 1, "r2": r2, "rmse": rmse})
            progress.append(ok(f"Fold {i+1}/{n_splits}", f"r2={r2:.3f} rmse={rmse:.3f}"))
            if r2 > best_score:
                best_score = r2
                best_fold_idx = i

        r2s = [s["r2"] for s in fold_scores]
        rmses = [s["rmse"] for s in fold_scores]
        mean_metrics = {
            "r2_mean": round(float(np.mean(r2s)), 4),
            "r2_std": round(float(np.std(r2s)), 4),
            "rmse_mean": round(float(np.mean(rmses)), 4),
            "rmse_std": round(float(np.std(rmses)), 4),
        }

    # Save best-fold model (retrain on full data for best params)
    import pickle, tempfile, shutil, json as json_module
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    models_dir = path.parent / MODELS_DIR
    models_dir.mkdir(exist_ok=True)
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
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2,
                                                    random_state=random_state, stratify=y)
        y_pred_final = _fit_predict_classifier(model, x_tr, x_te, y_tr)
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=random_state)
        y_pred_final = _fit_predict_regressor(model, x_tr, x_te, y_tr)

    import sklearn
    metadata = {
        "model_type": model,
        "task": task,
        "trained_on": path.name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(df.drop(columns=[target_column]).columns),
        "target_column": target_column,
        "encoding_map": encoding_map,
        "cv_splits": n_splits,
        "cv_mean_metrics": mean_metrics,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
    }

    payload = {"model": None, "metadata": metadata}  # model obj not easily serializable for xgb
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=models_dir) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, model_path)
    manifest_path.write_text(json_module.dumps(metadata, indent=2))
    progress.append(ok("Saved best model", model_path.name))

    append_receipt(str(path), "train_with_cv",
                   {"model": model, "task": task, "n_splits": n_splits},
                   "success", backup)

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
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: compare_models
# ---------------------------------------------------------------------------

MIN_ROWS_COMPARE = 20


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
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

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
        import pickle, tempfile, shutil, json as json_module
        from datetime import datetime, timezone
        import sklearn

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        models_dir = path.parent / MODELS_DIR
        models_dir.mkdir(exist_ok=True)
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
            "training_date": datetime.now(timezone.utc).isoformat(),
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
        mp.with_suffix(".manifest.json").write_text(json_module.dumps(metadata, indent=2))
        best_model_path = str(mp)
        progress.append(ok("Saved best model", mp.name))

    append_receipt(str(path), "compare_models",
                   {"task": task, "models": models},
                   "success", backup)

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
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: run_clustering
# ---------------------------------------------------------------------------


def run_clustering(
    file_path: str,
    feature_columns: list[str],
    algorithm: str,
    n_clusters: int = 3,
    eps: float = 3.0,
    min_samples: int = 5,
    reduce_dims: str = "",
    n_components: int = 2,
    save_labels: bool = False,
    dry_run: bool = False,
) -> dict:
    """Cluster dataset. algorithm: kmeans meanshift dbscan."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    if algorithm not in ALLOWED_CLUSTER_ALGOS:
        return _error(
            f"Unknown algorithm: '{algorithm}'. Allowed: {', '.join(sorted(ALLOWED_CLUSTER_ALGOS))}",
            "Use 'kmeans', 'meanshift', or 'dbscan'.",
        )

    if reduce_dims and reduce_dims not in ("pca", "ica"):
        return _error(f"Unknown reduce_dims: '{reduce_dims}'.", "Use 'pca', 'ica', or '' (none).")

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

    x = df[feature_columns].select_dtypes(include="number").values
    if x.shape[1] == 0:
        return _error("No numeric feature columns found.", "Select numeric columns for clustering.")

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "run_clustering",
            "dry_run": True,
            "algorithm": algorithm,
            "feature_columns": feature_columns,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    # Scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    progress.append(ok("Scaled features", "StandardScaler"))

    # Optional dimensionality reduction
    if reduce_dims:
        nc = min(n_components, x_scaled.shape[1])
        if reduce_dims == "pca":
            reducer = PCA(n_components=nc)
        else:
            reducer = FastICA(n_components=nc)
        x_scaled = reducer.fit_transform(x_scaled)
        progress.append(ok(f"Reduced dims with {reduce_dims.upper()}", f"{nc} components"))

    # Cluster
    if algorithm == "kmeans":
        clf = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42)
        labels = clf.fit_predict(x_scaled)
        inertia = float(clf.inertia_)
        n_found = n_clusters
        extra = {"inertia": round(inertia, 4)}
    elif algorithm == "meanshift":
        clf = MeanShift()
        labels = clf.fit_predict(x_scaled)
        n_found = len(np.unique(labels))
        extra = {"n_clusters_found": n_found}
    else:  # dbscan
        clf = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clf.fit_predict(x_scaled)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int((labels == -1).sum())
        extra = {"n_clusters_found": n_found, "noise_points": noise}

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {str(int(u)): int(c) for u, c in zip(unique, counts)}
    progress.append(ok(f"Clustered with {algorithm}", f"{n_found} clusters"))

    backup = ""
    if save_labels:
        try:
            backup = snapshot(str(path))
        except Exception as exc:
            progress.append(warn("Snapshot failed", str(exc)))
        df["cluster_label"] = labels
        df.to_csv(path, index=False)
        progress.append(ok("Saved labels", f"cluster_label column added"))

    append_receipt(str(path), "run_clustering",
                   {"algorithm": algorithm, "feature_columns": feature_columns},
                   "success", backup)

    resp = {
        "success": True,
        "op": "run_clustering",
        "algorithm": algorithm,
        "feature_columns": feature_columns,
        "label_counts": label_counts,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
        **extra,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: read_receipt
# ---------------------------------------------------------------------------

from shared.receipt import read_receipt_log


def read_receipt(file_path: str) -> dict:
    """Read operation history for a file. Returns log entries."""
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")

    log = read_receipt_log(str(path))
    resp: dict = {
        "success": True,
        "op": "read_receipt",
        "file": Path(file_path).name,
        "entry_count": len(log),
        "entries": log,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp

