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




# ---------------------------------------------------------------------------
# Tool: generate_eda_report
# ---------------------------------------------------------------------------

# ---- Quality alert helpers ----

def _compute_quality_score(df: pd.DataFrame, alerts: list[dict]) -> float:
    """Score 0–100. Start at 100, deduct for each alert by severity."""
    severity_weights = {"high": 15, "medium": 8, "low": 3}
    deductions = sum(severity_weights.get(a.get("severity", "low"), 3) for a in alerts)
    # Additional structural deductions
    miss_pct = df.isnull().sum().sum() / max(len(df) * len(df.columns), 1) * 100
    dup_pct = df.duplicated().sum() / max(len(df), 1) * 100
    deductions += min(miss_pct * 0.5, 20)   # up to 20 pts for missingness
    deductions += min(dup_pct * 0.3, 10)    # up to 10 pts for duplicates
    return round(max(0.0, min(100.0, 100.0 - deductions)), 1)


def _run_quality_alerts(df: pd.DataFrame, target_column: str = "") -> list[dict]:
    """Run 8 quality checks. Returns list of alert dicts."""
    alerts: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # 1. Constant columns (single unique value)
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            alerts.append({
                "type": "constant_column",
                "severity": "high",
                "column": col,
                "message": f"Column '{col}' has only 1 unique value — provides no information.",
                "recommendation": f"Drop '{col}' with run_preprocessing op 'drop_column'.",
            })

    # 2. High missing data (>20%)
    for col in df.columns:
        miss_pct = df[col].isnull().mean() * 100
        if miss_pct > 20:
            alerts.append({
                "type": "high_missing",
                "severity": "high",
                "column": col,
                "missing_pct": round(miss_pct, 1),
                "message": f"Column '{col}' is {miss_pct:.1f}% missing.",
                "recommendation": f"Fill with run_preprocessing 'fill_nulls' (strategy: median/mode) or drop if >50%.",
            })

    # 3. Zero-inflated distributions
    for col in numeric_cols:
        if col == target_column:
            continue
        zero_pct = (df[col] == 0).mean() * 100
        if zero_pct > 50:
            alerts.append({
                "type": "zero_inflated",
                "severity": "medium",
                "column": col,
                "zero_pct": round(zero_pct, 1),
                "message": f"Column '{col}' is {zero_pct:.1f}% zeros — may need log transform.",
                "recommendation": f"Consider log1p transform or treat zeros as a separate indicator.",
            })

    # 4. High cardinality in categoricals
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target_column]
    for col in cat_cols:
        n_unique = df[col].nunique()
        ratio = n_unique / max(len(df), 1)
        if ratio > 0.5 and n_unique > 20:
            alerts.append({
                "type": "high_cardinality",
                "severity": "medium",
                "column": col,
                "unique_count": n_unique,
                "message": f"Column '{col}' has {n_unique} unique values ({ratio*100:.1f}% of rows) — likely an ID or free-text field.",
                "recommendation": f"Drop '{col}' or encode only top-N categories before training.",
            })

    # 5. Class imbalance (>90% dominance) — only for target or low-cardinality cols
    check_imbal = [target_column] if target_column and target_column in df.columns else []
    check_imbal += [c for c in cat_cols if df[c].nunique() <= 10]
    for col in check_imbal[:5]:  # cap to 5
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.90:
            alerts.append({
                "type": "class_imbalance",
                "severity": "high" if col == target_column else "medium",
                "column": col,
                "dominant_class": str(vc.index[0]),
                "dominant_pct": round(float(vc.iloc[0]) * 100, 1),
                "message": f"'{col}' is {vc.iloc[0]*100:.1f}% '{vc.index[0]}' — severe class imbalance.",
                "recommendation": "Use stratify=y in train split. Consider SMOTE or class_weight='balanced'.",
            })

    # 6. Extreme skewness (|skew| > 2)
    for col in numeric_cols:
        if col == target_column:
            continue
        try:
            skew = float(df[col].skew())
        except Exception:
            continue
        if abs(skew) > 2:
            direction = "right" if skew > 0 else "left"
            alerts.append({
                "type": "extreme_skewness",
                "severity": "medium",
                "column": col,
                "skewness": round(skew, 2),
                "message": f"Column '{col}' is {direction}-skewed (skew={skew:.2f}) — may hurt linear models.",
                "recommendation": f"Apply log transform or use run_preprocessing 'cap_outliers' to reduce skew.",
            })

    # 7. Multicollinearity (|r| > 0.9 between feature pairs)
    if len(numeric_cols) >= 2:
        feat_cols = [c for c in numeric_cols if c != target_column]
        if len(feat_cols) >= 2:
            corr = df[feat_cols].corr().abs()
            seen: set[tuple] = set()
            for i, c1 in enumerate(feat_cols):
                for c2 in feat_cols[i+1:]:
                    r = corr.loc[c1, c2]
                    if r > 0.9 and (c1, c2) not in seen:
                        seen.add((c1, c2))
                        alerts.append({
                            "type": "multicollinearity",
                            "severity": "medium",
                            "columns": [c1, c2],
                            "correlation": round(float(r), 3),
                            "message": f"'{c1}' and '{c2}' are highly correlated (r={r:.3f}).",
                            "recommendation": f"Consider dropping one of them, or use PCA via apply_dimensionality_reduction.",
                        })

    # 8. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / len(df) * 100, 1)
        alerts.append({
            "type": "duplicate_rows",
            "severity": "medium" if dup_pct > 5 else "low",
            "count": dup_count,
            "pct": dup_pct,
            "message": f"{dup_count} duplicate rows found ({dup_pct}% of data).",
            "recommendation": "Remove with run_preprocessing op 'drop_duplicates'.",
        })

    return alerts


def _alerts_html(alerts: list[dict], t: dict) -> str:
    """Render alerts as styled HTML cards."""
    if not alerts:
        return f'<div class="alert alert-success">✔ No quality issues detected.</div>'
    sev_class = {"high": "alert-danger", "medium": "alert-warning", "low": "alert-success"}
    sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    parts = []
    for a in alerts:
        sev = a.get("severity", "low")
        cls = sev_class.get(sev, "alert-warning")
        icon = sev_icon.get(sev, "●")
        msg = a.get("message", "")
        rec = a.get("recommendation", "")
        parts.append(
            f'<div class="alert {cls}">'
            f'<strong>{icon} {a["type"].replace("_", " ").title()}</strong> — {msg}'
            f'{"<br><em>💡 " + rec + "</em>" if rec else ""}'
            f'</div>'
        )
    return "\n".join(parts)


def _quality_score_html(score: float, alerts: list[dict], t: dict) -> str:
    """Render quality score gauge + alert summary cards."""
    high = sum(1 for a in alerts if a.get("severity") == "high")
    med  = sum(1 for a in alerts if a.get("severity") == "medium")
    low  = sum(1 for a in alerts if a.get("severity") == "low")

    color = t["success"] if score >= 80 else (t["warning"] if score >= 60 else t["danger"])
    score_card = (
        f'<div class="cards">'
        f'<div class="card" style="border-left:4px solid {color}">'
        f'  <div class="label">Quality Score</div>'
        f'  <div class="value" style="color:{color}">{score}</div>'
        f'  <div class="sub">out of 100</div>'
        f'</div>'
        f'<div class="card" style="border-left:4px solid {t["danger"]}">'
        f'  <div class="label">High Severity</div>'
        f'  <div class="value" style="color:{t["danger"]}">{high}</div>'
        f'  <div class="sub">critical issues</div>'
        f'</div>'
        f'<div class="card" style="border-left:4px solid {t["warning"]}">'
        f'  <div class="label">Medium Severity</div>'
        f'  <div class="value" style="color:{t["warning"]}">{med}</div>'
        f'  <div class="sub">warnings</div>'
        f'</div>'
        f'<div class="card" style="border-left:4px solid {t["success"]}">'
        f'  <div class="label">Low Severity</div>'
        f'  <div class="value" style="color:{t["success"]}">{low}</div>'
        f'  <div class="sub">minor notes</div>'
        f'</div>'
        f'</div>'
    )
    return score_card + _alerts_html(alerts, t)


def generate_eda_report(
    file_path: str,
    target_column: str = "",
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate interactive HTML EDA report with Plotly charts."""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    out_path_str = output_path or str(path.parent / f"{path.stem}_eda_report.html")
    try:
        out_path = resolve_path(out_path_str)
    except ValueError:
        out_path = Path(out_path_str)

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "generate_eda_report",
            "dry_run": True,
            "output_path": str(out_path),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    t = get_theme(theme)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    sections: list[dict] = []

    # ── 1. Quality Alerts + Score ──────────────────────────────────────────
    alerts = _run_quality_alerts(df, target_column)
    quality_score = _compute_quality_score(df, alerts)
    progress.append(ok("Quality analysis", f"score={quality_score}/100, {len(alerts)} alerts"))

    missing_total = int(df.isnull().sum().sum())
    missing_pct   = round(missing_total / max(len(df) * len(df.columns), 1) * 100, 2)
    dup_rows      = int(df.duplicated().sum())

    overview_html = metrics_cards_html({
        "quality_score": f"{quality_score}/100",
        "rows":          f"{len(df):,}",
        "columns":       len(df.columns),
        "numeric":       len(numeric_cols),
        "categorical":   len(cat_cols),
        "missing_cells": f"{missing_pct}%",
        "duplicate_rows": dup_rows,
    })
    sections.append({
        "id": "quality",
        "heading": "Data Quality",
        "html": overview_html + _quality_score_html(quality_score, alerts, t),
    })

    # ── 2. Missing values pattern ──────────────────────────────────────────
    if missing_total > 0:
        null_series = df.isnull().sum().sort_values(ascending=False)
        null_series = null_series[null_series > 0]

        fig_miss = px.bar(
            x=null_series.values, y=null_series.index, orientation="h",
            title="Missing Values per Column",
            labels={"x": "Missing Count", "y": "Column"},
            template=t["plotly_template"],
            color=null_series.values / len(df) * 100,
            color_continuous_scale="Reds",
        )
        fig_miss.update_coloraxes(colorbar_title="% Missing")
        fig_miss.update_layout(
            paper_bgcolor=t["paper_color"], plot_bgcolor=t["bg_color"],
            font_color=t["text_color"], height=max(300, len(null_series) * 30 + 80),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        sections.append({
            "id": "missing",
            "heading": "Missing Values",
            "html": plotly_div(fig_miss, height=max(300, len(null_series) * 30 + 80)),
        })
        progress.append(ok("Missing values chart", f"{len(null_series)} cols affected"))

    # ── 3. Distributions — histogram + box plot side by side ──────────────
    if numeric_cols:
        show_nums = numeric_cols[:12]
        n = len(show_nums)
        cols_per_row = min(2, n)  # 2 charts per col: histogram left, box right
        rows_n = n

        fig_dist = make_subplots(
            rows=rows_n, cols=2,
            subplot_titles=[f"{c} — histogram" if i % 2 == 0 else f"{c} — box"
                            for c in show_nums for i in range(2)],
            horizontal_spacing=0.08, vertical_spacing=0.05,
        )
        for i, col in enumerate(show_nums):
            r = i + 1
            clean = df[col].dropna()
            # histogram
            fig_dist.add_trace(
                go.Histogram(x=clean, name=col, showlegend=False,
                             marker_color=t["accent"]),
                row=r, col=1,
            )
            # box plot
            fig_dist.add_trace(
                go.Box(x=clean, name=col, showlegend=False,
                       marker_color=t["accent"], boxpoints="outliers"),
                row=r, col=2,
            )
        fig_dist.update_layout(
            title="Numeric Distributions (Histogram + Box Plot)",
            template=t["plotly_template"],
            paper_bgcolor=t["paper_color"], plot_bgcolor=t["bg_color"],
            font_color=t["text_color"], height=220 * rows_n + 60,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append({
            "id": "distributions",
            "heading": "Numeric Distributions",
            "html": plotly_div(fig_dist, height=220 * rows_n + 100),
        })
        progress.append(ok("Distribution charts", f"{len(show_nums)} cols (histogram + box)"))

    # ── 4. Correlation — Pearson AND Spearman ─────────────────────────────
    if len(numeric_cols) >= 2:
        feat_cols = [c for c in numeric_cols if c != target_column] or numeric_cols

        fig_corr = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Pearson Correlation", "Spearman Correlation"],
            horizontal_spacing=0.12,
        )
        for col_idx, method in enumerate(["pearson", "spearman"], start=1):
            corr = df[feat_cols].corr(method=method)
            fig_corr.add_trace(
                go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale="RdBu_r", zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}", showscale=(col_idx == 1),
                    name=method,
                ),
                row=1, col=col_idx,
            )
        fig_corr.update_layout(
            template=t["plotly_template"],
            paper_bgcolor=t["paper_color"], plot_bgcolor=t["bg_color"],
            font_color=t["text_color"], height=max(450, len(feat_cols) * 28 + 100),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append({
            "id": "correlation",
            "heading": "Correlation (Pearson + Spearman)",
            "html": plotly_div(fig_corr, height=max(450, len(feat_cols) * 28 + 140)),
        })
        progress.append(ok("Correlation heatmaps", f"Pearson + Spearman, {len(feat_cols)} features"))

    # ── 5. Categorical columns ─────────────────────────────────────────────
    show_cats = [c for c in cat_cols if c != target_column][:6]
    if show_cats:
        cat_html = ""
        for col in show_cats:
            vc = df[col].value_counts().head(15)
            fig_cat = px.bar(
                x=vc.values, y=vc.index.astype(str), orientation="h",
                title=f"{col} — Top Values",
                labels={"x": "Count", "y": col},
                template=t["plotly_template"],
                color=vc.values, color_continuous_scale="Blues",
            )
            fig_cat.update_layout(
                paper_bgcolor=t["paper_color"], plot_bgcolor=t["bg_color"],
                font_color=t["text_color"], height=max(250, len(vc) * 25 + 80),
                margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
            )
            cat_html += plotly_div(fig_cat, height=max(250, len(vc) * 25 + 80))
        sections.append({"id": "categorical", "heading": "Categorical Columns", "html": cat_html})
        progress.append(ok("Categorical charts", f"{len(show_cats)} cols"))

    # ── 6. Target column analysis ──────────────────────────────────────────
    if target_column and target_column in df.columns:
        tgt = df[target_column]
        if tgt.nunique() <= 20:
            vc = tgt.value_counts()
            fig_tgt = px.pie(
                names=vc.index.astype(str), values=vc.values,
                title=f"Target Distribution: {target_column}",
                template=t["plotly_template"],
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            fig_tgt = px.histogram(
                df, x=target_column, nbins=40,
                title=f"Target Distribution: {target_column}",
                template=t["plotly_template"],
            )
        fig_tgt.update_layout(
            paper_bgcolor=t["paper_color"], plot_bgcolor=t["bg_color"],
            font_color=t["text_color"], height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append({
            "id": "target",
            "heading": f"Target Column: {target_column}",
            "html": plotly_div(fig_tgt, height=420),
        })
        progress.append(ok("Target distribution", f"{tgt.nunique()} unique values"))

    # ── 7. Summary statistics table ────────────────────────────────────────
    if numeric_cols:
        desc = df[numeric_cols[:12]].describe().T.round(3)
        desc["skewness"] = df[numeric_cols[:12]].skew().round(3)
        desc.index.name = "column"
        rows_data = [{"column": idx, **row.to_dict()} for idx, row in desc.iterrows()]
        sections.append({
            "id": "stats",
            "heading": "Summary Statistics",
            "html": data_table_html(rows_data),
        })

    # ── Build and write report ─────────────────────────────────────────────
    from datetime import datetime
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    subtitle = (
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{len(df):,} rows · {len(df.columns)} columns · "
        f"Quality score: {quality_score}/100"
    )
    html = build_html_report(
        title=f"EDA Report — {path.name}",
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
    append_receipt(str(path), "generate_eda_report",
                   {"theme": theme, "target_column": target_column},
                   "success", "")

    resp = {
        "success": True,
        "op": "generate_eda_report",
        "output_path": str(out_path),
        "file_size_kb": file_size_kb,
        "quality_score": quality_score,
        "alerts_count": len(alerts),
        "alerts_high": sum(1 for a in alerts if a.get("severity") == "high"),
        "alerts_medium": sum(1 for a in alerts if a.get("severity") == "medium"),
        "alerts_low": sum(1 for a in alerts if a.get("severity") == "low"),
        "alerts": alerts,
        "charts_generated": len(sections),
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
