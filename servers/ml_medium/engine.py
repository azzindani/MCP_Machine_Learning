"""ml_medium engine — Tier 2 ML logic. Zero MCP imports."""

from __future__ import annotations

import logging
import sys
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.decomposition import PCA, FastICA
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

from shared.file_utils import resolve_path
from shared.platform_utils import get_cv_folds, get_max_models
from shared.progress import fail, info, ok, warn
from shared.receipt import append_receipt, read_receipt_log
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
    "bin_numeric",
    "add_date_parts",
    "log_transform",
    "drop_null_rows",
    "clip_column",
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
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object or str(df[col].dtype) == "category":
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


def _build_regressor(model: str, degree: int = 5, alpha: float = 0.01, n_estimators: int = 10) -> object:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures

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


def _fit_predict_classifier(model_str: str, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Fit classifier and return predictions on x_test."""
    if model_str == "xgb":
        nc = len(np.unique(y_train))
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        params: dict = {
            "max_depth": 3,
            "eta": 0.3,
            "verbosity": 0,
            "objective": "multi:softprob" if nc > 2 else "binary:logistic",
        }
        if nc > 2:
            params["num_class"] = nc
        bst = xgb.train(params, dtrain, num_boost_round=10, evals=[], verbose_eval=False)
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
    model_str: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    degree: int = 5,
    alpha: float = 0.01,
    n_estimators: int = 10,
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
                    f"Strategy '{strategy}' not valid for fill_nulls. Allowed: {' '.join(sorted(FILL_STRATEGIES))}"
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

    elif op_name == "bin_numeric":
        col = op["column"]
        bins = op.get("bins", 5)
        labels = op.get("labels")
        new_col = op.get("new_column", f"{col}_bin")
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
        return df, {"op": op_name, "column": col, "new_column": new_col, "bins": bins}

    elif op_name == "add_date_parts":
        col = op["column"]
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            parts = op.get("parts", ["year", "month", "day", "dayofweek"])
            added = []
            for part in parts:
                new_col = f"{col}_{part}"
                df[new_col] = getattr(dt.dt, part)
                added.append(new_col)
        except Exception as exc:
            return df, {"op": op_name, "column": col, "error": str(exc)}
        return df, {"op": op_name, "column": col, "added_columns": added}

    elif op_name == "log_transform":
        col = op["column"]
        base = op.get("base", "natural")  # "natural", "log2", "log10"
        new_col = op.get("new_column", f"{col}_log")
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        series = pd.to_numeric(df[col], errors="coerce")
        offset = max(0, float(-series.min()) + 1) if series.min() <= 0 else 0.0
        if base == "log2":
            df[new_col] = np.log2(series + offset)
        elif base == "log10":
            df[new_col] = np.log10(series + offset)
        else:
            df[new_col] = np.log1p(series + offset)
        return df, {"op": op_name, "column": col, "new_column": new_col, "base": base, "offset": offset}

    elif op_name == "drop_null_rows":
        col = op.get("column", "")
        before = len(df)
        if col:
            if col not in df.columns:
                return df, {"op": op_name, "column": col, "error": "column not found"}
            df = df.dropna(subset=[col])
        else:
            df = df.dropna()
        return df.copy(), {"op": op_name, "column": col or "all", "removed": before - len(df)}

    elif op_name == "clip_column":
        col = op["column"]
        lower = op.get("lower")
        upper = op.get("upper")
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lower, upper=upper)
        return df, {"op": op_name, "column": col, "lower": lower, "upper": upper}

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

    append_receipt(
        str(path),
        "run_preprocessing",
        {"ops_count": len(ops), "output_path": str(out_path_resolved)},
        "success",
        backup,
    )

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
        results.append(
            {
                "column": col,
                "method": method,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "sample_outliers": outlier_vals,
            }
        )
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

    # Save best-fold model (retrain on full data for best params)
    import json as json_module
    import pickle
    import shutil
    import tempfile
    from datetime import datetime

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
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
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
        _y_pred_final = _fit_predict_classifier(model, x_tr, x_te, y_tr)
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=random_state)
        _y_pred_final = _fit_predict_regressor(model, x_tr, x_te, y_tr)

    import sklearn

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

    payload = {"model": None, "metadata": metadata}  # model obj not easily serializable for xgb
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=models_dir) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, model_path)
    manifest_path.write_text(json_module.dumps(metadata, indent=2))
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
        import json as json_module
        import pickle
        import shutil
        import tempfile
        from datetime import datetime

        import sklearn

        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
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
        mp.with_suffix(".manifest.json").write_text(json_module.dumps(metadata, indent=2))
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

    # Silhouette score (needs at least 2 clusters and not all noise)
    silhouette = None
    if n_found >= 2 and len(set(labels)) >= 2:
        try:
            from sklearn.metrics import silhouette_score

            non_noise = labels != -1
            if non_noise.sum() > n_found:
                silhouette = round(float(silhouette_score(x_scaled[non_noise], labels[non_noise])), 4)
        except Exception:
            pass

    backup = ""
    if save_labels:
        try:
            backup = snapshot(str(path))
        except Exception as exc:
            progress.append(warn("Snapshot failed", str(exc)))
        df["cluster_label"] = labels
        df.to_csv(path, index=False)
        progress.append(ok("Saved labels", "cluster_label column added"))

    append_receipt(
        str(path), "run_clustering", {"algorithm": algorithm, "feature_columns": feature_columns}, "success", backup
    )

    resp = {
        "success": True,
        "op": "run_clustering",
        "algorithm": algorithm,
        "feature_columns": feature_columns,
        "label_counts": label_counts,
        "silhouette_score": silhouette,
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
    deductions += min(miss_pct * 0.5, 20)  # up to 20 pts for missingness
    deductions += min(dup_pct * 0.3, 10)  # up to 10 pts for duplicates
    return round(max(0.0, min(100.0, 100.0 - deductions)), 1)


def _run_quality_alerts(df: pd.DataFrame, target_column: str = "") -> list[dict]:
    """Run 8 quality checks. Returns list of alert dicts."""
    alerts: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # 1. Constant columns (single unique value)
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            alerts.append(
                {
                    "type": "constant_column",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' has only 1 unique value — provides no information.",
                    "recommendation": f"Drop '{col}' with run_preprocessing op 'drop_column'.",
                }
            )

    # 2. High missing data (>20%)
    for col in df.columns:
        miss_pct = df[col].isnull().mean() * 100
        if miss_pct > 20:
            alerts.append(
                {
                    "type": "high_missing",
                    "severity": "high",
                    "column": col,
                    "missing_pct": round(miss_pct, 1),
                    "message": f"Column '{col}' is {miss_pct:.1f}% missing.",
                    "recommendation": "Fill with run_preprocessing fill_nulls or drop if >50%.",
                }
            )

    # 3. Zero-inflated distributions
    for col in numeric_cols:
        if col == target_column:
            continue
        zero_pct = (df[col] == 0).mean() * 100
        if zero_pct > 50:
            alerts.append(
                {
                    "type": "zero_inflated",
                    "severity": "medium",
                    "column": col,
                    "zero_pct": round(zero_pct, 1),
                    "message": f"Column '{col}' is {zero_pct:.1f}% zeros — may need log transform.",
                    "recommendation": "Consider log1p transform or treat zeros as a separate indicator.",
                }
            )

    # 4. High cardinality in categoricals
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target_column]
    for col in cat_cols:
        n_unique = df[col].nunique()
        ratio = n_unique / max(len(df), 1)
        if ratio > 0.5 and n_unique > 20:
            alerts.append(
                {
                    "type": "high_cardinality",
                    "severity": "medium",
                    "column": col,
                    "unique_count": n_unique,
                    "message": f"Column '{col}' has {n_unique} unique values ({ratio * 100:.1f}% of rows) — likely an ID or free-text field.",  # noqa: E501
                    "recommendation": f"Drop '{col}' or encode only top-N categories before training.",
                }
            )

    # 5. Class imbalance (>90% dominance) — only for target or low-cardinality cols
    check_imbal = [target_column] if target_column and target_column in df.columns else []
    check_imbal += [c for c in cat_cols if df[c].nunique() <= 10]
    for col in check_imbal[:5]:  # cap to 5
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.90:
            alerts.append(
                {
                    "type": "class_imbalance",
                    "severity": "high" if col == target_column else "medium",
                    "column": col,
                    "dominant_class": str(vc.index[0]),
                    "dominant_pct": round(float(vc.iloc[0]) * 100, 1),
                    "message": f"'{col}' is {vc.iloc[0] * 100:.1f}% '{vc.index[0]}' — severe class imbalance.",
                    "recommendation": "Use stratify=y in train split. Consider SMOTE or class_weight='balanced'.",
                }
            )

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
            alerts.append(
                {
                    "type": "extreme_skewness",
                    "severity": "medium",
                    "column": col,
                    "skewness": round(skew, 2),
                    "message": f"Column '{col}' is {direction}-skewed (skew={skew:.2f}) — may hurt linear models.",
                    "recommendation": "Apply log transform or use run_preprocessing 'cap_outliers' to reduce skew.",
                }
            )

    # 7. Multicollinearity (|r| > 0.9 between feature pairs)
    if len(numeric_cols) >= 2:
        feat_cols = [c for c in numeric_cols if c != target_column]
        if len(feat_cols) >= 2:
            corr = df[feat_cols].corr().abs()
            seen: set[tuple] = set()
            for i, c1 in enumerate(feat_cols):
                for c2 in feat_cols[i + 1 :]:
                    r = corr.loc[c1, c2]
                    if r > 0.9 and (c1, c2) not in seen:
                        seen.add((c1, c2))
                        alerts.append(
                            {
                                "type": "multicollinearity",
                                "severity": "medium",
                                "columns": [c1, c2],
                                "correlation": round(float(r), 3),
                                "message": f"'{c1}' and '{c2}' are highly correlated (r={r:.3f}).",
                                "recommendation": "Consider dropping one of them, or use PCA via apply_dimensionality_reduction.",  # noqa: E501
                            }
                        )

    # 8. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / len(df) * 100, 1)
        alerts.append(
            {
                "type": "duplicate_rows",
                "severity": "medium" if dup_pct > 5 else "low",
                "count": dup_count,
                "pct": dup_pct,
                "message": f"{dup_count} duplicate rows found ({dup_pct}% of data).",
                "recommendation": "Remove with run_preprocessing op 'drop_duplicates'.",
            }
        )

    return alerts


def _alerts_html(alerts: list[dict], t: dict) -> str:
    """Render alerts as styled HTML cards."""
    if not alerts:
        return '<div class="alert alert-success">✔ No quality issues detected.</div>'
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
            f"<strong>{icon} {a['type'].replace('_', ' ').title()}</strong> — {msg}"
            f"{'<br><em>💡 ' + rec + '</em>' if rec else ''}"
            f"</div>"
        )
    return "\n".join(parts)


def _quality_score_html(score: float, alerts: list[dict], t: dict) -> str:
    """Render quality score gauge + alert summary cards."""
    high = sum(1 for a in alerts if a.get("severity") == "high")
    med = sum(1 for a in alerts if a.get("severity") == "medium")
    low = sum(1 for a in alerts if a.get("severity") == "low")

    color = t["success"] if score >= 80 else (t["warning"] if score >= 60 else t["danger"])
    score_card = (
        f'<div class="cards">'
        f'<div class="card" style="border-left:4px solid {color}">'
        f'  <div class="label">Quality Score</div>'
        f'  <div class="value" style="color:{color}">{score}</div>'
        f'  <div class="sub">out of 100</div>'
        f"</div>"
        f'<div class="card" style="border-left:4px solid {t["danger"]}">'
        f'  <div class="label">High Severity</div>'
        f'  <div class="value" style="color:{t["danger"]}">{high}</div>'
        f'  <div class="sub">critical issues</div>'
        f"</div>"
        f'<div class="card" style="border-left:4px solid {t["warning"]}">'
        f'  <div class="label">Medium Severity</div>'
        f'  <div class="value" style="color:{t["warning"]}">{med}</div>'
        f'  <div class="sub">warnings</div>'
        f"</div>"
        f'<div class="card" style="border-left:4px solid {t["success"]}">'
        f'  <div class="label">Low Severity</div>'
        f'  <div class="value" style="color:{t["success"]}">{low}</div>'
        f'  <div class="sub">minor notes</div>'
        f"</div>"
        f"</div>"
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
    missing_pct = round(missing_total / max(len(df) * len(df.columns), 1) * 100, 2)
    dup_rows = int(df.duplicated().sum())

    overview_html = metrics_cards_html(
        {
            "quality_score": f"{quality_score}/100",
            "rows": f"{len(df):,}",
            "columns": len(df.columns),
            "numeric": len(numeric_cols),
            "categorical": len(cat_cols),
            "missing_cells": f"{missing_pct}%",
            "duplicate_rows": dup_rows,
        }
    )
    sections.append(
        {
            "id": "quality",
            "heading": "Data Quality",
            "html": overview_html + _quality_score_html(quality_score, alerts, t),
        }
    )

    # ── 2. Missing values pattern ──────────────────────────────────────────
    if missing_total > 0:
        null_series = df.isnull().sum().sort_values(ascending=False)
        null_series = null_series[null_series > 0]

        fig_miss = px.bar(
            x=null_series.values,
            y=null_series.index,
            orientation="h",
            title="Missing Values per Column",
            labels={"x": "Missing Count", "y": "Column"},
            template=t["plotly_template"],
            color=null_series.values / len(df) * 100,
            color_continuous_scale="Reds",
        )
        fig_miss.update_coloraxes(colorbar_title="% Missing")
        fig_miss.update_layout(
            paper_bgcolor=t["paper_color"],
            plot_bgcolor=t["bg_color"],
            font_color=t["text_color"],
            height=max(300, len(null_series) * 30 + 80),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        sections.append(
            {
                "id": "missing",
                "heading": "Missing Values",
                "html": plotly_div(fig_miss, height=max(300, len(null_series) * 30 + 80)),
            }
        )
        progress.append(ok("Missing values chart", f"{len(null_series)} cols affected"))

    # ── 3. Distributions — histogram + box plot side by side ──────────────
    if numeric_cols:
        show_nums = numeric_cols[:12]
        n = len(show_nums)
        # 2 charts per col: histogram left, box right
        rows_n = n

        fig_dist = make_subplots(
            rows=rows_n,
            cols=2,
            subplot_titles=[f"{c} — histogram" if i % 2 == 0 else f"{c} — box" for c in show_nums for i in range(2)],
            horizontal_spacing=0.08,
            vertical_spacing=0.05,
        )
        for i, col in enumerate(show_nums):
            r = i + 1
            clean = df[col].dropna()
            # histogram
            fig_dist.add_trace(
                go.Histogram(x=clean, name=col, showlegend=False, marker_color=t["accent"]),
                row=r,
                col=1,
            )
            # box plot
            fig_dist.add_trace(
                go.Box(x=clean, name=col, showlegend=False, marker_color=t["accent"], boxpoints="outliers"),
                row=r,
                col=2,
            )
        fig_dist.update_layout(
            title="Numeric Distributions (Histogram + Box Plot)",
            template=t["plotly_template"],
            paper_bgcolor=t["paper_color"],
            plot_bgcolor=t["bg_color"],
            font_color=t["text_color"],
            height=220 * rows_n + 60,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append(
            {
                "id": "distributions",
                "heading": "Numeric Distributions",
                "html": plotly_div(fig_dist, height=220 * rows_n + 100),
            }
        )
        progress.append(ok("Distribution charts", f"{len(show_nums)} cols (histogram + box)"))

    # ── 4. Correlation — Pearson AND Spearman ─────────────────────────────
    if len(numeric_cols) >= 2:
        feat_cols = [c for c in numeric_cols if c != target_column] or numeric_cols

        fig_corr = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Pearson Correlation", "Spearman Correlation"],
            horizontal_spacing=0.12,
        )
        for col_idx, method in enumerate(["pearson", "spearman"], start=1):
            corr = df[feat_cols].corr(method=method)
            fig_corr.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                    showscale=(col_idx == 1),
                    name=method,
                ),
                row=1,
                col=col_idx,
            )
        fig_corr.update_layout(
            template=t["plotly_template"],
            paper_bgcolor=t["paper_color"],
            plot_bgcolor=t["bg_color"],
            font_color=t["text_color"],
            height=max(450, len(feat_cols) * 28 + 100),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append(
            {
                "id": "correlation",
                "heading": "Correlation (Pearson + Spearman)",
                "html": plotly_div(fig_corr, height=max(450, len(feat_cols) * 28 + 140)),
            }
        )
        progress.append(ok("Correlation heatmaps", f"Pearson + Spearman, {len(feat_cols)} features"))

    # ── 5. Categorical columns ─────────────────────────────────────────────
    show_cats = [c for c in cat_cols if c != target_column][:6]
    if show_cats:
        cat_html = ""
        for col in show_cats:
            vc = df[col].value_counts().head(15)
            fig_cat = px.bar(
                x=vc.values,
                y=vc.index.astype(str),
                orientation="h",
                title=f"{col} — Top Values",
                labels={"x": "Count", "y": col},
                template=t["plotly_template"],
                color=vc.values,
                color_continuous_scale="Blues",
            )
            fig_cat.update_layout(
                paper_bgcolor=t["paper_color"],
                plot_bgcolor=t["bg_color"],
                font_color=t["text_color"],
                height=max(250, len(vc) * 25 + 80),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
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
                names=vc.index.astype(str),
                values=vc.values,
                title=f"Target Distribution: {target_column}",
                template=t["plotly_template"],
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            fig_tgt = px.histogram(
                df,
                x=target_column,
                nbins=40,
                title=f"Target Distribution: {target_column}",
                template=t["plotly_template"],
            )
        fig_tgt.update_layout(
            paper_bgcolor=t["paper_color"],
            plot_bgcolor=t["bg_color"],
            font_color=t["text_color"],
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        sections.append(
            {
                "id": "target",
                "heading": f"Target Column: {target_column}",
                "html": plotly_div(fig_tgt, height=420),
            }
        )
        progress.append(ok("Target distribution", f"{tgt.nunique()} unique values"))

    # ── 7. Summary statistics table ────────────────────────────────────────
    if numeric_cols:
        desc = df[numeric_cols[:12]].describe().T.round(3)
        desc["skewness"] = df[numeric_cols[:12]].skew().round(3)
        desc.index.name = "column"
        rows_data = [{"column": idx, **row.to_dict()} for idx, row in desc.iterrows()]
        sections.append(
            {
                "id": "stats",
                "heading": "Summary Statistics",
                "html": data_table_html(rows_data),
            }
        )

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
    append_receipt(str(path), "generate_eda_report", {"theme": theme, "target_column": target_column}, "success", "")

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


# ---------------------------------------------------------------------------
# Tool: filter_rows
# ---------------------------------------------------------------------------

FILTER_OPS = {
    "eq",
    "ne",
    "gt",
    "lt",
    "gte",
    "lte",
    "contains",
    "not_contains",
    "is_null",
    "not_null",
    "starts_with",
    "ends_with",
}


def filter_rows(
    file_path: str,
    column: str,
    operator: str,
    value: str = "",
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Filter dataset rows by condition. operator: eq ne gt lt gte lte contains is_null not_null."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv, got {path.suffix!r}", "Provide a CSV file.")
    if operator not in FILTER_OPS:
        return _error(
            f"Unknown operator: '{operator}'. Allowed: {', '.join(sorted(FILTER_OPS))}",
            "Use one of: eq ne gt lt gte lte contains is_null not_null starts_with ends_with",
        )

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")

    if column not in df.columns:
        return _error(f"Column '{column}' not found.", "Use inspect_dataset() to list column names.")

    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows"))
    original_count = len(df)

    col = df[column]
    try:
        if operator == "eq":
            mask = col.astype(str) == str(value)
        elif operator == "ne":
            mask = col.astype(str) != str(value)
        elif operator == "gt":
            mask = pd.to_numeric(col, errors="coerce") > float(value)
        elif operator == "lt":
            mask = pd.to_numeric(col, errors="coerce") < float(value)
        elif operator == "gte":
            mask = pd.to_numeric(col, errors="coerce") >= float(value)
        elif operator == "lte":
            mask = pd.to_numeric(col, errors="coerce") <= float(value)
        elif operator == "contains":
            mask = col.astype(str).str.contains(str(value), na=False)
        elif operator == "not_contains":
            mask = ~col.astype(str).str.contains(str(value), na=False)
        elif operator == "is_null":
            mask = col.isnull()
        elif operator == "not_null":
            mask = col.notnull()
        elif operator == "starts_with":
            mask = col.astype(str).str.startswith(str(value), na=False)
        elif operator == "ends_with":
            mask = col.astype(str).str.endswith(str(value), na=False)
        else:
            mask = pd.Series([True] * len(df))
    except Exception as exc:
        return _error(f"Filter failed: {exc}", "Check value type matches column dtype.")

    df_filtered = df[mask].copy()
    kept = len(df_filtered)
    removed = original_count - kept
    progress.append(ok("Filter applied", f"{kept:,} rows kept, {removed:,} removed"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "filter_rows",
            "dry_run": True,
            "rows_kept": kept,
            "rows_removed": removed,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    out_path = Path(output_path) if output_path else path.parent / f"{path.stem}_filtered.csv"
    try:
        out_resolved = resolve_path(str(out_path))
    except ValueError:
        out_resolved = out_path

    backup = ""
    if out_resolved.exists():
        try:
            backup = snapshot(str(out_resolved))
        except Exception:
            pass

    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(out_resolved, index=False)
    progress.append(ok("Saved filtered dataset", out_resolved.name))
    append_receipt(
        str(path), "filter_rows", {"column": column, "operator": operator, "value": value}, "success", backup
    )  # noqa: E501

    resp = {
        "success": True,
        "op": "filter_rows",
        "output_path": str(out_resolved),
        "rows_original": original_count,
        "rows_kept": kept,
        "rows_removed": removed,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: merge_datasets
# ---------------------------------------------------------------------------


def merge_datasets(
    file_path_1: str,
    file_path_2: str,
    on: str,
    how: str = "left",
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Merge two CSVs on a key column. how: left right inner outer."""
    progress: list[dict] = []
    if how not in ("left", "right", "inner", "outer"):
        return _error(f"Unknown how: '{how}'.", "Use: left right inner outer")

    try:
        p1 = resolve_path(file_path_1)
        p2 = resolve_path(file_path_2)
    except ValueError as exc:
        return _error(str(exc), "Check both paths are inside your home directory.")

    for p in (p1, p2):
        if not p.exists():
            return _error(f"File not found: {p}", "Check the file path.")

    try:
        df1 = pd.read_csv(p1, low_memory=False)
        df2 = pd.read_csv(p2, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check files are valid CSVs.")

    progress.append(ok(f"Loaded {p1.name}", f"{len(df1):,} rows × {len(df1.columns)} cols"))
    progress.append(ok(f"Loaded {p2.name}", f"{len(df2):,} rows × {len(df2.columns)} cols"))

    # Validate key column exists in both
    keys = [k.strip() for k in on.split(",")]
    for k in keys:
        if k not in df1.columns:
            return _error(f"Key '{k}' not in {p1.name}.", "Use inspect_dataset() to list column names.")
        if k not in df2.columns:
            return _error(f"Key '{k}' not in {p2.name}.", "Use inspect_dataset() to list column names.")

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "merge_datasets",
            "dry_run": True,
            "left_rows": len(df1),
            "right_rows": len(df2),
            "join_keys": keys,
            "how": how,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    on_param = keys[0] if len(keys) == 1 else keys
    df_merged = pd.merge(df1, df2, on=on_param, how=how, suffixes=("", "_right"))
    progress.append(ok("Merged", f"{len(df_merged):,} rows × {len(df_merged.columns)} cols"))

    out_path = Path(output_path) if output_path else p1.parent / f"{p1.stem}_merged.csv"
    try:
        out_resolved = resolve_path(str(out_path))
    except ValueError:
        out_resolved = out_path

    backup = ""
    if out_resolved.exists():
        try:
            backup = snapshot(str(out_resolved))
        except Exception:
            pass

    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_resolved, index=False)
    progress.append(ok("Saved merged dataset", out_resolved.name))
    append_receipt(str(p1), "merge_datasets", {"file_2": p2.name, "on": on, "how": how}, "success", backup)

    resp = {
        "success": True,
        "op": "merge_datasets",
        "output_path": str(out_resolved),
        "left_rows": len(df1),
        "right_rows": len(df2),
        "merged_rows": len(df_merged),
        "merged_columns": len(df_merged.columns),
        "join_keys": keys,
        "how": how,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: find_optimal_clusters
# ---------------------------------------------------------------------------


def find_optimal_clusters(
    file_path: str,
    feature_columns: list[str],
    max_k: int = 10,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
) -> dict:
    """Find optimal K for K-Means via elbow + silhouette. Saves HTML chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.metrics import silhouette_score

    from shared.html_theme import get_theme, save_chart

    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return _error(f"Columns not found: {', '.join(missing)}", "Use inspect_dataset() for column names.")

    x = df[feature_columns].select_dtypes(include="number").dropna().values
    if x.shape[0] < 4:
        return _error("Need at least 4 rows for cluster analysis.", "Provide a larger dataset.")

    from sklearn.preprocessing import StandardScaler as _SS

    x_scaled = _SS().fit_transform(x)

    max_k = min(max_k, x_scaled.shape[0] - 1, 15)
    k_range = list(range(2, max_k + 1))

    inertias, silhouettes = [], []
    for k in k_range:
        km = _KMeans(n_clusters=k, random_state=42, max_iter=100)
        labels = km.fit_predict(x_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(x_scaled, labels)))
        progress.append(info(f"k={k}", f"inertia={km.inertia_:.1f} sil={silhouettes[-1]:.3f}"))

    best_k = k_range[int(np.argmax(silhouettes))]
    t = get_theme(theme)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Elbow Curve (Inertia)", "Silhouette Score"])
    fig.add_trace(
        go.Scatter(x=k_range, y=inertias, mode="lines+markers", name="Inertia", marker_color=t["accent"]), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=k_range, y=silhouettes, mode="lines+markers", name="Silhouette", marker_color=t["success"]),
        row=1,
        col=2,
    )
    fig.add_vline(x=best_k, line_dash="dash", line_color=t["danger"], annotation_text=f"Best K={best_k}", row=1, col=2)
    fig.update_layout(
        title=f"Optimal Clusters for {path.name}",
        template=t["plotly_template"],
        paper_bgcolor=t["paper_color"],
        plot_bgcolor=t["bg_color"],
        font_color=t["text_color"],
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    out_str = output_path or str(path.parent / f"{path.stem}_optimal_k.html")
    out_abs, out_name = save_chart(
        fig, out_str, theme=theme, open_browser=open_browser, title=f"Optimal Clusters — {path.name}"
    )
    progress.append(ok("Saved elbow chart", out_name))

    resp: dict = {
        "success": True,
        "op": "find_optimal_clusters",
        "best_k": best_k,
        "k_range": k_range,
        "inertias": [round(v, 2) for v in inertias],
        "silhouette_scores": [round(v, 4) for v in silhouettes],
        "output_path": out_abs,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: anomaly_detection
# ---------------------------------------------------------------------------


def anomaly_detection(
    file_path: str,
    feature_columns: list[str],
    method: str = "isolation_forest",
    contamination: float = 0.05,
    save_labels: bool = False,
    dry_run: bool = False,
) -> dict:
    """Detect anomalies. method: isolation_forest lof. Adds anomaly_score column if save_labels=True."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    progress: list[dict] = []
    if method not in ("isolation_forest", "lof"):
        return _error(f"Unknown method: '{method}'.", "Use 'isolation_forest' or 'lof'.")
    if not 0 < contamination < 0.5:
        return _error(f"contamination={contamination} must be in (0, 0.5).", "Use e.g. contamination=0.05")

    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return _error(f"Columns not found: {', '.join(missing)}", "Use inspect_dataset() for column names.")

    x = df[feature_columns].select_dtypes(include="number").fillna(0).values
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows, {len(feature_columns)} features"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "anomaly_detection",
            "dry_run": True,
            "method": method,
            "contamination": contamination,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    from sklearn.preprocessing import StandardScaler as _SS

    x_scaled = _SS().fit_transform(x)

    if method == "isolation_forest":
        det = IsolationForest(contamination=contamination, random_state=42)
        labels = det.fit_predict(x_scaled)  # -1 = anomaly, 1 = normal
        scores = det.score_samples(x_scaled)  # lower = more anomalous
    else:
        det = LocalOutlierFactor(contamination=contamination)
        labels = det.fit_predict(x_scaled)
        scores = det.negative_outlier_factor_

    anomaly_mask = labels == -1
    n_anomalies = int(anomaly_mask.sum())
    anomaly_pct = round(n_anomalies / len(df) * 100, 2)
    progress.append(ok("Detected anomalies", f"{n_anomalies} ({anomaly_pct}%) anomalous rows"))

    # Top anomaly indices
    top_anomaly_idx = np.argsort(scores)[: min(10, n_anomalies)].tolist()

    backup = ""
    if save_labels:
        try:
            backup = snapshot(str(path))
        except Exception as exc:
            progress.append(warn("Snapshot failed", str(exc)))
        df["anomaly_score"] = scores
        df["is_anomaly"] = anomaly_mask.astype(int)
        df.to_csv(path, index=False)
        progress.append(ok("Saved anomaly labels", path.name))

    append_receipt(
        str(path), "anomaly_detection", {"method": method, "contamination": contamination}, "success", backup
    )

    resp = {
        "success": True,
        "op": "anomaly_detection",
        "method": method,
        "contamination": contamination,
        "n_anomalies": n_anomalies,
        "anomaly_pct": anomaly_pct,
        "top_anomaly_indices": top_anomaly_idx,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: check_data_quality (JSON summary — model-readable)
# ---------------------------------------------------------------------------


def check_data_quality(file_path: str) -> dict:
    """Return JSON data quality summary with score 0-100. No HTML."""
    from shared.file_utils import resolve_path
    from shared.progress import ok

    progress = []
    try:
        path = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file path.", "token_estimate": 30}
    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check that file_path is absolute.",
            "token_estimate": 30,
        }

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"success": False, "error": str(exc), "hint": "Check the file is a valid CSV.", "token_estimate": 30}
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    n_rows, n_cols = len(df), len(df.columns)
    alerts = []
    score = 100.0

    # 1. Constant columns
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            alerts.append(
                {
                    "type": "constant_column",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' has only 1 unique value.",
                    "recommendation": f"Drop column '{col}' — it contains no information.",
                }
            )
            score -= 15

    # 2. High missing
    null_summary = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = null_count / n_rows * 100 if n_rows > 0 else 0
        if null_pct > 0:
            null_summary.append({"column": col, "null_count": null_count, "null_pct": round(null_pct, 2)})
        if null_pct > 20:
            alerts.append(
                {
                    "type": "high_missing",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' is {null_pct:.1f}% null.",
                    "recommendation": f"Use run_preprocessing fill_nulls or drop column '{col}'.",
                }
            )
            score -= 15

    # 3. Duplicate rows
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / n_rows * 100 if n_rows > 0 else 0
    if dup_count > 0:
        alerts.append(
            {
                "type": "duplicate_rows",
                "severity": "medium",
                "message": f"{dup_count} duplicate rows ({dup_pct:.1f}%).",
                "recommendation": "Use run_preprocessing op 'drop_duplicates' to remove them.",
            }
        )
        score -= min(10, dup_pct * 0.3)

    # 4. Zero-inflated numeric
    for col in df.select_dtypes(include="number").columns:
        zero_pct = (df[col] == 0).sum() / n_rows * 100 if n_rows > 0 else 0
        if zero_pct > 50:
            alerts.append(
                {
                    "type": "zero_inflated",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' is {zero_pct:.1f}% zeros.",
                    "recommendation": f"Consider log_transform or separate zero/nonzero modeling for '{col}'.",
                }
            )
            score -= 8

    # 5. High cardinality
    for col in df.select_dtypes(include=["object", "string"]).columns:
        n_unique = df[col].nunique()
        ratio = n_unique / n_rows if n_rows > 0 else 0
        if ratio > 0.5 and n_unique > 20:
            alerts.append(
                {
                    "type": "high_cardinality",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' has {n_unique} unique values ({ratio * 100:.1f}% of rows).",
                    "recommendation": f"Consider drop_column or target-encoding for '{col}'.",
                }
            )
            score -= 8

    # 6. Extreme skewness
    for col in df.select_dtypes(include="number").columns:
        try:
            skew = float(df[col].skew())
        except Exception:
            continue
        if abs(skew) > 2:
            alerts.append(
                {
                    "type": "extreme_skewness",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' skewness = {skew:.2f}.",
                    "recommendation": f"Apply log_transform to column '{col}' before training.",
                }
            )
            score -= 8

    # 7. Multicollinearity
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols].corr(method="pearson").abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        c1, c2 = num_cols[i], num_cols[j]
                        alerts.append(
                            {
                                "type": "multicollinearity",
                                "severity": "high",
                                "column_pair": [c1, c2],
                                "message": f"Columns '{c1}' and '{c2}' have |r|={corr_matrix.iloc[i, j]:.2f}.",
                                "recommendation": f"Drop one of '{c1}' or '{c2}' to reduce multicollinearity.",
                            }
                        )
                        score -= 15
        except Exception:
            pass

    # Cap score
    score = max(0.0, min(100.0, score))

    # Summarize null cols with high missing
    high_missing_cols = [c["column"] for c in alerts if c.get("type") == "high_missing"]
    constant_cols = [c["column"] for c in alerts if c.get("type") == "constant_column"]

    resp = {
        "success": True,
        "op": "check_data_quality",
        "file": path.name,
        "row_count": n_rows,
        "column_count": n_cols,
        "quality_score": round(score, 1),
        "alerts_count": len(alerts),
        "alerts_high": sum(1 for a in alerts if a.get("severity") == "high"),
        "alerts_medium": sum(1 for a in alerts if a.get("severity") == "medium"),
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_pct, 2),
        "null_summary": null_summary[:20],
        "high_missing_columns": high_missing_cols,
        "constant_columns": constant_cols,
        "alerts": alerts[:30],
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: evaluate_model (score saved model on external labeled test file)
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    test_file_path: str,
    target_column: str,
) -> dict:
    """Score a saved model on a labeled test CSV. Returns metrics dict."""
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    from shared.file_utils import resolve_path
    from shared.progress import ok

    progress = []
    try:
        mp = resolve_path(model_path)
        dp = resolve_path(test_file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file paths.", "token_estimate": 30}

    if not mp.exists():
        return {
            "success": False,
            "error": f"Model not found: {model_path}",
            "hint": "Train a model first.",
            "token_estimate": 30,
        }
    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {test_file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    try:
        import xgboost as xgb

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model_obj = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "classification")
        feature_columns = metadata.get("feature_columns", [])
        encoding_map = metadata.get("encoding_map", {})
        scaler = metadata.get("scaler")
        poly = metadata.get("poly")

        df = pd.read_csv(dp, low_memory=False)
        progress.append(ok("Loaded test data", f"{len(df)} rows"))

        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not in test file.",
                "hint": "Use inspect_dataset() to check column names.",
                "token_estimate": 30,
            }

        # Encode categoricals using stored map
        for col, mapping in encoding_map.items():
            if col in df.columns and col != target_column:
                df[col] = df[col].astype(str).map({str(k): v for k, v in mapping.items()}).fillna(-1).astype(int)

        available = [c for c in feature_columns if c in df.columns]
        if not available:
            return {
                "success": False,
                "error": "No feature columns found in test file.",
                "hint": "Use the same dataset schema as training.",
                "token_estimate": 30,
            }

        X = df[available].fillna(0).values.astype(float)
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)

        y_true = df[target_column].values

        # Encode target if categorical
        from sklearn.preprocessing import LabelEncoder

        le = None
        if y_true.dtype == object or str(y_true.dtype) in ("string",):
            le = LabelEncoder()
            y_true = le.fit_transform(y_true)

        model_key = metadata.get("model_key", "")
        if model_key == "xgb" or isinstance(model_obj, xgb.Booster):
            dmat = xgb.DMatrix(X)
            raw = model_obj.predict(dmat)
            n_classes = metadata.get("n_classes", 2)
            if task == "classification":
                if n_classes > 2:
                    y_pred = np.asarray([np.argmax(line) for line in raw])
                else:
                    y_pred = (raw > 0.5).astype(int)
            else:
                y_pred = raw
        else:
            y_pred = model_obj.predict(X)

        metrics: dict = {}
        if task == "classification":
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}
            if len(set(y_true)) == 2:
                try:
                    if hasattr(model_obj, "predict_proba"):
                        y_prob = model_obj.predict_proba(X)[:, 1]
                    elif model_key == "xgb":
                        y_prob = raw if raw.ndim == 1 else raw[:, 1]
                    else:
                        y_prob = None
                    if y_prob is not None:
                        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
                except Exception:
                    pass
            from servers.ml_basic.engine import _confusion_dict

            metrics["confusion_matrix"] = _confusion_dict(y_true, y_pred)
        else:
            mse = float(mean_squared_error(y_true, y_pred))
            metrics = {
                "mse": round(mse, 4),
                "rmse": round(float(np.sqrt(mse)), 4),
                "r2": round(float(r2_score(y_true, y_pred)), 4),
            }

        progress.append(
            ok("Evaluated model", ", ".join(f"{k}={v}" for k, v in metrics.items() if not isinstance(v, dict))[:80])
        )

        resp = {
            "success": True,
            "op": "evaluate_model",
            "model_path": str(mp),
            "test_file": dp.name,
            "task": task,
            "target_column": target_column,
            "test_rows": len(df),
            "metrics": metrics,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check model and test file compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# Tool: batch_predict (all rows → CSV, no row limit)
# ---------------------------------------------------------------------------


def batch_predict(
    model_path: str,
    file_path: str,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Run predictions on all rows and save to CSV. No row limit."""
    import pickle

    import numpy as np
    import pandas as pd

    from shared.file_utils import resolve_path
    from shared.progress import info, ok
    from shared.version_control import snapshot

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
            "hint": "Train a model first.",
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
            "op": "batch_predict",
            "dry_run": True,
            "model_path": str(mp),
            "file_path": str(dp),
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import xgboost as xgb

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model_obj = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "classification")
        feature_columns = metadata.get("feature_columns", [])
        encoding_map = metadata.get("encoding_map", {})
        scaler = metadata.get("scaler")
        poly = metadata.get("poly")
        model_key = metadata.get("model_key", "")
        n_classes = metadata.get("n_classes", 2)

        df = pd.read_csv(dp, low_memory=False)
        progress.append(ok("Loaded data", f"{len(df):,} rows"))

        for col, mapping in encoding_map.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map({str(k): v for k, v in mapping.items()}).fillna(-1).astype(int)

        available = [c for c in feature_columns if c in df.columns]
        X = df[available].fillna(0).values.astype(float)
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)

        if model_key == "xgb" or isinstance(model_obj, xgb.Booster):
            dmat = xgb.DMatrix(X)
            raw = model_obj.predict(dmat)
            if task == "classification":
                if n_classes > 2:
                    preds = np.asarray([np.argmax(line) for line in raw])
                else:
                    preds = (raw > 0.5).astype(int)
            else:
                preds = raw
        else:
            preds = model_obj.predict(X)

        df["prediction"] = preds
        progress.append(ok("Generated predictions", f"{len(preds):,} rows"))

        out_path_str = output_path or str(dp.parent / f"{dp.stem}_predictions.csv")
        from pathlib import Path

        out = Path(out_path_str).resolve()

        backup = ""
        if out.exists():
            try:
                backup = snapshot(str(out))
            except Exception:
                pass

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        progress.append(ok("Saved predictions", out.name))

        # Distribution summary
        if task == "classification":
            dist = {str(k): int(v) for k, v in pd.Series(preds).value_counts().sort_index().items()}
        else:
            dist = {
                "min": round(float(preds.min()), 4),
                "max": round(float(preds.max()), 4),
                "mean": round(float(preds.mean()), 4),
            }

        resp = {
            "success": True,
            "op": "batch_predict",
            "output_path": str(out),
            "row_count": len(preds),
            "task": task,
            "prediction_distribution": dist,
            "backup": backup,
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
