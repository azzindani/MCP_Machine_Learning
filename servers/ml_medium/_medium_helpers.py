"""ml_medium shared helpers — imports, constants, and utility functions."""

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

from shared.file_utils import get_output_dir, resolve_path
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
# Preprocessing op validation
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


__all__ = [
    # re-exports from shared
    "get_output_dir",
    "resolve_path",
    "get_cv_folds",
    "get_max_models",
    "fail",
    "info",
    "ok",
    "warn",
    "append_receipt",
    "read_receipt_log",
    "snapshot",
    # numpy / pandas / psutil
    "np",
    "pd",
    "psutil",
    "xgb",
    # sklearn
    "DBSCAN",
    "KMeans",
    "MeanShift",
    "PCA",
    "FastICA",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
    "accuracy_score",
    "f1_score",
    "mean_squared_error",
    "r2_score",
    "KFold",
    "StratifiedKFold",
    "train_test_split",
    "GaussianNB",
    "KNeighborsClassifier",
    "LabelEncoder",
    "MinMaxScaler",
    "StandardScaler",
    "SVC",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    # constants
    "ALLOWED_CLASSIFIERS",
    "ALLOWED_REGRESSORS",
    "ALLOWED_CLUSTER_ALGOS",
    "ALLOWED_OPS",
    "FILL_STRATEGIES",
    "SCALE_METHODS",
    "MODELS_DIR",
    "MAX_OPS",
    # helpers
    "_error",
    "_check_memory",
    "_auto_preprocess",
    "_build_classifier",
    "_build_regressor",
    "_fit_predict_classifier",
    "_fit_predict_regressor",
    "_validate_ops",
    "_apply_op",
    # stdlib re-exports used in sub-modules
    "sys",
    "UTC",
    "Path",
    "logger",
]
