"""Shared helpers for ml_basic — imported by engine, _basic_train, _basic_predict."""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shared.file_utils import atomic_write_json, get_output_dir, resolve_path
from shared.platform_utils import get_max_columns, get_max_results, get_max_rows
from shared.progress import info, ok
from shared.progress import name as pname
from shared.receipt import append_receipt
from shared.version_control import restore_version as _restore_version
from shared.version_control import snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_ROWS_CLASSIFIER = 20
MIN_ROWS_REGRESSOR = 10

ALLOWED_CLASSIFIERS = {"lr", "svm", "rf", "dtc", "knn", "nb", "xgb"}
ALLOWED_REGRESSORS = {"lir", "pr", "lar", "rr", "dtr", "rfr", "xgb"}

MODELS_DIR = ".mcp_models"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _check_memory(required_gb: float) -> dict | None:
    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < required_gb:
        return {
            "success": False,
            "error": f"Insufficient RAM: need ~{required_gb:.1f} GB, available ~{available_gb:.1f} GB.",
            "hint": "Use read_rows() to sample a subset, or increase system RAM.",
            "token_estimate": 60,
        }
    return None


def _error(error: str, hint: str, backup: str | None = None) -> dict:
    base: dict = {"success": False, "error": error, "hint": hint, "progress": []}
    if backup:
        base["backup"] = backup
    base["token_estimate"] = len(str(base)) // 4
    return base


def _auto_preprocess(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict, list[str]]:
    """Drop null targets, label-encode categoricals, fill numeric nulls.

    Returns: (processed_df, encoding_map, encoded_columns)
    """
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

    # Encode target column if it is categorical (handles string labels like "yes"/"no")
    if (
        pd.api.types.is_string_dtype(df[target_column])
        or df[target_column].dtype == object
        or str(df[target_column].dtype) == "category"
    ):
        le_tgt = LabelEncoder()
        df[target_column] = le_tgt.fit_transform(df[target_column].astype(str))
        encoding_map[f"__target__{target_column}"] = {str(cls): int(idx) for idx, cls in enumerate(le_tgt.classes_)}

    # fill numeric nulls with median (vectorized — single pass)
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)

    return df, encoding_map, encoded_cols


def _confusion_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return confusion matrix as named dict (not raw 2D array)."""
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if len(classes) == 2:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    # multiclass — per class stats from classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    result = {}
    for cls in classes[:10]:  # max 10 classes
        key = str(cls)
        if key in report:
            r = report[key]
            result[f"class_{cls}"] = {
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1-score"], 4),
                "support": int(r["support"]),
            }
    return result


def _save_model(model: Any, path: Path, metadata: dict) -> None:
    """Atomically save model pickle + manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": metadata}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", dir=path.parent) as tmp:
        pickle.dump(payload, tmp)
        tmp_path = tmp.name
    shutil.move(tmp_path, str(path))
    manifest_path = path.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, metadata)


def _load_model(model_path: str) -> tuple[Any, dict]:
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["metadata"]


__all__ = [
    # re-exported stdlib / third-party symbols used by sub-modules
    "UTC",
    "datetime",
    "Path",
    "Any",
    "np",
    "pd",
    "xgb",
    "sys",
    "sklearn",
    "train_test_split",
    "StandardScaler",
    "PolynomialFeatures",
    "LabelEncoder",
    "LogisticRegression",
    "Ridge",
    "Lasso",
    "LinearRegression",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "KNeighborsClassifier",
    "GaussianNB",
    "SVC",
    "accuracy_score",
    "f1_score",
    "mean_squared_error",
    "r2_score",
    "resolve_path",
    "get_output_dir",
    "get_max_rows",
    "get_max_results",
    "get_max_columns",
    "ok",
    "info",
    "pname",
    "append_receipt",
    "snapshot",
    "_restore_version",
    "logger",
    # constants
    "MIN_ROWS_CLASSIFIER",
    "MIN_ROWS_REGRESSOR",
    "ALLOWED_CLASSIFIERS",
    "ALLOWED_REGRESSORS",
    "MODELS_DIR",
    # helpers
    "_check_memory",
    "_error",
    "_auto_preprocess",
    "_confusion_dict",
    "_save_model",
    "_load_model",
]
