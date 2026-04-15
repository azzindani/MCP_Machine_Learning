"""ml_advanced helpers — constants, imports, and shared helper functions."""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np  # noqa: F401  (re-exported for sub-modules)
import pandas as pd
import psutil
import sklearn
import xgboost as xgb
from sklearn.decomposition import PCA, FastICA  # noqa: F401
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # noqa: F401
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: F401
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shared.file_utils import atomic_write_json, get_output_dir, resolve_path
from shared.html_layout import get_output_path  # noqa: F401  (re-exported)
from shared.html_theme import _open_file, save_chart  # noqa: F401  (re-exported)
from shared.platform_utils import get_cv_folds, is_constrained_mode  # noqa: F401
from shared.progress import info, ok, warn  # noqa: F401
from shared.receipt import append_receipt  # noqa: F401
from shared.version_control import snapshot  # noqa: F401

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


def _save_chart(
    fig: object,
    output_path: str,
    stem_suffix: str,
    input_path: Path,
    open_after: bool,
    theme: str,
) -> tuple[str, str]:
    """Thin wrapper — saves a Plotly chart via the shared save_chart helper."""
    return save_chart(fig, output_path, stem_suffix, input_path, theme, open_after, _open_file)


def _error(error: str, hint: str, backup: str | None = None) -> dict:
    base: dict = {"success": False, "error": error, "hint": hint, "progress": []}
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
    atomic_write_json(manifest_path, metadata)


def _load_model(model_path: str) -> tuple[object, dict]:
    path = resolve_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload.get("model"), payload.get("metadata", {})


__all__ = [
    # constants
    "ALLOWED_CLASSIFIERS",
    "ALLOWED_REGRESSORS",
    "DEFAULT_PARAMS",
    "MODELS_DIR",
    # stdlib / third-party re-exports used by engine.py
    "UTC",
    "datetime",
    "json",
    "logger",
    "np",
    "pd",
    "sys",
    "sklearn",
    "xgb",
    "Path",
    # shared re-exports
    "append_receipt",
    "get_cv_folds",
    "get_output_dir",
    "get_output_path",
    "info",
    "is_constrained_mode",
    "ok",
    "resolve_path",
    "snapshot",
    "warn",
    # helpers
    "_auto_preprocess",
    "_build_estimator",
    "_check_memory",
    "_error",
    "_load_model",
    "_save_chart",
    "_save_model",
    # sklearn classes (used in engine.py tool bodies)
    "FastICA",
    "GridSearchCV",
    "LabelEncoder",
    "PCA",
    "RandomizedSearchCV",
    "StandardScaler",
]
