"""Shared helpers for ml_basic — imported by engine, _basic_train, _basic_predict."""

from __future__ import annotations

import json
import logging
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

from shared.file_utils import apply_default_mode, atomic_write_json, get_output_dir, resolve_path
from shared.file_utils import read_csv as _read_csv
from shared.ml_utils import _auto_preprocess, leakage_warning, typical_row
from shared.model_output import save_model
from shared.model_signing import dump_signed, load_signed
from shared.platform_utils import get_max_columns, get_max_results, get_max_rows
from shared.progress import info, ok, warn
from shared.progress import name as pname
from shared.receipt import append_receipt
from shared.registry import allowed_classifiers as _allowed_classifiers
from shared.registry import allowed_regressors as _allowed_regressors
from shared.version_control import restore_version as _restore_version
from shared.version_control import snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_ROWS_CLASSIFIER = 20
MIN_ROWS_REGRESSOR = 10

ALLOWED_CLASSIFIERS = _allowed_classifiers()
ALLOWED_REGRESSORS = _allowed_regressors()

MODELS_DIR = ".mcp_models"

# Which models actually read each hyper-parameter the trainers declare.
#
# The tool schema describes the tool; the vocabulary is per model. Four of the
# seven classifiers pass class_weight to sklearn and three do not, so
# train_classifier(model="nb", class_weight="balanced") trained a GaussianNB
# that has no such parameter and answered success: true without a word --
# every schema check there is passes, because the argument is valid for the
# tool. Same shape on the regressor: degree is read by pr alone, alpha by lar
# and rr, n_estimators by rfr and (since this table was written) xgb.
#
# One table, next to the model lists it refers to, so a new model cannot be
# added to one copy.
CLASSIFIER_ARG_MODELS: dict[str, frozenset[str]] = {
    "class_weight": frozenset({"lr", "svm", "rf", "dtc"}),
}
REGRESSOR_ARG_MODELS: dict[str, frozenset[str]] = {
    "degree": frozenset({"pr"}),
    "alpha": frozenset({"lar", "rr"}),
    "n_estimators": frozenset({"rfr", "xgb"}),
}

# `cw = class_weight if class_weight in ("balanced",) else None` turned every
# other spelling into the default in silence: "balance", "auto" and
# "class_weight" all trained an unweighted model under success: true.
CLASS_WEIGHTS = frozenset({"balanced"})


def unread_arg_error(
    table: dict[str, frozenset[str]], model: str, given: dict, defaults: dict
) -> tuple[str, str] | None:
    """(error, hint) when an argument was set that this model never reads."""
    ignored = sorted(
        name for name, value in given.items() if value != defaults[name] and model not in table.get(name, frozenset())
    )
    if not ignored:
        return None
    parts = [f"{name} (read by: {', '.join(sorted(table.get(name, ())))})" for name in ignored]
    return (
        f"model='{model}' does not read {', '.join(ignored)}",
        f"Drop the argument, or choose a model that uses it — {'; '.join(parts)}.",
    )


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


def _save_model(model: Any, path: Path, metadata: dict) -> Path:
    """Atomically save model pickle + manifest JSON. Returns the manifest path.

    Kept as a name this module already exported; the implementation is shared
    with ml_advanced, which had a second copy of it that had drifted -- this
    one made the parent directory and that one did not. See
    shared.model_output.
    """
    return save_model(model, path, metadata)


def _load_model(model_path: str) -> tuple[Any, dict]:
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        payload = load_signed(f)
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
    "_read_csv",
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
    "CLASSIFIER_ARG_MODELS",
    "CLASS_WEIGHTS",
    "REGRESSOR_ARG_MODELS",
    "unread_arg_error",
    "MODELS_DIR",
    # helpers
    "_check_memory",
    "_error",
    "_auto_preprocess",
    "leakage_warning",
    "typical_row",
    "warn",
    "_confusion_dict",
    "_save_model",
    "_load_model",
]
