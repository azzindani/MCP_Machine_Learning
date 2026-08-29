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
from shared.file_utils import read_csv as _read_csv
from shared.html_layout import get_output_path
from shared.html_theme import _open_file, save_chart
from shared.ml_utils import _auto_preprocess, baseline_warning, bounded_silhouette, leakage_warning, typical_row
from shared.platform_utils import get_cv_folds, get_max_models
from shared.progress import fail, info, ok, warn
from shared.receipt import append_receipt, read_receipt_log
from shared.registry import CLUSTERERS
from shared.registry import allowed_classifiers as _allowed_classifiers
from shared.registry import allowed_regressors as _allowed_regressors
from shared.version_control import snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CLASSIFIERS = _allowed_classifiers()
ALLOWED_REGRESSORS = _allowed_regressors()
ALLOWED_CLUSTER_ALGOS = CLUSTERERS

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


def fit_final_estimator(model_str: str, x: np.ndarray, y: np.ndarray, task: str) -> object:
    """Fit one estimator on the whole dataset and return it, ready to save.

    Cross-validation fits a throwaway estimator per fold and keeps only its
    predictions, so nothing survives the loop that could be persisted --
    train_with_cv and compare_models were both writing {"model": None} into a
    .pkl while reporting "Saved best model". Refitting on all the data is the
    normal way to produce the shipped model once CV has estimated how well it
    generalises.

    Returns None for model types with no plain estimator to fit (xgb goes
    through the Booster API), so callers must still handle that case rather
    than assume a model came back.
    """
    builder = _build_classifier if task == "classification" else _build_regressor
    estimator = builder(model_str)
    if estimator is None:
        return None
    # The scaled builders return a ("name", scaler, estimator) tuple rather than
    # an assembled object; wrap it so the saved model applies the same scaling
    # at predict time that it was fitted with.
    if isinstance(estimator, tuple):
        from sklearn.pipeline import Pipeline

        _, scaler, inner = estimator
        estimator = Pipeline([("scaler", scaler), ("model", inner)])
    estimator.fit(x, y)  # type: ignore[attr-defined]
    return estimator


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
            return np.argmax(preds, axis=1)
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

# Quality scores deduct per alert, and alerts are raised per column, so the
# penalty grows with the width of the frame. The real ad dataset raises four
# extreme_skewness and three multicollinearity alerts -- 56 points on their own
# -- and scored 5.6, sitting alongside a frame of eight constant columns and
# 100% duplicates on 0.0. Almost the whole scale went unused and the two scorers
# in this server disagreed with each other and with the sibling report in
# MCP_Data_Analyst (41 for the same file).
#
# Capping the alert term is what the missingness and duplicate terms already
# did, and it means no single axis can flatten the score on its own. The caps
# sum to 100 in generate_eda_report, which carries all three terms;
# check_data_quality has no separate missingness term, so its floor is 20.
# Ordering is what matters and it holds: 92 clean, 77 one constant column,
# ~29.5 the real ad dataset, below that a frame that is bad on every axis.
ALERT_DEDUCTION_CAP = 70.0
MISSINGNESS_DEDUCTION_CAP = 20.0
DUPLICATE_DEDUCTION_CAP = 10.0

# Below this, an anomaly detector still returns a verdict but has little to
# base it on: LocalOutlierFactor compares each point to its 20 nearest
# neighbours by default, and an isolation tree over a handful of points
# separates all of them in a split or two, so every point looks equally
# isolated. Not a refusal -- 19 rows is real data -- but the caller should be
# told which side of that line the answer came from.
MIN_ROWS_FOR_ANOMALY_CONFIDENCE = 20

# The smallest sample this server will make a claim about a column's
# *distribution* from. scipy's skewness sets it: it is undefined below three
# values, so extreme_skewness has always been silent there. zero_inflated asks
# the same kind of question -- does this column hold more zeros than its shape
# would predict -- and so answers at the same n. Below it, "100% zeros" is a
# fact about the row count wearing a percentage.
MIN_ROWS_FOR_DISTRIBUTION = 3

_OP_KEY_ALIASES: dict[str, str] = {"operation": "op", "column_name": "column", "col": "column"}
_OP_NAME_ALIASES: dict[str, str] = {"impute_missing": "fill_nulls", "fillna": "fill_nulls"}
_FILL_KEY_ALIASES: dict[str, str] = {"method": "strategy"}

# Every field each op's handler actually reads. Until this existed, run_preprocessing
# validated op names and a couple of enumerated values and nothing else, so any
# other key was dropped in silence -- the same hole the Data_Analyst repo closed
# for apply_patch in round 11 and that was never ported here. Round 14 measured
# what it costs, and every one of these reported success:
#
#     clip_column  column=n min=0 max=100   -> nothing clipped, 900 still 900
#     label_encode column=cat new_column=e  -> no column e; cat overwritten
#     log_transform column=n base=log5      -> natural log, silently
#     log_transform column=n method=log10   -> natural log, silently
#     drop_outliers column=n threshold=5    -> threshold ignored, 1.5*IQR used
#
# Four of the five are a caller using the spelling its sibling tool in the
# Data_Analyst repo documents -- clip_values takes min/max, log_transform takes
# method, cast_column takes dtype. The two servers are used in the same session
# by the same model, so those are the spellings to expect, not to punish.
OP_FIELDS: dict[str, frozenset[str]] = {
    "add_date_parts": frozenset({"column", "parts"}),
    "bin_numeric": frozenset({"bins", "column", "labels", "new_column"}),
    "clip_column": frozenset({"column", "lower", "upper"}),
    "convert_dtype": frozenset({"column", "to"}),
    "drop_column": frozenset({"column"}),
    "drop_duplicates": frozenset({"subset"}),
    "drop_null_rows": frozenset({"column"}),
    # threshold: read now (IQR multiplier, or sigma count for method=std)
    # instead of the hardcoded 1.5 and 3.
    "drop_outliers": frozenset({"column", "method", "threshold"}),
    "fill_nulls": frozenset({"column", "strategy"}),
    # new_column: advertised by the sibling tool and never read here, so the
    # codes were written over the categorical they encode.
    "label_encode": frozenset({"column", "new_column"}),
    "log_transform": frozenset({"base", "column", "new_column"}),
    "onehot_encode": frozenset({"column"}),
    "rename_column": frozenset({"from", "to"}),
    "scale": frozenset({"columns", "method"}),
}

# Spellings a caller writes that mean exactly one thing here. Aliased rather
# than renamed: the documented spelling keeps working unchanged.
OP_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "clip_column": {"min": "lower", "max": "upper"},
    "convert_dtype": {"dtype": "to"},
    "log_transform": {"method": "base"},
    "rename_column": {"old_name": "from", "new_name": "to", "old": "from", "new": "to"},
    "scale": {"column": "columns"},
}

# `natural`, `log` and `log1p` all mean the log1p branch; the other two are
# themselves. An unlisted base used to fall through a bare `else` to natural log
# and report the base it had been given, so the response named a transform that
# had not been applied.
LOG_BASES: frozenset[str] = frozenset({"natural", "log", "log1p", "log2", "log10"})
# Same bare-else shape: anything that was not "iqr" was treated as std.
OUTLIER_METHODS: frozenset[str] = frozenset({"iqr", "std"})
DTYPE_TARGETS: frozenset[str] = frozenset({"datetime", "numeric", "str", "string", "int", "float", "bool"})

_UNIVERSAL_OP_FIELDS: frozenset[str] = frozenset({"op"})


def known_op_fields(op_name: str) -> list[str]:
    """Every field this op reads, plus its accepted aliases."""
    return sorted(_UNIVERSAL_OP_FIELDS | OP_FIELDS.get(op_name, frozenset()) | set(OP_FIELD_ALIASES.get(op_name, {})))


def _did_you_mean(unknown: str, known: list[str]) -> str:
    """The closest accepted name, when one is obviously close."""
    import difflib

    for k in known:
        if k in unknown or unknown in k:
            return k
    close = difflib.get_close_matches(unknown, known, n=1, cutoff=0.75)
    return close[0] if close else ""


def _apply_op_aliases(op: dict) -> dict:
    """Fill a canonical field from the spelling the caller used."""
    for given, canonical in OP_FIELD_ALIASES.get(op.get("op", ""), {}).items():
        if canonical not in op and given in op:
            op[canonical] = op.pop(given)
    return op


def _normalize_op(op: dict) -> dict:
    """Return a copy of op with aliased keys/values normalized to canonical form."""
    normalized = {_OP_KEY_ALIASES.get(k, k): v for k, v in op.items()}
    if "op" in normalized:
        normalized["op"] = _OP_NAME_ALIASES.get(normalized["op"], normalized["op"])
    if normalized.get("op") == "fill_nulls":
        normalized = {_FILL_KEY_ALIASES.get(k, k): v for k, v in normalized.items()}
    return _apply_op_aliases(normalized)


# Every op whose handler reads op["column"] without a default. The check used
# to be a hand-written set naming five of them, so the four added later --
# add_date_parts, bin_numeric, clip_column, log_transform -- reached the handler
# with no column and raised a bare KeyError('column') straight out of the tool,
# past the return-value contract entirely. Derived from the handlers rather than
# re-typed, and asserted against them by a test that calls every allowed op with
# no arguments at all.
OPS_REQUIRING_COLUMN: frozenset[str] = frozenset(
    {
        "add_date_parts",
        "bin_numeric",
        "clip_column",
        "convert_dtype",
        "drop_column",
        "drop_outliers",
        "fill_nulls",
        "label_encode",
        "log_transform",
        "onehot_encode",
    }
)


def _validate_ops(ops: list[dict]) -> tuple[bool, list[dict], str]:
    """Validate and normalize preprocessing ops. Returns (ok, normalized_ops, error_msg)."""
    if not isinstance(ops, list):
        return False, ops, "ops must be a list of dicts."
    if len(ops) > MAX_OPS:
        return False, ops, f"Too many ops: {len(ops)}. Max is {MAX_OPS}."
    normalized: list[dict] = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return False, ops, f"Op #{i} is not a dict."
        op = _normalize_op(op)
        normalized.append(op)
        op_name = op.get("op", "")
        if op_name not in ALLOWED_OPS:
            return False, ops, f"Unknown op: '{op_name}'. Allowed: {', '.join(sorted(ALLOWED_OPS))}"
        if op_name in OPS_REQUIRING_COLUMN and "column" not in op:
            return False, ops, f"Op #{i} ('{op_name}') missing required field: 'column'"

        # Names before values: a field the op does not read is dropped, and the
        # dropped field is often the one that decides what gets written. Checked
        # before the enumerated values so a typo is reported as itself rather
        # than as a complaint about some other field being absent.
        known = known_op_fields(op_name)
        unknown = sorted(k for k in op if k not in known)
        if unknown:
            suggestion = _did_you_mean(unknown[0], known)
            lead = f"did you mean {suggestion}? " if suggestion else ""
            return (
                False,
                ops,
                f"Op #{i} ('{op_name}'): unknown field(s) {', '.join(unknown)} -- "
                f"{lead}{op_name} accepts: {', '.join(known)}",
            )

        if op_name == "log_transform":
            base = op.get("base", "natural")
            if base not in LOG_BASES:
                return (
                    False,
                    ops,
                    f"Op #{i} (log_transform): invalid base '{base}'. Allowed: {', '.join(sorted(LOG_BASES))}",
                )
        elif op_name == "drop_outliers":
            method = op.get("method", "iqr")
            if method not in OUTLIER_METHODS:
                return (
                    False,
                    ops,
                    f"Op #{i} (drop_outliers): invalid method '{method}'. "
                    f"Allowed: {', '.join(sorted(OUTLIER_METHODS))}",
                )
        elif op_name == "convert_dtype":
            target = op.get("to", "")
            if target not in DTYPE_TARGETS:
                return (
                    False,
                    ops,
                    f"Op #{i} (convert_dtype): invalid target '{target}'. Allowed: {', '.join(sorted(DTYPE_TARGETS))}",
                )

        if op_name == "fill_nulls":
            strategy = op.get("strategy", "median")
            if strategy not in FILL_STRATEGIES:
                return (
                    False,
                    ops,
                    (f"Strategy '{strategy}' not valid for fill_nulls. Allowed: {' '.join(sorted(FILL_STRATEGIES))}"),
                )
        elif op_name == "scale":
            if "columns" not in op:
                return False, ops, f"Op '{op_name}' missing required field: 'columns'"
            # One column named as a bare string reached StandardScaler as a
            # 1-D Series and raised out of the tool.
            if isinstance(op["columns"], str):
                op["columns"] = [op["columns"]]
            if not isinstance(op["columns"], list) or not op["columns"]:
                return False, ops, f"Op #{i} (scale): 'columns' must be a non-empty list of column names"
            method = op.get("method", "standard")
            if method not in SCALE_METHODS:
                return False, ops, f"Method '{method}' not valid for scale. Allowed: standard minmax"
        elif op_name == "rename_column":
            for field in ("from", "to"):
                if field not in op:
                    return False, ops, f"Op '{op_name}' missing required field: '{field}'"
    return True, normalized, ""


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
        # threshold was accepted and never read: 1.5 and 3 were hardcoded, so
        # asking for a wider or tighter fence changed nothing and said nothing.
        # The validator has already refused any method outside OUTLIER_METHODS,
        # so this is a choice between two, not a fallthrough.
        if method == "iqr":
            multiplier = float(op.get("threshold", 1.5))
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        else:
            multiplier = float(op.get("threshold", 3.0))
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - multiplier * std, mean + multiplier * std
        df = df[(df[col] >= lower) & (df[col] <= upper)].copy()  # type: ignore[assignment]
        return df, {
            "op": op_name,
            "column": col,
            "method": method,
            "threshold": multiplier,
            "removed": before - len(df),
        }

    elif op_name == "label_encode":
        col = op["column"]
        if col not in df.columns:
            return df, {"op": op_name, "column": col, "error": "column not found"}
        # new_column was accepted and never read, so the codes were written over
        # the categorical they encode and the original was gone.
        new_col = op.get("new_column") or col
        le = LabelEncoder()
        df[new_col] = le.fit_transform(df[col].fillna("nan").astype(str))
        return df, {
            "op": op_name,
            "column": col,
            "new_column": new_col,
            "replaced_source": new_col == col,
            "classes": list(le.classes_[:10]),
        }

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
        return df, {"op": op_name, "removed": before - len(df)}

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
        # The validator has already refused any base outside LOG_BASES. It used
        # to end in a bare `else`, so an unrecognised base -- including `log10`
        # written as the sibling tool's `method=log10` -- came back reported as
        # the base asked for and computed as a natural log.
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


def receipt_for_created(output_path: str, source_path: object, tool: str, args: dict) -> None:
    """Give a file this tool created its own provenance entry.

    A tool's receipt is filed against its input. When the output is a different
    file, that new file was left with no history at all, so read_receipt on it
    answered entry_count 0 with success true -- making "nothing has been done to
    this file" and "its history is filed under another name" the same answer.
    Found on run_clustering; run_preprocessing and merge_datasets did it too,
    which is why this lives here rather than being written out three times.
    """
    if not output_path:
        return
    out = Path(output_path)
    src = Path(str(source_path))
    if out == src:
        return  # in place: the input's own receipt already covers it
    append_receipt(str(out), tool, {"source": src.name, **args}, f"created from {src.name}")


__all__ = [
    "receipt_for_created",
    "baseline_warning",
    "leakage_warning",
    "typical_row",
    "bounded_silhouette",
    # re-exports from shared
    "get_output_dir",
    "get_output_path",
    "resolve_path",
    "_read_csv",
    # chart helper
    "_save_chart",
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
    "ALERT_DEDUCTION_CAP",
    "MISSINGNESS_DEDUCTION_CAP",
    "DUPLICATE_DEDUCTION_CAP",
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
