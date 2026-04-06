"""ml_basic engine — Tier 1 ML logic. Zero MCP imports."""

import json
import logging
import pickle
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
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
import xgboost as xgb

from shared.file_utils import resolve_path
from shared.platform_utils import get_max_columns, get_max_results, get_max_rows
from shared.progress import fail, info, ok, warn
from shared.progress import name as pname
from shared.receipt import append_receipt
from shared.version_control import list_snapshots
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
    base: dict = {"success": False, "error": error, "hint": hint}
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
        if (
            pd.api.types.is_string_dtype(df[col])
            or df[col].dtype == object
            or str(df[col].dtype) == "category"
        ):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("nan").astype(str))
            encoding_map[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            encoded_cols.append(col)

    # fill numeric nulls with median
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

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
    manifest_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _load_model(model_path: str) -> tuple[Any, dict]:
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["metadata"]



# ---------------------------------------------------------------------------
# 1. inspect_dataset
# ---------------------------------------------------------------------------
def inspect_dataset(file_path: str) -> dict:
    """Inspect dataset schema, row count, dtypes, null summary."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        df = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows × {len(df.columns)} cols"))

        max_cols = get_max_columns()
        all_columns = list(df.columns)
        truncated = len(all_columns) > max_cols
        display_cols = all_columns[:max_cols]

        col_info = []
        for col in display_cols:
            null_count = int(df[col].isnull().sum())
            col_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
            })

        # target candidates: ≤20 unique values or bool dtype
        target_candidates = [
            c for c in all_columns
            if df[c].dtype == bool or df[c].nunique() <= 20
        ]

        response = {
            "success": True,
            "op": "inspect_dataset",
            "file": pname(file_path),
            "row_count": len(df),
            "column_count": len(all_columns),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "columns": col_info,
            "target_candidates": target_candidates[:get_max_results()],
            "truncated": truncated,
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except ValueError as exc:
        return _error(str(exc), "Check that file_path points to a valid CSV file.")
    except Exception as exc:
        logger.debug("inspect_dataset error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() with an absolute path to a .csv file.")


# ---------------------------------------------------------------------------
# 2. read_column_profile
# ---------------------------------------------------------------------------
def read_column_profile(file_path: str, column_name: str) -> dict:
    """Profile one column. Returns stats, null count, top values."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        df = pd.read_csv(path, low_memory=False)
        if column_name not in df.columns:
            return _error(
                f"Column '{column_name}' not found. Available: {', '.join(list(df.columns)[:10])}",
                "Use inspect_dataset() to list all column names.",
            )

        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows"))
        series = df[column_name]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / len(df) * 100, 2) if len(df) else 0.0
        dtype_str = str(series.dtype)

        if series.dtype == bool or (series.nunique() <= 2 and set(series.dropna().unique()) <= {0, 1, True, False}):
            true_count = int(series.sum()) if series.dtype != bool else int(series.sum())
            false_count = int(len(series.dropna())) - true_count
            profile: dict = {
                "dtype": dtype_str,
                "kind": "boolean",
                "true_count": true_count,
                "false_count": false_count,
                "null_count": null_count,
                "null_pct": null_pct,
                "balance_ratio": round(true_count / max(false_count, 1), 4),
            }
        elif pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            profile = {
                "dtype": dtype_str,
                "kind": "numeric",
                "mean": round(float(clean.mean()), 4) if len(clean) else None,
                "std": round(float(clean.std()), 4) if len(clean) else None,
                "min": round(float(clean.min()), 4) if len(clean) else None,
                "max": round(float(clean.max()), 4) if len(clean) else None,
                "median": round(float(clean.median()), 4) if len(clean) else None,
                "q25": round(float(clean.quantile(0.25)), 4) if len(clean) else None,
                "q75": round(float(clean.quantile(0.75)), 4) if len(clean) else None,
                "skewness": round(float(clean.skew()), 4) if len(clean) else None,
                "null_count": null_count,
                "null_pct": null_pct,
            }
        else:
            top_vals = series.value_counts().head(10)
            profile = {
                "dtype": dtype_str,
                "kind": "categorical",
                "unique_count": int(series.nunique()),
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
                "mode": str(series.mode().iloc[0]) if len(series.dropna()) else None,
                "null_count": null_count,
                "null_pct": null_pct,
            }

        progress.append(ok(f"Profiled '{column_name}'", profile["kind"]))
        response = {
            "success": True,
            "op": "read_column_profile",
            "file": pname(file_path),
            "column": column_name,
            "profile": profile,
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except ValueError as exc:
        return _error(str(exc), "Use inspect_dataset() to verify column names.")
    except Exception as exc:
        logger.debug("read_column_profile error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() to verify column names and file path.")


# ---------------------------------------------------------------------------
# 3. search_columns
# ---------------------------------------------------------------------------
def search_columns(
    file_path: str,
    has_nulls: bool = False,
    dtype: str = "",
    name_contains: str = "",
    max_results: int = 20,
) -> dict:
    """Search columns by condition. Returns names only, no data."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        df = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df.columns)} columns"))

        cap = min(max_results, get_max_results())
        matches: list[str] = []

        for col in df.columns:
            series = df[col]
            if has_nulls and not series.isnull().any():
                continue
            if dtype:
                if dtype == "numeric" and not pd.api.types.is_numeric_dtype(series):
                    continue
                elif dtype == "categorical" and (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
                    continue
                elif dtype == "bool" and not pd.api.types.is_bool_dtype(series):
                    continue
                elif dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
                    continue
            if name_contains and name_contains.lower() not in col.lower():
                continue
            matches.append(col)

        truncated = len(matches) > cap
        response = {
            "success": True,
            "op": "search_columns",
            "file": pname(file_path),
            "columns": matches[:cap],
            "returned": len(matches[:cap]),
            "total_matched": len(matches),
            "truncated": truncated,
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except ValueError as exc:
        return _error(str(exc), "Check file_path and dtype parameter.")
    except Exception as exc:
        logger.debug("search_columns error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() to verify the file path.")



# ---------------------------------------------------------------------------
# 4. read_rows
# ---------------------------------------------------------------------------
def read_rows(file_path: str, start: int, end: int) -> dict:
    """Read bounded row slice. Max rows enforced by hardware mode."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        df = pd.read_csv(path, low_memory=False)
        total = len(df)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{total:,} rows total"))

        cap = get_max_rows()
        requested = max(0, end - start)
        actual = min(requested, cap)
        truncated = requested > actual

        slice_df = df.iloc[start: start + actual]
        rows = slice_df.where(slice_df.notna(), other=None).to_dict(orient="records")

        response = {
            "success": True,
            "op": "read_rows",
            "file": pname(file_path),
            "rows": rows,
            "returned": len(rows),
            "total_available": total,
            "start": start,
            "end": start + len(rows),
            "truncated": truncated,
            "progress": progress,
        }
        if truncated:
            response["hint"] = f"Results capped at {cap}. Use start/end parameters to page through the data."
        response["token_estimate"] = len(str(response)) // 4
        return response

    except ValueError as exc:
        return _error(str(exc), "Provide a valid CSV file path.")
    except Exception as exc:
        logger.debug("read_rows error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() to verify row count before slicing.")


# ---------------------------------------------------------------------------
# 5. train_classifier
# ---------------------------------------------------------------------------
def train_classifier(
    file_path: str,
    target_column: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train classifier on CSV. model: lr svm rf dtc knn nb xgb."""
    progress: list[dict] = []
    backup: str | None = None
    try:
        # --- validation ---
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        model = model.strip().lower()
        if model not in ALLOWED_CLASSIFIERS:
            return _error(
                f"Unknown algorithm: '{model}'. Allowed: {', '.join(sorted(ALLOWED_CLASSIFIERS))}",
                "Use one of: lr svm rf dtc knn nb xgb",
            )

        df_raw = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df_raw):,} rows × {len(df_raw.columns)} cols"))

        # RAM check
        required_gb = df_raw.memory_usage(deep=True).sum() / 1e9 * 3
        mem_err = _check_memory(required_gb)
        if mem_err:
            return mem_err

        if target_column not in df_raw.columns:
            return _error(
                f"Column '{target_column}' not found. Available: {', '.join(list(df_raw.columns)[:10])}",
                "Use inspect_dataset() to list all column names.",
            )

        df, encoding_map, encoded_cols = _auto_preprocess(df_raw, target_column)
        if encoded_cols:
            progress.append(ok(f"Encoded {len(encoded_cols)} categorical columns", "LabelEncoder"))

        if len(df) < MIN_ROWS_CLASSIFIER:
            return _error(
                f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_CLASSIFIER}.",
                "Provide a dataset with more samples before training.",
            )

        n_classes = df[target_column].nunique()
        if n_classes < 2:
            return _error(
                f"Target column '{target_column}' has only {n_classes} unique value — cannot train classifier.",
                "Choose a column with at least 2 distinct class values.",
            )

        feature_cols = [c for c in df.columns if c != target_column]
        x = df[feature_cols].values
        y = df[target_column].values

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "op": "train_classifier",
                "model": model,
                "target_column": target_column,
                "feature_columns": feature_cols,
                "row_count": len(df),
                "would_train": True,
                "progress": progress,
                "token_estimate": 80,
            }

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        progress.append(ok("Split dataset", f"{len(x_train):,} train / {len(x_test):,} test (stratified)"))

        # --- model training ---
        scaler: StandardScaler | None = None
        model_class_name = ""

        if model == "lr":
            clf = LogisticRegression(random_state=42, max_iter=200)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "LogisticRegression"
            trained = clf

        elif model == "svm":
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            clf = SVC(kernel="rbf", gamma="auto", random_state=42)
            clf.fit(x_train_s, y_train)
            y_pred = clf.predict(x_test_s)
            model_class_name = "SVC"
            trained = clf

        elif model == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "RandomForestClassifier"
            trained = clf

        elif model == "dtc":
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "DecisionTreeClassifier"
            trained = clf

        elif model == "knn":
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
            clf.fit(x_train_s, y_train)
            y_pred = clf.predict(x_test_s)
            model_class_name = "KNeighborsClassifier"
            trained = clf

        elif model == "nb":
            clf = GaussianNB()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            model_class_name = "GaussianNB"
            trained = clf

        else:  # xgb
            nc = int(n_classes)
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test, label=y_test)
            params: dict = {
                "max_depth": 3, "eta": 0.3, "verbosity": 0,
                "objective": "multi:softprob" if nc > 2 else "binary:logistic",
            }
            if nc > 2:
                params["num_class"] = nc
            xgb_model = xgb.train(params, dtrain, num_boost_round=10)
            preds = xgb_model.predict(dtest)
            if nc > 2:
                y_pred = np.asarray([np.argmax(line) for line in preds])
            else:
                y_pred = (preds > 0.5).astype(int)
            model_class_name = "XGBClassifier"
            trained = xgb_model

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = _confusion_dict(y_test, y_pred)
        metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "confusion_matrix": cm}
        progress.append(ok(f"Trained {model_class_name}", f"accuracy={acc:.3f}, f1={f1:.3f}"))

        # --- save model ---
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        models_dir = path.parent / MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        model_filename = f"{path.stem}_{model}_{ts}.pkl"
        model_path = models_dir / model_filename

        # snapshot if overwriting
        if model_path.exists():
            backup = snapshot(str(model_path))

        metadata: dict = {
            "model_type": model_class_name,
            "task": "classification",
            "model_key": model,
            "trained_on": path.name,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "feature_columns": feature_cols,
            "target_column": target_column,
            "encoding_map": encoding_map,
            "scaler": scaler,
            "metrics": metrics,
            "n_classes": int(n_classes),
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
        }
        _save_model(trained, model_path, metadata)
        progress.append(ok("Saved model", pname(str(model_path))))

        append_receipt(file_path, "train_classifier", {"target": target_column, "model": model}, f"accuracy={acc:.3f}", backup)

        response: dict = {
            "success": True,
            "op": "train_classifier",
            "model": model,
            "model_class": model_class_name,
            "task": "classification",
            "target_column": target_column,
            "feature_columns": feature_cols,
            "row_count": len(df),
            "train_size": len(x_train),
            "test_size": len(x_test),
            "metrics": metrics,
            "model_path": str(model_path),
            "backup": backup or "",
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except Exception as exc:
        logger.debug("train_classifier error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() and read_column_profile() to verify your data first.", backup)



# ---------------------------------------------------------------------------
# 6. train_regressor
# ---------------------------------------------------------------------------
def train_regressor(
    file_path: str,
    target_column: str,
    model: str,
    degree: int = 5,
    alpha: float = 0.01,
    n_estimators: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train regressor on CSV. model: lir pr lar rr dtr rfr xgb."""
    progress: list[dict] = []
    backup: str | None = None
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        model = model.strip().lower()
        if model not in ALLOWED_REGRESSORS:
            return _error(
                f"Unknown algorithm: '{model}'. Allowed: {', '.join(sorted(ALLOWED_REGRESSORS))}",
                "Use one of: lir pr lar rr dtr rfr xgb",
            )

        df_raw = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df_raw):,} rows × {len(df_raw.columns)} cols"))

        required_gb = df_raw.memory_usage(deep=True).sum() / 1e9 * 3
        mem_err = _check_memory(required_gb)
        if mem_err:
            return mem_err

        if target_column not in df_raw.columns:
            return _error(
                f"Column '{target_column}' not found. Available: {', '.join(list(df_raw.columns)[:10])}",
                "Use inspect_dataset() to list all column names.",
            )

        df, encoding_map, encoded_cols = _auto_preprocess(df_raw, target_column)
        if encoded_cols:
            progress.append(ok(f"Encoded {len(encoded_cols)} categorical columns", "LabelEncoder"))

        if len(df) < MIN_ROWS_REGRESSOR:
            return _error(
                f"Dataset has only {len(df)} rows. Need at least {MIN_ROWS_REGRESSOR}.",
                "Provide a dataset with more samples before training.",
            )

        feature_cols = [c for c in df.columns if c != target_column]
        x = df[feature_cols].values.astype(float)
        y = df[target_column].values.astype(float)

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "op": "train_regressor",
                "model": model,
                "target_column": target_column,
                "feature_columns": feature_cols,
                "row_count": len(df),
                "would_train": True,
                "progress": progress,
                "token_estimate": 80,
            }

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        progress.append(ok("Split dataset", f"{len(x_train):,} train / {len(x_test):,} test"))

        poly: PolynomialFeatures | None = None
        model_class_name = ""
        scaler: StandardScaler | None = None

        if model == "lir":
            reg = LinearRegression()
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "LinearRegression"
            trained = reg

        elif model == "pr":
            poly = PolynomialFeatures(degree=degree)
            x_train_p = poly.fit_transform(x_train)
            x_test_p = poly.transform(x_test)
            reg = LinearRegression()
            reg.fit(x_train_p, y_train)
            y_pred = reg.predict(x_test_p)
            model_class_name = "PolynomialRegression"
            trained = reg

        elif model == "lar":
            reg = Lasso(alpha=alpha, max_iter=200, tol=0.1)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "Lasso"
            trained = reg

        elif model == "rr":
            reg = Ridge(alpha=alpha, max_iter=100, tol=0.1)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "Ridge"
            trained = reg

        elif model == "dtr":
            reg = DecisionTreeRegressor(random_state=42)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "DecisionTreeRegressor"
            trained = reg

        elif model == "rfr":
            reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            model_class_name = "RandomForestRegressor"
            trained = reg

        else:  # xgb
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test, label=y_test)
            params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.3, "verbosity": 0}
            xgb_model = xgb.train(params, dtrain, num_boost_round=5)
            y_pred = xgb_model.predict(dtest)
            model_class_name = "XGBRegressor"
            trained = xgb_model

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        metrics = {"mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
        progress.append(ok(f"Trained {model_class_name}", f"r2={r2:.3f}, rmse={rmse:.2f}"))

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        models_dir = path.parent / MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{path.stem}_{model}_{ts}.pkl"

        if model_path.exists():
            backup = snapshot(str(model_path))

        metadata: dict = {
            "model_type": model_class_name,
            "task": "regression",
            "model_key": model,
            "trained_on": path.name,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "feature_columns": feature_cols,
            "target_column": target_column,
            "encoding_map": encoding_map,
            "poly": poly,
            "scaler": scaler,
            "metrics": metrics,
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
        }
        _save_model(trained, model_path, metadata)
        progress.append(ok("Saved model", pname(str(model_path))))

        append_receipt(file_path, "train_regressor", {"target": target_column, "model": model}, f"r2={r2:.3f}", backup)

        response: dict = {
            "success": True,
            "op": "train_regressor",
            "model": model,
            "model_class": model_class_name,
            "task": "regression",
            "target_column": target_column,
            "feature_columns": feature_cols,
            "row_count": len(df),
            "train_size": len(x_train),
            "test_size": len(x_test),
            "metrics": metrics,
            "model_path": str(model_path),
            "backup": backup or "",
            "progress": progress,
        }
        response["token_estimate"] = len(str(response)) // 4
        return response

    except Exception as exc:
        logger.debug("train_regressor error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() and read_column_profile() to verify your data first.", backup)



# ---------------------------------------------------------------------------
# 7. get_predictions
# ---------------------------------------------------------------------------
def get_predictions(model_path: str, file_path: str, max_rows: int = 20) -> dict:
    """Run predictions with saved model. Returns bounded prediction list."""
    progress: list[dict] = []
    try:
        mpath = Path(model_path).resolve()
        if not mpath.exists():
            return _error(
                f"Model file not found: {model_path}",
                "Use train_classifier() or train_regressor() to train a model first.",
            )

        model_obj, metadata = _load_model(model_path)
        progress.append(ok(f"Loaded model {pname(model_path)}", metadata.get("model_type", "")))

        data_path = resolve_path(file_path, (".csv",))
        if not data_path.exists():
            return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")

        df = pd.read_csv(data_path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows"))

        # apply same encoding as training
        encoding_map: dict = metadata.get("encoding_map", {})
        for col, mapping in encoding_map.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    {str(k): v for k, v in mapping.items()}
                ).fillna(-1).astype(int)

        feature_cols: list[str] = metadata.get("feature_columns", [])
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return _error(
                f"Feature columns missing in data: {', '.join(missing)}",
                "Ensure the data file has the same columns used during training.",
            )

        x = df[feature_cols].fillna(0).values.astype(float)

        # apply scaler if used
        sc = metadata.get("scaler")
        if sc is not None:
            x = sc.transform(x)

        # apply poly if used
        poly = metadata.get("poly")
        if poly is not None:
            x = poly.transform(x)

        cap = min(max_rows, get_max_rows())
        x_slice = x[:cap]
        truncated = len(x) > cap

        task = metadata.get("task", "classification")
        model_key = metadata.get("model_key", "")

        if model_key == "xgb":
            dmat = xgb.DMatrix(x_slice)
            raw_preds = model_obj.predict(dmat)
            if task == "classification":
                n_classes = metadata.get("n_classes", 2)
                if n_classes > 2:
                    preds_list = [int(np.argmax(line)) for line in raw_preds]
                else:
                    preds_list = [int(p > 0.5) for p in raw_preds]
            else:
                preds_list = [float(p) for p in raw_preds]
        else:
            raw = model_obj.predict(x_slice)
            if task == "classification":
                preds_list = [int(p) for p in raw]
            else:
                preds_list = [float(p) for p in raw]

        predictions = [{"row": i, "prediction": preds_list[i]} for i in range(len(preds_list))]
        progress.append(ok("Generated predictions", f"{len(predictions)} rows"))

        response: dict = {
            "success": True,
            "op": "get_predictions",
            "model_path": pname(model_path),
            "task": task,
            "predictions": predictions,
            "returned": len(predictions),
            "total_rows": len(x),
            "truncated": truncated,
            "progress": progress,
        }
        if truncated:
            response["hint"] = f"Results capped at {cap}. Pass max_rows to increase (up to {get_max_rows()})."
        response["token_estimate"] = len(str(response)) // 4
        return response

    except Exception as exc:
        logger.debug("get_predictions error: %s", exc)
        return _error(str(exc), "Check model_path and that the data file matches the training schema.")


# ---------------------------------------------------------------------------
# 8. restore_version
# ---------------------------------------------------------------------------
def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file/model to previous snapshot. Empty timestamp = list."""
    progress: list[dict] = []
    try:
        result = _restore_version(file_path, timestamp)
        if result.get("success"):
            if timestamp:
                progress.append(ok(f"Restored {pname(file_path)}", result.get("restored_from", "")))
            else:
                progress.append(info(f"Listed snapshots for {pname(file_path)}", f"{len(result.get('snapshots', []))} available"))
        result["progress"] = progress
        result["token_estimate"] = len(str(result)) // 4
        return result
    except Exception as exc:
        logger.debug("restore_version error: %s", exc)
        return _error(str(exc), "Check that the file path is correct and snapshots exist in .mcp_versions/.")
