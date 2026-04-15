"""ml_basic engine — Tier 1 ML logic. Zero MCP imports."""

from __future__ import annotations

import logging

import pandas as pd

from shared.file_utils import resolve_path
from shared.handover import make_handover
from shared.platform_utils import get_max_columns, get_max_results, get_max_rows
from shared.progress import name as pname
from shared.progress import ok

from ._basic_helpers import _confusion_dict, _error
from ._basic_predict import (
    get_predictions,
    list_models,
    predict_single,
    restore_version,
    split_dataset,
)
from ._basic_train import train_classifier, train_regressor

logger = logging.getLogger(__name__)

__all__ = [
    "inspect_dataset",
    "read_column_profile",
    "search_columns",
    "read_rows",
    "train_classifier",
    "train_regressor",
    "get_predictions",
    "restore_version",
    "predict_single",
    "list_models",
    "split_dataset",
    "_confusion_dict",
]


# ---------------------------------------------------------------------------
# 1. inspect_dataset
# ---------------------------------------------------------------------------
def inspect_dataset(file_path: str) -> dict:
    """Inspect dataset schema, row count, dtypes, null summary."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

        df = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows × {len(df.columns)} cols"))

        max_cols = get_max_columns()
        all_columns = list(df.columns)
        truncated = len(all_columns) > max_cols
        display_cols = all_columns[:max_cols]

        col_info = []
        for col in display_cols:
            null_count = int(df[col].isnull().sum())
            col_info.append(
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "null_count": null_count,
                    "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
                }
            )

        # target candidates: ≤20 unique values or bool dtype
        target_candidates = [c for c in all_columns if df[c].dtype == bool or df[c].nunique() <= 20]

        response = {
            "success": True,
            "op": "inspect_dataset",
            "file": pname(file_path),
            "row_count": len(df),
            "column_count": len(all_columns),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "columns": col_info,
            "target_candidates": target_candidates[: get_max_results()],
            "truncated": truncated,
            "progress": progress,
        }
        response["handover"] = make_handover(
            "LOCATE",
            ["read_column_profile", "search_columns", "read_rows"],
            {"file_path": file_path},
        )
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
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

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
            true_count = int(series.sum())
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
        response["handover"] = make_handover(
            "INSPECT",
            ["train_classifier", "train_regressor", "run_preprocessing"],
            {"file_path": file_path, "column_name": column_name},
        )
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
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

        df = pd.read_csv(path, low_memory=False)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df.columns)} columns"))

        cap = min(max_results, get_max_results())
        matches: list[str] = []

        for col in df.columns:
            series = df[col]
            if has_nulls and not bool(series.isnull().any()):
                continue
            if dtype:
                if dtype == "numeric" and not pd.api.types.is_numeric_dtype(series):
                    continue
                elif dtype == "categorical" and (
                    pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                ):
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
        response["handover"] = make_handover(
            "LOCATE",
            ["read_column_profile", "read_rows"],
            {"file_path": file_path},
        )
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
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )

        df = pd.read_csv(path, low_memory=False)
        total = len(df)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{total:,} rows total"))

        cap = get_max_rows()
        requested = max(0, end - start)
        actual = min(requested, cap)
        truncated = requested > actual

        slice_df = df.iloc[start : start + actual]
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
        response["handover"] = make_handover(
            "INSPECT",
            ["train_classifier", "train_regressor", "run_preprocessing"],
            {"file_path": file_path},
        )
        response["token_estimate"] = len(str(response)) // 4
        return response

    except ValueError as exc:
        return _error(str(exc), "Provide a valid CSV file path.")
    except Exception as exc:
        logger.debug("read_rows error: %s", exc)
        return _error(str(exc), "Use inspect_dataset() to verify row count before slicing.")
