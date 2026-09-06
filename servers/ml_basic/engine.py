"""ml_basic engine — Tier 1 ML logic. Zero MCP imports."""

from __future__ import annotations

import logging

import pandas as pd

from shared.counts import counted
from shared.file_utils import read_csv as _read_csv
from shared.file_utils import resolve_path
from shared.handover import make_context, make_handover
from shared.platform_utils import get_max_columns, get_max_results, get_max_rows
from shared.progress import name as pname
from shared.progress import ok, warn
from shared.version_control import size_kb

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

# The vocabulary search_columns' `dtype` actually filters by, and the concrete
# pandas names a caller reads off inspect_dataset mapped onto it. Kept
# deliberately in step with MCP_Data_Analyst's DTYPE_FILTER_ALIASES, because
# the two repos expose a tool of the same name with the same description and a
# caller cannot tell which one they are holding. The groups differ -- this tier
# separates `bool` and calls the string group `categorical` -- so the alias
# targets differ, but every pandas name accepted there is accepted here.
DTYPE_FILTERS: frozenset[str] = frozenset({"numeric", "categorical", "bool", "datetime"})
DTYPE_FILTER_ALIASES: dict[str, str] = {
    "float": "numeric",
    "float64": "numeric",
    "int": "numeric",
    "int64": "numeric",
    "number": "numeric",
    "numerical": "numeric",
    "category": "categorical",
    "object": "categorical",
    "str": "categorical",
    "string": "categorical",
    "text": "categorical",
    "boolean": "bool",
    "date": "datetime",
    "datetime64": "datetime",
    "timestamp": "datetime",
}


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
        if path.stat().st_size == 0:
            return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

        df = _read_csv(str(path))
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows × {len(df.columns)} cols"))

        max_cols = get_max_columns()
        all_columns = list(df.columns)
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
        # Cut by get_max_results() while `truncated` above was derived from
        # get_max_columns(). Two lists, two caps, one flag: a caller whose
        # columns all fitted read `truncated: false` and could still be missing
        # target candidates, which is the field they came here to read.
        candidates_shown = target_candidates[: get_max_results()]

        response = {
            "success": True,
            "op": "inspect_dataset",
            "file": pname(file_path),
            "row_count": len(df),
            "column_count": len(all_columns),
            "file_size_kb": size_kb(path.stat().st_size),
            "columns": col_info,
            "target_candidates": candidates_shown,
            "target_candidates_total": len(target_candidates),
            "target_candidates_truncated": len(candidates_shown) < len(target_candidates),
            **counted(len(col_info), len(all_columns)),
            "progress": progress,
        }
        response["context"] = make_context(
            "inspect_dataset",
            f"Inspected {pname(file_path)}: {len(df):,} rows × {len(all_columns)} cols",
        )
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
def _top_values(series: pd.Series, limit: int = 10) -> dict:
    """The most frequent values and their counts, for any dtype.

    read_column_profile's docstring says it returns "stats, null count, top
    values"; the third only ever appeared for categorical columns, so a numeric
    column got two of the three things it was told it would get.
    """
    counts = series.value_counts().head(limit)
    return {str(k): int(v) for k, v in counts.items()}


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
        if path.stat().st_size == 0:
            return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

        df = _read_csv(str(path))
        if column_name not in df.columns:
            return _error(
                f"Column '{column_name}' not found. Available: {', '.join(list(df.columns)[:10])}",
                "Use inspect_dataset() to list all column names.",
            )

        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df):,} rows"))
        series = df[column_name]
        null_count = int(series.isnull().sum())
        observed = int(series.notna().sum())
        # 0.0 is a real percentage and reads as a healthy column. With nothing
        # observed there is no percentage to report, so report none.
        null_pct = round(null_count / len(df) * 100, 2) if len(df) else None
        dtype_str = str(series.dtype)

        if observed == 0:
            # The boolean test below asks whether the observed values are a
            # subset of {0, 1, True, False}. The empty set is a subset of
            # everything, so a column holding nothing passed it -- and a plainly
            # numeric column came back kind="boolean" with a balance_ratio
            # computed from two zeroes, under success: true. Emptiness is not a
            # type, and it is not something to infer a type from.
            reason = f"all {null_count} rows are null" if null_count else "the file has no data rows"
            profile = {
                "dtype": dtype_str,
                "kind": "empty",
                "count": 0,
                "null_count": null_count,
                "null_pct": null_pct,
                "note": f"'{column_name}' has no observed values ({reason}); nothing was inferred.",
            }
            progress.append(warn(f"'{column_name}' is empty", "no type or statistics inferred"))
            response = {
                "success": True,
                "op": "read_column_profile",
                "file": pname(file_path),
                "column": column_name,
                "profile": profile,
                "hint": ("Load a file with data rows, or use inspect_dataset() to see which columns hold values."),
                "progress": progress,
            }
            response["token_estimate"] = len(str(response)) // 4
            return response

        if series.dtype == bool or (series.nunique() <= 2 and set(series.dropna().unique()) <= {0, 1, True, False}):
            true_count = int(series.sum())
            false_count = int(len(series.dropna())) - true_count
            profile: dict = {
                "dtype": dtype_str,
                "kind": "boolean",
                "true_count": true_count,
                "false_count": false_count,
                "top_values": _top_values(series),
                "null_count": null_count,
                "null_pct": null_pct,
                "balance_ratio": round(true_count / max(false_count, 1), 4),
            }
        elif pd.api.types.is_numeric_dtype(series):
            dropped = series.dropna()
            inf_count = int(dropped.isin([float("inf"), float("-inf")]).sum())
            clean = dropped[~dropped.isin([float("inf"), float("-inf")])]
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
                # The docstring promises "stats, null count, top values" and
                # only the categorical branch produced the third. On the
                # reference dataset link_clicks comes back with median, q25 and
                # q75 all 0.0 -- a heavy-zero column whose shape the top values
                # state outright, from the one tool that had been asked for
                # them and answered with the other two thirds.
                "top_values": _top_values(series),
                "inf_count": inf_count,
                "null_count": null_count,
                "null_pct": null_pct,
            }
        else:
            profile = {
                "dtype": dtype_str,
                "kind": "categorical",
                "unique_count": int(series.nunique()),
                "top_values": _top_values(series),
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
        response["context"] = make_context(
            "read_column_profile",
            f"Profiled column '{column_name}' ({profile['kind']}) in {pname(file_path)}",
        )
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
    """Search columns: dtype numeric/categorical/bool/datetime. Names only."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path, (".csv",))
        if not path.exists():
            return _error(
                f"File not found: {file_path}",
                "Check that file_path is absolute and the CSV file exists.",
            )
        if path.stat().st_size == 0:
            return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

        df = _read_csv(str(path))
        progress.append(ok(f"Loaded {pname(file_path)}", f"{len(df.columns)} columns"))

        # A dtype this tool cannot filter by used to fall through every branch
        # of the chain below and therefore filter NOTHING, so the ordinary
        # pandas name a caller reads off inspect_dataset came back as a clean
        # pass over the whole frame:
        #
        #     search_columns(f, dtype="float64")  -> all 16 columns, success
        #
        # Refuse an unlisted value and name the vocabulary, and accept the
        # concrete pandas names as aliases so the obvious call works. The
        # sibling tool in MCP_Data_Analyst answers dtype="float64" with the
        # four numeric columns; a caller cannot be expected to know which of
        # two identically-described tools they are holding.
        dtype_key = DTYPE_FILTER_ALIASES.get(dtype.strip().lower(), dtype.strip().lower()) if dtype else ""
        if dtype and dtype_key not in DTYPE_FILTERS:
            return _error(
                f"Cannot filter by dtype '{dtype}'.",
                f"Use one of: {', '.join(sorted(DTYPE_FILTERS))}. "
                f"Concrete pandas names are accepted too ({', '.join(sorted(DTYPE_FILTER_ALIASES))}).",
            )
        # An alias widens the filter -- float64 means "numeric", which also
        # matches int columns. Say so, or the count disagrees with the name
        # the caller used and nothing explains why.
        if dtype and dtype_key != dtype.strip().lower():
            progress.append(
                warn(
                    f"Filtered by '{dtype_key}', not '{dtype.strip()}' exactly",
                    f"this tool groups dtypes into {', '.join(sorted(DTYPE_FILTERS))}",
                )
            )

        cap = min(max_results, get_max_results())
        matches: list[str] = []

        for col in df.columns:
            series = df[col]
            if has_nulls and not bool(series.isnull().any()):
                continue
            if dtype_key:
                if dtype_key == "numeric" and not pd.api.types.is_numeric_dtype(series):
                    continue
                elif dtype_key == "categorical" and (
                    pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                ):
                    continue
                elif dtype_key == "bool" and not pd.api.types.is_bool_dtype(series):
                    continue
                elif dtype_key == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
                    continue
            if name_contains and name_contains.lower() not in col.lower():
                continue
            matches.append(col)

        shown = matches[:cap]
        response = {
            "success": True,
            "op": "search_columns",
            "file": pname(file_path),
            "columns": shown,
            "total_matched": len(matches),
            **counted(len(shown), len(matches)),
            "progress": progress,
        }
        response["context"] = make_context(
            "search_columns",
            f"Found {len(matches)} column(s) matching criteria in {pname(file_path)}",
        )
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
        if path.stat().st_size == 0:
            return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

        df = _read_csv(str(path))
        total = len(df)
        progress.append(ok(f"Loaded {pname(file_path)}", f"{total:,} rows total"))

        cap = get_max_rows()
        requested = max(0, end - start)
        actual = min(requested, cap)
        truncated = requested > actual

        slice_df = df.iloc[start : start + actual]
        rows = slice_df.where(slice_df.notna(), other=None).to_dict(orient="records")

        # What the caller could have had from this window: their own range,
        # bounded by where the file ends. Running out of rows is not truncation
        # -- asking for 200 from a 50-row file and getting 50 is the complete
        # answer -- so the denominator is the window, not the file.
        eligible = min(requested, max(0, total - start))

        response = {
            "success": True,
            "op": "read_rows",
            "file": pname(file_path),
            "rows": rows,
            "total_available": total,
            "start": start,
            "end": start + len(rows),
            **counted(len(rows), eligible),
            "progress": progress,
        }
        if truncated:
            response["hint"] = f"Results capped at {cap}. Use start/end parameters to page through the data."
        response["context"] = make_context(
            "read_rows",
            f"Read rows {start}–{start + len(rows)} of {total:,} from {pname(file_path)}",
        )
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
