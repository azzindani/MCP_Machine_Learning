"""ml_medium preprocessing tools — run_preprocessing, detect_outliers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.file_utils import embed_content
from shared.handover import make_context, make_handover
from shared.small_sample import MIN_N_IQR, min_n_for_zscore, rounded

from ._medium_helpers import (
    _apply_op,
    _error,
    _read_csv,
    _validate_ops,
    append_receipt,
    info,
    ok,
    resolve_path,
    snapshot,
    warn,
)


def run_preprocessing(
    file_path: str,
    ops: list[dict],
    output_path: str = "",
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Apply preprocessing pipeline ops to dataset. Snapshot before write."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.stat().st_size == 0:
        return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    valid, ops, err_msg = _validate_ops(ops)
    if not valid:
        # The old hint sent the caller to "the run_preprocessing docstring",
        # which is the 80-character tool description -- "Apply preprocessing ops
        # to dataset. Snapshot before write." It has never listed an op, so the
        # one place the hint pointed was the one place the answer was not. The
        # error above already names the valid ops; a hint has to add something
        # the error does not, so it gives the shape and names the tool whose
        # recommendations quote the op to use for each problem it finds.
        return _error(
            err_msg,
            "Each op is a dict like {'op': 'drop_duplicates'} or "
            "{'op': 'fill_nulls', 'column': 'link_clicks', 'strategy': 'median'}. "
            "check_data_quality names the op to use for each issue it reports.",
        )

    progress.append(info("Validated ops", f"{len(ops)} ops"))

    try:
        df = _read_csv(str(path))
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
    for i, op in enumerate(ops):
        # A handler that raises used to take the whole tool with it: the four
        # ops missing a `column` guard raised KeyError('column') past the return
        # contract, so the caller got a traceback instead of a dict. The guard
        # in _validate_ops closes those four; this closes the fifteenth op
        # nobody has written yet.
        try:
            df, summary = _apply_op(df, op)
        except Exception as exc:
            progress.append(warn(f"Op {i} ({op.get('op', '?')}) failed", str(exc)))
            return _error(
                f"Op {i} ({op.get('op', '?')}) failed: {exc}",
                f"Nothing was written. Fix that op and retry; {path.name} is unchanged.",
            )
        ops_summary.append(summary)
        # A summary carrying `error` is an op that did not do what it was asked.
        # It was still counted in `applied`, still announced as "Applied
        # drop_column", and still left the run reporting success -- so asking to
        # drop a column that does not exist wrote a full output file and called
        # it preprocessed.
        if summary.get("error"):
            progress.append(warn(f"Op {i} ({op.get('op', '?')}) did nothing", str(summary["error"])))
            return _error(
                f"Op {i} ({op.get('op', '?')}): {summary['error']}",
                f"Available columns: {', '.join(map(str, df.columns))}. Nothing was written.",
            )
        progress.append(ok(f"Applied {op['op']}", str(summary.get("filled", summary.get("removed", "")))))

    if output_path:
        out_path = Path(output_path)
    else:
        out_path = path.parent / f"{path.stem}_preprocessed{path.suffix}"
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
    resp["context"] = make_context(
        "run_preprocessing",
        f"Applied {len(ops)} preprocessing op(s) to {path.name}, saved to {out_path_resolved.name}",
        [{"type": "csv", "path": str(out_path_resolved), "role": "preprocessed_dataset"}],
    )
    resp["handover"] = make_handover(
        "CLEAN",
        ["train_classifier", "train_regressor", "train_with_cv", "detect_outliers"],
        {"file_path": str(out_path_resolved)},
    )
    embed_content(resp, out_path_resolved, return_content)
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


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
    if path.stat().st_size == 0:
        return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

    if method not in ("iqr", "std"):
        return _error(f"Unknown method: '{method}'.", "Use 'iqr' or 'std'.")

    try:
        df = _read_csv(str(path))
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        return _error(
            f"Columns not found: {', '.join(missing[:5])}",
            "Use inspect_dataset() to list valid column names.",
        )

    results: list[dict] = []
    undetermined: list[str] = []
    # Both minimums are properties of the arithmetic rather than of the data --
    # see shared/small_sample.py. Below them the bounds land outside every value
    # in the sample, so "0 outliers" was settled before the file was read.
    min_n = MIN_N_IQR if method == "iqr" else min_n_for_zscore(3.0)
    for col in columns:
        import pandas as _pd

        series = _pd.to_numeric(df[col], errors="coerce").dropna()
        n = int(len(series))
        if n < min_n:
            reason = (
                f"undetermined at n={n}: the 1.5*IQR fence cannot fall inside a sample smaller than {min_n}"
                if method == "iqr"
                else (
                    f"undetermined at n={n}: the largest z-score attainable by any of n points is "
                    f"(n-1)/sqrt(n), which first exceeds 3 at n={min_n}"
                )
            )
            results.append(
                {
                    "column": col,
                    "method": method,
                    "n": n,
                    "lower_bound": None,
                    "upper_bound": None,
                    "outlier_count": None,
                    "sample_outliers": [],
                    "status": reason,
                }
            )
            undetermined.append(col)
            progress.append(warn(f"Cannot scan {col}", reason))
            continue

        if method == "iqr":
            q1 = series.quantile(th1)
            q3 = series.quantile(th3)
            iqr_val = q3 - q1
            lower = float(q1 - 1.5 * iqr_val)
            upper = float(q3 + 1.5 * iqr_val)
            zero_spread = float(iqr_val) == 0.0
        else:  # std
            mean, std = series.mean(), series.std()
            lower = float(mean - 3 * std)
            upper = float(mean + 3 * std)
            zero_spread = float(std) == 0.0

        mask = (series < lower) | (series > upper)
        outlier_vals = series[mask].head(5).tolist()
        entry: dict = {
            "column": col,
            "method": method,
            "n": n,
            "lower_bound": rounded(lower, 6),
            "upper_bound": rounded(upper, 6),
            "outlier_count": int(mask.sum()),
            "sample_outliers": outlier_vals,
        }
        if zero_spread:
            # Enough rows, no spread: zero is a real answer here, and the bounds
            # still sit on the data rather than around it.
            entry["status"] = "zero spread: every value is identical, so the bounds have no width"
        results.append(entry)
        progress.append(ok(f"Analyzed {col}", f"{int(mask.sum())} outliers"))

    total_outliers = sum(r["outlier_count"] or 0 for r in results)
    resp: dict = {
        "success": True,
        "op": "detect_outliers",
        "method": method,
        "columns_checked": len(columns),
        "columns_undetermined": undetermined,
        "results": results,
        "progress": progress,
        "token_estimate": 0,
    }
    if undetermined:
        resp["hint"] = (
            f"{len(undetermined)} of {len(columns)} column(s) had too few rows for {method} to flag anything, "
            "so a count of 0 there is not a finding. Each carries the n it had."
        )
    summary = (
        f"Detected {total_outliers} outlier(s) across {len(columns) - len(undetermined)} scannable "
        f"column(s) in {path.name} using {method}; {len(undetermined)} undetermined"
        if undetermined
        else f"Detected {total_outliers} outlier(s) across {len(columns)} column(s) in {path.name} using {method}"
    )
    resp["context"] = make_context("detect_outliers", summary)
    resp["handover"] = make_handover(
        "INSPECT",
        ["run_preprocessing", "train_classifier", "train_regressor"],
        {"file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
