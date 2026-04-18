"""ml_medium preprocessing tools — run_preprocessing, detect_outliers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.handover import make_context, make_handover

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

    valid, err_msg = _validate_ops(ops)
    if not valid:
        return _error(err_msg, "Check the op array. See run_preprocessing docstring for valid ops.")

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
    for col in columns:
        import pandas as _pd

        series = _pd.to_numeric(df[col], errors="coerce").dropna()
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

    total_outliers = sum(r["outlier_count"] for r in results)
    resp: dict = {
        "success": True,
        "op": "detect_outliers",
        "method": method,
        "columns_checked": len(columns),
        "results": results,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "detect_outliers",
        f"Detected {total_outliers} outlier(s) across {len(columns)} column(s) in {path.name} using {method}",
    )
    resp["handover"] = make_handover(
        "INSPECT",
        ["run_preprocessing", "train_classifier", "train_regressor"],
        {"file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
