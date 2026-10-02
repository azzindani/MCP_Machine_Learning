"""inspect_dataset and read_column_profile for a table too big to load: the same fields, read in chunks.

`shared/big_table.py` says whether a file is too big and answers the questions a pass at a time. These
functions put those answers in the shape the pandas path gives, so a caller reads one response whichever
way the file was read. The pandas path stays the one for every file that fits; a test holds the two to
the same numbers.
"""

from __future__ import annotations

from typing import Any

from shared.big_table import BigTable


def _dtype(kind: str, nulls: int, rows: int) -> str:
    """The pandas dtype name. A column of whole numbers with a gap is float64 (NaN is a float), so is a
    column with nothing in it, a boolean with a gap is object, and text and dates are `str`, as read."""
    if nulls == rows:
        return "float64"
    if kind == "integer":
        return "float64" if nulls else "int64"
    if kind == "float":
        return "float64"
    if kind == "boolean":
        return "object" if nulls else "bool"
    return "str"


def _shown(table: BigTable) -> dict[str, str]:
    """{name as the pandas path shows it: name in the file}; pandas strips a header's surrounding spaces."""
    return {name.strip(): name for name in table.names()}


def inspect_facts(table: BigTable) -> dict[str, Any]:
    """Everything inspect_dataset reads from the frame: row count and, per column, dtype, nulls and uniques."""
    shown = _shown(table)
    kinds = {c["name"]: c["kind"] for c in table.columns()}
    counts = table.counts()
    rows = table.rows()
    columns = []
    for name, raw in shown.items():
        nulls = rows - counts[raw]["non_null"]
        columns.append(
            {
                "name": name,
                "dtype": _dtype(kinds[raw], nulls, rows),
                "null_count": nulls,
                "null_pct": round(nulls / rows * 100, 2) if rows else 0.0,
                "target_candidate": kinds[raw] == "boolean" or counts[raw]["distinct"] <= 20,
            }
        )
    return {"rows": rows, "columns": columns}


def column_profile(table: BigTable, column: str) -> dict[str, Any] | None:
    """The `profile` of read_column_profile, or None when the file has no such column."""
    raw = _shown(table).get(column)
    if raw is None:
        return None
    kind = next(c["kind"] for c in table.columns() if c["name"] == raw)
    rows = table.rows()
    counted = table.counts([raw])[raw]
    observed, distinct = counted["non_null"], counted["distinct"]
    nulls = rows - observed
    null_pct = round(nulls / rows * 100, 2) if rows else None
    dtype = _dtype(kind, nulls, rows)
    if observed == 0:
        reason = f"all {nulls} rows are null" if nulls else "the file has no data rows"
        return {
            "dtype": dtype,
            "kind": "empty",
            "count": 0,
            "null_count": nulls,
            "null_pct": null_pct,
            "note": f"'{column}' has no observed values ({reason}); nothing was inferred.",
        }
    numeric = kind in ("integer", "float")
    zero_one = numeric and distinct <= 2 and set(table.distinct_values(raw)) <= {0, 1}
    if kind == "boolean" or zero_one:
        true_count = table.true_count(raw)
        top = table.top_values(raw)
        if kind == "boolean":  # DuckDB writes true / false; pandas, and so every other answer here, True / False
            top = {k.capitalize(): v for k, v in top.items()}
        return {
            "dtype": dtype,
            "kind": "boolean",
            "true_count": true_count,
            "false_count": observed - true_count,
            "top_values": top,
            "null_count": nulls,
            "null_pct": null_pct,
            "balance_ratio": round(true_count / max(observed - true_count, 1), 4),
        }
    if numeric:
        s = table.numeric(raw, finite=True)

        def r(value: Any) -> Any:
            return round(float(value), 4) if s["n"] and value is not None else None

        return {
            "dtype": dtype,
            "kind": "numeric",
            "mean": r(s["mean"]),
            "std": r(s["std"]),
            "min": r(s["lo"]),
            "max": r(s["hi"]),
            "median": r(s["median"]),
            "q25": r(s["q1"]),
            "q75": r(s["q3"]),
            "skewness": r(0.0 if s["skew"] is None and s["n"] >= 3 else s["skew"]),  # pandas: 0.0 for a constant
            "top_values": table.top_values(raw),
            "inf_count": int(s["non_finite"]),
            "null_count": nulls,
            "null_pct": null_pct,
        }
    top = table.top_values(raw)
    return {
        "dtype": dtype,
        "kind": "categorical",
        "unique_count": distinct,
        "top_values": top,
        "mode": next(iter(top)) if top else None,
        "null_count": nulls,
        "null_pct": null_pct,
    }
