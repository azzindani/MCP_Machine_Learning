"""Accept the name a caller would reasonably have guessed for an argument.

A census of every `@mcp.tool()` signature in this repo, grouped by meaning:

    a list of columns    feature_columns  5    columns  1

The one outlier is ml_medium.detect_outliers, which sits in the same module as
ml_medium.anomaly_detection and does conceptually the same job. Against the
same file and the same two columns:

    anomaly_detection(file_path=..., feature_columns=["spends","clicks"])
      -> success: true, 842 anomalies

    detect_outliers(file_path=..., feature_columns=["spends","clicks"])
      -> 2 validation errors for call[detect_outliers]
         columns          Missing required argument
         feature_columns  Unexpected keyword argument

pydantic refuses that before any engine code runs, so the tool cannot suggest
the name it wanted, and the live schemas carry no property descriptions -- the
parameter name is the whole contract. The Office repo lost three sweep phase
attempts to exactly this shape.

Renaming would break existing callers, so the outlier accepts both spellings
and resolves here, canonical being what the majority of its siblings use.
"""

from __future__ import annotations


def pick_list(op: str, field: str, primary: list[str], alias: list[str]) -> tuple[list[str], str]:
    """Resolve a list argument given under either spelling.

    Returns (value, note). `note` is empty unless the alias supplied the value.
    An empty result means neither was given; the caller turns that into its own
    error dict rather than raising.
    """
    if primary:
        return list(primary), ""
    if alias:
        return list(alias), f"Read {field} from the alias spelling; {field}= is the documented one"
    return [], ""


def missing_list(op: str, field: str, alias: str) -> dict:
    """The error dict for a list argument given under neither spelling."""
    return {
        "success": False,
        "op": op,
        "error": f"{op} needs a {field} list",
        "hint": f"Pass {field}=['col1','col2']. The spelling {alias}= is also accepted.",
        "progress": [],
        "token_estimate": 20,
    }
