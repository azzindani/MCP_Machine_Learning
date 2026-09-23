"""Is this column dates? The rule MCP_Data_Analyst's tools share.

A CSV's dates arrive as text, so a check on the dtype alone never finds one:
search_columns(dtype="datetime") found 0 columns in Ad_Data.csv, whose `Date`
holds 257 ISO dates, while every Data_Analyst tool calls it a date. This is
that repo's rule (shared/column_utils.parse_date_column there): a text column
is dates when at least 90% of a sample has a date's shape -- a four-digit year,
d/m/y digits, or a month name -- and at least 90% of the column parses.
"""

from __future__ import annotations

import re

import pandas as pd

_DATE_SHAPE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?!\d)|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|(?i:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)"
)
_THRESHOLD = 0.9


def date_like(series: pd.Series) -> bool:
    """True when a column holds dates: a datetime dtype, or text that reads as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    values = series.dropna().astype(str)
    if values.empty:
        return False
    sample = values.sample(200, random_state=0) if len(values) > 200 else values
    if sample.str.contains(_DATE_SHAPE).mean() < _THRESHOLD:
        return False
    present = int(series.notna().sum())
    for fmt in ("ISO8601", "mixed"):
        try:
            if pd.to_datetime(sample, errors="coerce", format=fmt).notna().mean() < _THRESHOLD:
                continue
            parsed = pd.to_datetime(series, errors="coerce", format=fmt)
        except ValueError, TypeError, OverflowError:
            continue
        if present and int(parsed.notna().sum()) / present >= _THRESHOLD:
            return True
    return False
