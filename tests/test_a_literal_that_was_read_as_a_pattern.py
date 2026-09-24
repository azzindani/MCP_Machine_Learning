"""filter_rows matches a `contains` value literally.

pandas reads `str.contains(value)` as a regular expression unless told not to,
and filter_rows passed the caller's value straight through, though its
operators are eq ne gt lt gte lte contains is_null not_null -- no pattern. So
`contains 'C++'` failed ("bad repetition operator"), `contains '(US)'` also kept
"US office", and `contains 'a.b'` kept "axb". The same fix as DA's filters.
"""

from __future__ import annotations

import pandas as pd
import pytest

from servers.ml_medium._medium_data import filter_rows

TITLES = ["C++ dev", "Java dev", "(US) office", "US office", "a.b", "axb"]
LITERAL = {"C++": ["C++ dev"], "(US)": ["(US) office"], "a.b": ["a.b"]}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    f = tmp_path / "jobs.csv"
    pd.DataFrame({"title": TITLES, "n": range(len(TITLES))}).to_csv(f, index=False)
    return f


@pytest.mark.parametrize("value", list(LITERAL))
@pytest.mark.parametrize("operator", ["contains", "not_contains"])
def test_contains_is_literal(jobs, value, operator):
    out = jobs.parent / "out.csv"
    r = filter_rows(str(jobs), "title", operator, value, output_path=str(out))
    assert r["success"] is True, r
    want = LITERAL[value] if operator == "contains" else [t for t in TITLES if t not in LITERAL[value]]
    assert pd.read_csv(out)["title"].tolist() == want
