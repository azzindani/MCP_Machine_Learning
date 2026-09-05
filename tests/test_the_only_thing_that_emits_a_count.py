"""Every response that reports a count reports it the same way.

The sibling of this file in MCP_Data_Analyst carries the full story. The short
version: `shared/counts.py` derives `truncated` from `returned` and `total`
rather than accepting it, because the original defect was one cap doing the
cutting while a different limit set the flag -- twenty of twenty-five groups,
reported as all of them.

The static rule below is deliberately repeated per repo rather than shared. It
has to read *this* repo's source on *this* repo's CI runner, where the sibling
repos do not exist; a single copy somewhere else would skip everywhere. It is
one regex, not a table, so the usual objection to a second copy does not apply.

Wiring it here turned up two defects of its own:

* `inspect_dataset` cut `columns` with `get_max_columns()` and
  `target_candidates` with `get_max_results()`, then reported a single
  `truncated` derived from the first. A caller whose columns all fitted read
  `truncated: false` and could still be missing target candidates -- which is
  the field they called this tool to read.
* `read_rows` measured truncation against the caller's requested range, so
  asking for 200 rows of a 50-row file came back `truncated: true`. That is the
  complete answer; the file simply ends. The denominator is now the window
  bounded by where the data stops, so running out of rows is not reported as
  something being withheld.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVERS = ROOT / "servers"
SHARED = ROOT / "shared"

_HAND_WRITTEN = re.compile(r'"truncated"\s*:')


def _py_files() -> list[Path]:
    files = [p for p in SERVERS.rglob("*.py") if "__pycache__" not in p.parts]
    files += [p for p in SHARED.rglob("*.py") if "__pycache__" not in p.parts]
    return [p for p in files if p.name != "counts.py"]


def test_no_module_writes_the_truncated_key_by_hand():
    offenders: list[str] = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue  # several modules quote the banned string while explaining it
            if _HAND_WRITTEN.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these write `truncated` by hand instead of calling counted():\n  "
        + "\n  ".join(offenders)
        + "\n\ncounted(returned, total) derives it, so the flag cannot disagree "
        "with the numbers printed beside it."
    )


@pytest.fixture
def unconstrained(monkeypatch):
    """A test about a limit must pin the mode, never inherit it from the runner."""
    monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)


@pytest.fixture
def short_csv(tmp_path):
    p = tmp_path / "short.csv"
    pd.DataFrame({"a": range(50), "b": range(50)}).to_csv(p, index=False)
    return p


@pytest.fixture
def wide_csv(tmp_path):
    """More columns than get_max_columns(), each with few enough uniques to be a target."""
    p = tmp_path / "wide.csv"
    pd.DataFrame({f"c_{i:03d}": [0, 1, 0, 1] for i in range(200)}).to_csv(p, index=False)
    return p


def test_running_out_of_rows_is_not_truncation(unconstrained, short_csv):
    """Asking for 200 rows of a 50-row file and getting 50 is the whole answer."""
    from servers.ml_basic.engine import read_rows

    r = read_rows(str(short_csv), start=0, end=200)
    assert r["success"] is True
    assert r["returned"] == 50
    assert r["total"] == 50, "the window is bounded by where the file ends"
    assert r["truncated"] is False
    assert r["total_available"] == 50


def test_a_window_the_cap_really_did_cut_says_so(short_csv, monkeypatch):
    from servers.ml_basic.engine import read_rows

    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
    r = read_rows(str(short_csv), start=0, end=50)
    assert r["returned"] < r["total"]
    assert r["truncated"] is True


def test_target_candidates_report_their_own_cut(unconstrained, wide_csv):
    """Two lists cut by two caps cannot share one flag."""
    from servers.ml_basic.engine import inspect_dataset

    r = inspect_dataset(str(wide_csv))
    assert r["success"] is True
    assert r["column_count"] == 200
    assert r["total"] == 200, "the total is every column, not the ones that fitted"
    assert r["returned"] == len(r["columns"])
    # The field that used to be silently cut now carries its own denominator.
    assert r["target_candidates_total"] == 200
    assert r["target_candidates_truncated"] == (len(r["target_candidates"]) < 200)


def test_the_columns_total_is_not_the_page_size(unconstrained, wide_csv):
    from servers.ml_basic.engine import inspect_dataset

    r = inspect_dataset(str(wide_csv))
    assert r["returned"] < r["total"], "200 columns cannot all fit under the cap"
    assert r["truncated"] is True
