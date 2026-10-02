"""inspect_dataset and read_column_profile on a file too big to load answer from chunks, with pandas' numbers.

Both tools read the whole file into pandas. At the live 1 GB a file of a few hundred MB is a frame of
a gigabyte and the call dies before it answers -- on the very first question anyone asks of a table.
`shared/big_table.py` says when a file is too big (its size against what a call may hold) and answers
the same questions from DuckDB a pass at a time. The pandas path stays the one for every file that fits,
so the two are held to the same numbers here, on one file read both ways (`MCP_BIG_TABLE=always|never`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from servers.ml_basic import engine
from shared import big_table

ROWS = 6_000


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("MCP_BIG_TABLE", raising=False)
    monkeypatch.delenv("MCP_CALL_MEMORY_MB", raising=False)
    return tmp_path


@pytest.fixture
def table(_home):
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "id": np.arange(ROWS),
            "units": rng.integers(0, 40, ROWS),
            "price": rng.normal(20, 4, ROWS).round(2),
            "gappy": np.where(rng.random(ROWS) < 0.1, np.nan, rng.integers(1, 9, ROWS)),
            "region": rng.choice(["north", "south", "east", "west", "centre"], ROWS, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
            "when": (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, ROWS), unit="D")).strftime(
                "%Y-%m-%d"
            ),
            "flag": rng.random(ROWS) < 0.3,
            "churned": rng.integers(0, 2, ROWS),
            "same": 7,
            "empty": np.nan,
            " padded ": rng.integers(0, 5, ROWS),
        }
    )
    frame.loc[::50, "price"] = 0.0
    frame.loc[3, "price"] = 900.0
    frame.loc[7, "price"] = np.inf  # profiled apart from the finite values
    frame["region"] = frame["region"].astype(object)
    frame.loc[::97, "region"] = "NA"  # pandas reads these as empty cells; so must the chunked read
    frame.loc[::89, "units"] = np.nan
    path = _home / "t.csv"
    frame.to_csv(path, index=False)
    return path


def both(monkeypatch, call, *args, **kwargs):
    monkeypatch.setenv("MCP_BIG_TABLE", "never")
    loaded = call(*args, **kwargs)
    monkeypatch.setenv("MCP_BIG_TABLE", "always")
    chunked = call(*args, **kwargs)
    return loaded, chunked


class TestInspectDatasetIsTheSameAnswer:
    def test_every_field(self, table, monkeypatch):
        loaded, chunked = both(monkeypatch, engine.inspect_dataset, str(table))
        assert loaded["success"] is True and chunked["success"] is True, chunked.get("error")
        for field in (
            "row_count",
            "column_count",
            "columns",
            "target_candidates",
            "target_candidates_total",
            "target_candidates_truncated",
            "truncated",
        ):
            assert chunked[field] == loaded[field], field
        assert "chunked" not in loaded
        assert chunked["chunked"]["engine"] == "duckdb" and "MCP_BIG_TABLE" in chunked["chunked"]["why"]

    def test_a_cell_pandas_reads_as_empty_is_empty_here_too(self, table, monkeypatch):
        _, chunked = both(monkeypatch, engine.inspect_dataset, str(table))
        region = next(c for c in chunked["columns"] if c["name"] == "region")
        assert region["null_count"] == len(range(0, ROWS, 97))

    def test_the_file_is_never_loaded_whole(self, table, monkeypatch):
        monkeypatch.setenv("MCP_BIG_TABLE", "always")

        def refuse(*_a, **_k):
            raise AssertionError("loaded the whole file")

        monkeypatch.setattr(engine, "_read_csv", refuse)
        assert engine.inspect_dataset(str(table))["success"] is True
        assert engine.read_column_profile(str(table), "price")["success"] is True


class TestAColumnProfileIsTheSameAnswer:
    @pytest.mark.parametrize(
        "column", ["price", "units", "gappy", "id", "same", "padded", "region", "when", "flag", "churned", "empty"]
    )
    def test_each_kind_of_column(self, table, monkeypatch, column):
        loaded, chunked = both(monkeypatch, engine.read_column_profile, str(table), column)
        assert loaded["success"] is True and chunked["success"] is True, chunked.get("error")
        want, got = loaded["profile"], chunked["profile"]
        assert set(got) == set(want)
        for key, expected in want.items():
            if key == "top_values":
                assert sorted(got[key].values()) == sorted(expected.values()), key
                if len(set(expected.values())) == len(expected):  # no ties: the order and the names are settled
                    assert got[key] == expected, key
                continue
            if isinstance(expected, float):
                expected = pytest.approx(expected, abs=1e-3)
            assert got[key] == expected, key
        assert chunked["chunked"]["engine"] == "duckdb" and "chunked" not in loaded

    def test_infinities_are_counted_apart_and_left_out_of_the_statistics(self, table, monkeypatch):
        _, chunked = both(monkeypatch, engine.read_column_profile, str(table), "price")
        profile = chunked["profile"]
        assert profile["inf_count"] == 1 and profile["max"] == 900.0

    def test_a_column_that_is_not_there_is_named_with_the_ones_that_are(self, table, monkeypatch):
        loaded, chunked = both(monkeypatch, engine.read_column_profile, str(table), "nope")
        assert chunked["success"] is False and chunked["error"] == loaded["error"]


class TestReadRowsIsTheSameAnswer:
    @pytest.mark.parametrize("window", [(0, 10), (5, 15), (5000, 6000), (5995, 6010), (100, 100)])
    def test_the_rows_asked_for(self, table, monkeypatch, window):
        loaded, chunked = both(monkeypatch, engine.read_rows, str(table), *window)
        assert loaded["success"] is True and chunked["success"] is True, chunked.get("error")
        for field in ("total_available", "start", "end", "truncated"):
            assert chunked[field] == loaded[field], field
        assert chunked["rows"] == loaded["rows"]
        assert chunked["chunked"]["engine"] == "duckdb" and "chunked" not in loaded

    def test_an_infinity_is_text_and_a_gap_is_null(self, table, monkeypatch):
        _, chunked = both(monkeypatch, engine.read_rows, str(table), 6, 9)
        assert chunked["rows"][1]["price"] == "inf"  # row 7
        assert any(row["gappy"] is None for row in both(monkeypatch, engine.read_rows, str(table), 0, 60)[1]["rows"])

    def test_a_reversed_range_is_refused_as_before(self, table, monkeypatch):
        loaded, chunked = both(monkeypatch, engine.read_rows, str(table), 10, 5)
        assert chunked["success"] is False and chunked["error"] == loaded["error"]


class TestAConstantColumnHasNoSkew:
    def test_whatever_noise_the_engine_reports(self, table, monkeypatch):
        """DuckDB gave NaN for a constant column on one CPU and -3.4e8 (rounding noise) on another; pandas says 0.0."""
        chunked = big_table.BigTable(table)
        real = chunked._ask

        def noisy(sql, limit=1000):
            rows = real(sql, limit)
            if "skewness" in sql:
                rows[0]["skew"] = -346184879.9163
            return rows

        monkeypatch.setattr(chunked, "_ask", noisy)
        assert chunked.numeric("same", finite=True)["skew"] == 0.0
        assert chunked.numeric("price", finite=True)["skew"] == -346184879.9163, "only a constant column is overruled"


class TestWhenItIsChunked:
    def test_a_file_that_fits_is_loaded_as_it_always_was(self, table, monkeypatch):
        monkeypatch.setenv("MCP_CALL_MEMORY_MB", "4096")
        assert big_table.reason(table) == ""
        assert "chunked" not in engine.inspect_dataset(str(table))

    def test_a_file_that_does_not_fit_in_what_a_call_may_hold_is_chunked(self, table, monkeypatch):
        monkeypatch.setenv("MCP_CALL_MEMORY_MB", "64")
        monkeypatch.setattr(big_table, "LOAD_FACTORS", {".csv": 6000})  # this file stands in for a big one
        why = big_table.reason(table)
        assert "needs about" in why and "may hold 64 MB" in why
        result = engine.inspect_dataset(str(table))
        assert result["success"] is True and result["chunked"]["why"] == why

    def test_with_no_limit_known_nothing_says_it_will_not_fit(self, table, monkeypatch):
        monkeypatch.setattr(big_table, "memory_budget_mb", lambda default=1024: 0 if default == 0 else default)
        assert big_table.reason(table) == ""

    def test_never_means_never(self, table, monkeypatch):
        monkeypatch.setenv("MCP_CALL_MEMORY_MB", "64")
        monkeypatch.setattr(big_table, "LOAD_FACTORS", {".csv": 6000})
        monkeypatch.setenv("MCP_BIG_TABLE", "never")
        assert big_table.reason(table) == ""


class TestTheModuleIsTheOneTheDataServerHas:
    def test_big_table(self):
        sibling = Path("/root/MCP_Data_Analyst/shared/big_table.py")
        if not sibling.is_file():
            pytest.skip("the sibling repo is not beside this one (CI)")
        assert sibling.read_bytes() == Path(big_table.__file__).read_bytes()
