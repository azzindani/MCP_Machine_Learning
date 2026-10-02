"""A table is read by what its own bytes say: a byte-order mark names the encoding, a header names the delimiter.

The 2026-10-02 corpus sweep: a UTF-16 file "decoded" as cp1252 (every character followed by a NUL) and two
semicolon-separated files read as one column holding every row, so every tool reported a one-column table
without saying why.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shared.file_utils import read_csv, sniff_encoding, sniff_separator
from shared.sql_query import run_query


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"id": range(30), "label": ["a", "b", "c"] * 10, "value": [i * 1.5 for i in range(30)]})


def test_a_utf16_file_is_read_by_its_byte_order_mark(tmp_path, frame):
    path = tmp_path / "u16.csv"
    path.write_bytes(b"\xff\xfe" + frame.to_csv(index=False).encode("utf-16-le"))
    assert sniff_encoding(path) == "utf-16"
    got = read_csv(str(path))
    assert list(got.columns) == list(frame.columns) and len(got) == 30


@pytest.mark.parametrize("mark", [";", "\t", "|"])
def test_a_delimiter_the_header_agrees_on_is_used(tmp_path, frame, mark):
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False, sep=mark)
    assert sniff_separator(path) == mark
    assert read_csv(str(path)).shape == frame.shape


def test_a_comma_file_and_a_quoted_comma_stay_commas(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text('name,note\n"a;b",x\n"c;d",y\n')
    assert sniff_separator(path) == "," and read_csv(str(path)).shape == (2, 2)


def test_duckdb_reads_a_utf16_table_too(tmp_path, frame):
    path = tmp_path / "u16.csv"
    path.write_bytes(b"\xff\xfe" + frame.to_csv(index=False).encode("utf-16-le"))
    assert run_query("SELECT count(*) AS n FROM data", tables={"data": path})["rows"] == [{"n": 30}]
