"""run_preprocessing tells the caller when an op did not do its job.

Three defects, all in the gap between "the op ran" and "the op worked".

1. Four of the fourteen allowed ops -- add_date_parts, bin_numeric,
   clip_column, log_transform -- read op["column"] with no guard, because the
   guard was a hand-written set naming the five that existed when it was
   written. Called without a column they raised KeyError('column') straight out
   of the tool, past the return-value contract: the caller got a traceback,
   not a dict.

2. Nothing wrapped the apply loop, so op #15 would do it again.

3. _apply_op reports a failure it can handle by putting `error` in its summary
   dict and returning the frame untouched. run_preprocessing never looked. So
   asking it to drop a column that does not exist gave:

       success: true, applied: 1, progress: "✔ Applied drop_column"
       ops: [{"op": "drop_column", "error": "column not found"}]

   plus a full output file written as though the dataset had been preprocessed.
   The only trace was one level down in a list nobody reads when success is
   true. That is the same shape as run_cleaning_pipeline's ops_with_no_effect
   in the Data_Analyst repo, fixed there and not here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from servers.ml_medium._medium_helpers import ALLOWED_OPS, OPS_REQUIRING_COLUMN
from servers.ml_medium._medium_preprocess import run_preprocessing


@pytest.fixture
def csv(tmp_path):
    f = tmp_path / "in.csv"
    pd.DataFrame(
        {
            "region": [f"W{i % 2}" for i in range(1, 9)],
            "spend": [i * 10 for i in range(1, 9)],
            "clicks": [i * 3 for i in range(1, 9)],
        }
    ).to_csv(f, index=False)
    return f


@pytest.mark.parametrize("op_name", sorted(ALLOWED_OPS))
def test_no_op_can_raise_out_of_the_tool(tmp_path, csv, op_name):
    """Called with no arguments at all, every op returns a dict.

    This is the guard that makes the fix durable: it enumerates ALLOWED_OPS
    rather than a list someone maintains, so a fifteenth op is covered the day
    it is added.
    """
    result = run_preprocessing(str(csv), [{"op": op_name}], output_path=str(tmp_path / "out.csv"))
    assert isinstance(result, dict)
    assert "success" in result
    if op_name in OPS_REQUIRING_COLUMN:
        assert result["success"] is False
        assert "column" in result["error"]
        assert op_name in result["error"]


def test_every_op_that_reads_a_column_declares_it():
    """OPS_REQUIRING_COLUMN is what the guard consults; an op missing from it
    reaches the handler unguarded, which is exactly how the first four got
    through."""
    assert OPS_REQUIRING_COLUMN <= ALLOWED_OPS
    for op_name in OPS_REQUIRING_COLUMN:
        assert op_name in ALLOWED_OPS


def test_dropping_a_column_that_is_not_there_is_not_a_success(tmp_path, csv):
    out = tmp_path / "out.csv"
    r = run_preprocessing(str(csv), [{"op": "drop_column", "column": "does_not_exist"}], output_path=str(out))
    assert r["success"] is False
    assert "drop_column" in r["error"]
    assert "column not found" in r["error"]
    # The available names belong in the hint, since that is what fixes the call.
    assert "spend" in r["hint"]
    assert not out.exists()


def test_a_later_op_failing_does_not_leave_a_half_written_file(tmp_path, csv):
    out = tmp_path / "out.csv"
    r = run_preprocessing(
        str(csv),
        [
            {"op": "drop_duplicates"},
            {"op": "label_encode", "column": "region"},
            {"op": "drop_column", "column": "ghost"},
        ],
        output_path=str(out),
    )
    assert r["success"] is False
    assert "Op 2" in r["error"]
    assert not out.exists()


@pytest.mark.parametrize(
    "ops",
    [
        [{"op": "drop_duplicates"}],
        [{"op": "fill_nulls", "column": "spend", "strategy": "median"}],
        [{"op": "log_transform", "column": "spend"}],
        [{"op": "bin_numeric", "column": "spend", "bins": 3}],
        [{"op": "clip_column", "column": "spend", "min": 20, "max": 60}],
        [{"op": "drop_duplicates"}, {"op": "label_encode", "column": "region"}, {"op": "scale", "columns": ["spend"]}],
    ],
)
def test_a_pipeline_that_works_still_works(tmp_path, csv, ops):
    out = tmp_path / "out.csv"
    r = run_preprocessing(str(csv), ops, output_path=str(out))
    assert r["success"] is True, r.get("error")
    assert r["applied"] == len(ops)
    assert out.exists()
    assert len(pd.read_csv(out)) > 0
