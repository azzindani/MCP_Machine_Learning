"""run_preprocessing's hint sent the caller to documentation that does not exist.

A coverage sweep passed `[{"op": "dropna"}]` and got back:

    error: Unknown op: 'dropna'. Allowed: add_date_parts, bin_numeric, ...
    hint:  Check the op array. See run_preprocessing docstring for valid ops.

The error is good -- it names every valid op. The hint is not: the
run_preprocessing docstring is the 80-character tool description, "Apply
preprocessing ops to dataset. Snapshot before write.", and it has never listed
an op. The one place the hint pointed was the one place the answer was not.

A hint has to add something the error does not, so it now gives the shape of an
op dict (the sweep's other two failures were a missing 'strategy' key and a
missing 'column' key, neither of which the op-name list helps with) and names
check_data_quality, whose recommendations quote the exact op to use for each
problem it reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from servers.ml_medium.engine import check_data_quality, run_preprocessing

BAD_OPS = [
    pytest.param([{"op": "dropna"}], id="unknown-op"),
    pytest.param([{"op": "fill_nulls"}], id="missing-column"),
    pytest.param([{"op": "fill_nulls", "column": "score", "strategy": "guess"}], id="bad-strategy"),
    pytest.param(["drop_duplicates"], id="not-a-dict"),
]


class TestTheHintDoesNotPointAtTheDocstring:
    @pytest.mark.parametrize("ops", BAD_OPS)
    def test_it_fails(self, classification_simple: Path, ops: list):
        assert run_preprocessing(str(classification_simple), ops)["success"] is False

    @pytest.mark.parametrize("ops", BAD_OPS)
    def test_the_hint_no_longer_names_a_docstring(self, classification_simple: Path, ops: list):
        hint = run_preprocessing(str(classification_simple), ops)["hint"]
        assert "docstring" not in hint.lower(), hint

    @pytest.mark.parametrize("ops", BAD_OPS)
    def test_the_hint_shows_the_shape_of_an_op(self, classification_simple: Path, ops: list):
        """Two of the four failures are a missing key, which an op-name list
        cannot fix -- the caller needs to see a whole op."""
        hint = run_preprocessing(str(classification_simple), ops)["hint"]
        assert "'op'" in hint and "{" in hint, hint

    @pytest.mark.parametrize("ops", BAD_OPS)
    def test_the_hint_names_a_tool_that_exists(self, classification_simple: Path, ops: list):
        hint = run_preprocessing(str(classification_simple), ops)["hint"]
        assert "check_data_quality" in hint, hint

    @pytest.mark.parametrize("ops", BAD_OPS)
    def test_the_error_still_carries_the_detail(self, classification_simple: Path, ops: list):
        r = run_preprocessing(str(classification_simple), ops)
        assert r["error"].strip() not in ("", "''", '""')


class TestTheToolTheHintNamesReallyQuotesOps:
    """The hint is only true if check_data_quality's recommendations name ops."""

    def test_its_recommendations_name_run_preprocessing_ops(self, classification_messy: Path):
        r = check_data_quality(str(classification_messy))
        assert r["success"] is True, r.get("error")
        texts = " ".join(str(a.get("recommendation", "")) for a in r.get("alerts", []))
        assert any(op in texts for op in ("drop_duplicates", "fill_nulls", "log_transform", "drop_column")), texts


class TestValidOpsStillRun:
    def test_a_good_op_list_is_accepted(self, classification_messy: Path, tmp_path: Path):
        r = run_preprocessing(
            str(classification_messy),
            [{"op": "drop_duplicates"}],
            output_path=str(tmp_path / "out.csv"),
        )
        assert r["success"] is True, r.get("error")
