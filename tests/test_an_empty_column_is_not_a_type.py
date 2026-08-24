"""A column of nothing came back typed, scored and recommended.

Round 12 gave every tool a header-only CSV. Two ML tools answered it with
confident numbers.

**read_column_profile typed the emptiness.** The boolean branch asks whether the
observed values are a subset of {0, 1, True, False}:

    series.nunique() <= 2 and set(series.dropna().unique()) <= {0, 1, True, False}

The empty set is a subset of everything, and 0 <= 2, so a column holding nothing
always took that branch. `spends` -- plainly numeric in the reference dataset --
came back kind="boolean", true_count 0, false_count 0, balance_ratio 0.0, and
null_pct 0.0, which reads as a *healthy* column rather than an absent one. The
tool's own progress line said "0 rows" while it did this, and the handover then
recommended train_classifier on the file.

**check_data_quality scored it 30.0/100.** Its empty-file guard tests
`st.st_size == 0`, which a header row defeats. Every column then failed the
constant-column check, whose condition is `<= 1` while its message says "1" --
so each column produced a HIGH alert claiming a unique-value count it does not
have (an all-null column has 0), a "drop it, no information" recommendation, and
a 15-point penalty. An empty file diagnosed as a badly built one.

Emptiness is not a type and not a data defect. Both tools say so now, and the
constant-column alert reports the number it actually found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_basic import engine as basic  # noqa: E402
from servers.ml_medium import engine as medium  # noqa: E402


@pytest.fixture
def header_only(tmp_path) -> str:
    p = tmp_path / "empty.csv"
    p.write_text("Date,product,spends,clicks\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def all_null_column(tmp_path) -> str:
    """Rows exist; one column is null in every one of them."""
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "spends": [None, None, None, None]})
    p = tmp_path / "nulls.csv"
    frame.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def real(tmp_path) -> str:
    frame = pd.DataFrame({"spends": [1.5, 2.5, 3.5, 4.5], "flag": [0, 1, 0, 1], "name": list("abcd")})
    p = tmp_path / "real.csv"
    frame.to_csv(p, index=False)
    return str(p)


class TestAnEmptyColumnIsNotBoolean:
    def test_a_header_only_file_does_not_yield_a_type(self, header_only):
        r = basic.read_column_profile(header_only, "spends")
        assert r["success"] is True, r.get("error")
        assert r["profile"]["kind"] == "empty", r["profile"]

    def test_no_balance_ratio_is_invented(self, header_only):
        r = basic.read_column_profile(header_only, "spends")
        assert "balance_ratio" not in r["profile"], r["profile"]

    def test_no_true_false_counts_are_invented(self, header_only):
        r = basic.read_column_profile(header_only, "spends")
        assert "true_count" not in r["profile"] and "false_count" not in r["profile"]

    def test_null_pct_is_not_reported_as_zero(self, header_only):
        # 0.0 reads as "no nulls", i.e. a healthy column.
        r = basic.read_column_profile(header_only, "spends")
        assert r["profile"]["null_pct"] is None, r["profile"]

    def test_it_says_why_it_is_empty(self, header_only):
        r = basic.read_column_profile(header_only, "spends")
        assert "no data rows" in r["profile"]["note"], r["profile"]["note"]

    def test_the_hint_points_somewhere_useful(self, header_only):
        r = basic.read_column_profile(header_only, "spends")
        assert "inspect_dataset" in r.get("hint", "")


class TestAnAllNullColumnInARealFile:
    def test_it_is_also_not_boolean(self, all_null_column):
        # Same bug, without an empty file: nunique() is 0 here too.
        r = basic.read_column_profile(all_null_column, "spends")
        assert r["profile"]["kind"] == "empty", r["profile"]

    def test_the_note_names_the_nulls_not_the_file(self, all_null_column):
        r = basic.read_column_profile(all_null_column, "spends")
        assert "null" in r["profile"]["note"], r["profile"]["note"]

    def test_the_null_count_is_real(self, all_null_column):
        r = basic.read_column_profile(all_null_column, "spends")
        assert r["profile"]["null_count"] == 4, r["profile"]


class TestRealColumnsAreUnaffected:
    def test_numeric_still_profiles_as_numeric(self, real):
        r = basic.read_column_profile(real, "spends")
        assert r["profile"]["kind"] == "numeric"
        assert r["profile"]["mean"] == pytest.approx(3.0)

    def test_a_genuine_boolean_still_profiles_as_boolean(self, real):
        r = basic.read_column_profile(real, "flag")
        assert r["profile"]["kind"] == "boolean"
        assert r["profile"]["true_count"] == 2
        assert r["profile"]["balance_ratio"] == pytest.approx(1.0)

    def test_categorical_still_profiles_as_categorical(self, real):
        r = basic.read_column_profile(real, "name")
        assert r["profile"]["kind"] == "categorical"
        assert r["profile"]["unique_count"] == 4

    def test_null_pct_is_still_a_number_when_there_are_rows(self, real):
        r = basic.read_column_profile(real, "spends")
        assert r["profile"]["null_pct"] == 0.0


class TestQualityIsNotScoredOnNothing:
    def test_a_header_only_file_is_refused(self, header_only):
        r = medium.check_data_quality(header_only)
        assert r["success"] is False, f"scored an empty file: {r.get('quality_score')}"

    def test_no_score_is_returned(self, header_only):
        r = medium.check_data_quality(header_only)
        assert "quality_score" not in r, r

    def test_no_column_is_accused_of_being_constant(self, header_only):
        r = medium.check_data_quality(header_only)
        assert not r.get("alerts"), r.get("alerts")

    def test_the_error_names_the_row_count(self, header_only):
        r = medium.check_data_quality(header_only)
        assert "no data rows" in r["error"], r["error"]

    def test_the_hint_points_at_the_producer_not_the_data(self, header_only):
        r = medium.check_data_quality(header_only)
        assert "inspect_dataset" in r["hint"], r["hint"]

    def test_the_column_count_is_still_reported(self, header_only):
        r = medium.check_data_quality(header_only)
        assert r["columns"] == 4 and r["rows"] == 0, r


class TestTheConstantAlertReportsWhatItFound:
    def test_an_all_null_column_is_not_called_constant(self, all_null_column):
        r = medium.check_data_quality(all_null_column)
        assert r["success"] is True, r.get("error")
        kinds = {a["type"] for a in r["alerts"] if a.get("column") == "spends"}
        assert "all_null_column" in kinds, r["alerts"]
        assert "constant_column" not in kinds, r["alerts"]

    def test_it_does_not_claim_one_unique_value(self, all_null_column):
        r = medium.check_data_quality(all_null_column)
        msgs = [a["message"] for a in r["alerts"] if a.get("column") == "spends"]
        assert msgs and "1 unique value" not in msgs[0], msgs

    def test_the_advice_distinguishes_missing_from_constant(self, all_null_column):
        r = medium.check_data_quality(all_null_column)
        rec = [a["recommendation"] for a in r["alerts"] if a.get("column") == "spends"][0]
        assert "missing data" in rec, rec

    def test_a_genuinely_constant_column_still_says_one(self, tmp_path):
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "same": ["x", "x", "x", "x"]})
        p = tmp_path / "const.csv"
        frame.to_csv(p, index=False)
        r = medium.check_data_quality(str(p))
        msgs = [a["message"] for a in r["alerts"] if a.get("column") == "same"]
        assert msgs and "only 1 unique value" in msgs[0], msgs


class TestTheResponseContract:
    def test_profile_carries_a_token_estimate(self, header_only):
        r = basic.read_column_profile(header_only, "spends")
        assert isinstance(r["token_estimate"], int) and r["token_estimate"] > 0

    def test_quality_refusal_carries_a_token_estimate(self, header_only):
        r = medium.check_data_quality(header_only)
        assert isinstance(r["token_estimate"], int)

    def test_both_name_their_op(self, header_only):
        assert basic.read_column_profile(header_only, "spends")["op"] == "read_column_profile"
        assert medium.check_data_quality(header_only)["op"] == "check_data_quality"
