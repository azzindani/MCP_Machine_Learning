"""One row is not a population, and these tools stopped pretending otherwise.

Handed a valid single-row CSV, four tools answered with confidence about things
the row count had already settled:

  detect_outliers      lower_bound == upper_bound == the value, "0 outliers"
  anomaly_detection    IsolationForest fitted on the one row it then scores,
                       so "not an anomaly" is a statement about the model's
                       only training example
  check_data_quality   30/100, every column flagged constant and marked for
                       dropping -- the row-count fix from the empty-file round
                       caught the header-only file and stopped one row short
  batch_predict        min, max and mean of a single prediction, under the key
                       prediction_distribution

Two of the thresholds are arithmetic, not judgement, and the tests below check
them against their own definitions rather than against a remembered number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_medium._medium_data import anomaly_detection, check_data_quality  # noqa: E402
from servers.ml_medium._medium_preprocess import detect_outliers  # noqa: E402
from shared.small_sample import MIN_N_IQR, min_n_for_zscore  # noqa: E402


def _csv(tmp_path, n_rows: int, name: str = "rows") -> Path:
    f = tmp_path / f"{name}_{n_rows}.csv"
    rows = "\n".join(f"r{i},{i * 10},{i * 3}" for i in range(1, n_rows + 1))
    f.write_text(f"label,spend,clicks\n{rows}\n")
    return f


def test_the_thresholds_match_their_own_definitions():
    for threshold in (1.0, 2.0, 3.0, 4.0):
        n = min_n_for_zscore(threshold)
        assert (n - 1) / n**0.5 > threshold, (threshold, n)
        assert (n - 2) / (n - 1) ** 0.5 <= threshold, (threshold, n)
    assert min_n_for_zscore(3.0) == 11
    assert MIN_N_IQR == 4


# --- detect_outliers --------------------------------------------------------


@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_iqr_withholds_a_verdict_below_four_rows(tmp_path, n_rows):
    r = detect_outliers(str(_csv(tmp_path, n_rows)), columns=["spend"], method="iqr")
    assert r["success"] is True
    entry = r["results"][0]
    assert entry["n"] == n_rows
    # None, not 0: nothing was measured.
    assert entry["outlier_count"] is None
    assert entry["lower_bound"] is None
    assert entry["upper_bound"] is None
    assert "undetermined" in entry["status"]
    assert r["columns_undetermined"] == ["spend"]
    assert "not a finding" in r["hint"]


def test_iqr_reports_a_real_count_from_four_rows(tmp_path):
    f = tmp_path / "four.csv"
    f.write_text("spend\n0\n0\n0\n100\n")
    entry = detect_outliers(str(f), columns=["spend"], method="iqr")["results"][0]
    assert entry["outlier_count"] == 1
    assert entry["lower_bound"] is not None


@pytest.mark.parametrize("n_rows", [1, 5, 10])
def test_three_sigma_withholds_a_verdict_below_eleven_rows(tmp_path, n_rows):
    entry = detect_outliers(str(_csv(tmp_path, n_rows)), columns=["spend"], method="std")["results"][0]
    assert entry["outlier_count"] is None, n_rows
    assert "11" in entry["status"]


def test_three_sigma_reports_a_real_count_from_eleven_rows(tmp_path):
    f = tmp_path / "eleven.csv"
    f.write_text("spend\n" + "1\n" * 10 + "1000\n")
    entry = detect_outliers(str(f), columns=["spend"], method="std")["results"][0]
    assert entry["outlier_count"] == 1


def test_a_constant_column_with_enough_rows_still_answers_zero(tmp_path):
    f = tmp_path / "flat.csv"
    f.write_text("spend\n" + "7\n" * 20)
    entry = detect_outliers(str(f), columns=["spend"], method="iqr")["results"][0]
    assert entry["outlier_count"] == 0
    assert "zero spread" in entry["status"]


# --- anomaly_detection ------------------------------------------------------


def test_anomaly_detection_refuses_a_single_row(tmp_path):
    r = anomaly_detection(str(_csv(tmp_path, 1)), feature_columns=["spend", "clicks"])
    assert r["success"] is False
    assert "1 usable row" in r["error"]
    assert "at least 2" in r["error"]


def test_anomaly_detection_flags_a_thin_sample_without_refusing(tmp_path):
    r = anomaly_detection(str(_csv(tmp_path, 8)), feature_columns=["spend", "clicks"])
    assert r["success"] is True
    assert r["rows_scored"] == 8
    assert r["low_confidence"] is True
    assert "little rest to compare with" in r["hint"]


def test_anomaly_detection_is_confident_on_a_real_sample(tmp_path):
    r = anomaly_detection(str(_csv(tmp_path, 60)), feature_columns=["spend", "clicks"])
    assert r["success"] is True
    assert r.get("low_confidence") is not True


def test_anomaly_detection_names_a_non_numeric_feature_set(tmp_path):
    r = anomaly_detection(str(_csv(tmp_path, 30)), feature_columns=["label"])
    assert r["success"] is False
    assert "numeric" in r["error"]


# --- check_data_quality -----------------------------------------------------


def test_one_row_is_not_scored_as_every_column_being_constant(tmp_path):
    r = check_data_quality(str(_csv(tmp_path, 1)))
    assert r["success"] is True
    assert r["row_count"] == 1
    assert r["constant_columns"] == []
    assert not any(a["type"] == "constant_column" for a in r["alerts"])
    # The checks a single row cannot answer are named, so a score is readable.
    assert any("constant_column" in s for s in r["checks_skipped"])
    assert any("duplicate_rows" in s for s in r["checks_skipped"])


def test_a_genuinely_constant_column_is_still_flagged(tmp_path):
    f = tmp_path / "const.csv"
    f.write_text("spend,flag\n1,X\n2,X\n3,X\n4,X\n")
    r = check_data_quality(str(f))
    assert r["constant_columns"] == ["flag"]
    assert r["checks_skipped"] == []


def test_an_all_null_column_is_still_flagged_at_one_row(tmp_path):
    """n_unique == 0 is about the data, not the row count -- it stays."""
    f = tmp_path / "null.csv"
    f.write_text("spend,empty\n5,\n")
    r = check_data_quality(str(f))
    assert any(a["type"] == "all_null_column" for a in r["alerts"])
