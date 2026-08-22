"""A usable dataset scored 5.6 out of 100.

check_data_quality on the real ad dataset -- 16,834 rows that train a classifier
without complaint -- returned:

    quality_score: 5.6, alerts_count: 11, alerts_high: 2

while a frame of eight constant columns, an all-null column and 100% duplicate
rows returned 0.0. Two datasets a person would describe completely differently
landed 5.6 apart, at the bottom of a 100-point scale.

The cause is that alerts are raised per column, so the penalty grows with the
width of the frame: four extreme_skewness alerts and three multicollinearity
alerts cost 56 points between them, for properties that are ordinary in real ad
data. A previous fix had already demoted multicollinearity from high to medium
for exactly this reason -- "a -15-per-pair 'high' penalty made realistic, usable
data indistinguishable from actual garbage" -- which lowered the per-alert cost
but left the term unbounded.

The alert term was also the *only* unbounded one: missingness was capped at 20
and duplicates at 10 in the same function. Capping it is what the other two
already did.

Found by comparing three tools' scores for one file: this said 5.6,
generate_eda_report said 41 in the sibling repo, validate_dataset said 89.
"""

from __future__ import annotations

import csv as csvmod
import shutil
from pathlib import Path

import pytest

from servers.ml_medium._medium_helpers import (
    ALERT_DEDUCTION_CAP,
    DUPLICATE_DEDUCTION_CAP,
    MISSINGNESS_DEDUCTION_CAP,
)
from servers.ml_medium.engine import check_data_quality, generate_eda_report

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def ad_data_full_csv(tmp_path: Path) -> Path:
    """The real 16,834-row ad dataset, untouched (no derived ctr column)."""
    dst = tmp_path / "ad_data_full.csv"
    shutil.copy(FIXTURES_DIR / "ad_data_full.csv", dst)
    return dst


def write_csv(path: Path, header: list[str], rows: list[list]) -> str:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csvmod.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


@pytest.fixture()
def clean_csv(tmp_path: Path) -> str:
    return write_csv(tmp_path / "clean.csv", ["a", "b", "c"], [[i, i * 2, f"cat{i % 3}"] for i in range(200)])


@pytest.fixture()
def one_constant_csv(tmp_path: Path) -> str:
    return write_csv(tmp_path / "mid.csv", ["a", "b", "const"], [[i, i * 2, "same"] for i in range(200)])


@pytest.fixture()
def many_alerts_csv(tmp_path: Path) -> str:
    """Wide enough to raise more alert points than the cap allows."""
    header = [f"k{i}" for i in range(8)] + ["v"]
    return write_csv(tmp_path / "awful.csv", header, [["x"] * 8 + [""] for _ in range(200)])


class TestTheScoreSeparatesDatasets:
    def test_a_clean_frame_scores_well(self, clean_csv: str):
        assert check_data_quality(clean_csv)["quality_score"] >= 85

    def test_one_constant_column_costs_something_but_not_everything(self, one_constant_csv: str):
        score = check_data_quality(one_constant_csv)["quality_score"]
        assert 60 <= score < 85, score

    def test_a_frame_with_many_alerts_is_not_pinned_to_zero(self, many_alerts_csv: str):
        """Every alert type at once used to floor at 0.0, taking the ordering
        with it."""
        assert check_data_quality(many_alerts_csv)["quality_score"] > 0

    def test_worse_data_still_scores_lower(self, clean_csv: str, one_constant_csv: str, many_alerts_csv: str):
        clean = check_data_quality(clean_csv)["quality_score"]
        mid = check_data_quality(one_constant_csv)["quality_score"]
        bad = check_data_quality(many_alerts_csv)["quality_score"]
        assert clean > mid > bad, (clean, mid, bad)


class TestTheRealAdDataset:
    def test_it_is_no_longer_scored_as_near_worthless(self, ad_data_full_csv: Path):
        score = check_data_quality(str(ad_data_full_csv))["quality_score"]
        assert score > 20, score

    def test_it_still_scores_poorly(self, ad_data_full_csv: Path):
        """Two constant columns, 205 duplicates, four skewed columns: not good
        data, just not 5.6-out-of-100 data."""
        assert check_data_quality(str(ad_data_full_csv))["quality_score"] < 60

    def test_the_alerts_are_unchanged(self, ad_data_full_csv: Path):
        r = check_data_quality(str(ad_data_full_csv))
        assert r["alerts_count"] == 11
        assert r["alerts_high"] == 2
        assert sorted(r["constant_columns"]) == ["phase", "product"]

    def test_the_two_scorers_in_this_server_agree(self, ad_data_full_csv: Path, tmp_path: Path):
        """They compute the score in different modules; they used to disagree."""
        a = check_data_quality(str(ad_data_full_csv))["quality_score"]
        b = generate_eda_report(str(ad_data_full_csv), output_path=str(tmp_path / "eda.html"), open_after=False)[
            "quality_score"
        ]
        assert abs(a - b) <= 2, (a, b)


class TestTheCapsThemselves:
    def test_the_alert_cap_leaves_room_for_the_others(self):
        assert ALERT_DEDUCTION_CAP + MISSINGNESS_DEDUCTION_CAP + DUPLICATE_DEDUCTION_CAP == 100.0

    def test_the_alert_term_is_the_largest(self):
        assert ALERT_DEDUCTION_CAP > MISSINGNESS_DEDUCTION_CAP > DUPLICATE_DEDUCTION_CAP

    def test_no_score_leaves_the_range(self, clean_csv: str, many_alerts_csv: str, ad_data_full_csv: Path):
        for path in (clean_csv, many_alerts_csv, str(ad_data_full_csv)):
            score = check_data_quality(path)["quality_score"]
            assert 0.0 <= score <= 100.0, (path, score)
