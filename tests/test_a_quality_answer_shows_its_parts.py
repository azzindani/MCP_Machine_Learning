"""check_data_quality shows the parts of its score; a constant column is alerted once.

The sweep read 59.3 for Ad_Data.csv from check_data_quality with no way to see
that completeness and uniqueness were near 100 and every lost point went to
alerts, while MCP_Data_Analyst scored the same file 89 and 99. The answer now
carries the breakdown and what `validity` prices. generate_eda_report also
alerted each constant column twice, as constant and as a 100% class imbalance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from servers.ml_medium._medium_data import check_data_quality
from servers.ml_medium._medium_eda import _quality_score_html, _run_quality_alerts


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["same"] * 40,
            "region": ["north", "south", "east", "west"] * 10,
            "units": list(range(40)),
        }
    )


class TestTheBreakdownIsInTheAnswer:
    def test_the_score_is_the_breakdowns(self, frame, tmp_path):
        path = tmp_path / "f.csv"
        frame.to_csv(path, index=False)
        result = check_data_quality(str(path))
        assert result["success"] is True, result.get("error")
        breakdown = result["quality_breakdown"]
        assert breakdown["quality_score"] == result["quality_score"]
        assert set(breakdown["components"]) >= {"completeness", "validity", "uniqueness"}
        assert breakdown["weights"] and "validity" in breakdown["validity_note"]

    def test_the_constant_column_is_what_validity_lost(self, frame, tmp_path):
        path = tmp_path / "f.csv"
        frame.to_csv(path, index=False)
        breakdown = check_data_quality(str(path))["quality_breakdown"]
        assert breakdown["components"]["completeness"] == 100.0
        assert breakdown["components"]["uniqueness"] == 100.0
        # One constant column of three costs validity that column's share (#22).
        assert breakdown["components"]["validity"] == pytest.approx(100 - 100 / 3, abs=0.1)


class TestAConstantColumnIsAlertedOnce:
    def test_not_again_as_class_imbalance(self, frame):
        alerts = [a for a in _run_quality_alerts(frame, "") if a.get("column") == "product"]
        assert [a["type"] for a in alerts] == ["constant_column"]

    def test_a_real_imbalance_is_still_alerted(self):
        skewed = pd.DataFrame({"flag": ["a"] * 38 + ["b"] * 2, "units": list(range(40))})
        types = [a["type"] for a in _run_quality_alerts(skewed, "") if a.get("column") == "flag"]
        assert types == ["class_imbalance"]


class TestThePageSaysWhatWasNotScored:
    def test_the_advice_card_counts_the_advice(self):
        """Advice costs nothing since #22 (2026-09-24), so a high score can sit
        above a long list of alerts; the card says how many of them it left out."""
        skewed = pd.DataFrame({"x": list(range(40)), "y": [v * 3 for v in range(40)], "z": [0] * 36 + [1, 2, 3, 4]})
        alerts = _run_quality_alerts(skewed, "")
        advice = sum(1 for a in alerts if a["type"] in {"multicollinearity", "zero_inflated", "extreme_skewness"})
        assert advice > 0, "the fixture raises advice"
        html = _quality_score_html(100.0, alerts, {})
        assert f'<div class="num">{advice}</div><div class="lbl">Advice, not scored</div>' in html
