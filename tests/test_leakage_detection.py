"""A perfect score is a bug report, not a result.

A real training run targeted `campaign_type` while `campaign_platform` stayed in
the feature set — the same fact under two names. accuracy, f1 and AUC all came
back at 1.000 with a spotless confusion matrix, and neither the tool response nor
the generated HTML report said anything about it. These tests cover the check
that now explains such a score instead of presenting it as a good model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from servers.ml_basic.engine import train_classifier, train_regressor
from shared.ml_utils import find_determinant_features, leakage_warning


@pytest.fixture()
def leaky_csv(tmp_path: Path) -> Path:
    """`platform` fixes `campaign` exactly, exactly as the real dataset did."""
    rows = []
    for i in range(200):
        google = i % 2 == 0
        rows.append(
            {
                "platform": "Google Ads" if google else "Facebook Ads",
                "campaign": "Search" if google else "Conversions",
                "spend": float(i % 37),
                "clicks": i % 11,
            }
        )
    csv = tmp_path / "leaky.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


@pytest.fixture()
def honest_csv(tmp_path: Path) -> Path:
    """No feature determines the label — scores should land well short of 1.0."""
    rows = []
    for i in range(200):
        rows.append(
            {
                "a": float(i % 17),
                "b": float((i * 7) % 13),
                "c": float((i * 3) % 5),
                "label": i % 2,
            }
        )
    csv = tmp_path / "honest.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


class TestFindDeterminantFeatures:
    def test_finds_the_column_that_fixes_the_target(self):
        df = pd.DataFrame({"platform": ["A", "A", "B", "B"], "campaign": ["x", "x", "y", "y"], "n": [1, 2, 3, 4]})
        assert find_determinant_features(df, "campaign", ["platform", "n"]) == ["platform"]

    def test_ignores_a_column_with_one_value_per_row(self):
        """An id partitions any target perfectly by construction — flagging it
        would fire on every dataset that carries a row identifier."""
        df = pd.DataFrame({"row_id": range(50), "label": [i % 2 for i in range(50)]})
        assert find_determinant_features(df, "label", ["row_id"]) == []

    def test_returns_nothing_when_no_column_determines_the_target(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 3, 4], "label": [0, 1, 1, 0]})
        assert find_determinant_features(df, "label", ["a", "b"]) == []

    def test_unknown_target_is_not_an_error(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert find_determinant_features(df, "missing", ["a"]) == []


class TestLeakageWarning:
    def test_silent_on_an_ordinary_score(self):
        df = pd.DataFrame({"platform": ["A", "A", "B", "B"], "campaign": ["x", "x", "y", "y"]})
        assert leakage_warning(df, "campaign", ["platform"], 0.83) == ""

    def test_names_the_leaking_column(self):
        df = pd.DataFrame({"platform": ["A", "A", "B", "B"], "campaign": ["x", "x", "y", "y"]})
        message = leakage_warning(df, "campaign", ["platform"], 1.0)
        assert "'platform'" in message
        assert "campaign" in message

    def test_still_warns_when_no_single_column_explains_it(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 3, 4], "label": [0, 1, 1, 0]})
        message = leakage_warning(df, "label", ["a", "b"], 1.0)
        assert message
        assert "combinations" in message


class TestTrainersReportLeakage:
    def test_classifier_warns_on_a_perfect_score(self, leaky_csv: Path):
        result = train_classifier(str(leaky_csv), "campaign", "rf")
        assert result["success"] is True
        assert result["metrics"]["accuracy"] == 1.0
        assert "warning" in result
        assert "'platform'" in result["warning"]

    def test_the_warning_reaches_the_saved_model(self, leaky_csv: Path):
        """generate_training_report and read_model_report read the metadata, so
        persisting it there is what puts the caveat on the rendered page."""
        import json

        from servers.ml_basic._basic_helpers import _load_model

        result = train_classifier(str(leaky_csv), "campaign", "rf")
        _, metadata = _load_model(result["model_path"])
        assert "'platform'" in metadata["leakage_warning"]

        manifest = Path(result["model_path"]).with_suffix(".manifest.json")
        assert "'platform'" in json.loads(manifest.read_text(encoding="utf-8"))["leakage_warning"]

    def test_the_report_leads_with_the_caveat(self, leaky_csv: Path, tmp_path: Path):
        """The rendered page showed accuracy/f1/AUC of 1.000 beside a spotless
        confusion matrix and said nothing about why."""
        from servers.ml_advanced.engine import generate_training_report

        trained = train_classifier(str(leaky_csv), "campaign", "rf")
        out = tmp_path / "report.html"
        report = generate_training_report(trained["model_path"], output_path=str(out), open_after=False)
        assert report["success"] is True
        # The report inlines plotly.js, so the default Windows codec cannot read it.
        page = out.read_text(encoding="utf-8")
        assert "not trustworthy" in page.lower()
        assert page.index("not trustworthy") < page.index("Evaluation Metrics")

    def test_honest_model_carries_no_warning(self, honest_csv: Path):
        result = train_classifier(str(honest_csv), "label", "rf")
        assert result["success"] is True
        assert "warning" not in result

    def test_regressor_warns_on_a_perfect_r2(self, tmp_path: Path):
        rows = [{"group": f"g{i % 20}", "target": float(i % 20), "noise": float(i % 7)} for i in range(200)]
        csv = tmp_path / "leaky_reg.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        result = train_regressor(str(csv), "target", "rfr")
        assert result["success"] is True
        assert result["metrics"]["r2"] >= 0.999
        assert "warning" in result
        assert "'group'" in result["warning"]
