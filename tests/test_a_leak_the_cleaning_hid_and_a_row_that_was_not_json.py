"""Three defects the 2026-10-02 big-data sweep found in ML, each checked against the reply.

F18 -- training reported no leak the quality check found. `train_classifier` asked whether a
feature already held the answer AFTER the cleaning had filled every null with a median, so a
field that is empty for exactly one outcome (recorded after that outcome is known) looked like
an ordinary column. The loan file's `Interest_rate_spread` is null for all 29,311 defaults and
none of the 89,625 others; the model scored accuracy 1.0 and `leakage_suspects` was `[]`, while
`check_data_quality` on the same file and target named eight.

F16 -- `read_rows` wrote a bare `NaN` for every empty float cell, which JSON.parse refuses, so a
single gap made the whole reply unreadable to a strict client. A reversed range was clamped to
an empty slice and reported as success.

F22 -- the learning curve's published model list offered 13 names; a real run refused `nb`
and `xgb`, and `dry_run` reported success for both.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from servers.ml_advanced import server as _advanced
from servers.ml_basic import server as _basic
from servers.ml_basic.engine import train_classifier
from servers.ml_medium import server as _medium

read_rows = _basic.mcp._tool_manager._tools["read_rows"].fn
check_data_quality = _medium.mcp._tool_manager._tools["check_data_quality"].fn
plot_learning_curve = _advanced.mcp._tool_manager._tools["plot_learning_curve"].fn


def _loan_like(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """`spread` exists only for loans that did not default: null for every default."""
    rng = np.random.default_rng(seed)
    default = np.array([1] * (n // 4) + [0] * (n - n // 4))
    return pd.DataFrame(
        {
            "income": rng.normal(5000, 800, n),
            "score": rng.normal(650, 40, n),
            "spread": np.where(default == 1, np.nan, rng.normal(1.0, 0.2, n)),
            "status": default,
        }
    ).sample(frac=1, random_state=3, ignore_index=True)


@pytest.fixture
def loan(tmp_path) -> str:
    path = tmp_path / "loan.csv"
    _loan_like().to_csv(path, index=False)
    return str(path)


class TestTrainingSeesWhatTheCleaningHid:
    def test_the_quality_check_names_the_field(self, loan):
        result = check_data_quality(loan, target_column="status")
        assert "spread" in {s["feature"] for s in result["leakage_suspects"]}

    @pytest.mark.parametrize("model", ["rf", "lr"])
    def test_training_names_it_too(self, loan, model):
        result = train_classifier(loan, "status", model)
        assert result["success"] is True, result.get("error")
        suspects = {s["feature"]: s for s in result["leakage_suspects"]}
        assert "spread" in suspects
        assert suspects["spread"]["reason"] == "missingness_tracks_target"

    def test_a_dry_run_says_so_before_anything_is_fitted(self, loan):
        result = train_classifier(loan, "status", "rf", dry_run=True)
        assert "spread" in {s["feature"] for s in result["leakage_suspects"]}

    def test_a_clean_column_with_gaps_is_not_accused(self, tmp_path):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300), "y": rng.integers(0, 2, 300)})
        frame.loc[rng.choice(300, 40, replace=False), "a"] = np.nan  # gaps unrelated to y
        path = tmp_path / "clean.csv"
        frame.to_csv(path, index=False)
        result = train_classifier(str(path), "y", "dtc")
        assert "a" not in {s["feature"] for s in result["leakage_suspects"]}


class TestReadRowsIsJson:
    @pytest.fixture
    def gaps(self, tmp_path) -> str:
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"x": [1.0, np.nan, 3.0], "t": ["a", None, "c"], "n": [1, 2, 3]}).to_csv(path, index=False)
        return str(path)

    def test_a_null_cell_is_json_null(self, gaps):
        result = read_rows(gaps, 0, 3)
        text = json.dumps(result, allow_nan=False)  # raises on NaN / Infinity
        rows = json.loads(text)["rows"]
        assert rows[1]["x"] is None and rows[1]["t"] is None and rows[0]["x"] == 1.0

    def test_an_infinity_is_text_not_a_bare_token(self, tmp_path):
        path = tmp_path / "inf.csv"
        pd.DataFrame({"r": [np.inf, -np.inf, 2.0]}).to_csv(path, index=False)
        rows = json.loads(json.dumps(read_rows(str(path), 0, 3), allow_nan=False))["rows"]
        assert [r["r"] for r in rows] == ["inf", "-inf", 2.0]

    def test_a_reversed_range_is_refused_not_emptied(self, gaps):
        result = read_rows(gaps, 5, 1)
        assert result["success"] is False
        assert "before start" in result["error"]

    def test_an_empty_window_is_still_an_empty_answer(self, gaps):
        result = read_rows(gaps, 2, 2)
        assert result["success"] is True and result["rows"] == []


class TestTheLearningCurveDrawsWhatItOffers:
    @pytest.fixture
    def data(self, tmp_path) -> str:
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 160), "b": rng.normal(0, 1, 160)})
        frame["label"] = (frame.a + frame.b > 0).astype(int)
        frame["amount"] = frame.a * 3 + frame.b + rng.normal(0, 0.1, 160)
        path = tmp_path / "curve.csv"
        frame.to_csv(path, index=False)
        return str(path)

    @pytest.mark.parametrize("model", ["lr", "rf", "dtc", "knn", "svm", "nb", "xgb"])
    def test_every_classifier_in_the_list(self, data, model, tmp_path):
        out = tmp_path / f"lc_{model}.html"
        result = plot_learning_curve(
            data, "label", model, "classification", cv=3, output_path=str(out), open_after=False
        )
        assert result["success"] is True, result.get("error")
        assert out.exists()

    @pytest.mark.parametrize("model", ["lir", "rfr", "dtr", "lar", "rr", "pr", "xgb"])
    def test_every_regressor_in_the_list(self, data, model, tmp_path):
        out = tmp_path / f"lc_{model}.html"
        result = plot_learning_curve(data, "amount", model, "regression", cv=3, output_path=str(out), open_after=False)
        assert result["success"] is True, result.get("error")

    def test_a_dry_run_checks_the_model_against_the_task(self, data):
        result = plot_learning_curve(data, "label", "lir", "classification", dry_run=True)
        assert result["success"] is False
        assert "Allowed" in result["error"] and "nb" in result["error"]

    def test_a_dry_run_passes_a_model_that_will_run(self, data):
        assert plot_learning_curve(data, "label", "nb", "classification", dry_run=True)["success"] is True
