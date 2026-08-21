"""Every trainer must explain a near-perfect score, not just the basic two.

A coverage sweep tuned a classifier on Ad_Data.csv targeting campaign_type and
got best_score 1.0, reported as a clean PASS with a saved model. On the same
data train_classifier says:

    Score of 1.0000 is explained by leakage: 'campaign_platform',
    'communication_medium', 'subchannel' ... determine 'campaign_type' exactly,
    so the model is reading the answer rather than predicting it.

Only ml_basic called leakage_warning(). Three of the five training tools stayed
silent:

    train_classifier      basic     warned
    train_regressor       basic     warned
    train_with_cv         medium    silent
    compare_models        medium    silent
    tune_hyperparameters  advanced  silent

This is worse than an error, because it succeeds: the caller gets a perfect
model and no reason to doubt it. compare_models is the worst of the three -- it
invites you to pick a winner from a table of scores that are all meaningless.

The fixture below reproduces the real defect exactly: a feature that determines
the target, so every model scores 1.0 honestly and the warning must fire.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from servers.ml_advanced.engine import tune_hyperparameters
from servers.ml_medium.engine import compare_models, train_with_cv


@pytest.fixture()
def leaky(tmp_path: Path) -> Path:
    """`mirror` is the target spelled differently -- textbook leakage."""
    p = tmp_path / "leaky.csv"
    n = 120
    labels = ["a" if i % 2 else "b" for i in range(n)]
    pd.DataFrame(
        {
            "noise": [float(i % 13) for i in range(n)],
            "mirror": [0 if v == "a" else 1 for v in labels],
            "label": labels,
        }
    ).to_csv(p, index=False)
    return p


@pytest.fixture()
def honest(tmp_path: Path) -> Path:
    """No column determines the target, so no warning should appear."""
    p = tmp_path / "honest.csv"
    rows = 120
    pd.DataFrame(
        {
            "a": [float(i % 7) for i in range(rows)],
            "b": [float((i * 5) % 11) for i in range(rows)],
            "label": ["a" if (i * 7) % 3 else "b" for i in range(rows)],
        }
    ).to_csv(p, index=False)
    return p


class TestTuneHyperparameters:
    def test_it_warns_on_a_leaked_target(self, leaky: Path):
        r = tune_hyperparameters(str(leaky), "label", "lr", "classification")
        assert r["success"] is True, r.get("error")
        assert r.get("warning"), "a perfect tuned model must say why it is perfect"

    def test_the_warning_names_the_leaking_column(self, leaky: Path):
        r = tune_hyperparameters(str(leaky), "label", "lr", "classification")
        assert "mirror" in r["warning"]

    def test_it_stays_quiet_on_an_honest_score(self, honest: Path):
        r = tune_hyperparameters(str(honest), "label", "lr", "classification")
        assert r["success"] is True, r.get("error")
        assert "warning" not in r


class TestTrainWithCv:
    def test_it_warns_on_a_leaked_target(self, leaky: Path):
        r = train_with_cv(str(leaky), "label", "lr", "classification", n_splits=3)
        assert r["success"] is True, r.get("error")
        assert r.get("warning"), "perfect scores on every fold must be explained"

    def test_the_warning_names_the_leaking_column(self, leaky: Path):
        r = train_with_cv(str(leaky), "label", "lr", "classification", n_splits=3)
        assert "mirror" in r["warning"]

    def test_it_stays_quiet_on_an_honest_score(self, honest: Path):
        r = train_with_cv(str(honest), "label", "lr", "classification", n_splits=3)
        assert r["success"] is True, r.get("error")
        assert "warning" not in r


class TestCompareModels:
    def test_it_warns_on_a_leaked_target(self, leaky: Path):
        r = compare_models(str(leaky), "label", "classification", ["lr", "dtc"])
        assert r["success"] is True, r.get("error")
        assert r.get("warning"), "a table of perfect scores invites a meaningless choice"

    def test_the_warning_names_the_leaking_column(self, leaky: Path):
        r = compare_models(str(leaky), "label", "classification", ["lr", "dtc"])
        assert "mirror" in r["warning"]

    def test_it_stays_quiet_on_an_honest_score(self, honest: Path):
        r = compare_models(str(honest), "label", "classification", ["lr", "dtc"])
        assert r["success"] is True, r.get("error")
        assert "warning" not in r


class TestRegressionSideToo:
    @pytest.fixture()
    def leaky_numeric(self, tmp_path: Path) -> Path:
        p = tmp_path / "leaky_num.csv"
        n = 120
        target = [float(i % 23) for i in range(n)]
        pd.DataFrame(
            {
                "noise": [float(i % 5) for i in range(n)],
                "mirror": target,
                "value": target,
            }
        ).to_csv(p, index=False)
        return p

    def test_cv_warns_for_regression(self, leaky_numeric: Path):
        r = train_with_cv(str(leaky_numeric), "value", "lir", "regression", n_splits=3)
        assert r["success"] is True, r.get("error")
        assert r.get("warning")

    def test_compare_reads_r2_not_rmse(self, leaky_numeric: Path):
        """rmse is also a score key and is 0.0 at perfection, so picking the
        first key rather than r2 by name would score a perfect fit as 0."""
        r = compare_models(str(leaky_numeric), "value", "regression", ["lir", "dtr"])
        assert r["success"] is True, r.get("error")
        assert r.get("warning"), "a perfect regression fit must still be explained"


class TestEveryTrainerIsCovered:
    """The gap was 'someone added a trainer and forgot the warning'. Pin it."""

    TRAINERS = [
        ("servers.ml_basic._basic_train", "train_classifier"),
        ("servers.ml_basic._basic_train", "train_regressor"),
        ("servers.ml_medium._medium_train", "train_with_cv"),
        ("servers.ml_medium._medium_train", "compare_models"),
        ("servers.ml_advanced.engine", "tune_hyperparameters"),
    ]

    @pytest.mark.parametrize("module_name,func", TRAINERS)
    def test_the_module_calls_leakage_warning(self, module_name: str, func: str):
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module_name))
        assert "leakage_warning(" in src, f"{module_name} trains models but never checks for leakage"

    @pytest.mark.parametrize("module_name,func", TRAINERS)
    def test_it_surfaces_under_the_same_key(self, module_name: str, func: str):
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module_name))
        assert '"warning"' in src, f"{module_name} must report leakage under the same 'warning' key"
