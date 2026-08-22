"""The list of allowed models included models that can never be tuned.

A sweep called tune_hyperparameters with `lir`, taken from the tool's own
allowed list, and got:

    error: "No param grid available for model 'lir'."
    hint : "Provide a custom param_grid JSON string."

`allowed` is the whole model registry -- what train_regressor accepts -- but
tuning needs an entry in DEFAULT_PARAMS, and xgb is refused outright a few lines
later. So the advertised list carried three regressors that always fail (lir and
pr have no grid, xgb is blocked) and two classifiers (nb has no grid, xgb
blocked). Picking off the list was a coin flip, and the hint's only offer was to
write a parameter grid by hand.

Same shape as fs_write's op list and set_conditional_format's rule names: the
vocabulary a caller is handed has to be the vocabulary that works.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from servers.ml_advanced._adv_helpers import DEFAULT_PARAMS
from servers.ml_advanced.engine import tune_hyperparameters
from shared.registry import allowed_classifiers, allowed_regressors

FIXTURES_DIR = Path(__file__).parent / "fixtures"

TUNABLE_REGRESSORS = ["dtr", "lar", "rfr", "rr"]
TUNABLE_CLASSIFIERS = ["dtc", "knn", "lr", "rf", "svm"]


@pytest.fixture()
def csv_path(tmp_path: Path) -> str:
    dst = tmp_path / "regression_simple.csv"
    shutil.copy(FIXTURES_DIR / "regression_simple.csv", dst)
    return str(dst)


def target_for(csv_path: str) -> str:
    import csv as csvmod

    with open(csv_path, newline="", encoding="utf-8") as fh:
        return next(csvmod.reader(fh))[-1]


class TestAModelWithoutAGridSaysWhichHaveOne:
    @pytest.mark.parametrize("model", ["lir", "pr"])
    def test_the_error_is_specific(self, csv_path: str, model: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model=model, task="regression", dry_run=True)
        assert r["success"] is False
        assert model in r["error"], r["error"]

    @pytest.mark.parametrize("model", ["lir", "pr"])
    def test_the_hint_names_models_that_do_work(self, csv_path: str, model: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model=model, task="regression", dry_run=True)
        for good in TUNABLE_REGRESSORS:
            assert good in r["hint"], f"{good} missing from {r['hint']}"

    def test_it_does_not_offer_only_a_hand_written_grid(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="lir", task="regression", dry_run=True)
        assert r["hint"] != "Provide a custom param_grid JSON string."

    def test_it_still_mentions_param_grid_as_the_escape_hatch(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="lir", task="regression", dry_run=True)
        assert "param_grid" in r["hint"], r["hint"]

    def test_a_classifier_hint_names_classifiers(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="nb", task="classification", dry_run=True)
        assert r["success"] is False
        for good in TUNABLE_CLASSIFIERS:
            assert good in r["hint"], f"{good} missing from {r['hint']}"
        assert "rfr" not in r["hint"], "regressor leaked into a classification hint"


class TestXgbPointsAtTheRightTask:
    def test_the_regression_hint_lists_regressors(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="xgb", task="regression", dry_run=True)
        assert r["success"] is False
        assert "regression" in r["hint"]
        for good in TUNABLE_REGRESSORS:
            assert good in r["hint"], r["hint"]

    def test_it_no_longer_offers_classifiers_for_a_regression(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="xgb", task="regression", dry_run=True)
        assert "svm" not in r["hint"] and "knn" not in r["hint"], r["hint"]


class TestAnUnknownModelSaysWhichAreTunable:
    def test_the_error_still_lists_the_allowed_models(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="zzz", task="regression", dry_run=True)
        assert r["success"] is False
        assert "lir" in r["error"], r["error"]

    def test_the_hint_narrows_to_the_tunable_ones(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="zzz", task="regression", dry_run=True)
        assert "built-in grid" in r["hint"], r["hint"]
        assert "lir" not in r["hint"], "lir has no grid and must not be recommended"


class TestTheAdvertisedSetStaysHonest:
    def test_every_model_the_hint_names_really_has_a_grid(self):
        for m in TUNABLE_REGRESSORS + TUNABLE_CLASSIFIERS:
            assert DEFAULT_PARAMS.get(m), m

    def test_the_expected_sets_match_the_registry(self):
        """If a model gains or loses a grid, this test names it."""
        regs = sorted(m for m in allowed_regressors() if DEFAULT_PARAMS.get(m) and m != "xgb")
        clfs = sorted(m for m in allowed_classifiers() if DEFAULT_PARAMS.get(m) and m != "xgb")
        assert regs == TUNABLE_REGRESSORS, regs
        assert clfs == TUNABLE_CLASSIFIERS, clfs


class TestATunableModelStillTunes:
    def test_a_dry_run_succeeds(self, csv_path: str):
        r = tune_hyperparameters(csv_path, target_for(csv_path), model="rr", task="regression", dry_run=True)
        assert r["success"] is True, r.get("error")

    def test_a_custom_grid_still_overrides(self, csv_path: str):
        r = tune_hyperparameters(
            csv_path,
            target_for(csv_path),
            model="lir",
            task="regression",
            param_grid='{"fit_intercept": [true, false]}',
            dry_run=True,
        )
        assert r["success"] is True, r.get("error")
