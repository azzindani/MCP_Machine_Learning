"""An unknown model name should say which parameter, and spot the near-twin.

A coverage sweep trained a classifier with model="lr" (logistic regression,
valid), then trained a regressor with model="lr" and got:

    Unknown algorithm: 'lr'. Allowed: dtr, lar, lir, pr, rfr, rr, xgb

Two things were wrong with that, neither fatal:

  * it says "algorithm", but the parameter is called `model`. Every one of these
    sites took `model`; only run_clustering() actually has an `algorithm`
    parameter, and it keeps that wording.
  * 'lr' is not a typo or an unsupported algorithm -- it is the *classifier's*
    name for the same idea, and the regressor's near-twin is 'lir'. The message
    read as "linear regression is unsupported", which is the opposite of true.

The allowed values were already listed in both the error and the hint, so this
is a legibility fix rather than a missing-contract one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from servers.ml_basic._basic_helpers import ALLOWED_CLASSIFIERS, ALLOWED_REGRESSORS
from servers.ml_basic.engine import train_classifier, train_regressor


@pytest.fixture()
def csv(tmp_path: Path) -> Path:
    p = tmp_path / "train.csv"
    n = 60
    pd.DataFrame(
        {
            "x1": [float(i) for i in range(n)],
            "x2": [float(i % 7) for i in range(n)],
            "label": ["a" if i % 2 else "b" for i in range(n)],
            "value": [float(i * 3 % 11) for i in range(n)],
        }
    ).to_csv(p, index=False)
    return p


class TestTheMessageNamesTheParameter:
    def test_regressor_says_model_not_algorithm(self, csv: Path):
        r = train_regressor(str(csv), "value", "lr")
        assert r["success"] is False
        assert "Unknown model:" in r["error"]
        assert "Unknown algorithm" not in r["error"]

    def test_classifier_says_model_not_algorithm(self, csv: Path):
        r = train_classifier(str(csv), "label", "lir")
        assert r["success"] is False
        assert "Unknown model:" in r["error"]
        assert "Unknown algorithm" not in r["error"]

    def test_the_allowed_values_are_still_listed(self, csv: Path):
        r = train_regressor(str(csv), "value", "lr")
        for name in ALLOWED_REGRESSORS:
            assert name in r["error"]


class TestTheNearTwinIsPointedOut:
    def test_lr_on_the_regressor_names_the_classifier(self, csv: Path):
        r = train_regressor(str(csv), "value", "lr")
        assert "train_classifier()" in r["hint"]

    def test_lir_on_the_classifier_names_the_regressor(self, csv: Path):
        r = train_classifier(str(csv), "label", "lir")
        assert "train_regressor()" in r["hint"]

    def test_a_genuine_typo_keeps_the_plain_hint(self, csv: Path):
        """Only a name valid on the sibling earns the cross-reference."""
        r = train_regressor(str(csv), "value", "zzz")
        assert r["success"] is False
        assert "train_classifier()" not in r["hint"]
        assert "lir" in r["hint"]


def _hint():
    """Imported lazily so the message tests above fail on their own assertions
    rather than on a collection error when the helper is absent."""
    from servers.ml_basic._basic_train import _wrong_trainer_hint

    return _wrong_trainer_hint


class TestTheHelperItself:
    def test_it_fires_only_for_sibling_names(self):
        _wrong_trainer_hint = _hint()
        assert _wrong_trainer_hint("lr", ALLOWED_CLASSIFIERS, "train_classifier")
        assert _wrong_trainer_hint("nope", ALLOWED_CLASSIFIERS, "train_classifier") == ""

    def test_the_shared_name_can_never_trigger_it(self):
        """xgb is valid on both, so it is never rejected -- a hint saying "call
        the other tool" would be nonsense if it ever appeared."""
        shared = ALLOWED_CLASSIFIERS & ALLOWED_REGRESSORS
        assert shared == {"xgb"}
        for name in shared:
            assert name in ALLOWED_CLASSIFIERS and name in ALLOWED_REGRESSORS

    @pytest.mark.parametrize("name", sorted(ALLOWED_REGRESSORS - ALLOWED_CLASSIFIERS))
    def test_every_regressor_only_name_is_caught_on_the_classifier(self, name: str):
        assert _hint()(name, ALLOWED_REGRESSORS, "train_regressor")

    @pytest.mark.parametrize("name", sorted(ALLOWED_CLASSIFIERS - ALLOWED_REGRESSORS))
    def test_every_classifier_only_name_is_caught_on_the_regressor(self, name: str):
        assert _hint()(name, ALLOWED_CLASSIFIERS, "train_classifier")


class TestClusteringKeepsItsOwnWording:
    """run_clustering's parameter really is `algorithm`, so it must not be
    swept up in the rename."""

    def test_the_cluster_error_still_says_algorithm(self):
        import inspect

        from servers.ml_medium import _medium_cluster

        src = inspect.getsource(_medium_cluster)
        assert "Unknown algorithm" in src

    def test_no_model_parameter_site_still_says_algorithm(self):
        import inspect

        from servers.ml_advanced import engine as adv
        from servers.ml_basic import _basic_train
        from servers.ml_medium import _medium_train

        for mod in (_basic_train, _medium_train, adv):
            assert "Unknown algorithm" not in inspect.getsource(mod), (
                f"{mod.__name__} takes `model`, so its error must say model"
            )
