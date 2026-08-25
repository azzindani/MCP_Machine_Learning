"""Hyper-parameters the tool accepts and the chosen model ignores.

The tool schema describes the tool; the vocabulary is per model. That is the
blind spot below `list[dict]` found in round 14 on `aggregate_dataset`, and
ml_basic's two trainers have it four times over:

    train_classifier  class_weight   read by lr, svm, rf, dtc
                                     ignored by knn, nb, xgb
    train_regressor   degree         read by pr
                      alpha          read by lar, rr
                      n_estimators   read by rfr

`train_classifier(model="nb", class_weight="balanced")` trained a GaussianNB --
which has no such parameter in sklearn -- and answered success: true with
metrics, a manifest and a saved .pkl, mentioning the argument nowhere. Every
schema check passes, because the argument is valid for the tool.

Two more on the same line. `cw = class_weight if class_weight in ("balanced",)
else None` turned every other spelling into the default in silence, so
class_weight="balance" trained an unweighted model under success: true. And the
xgb regressor hardcoded `num_boost_round=5` while declaring `n_estimators`,
which for a boosted tree is that number -- so unlike the others, the fix there
is to read the argument rather than refuse it.

Last, the provenance half: n_estimators=15 reached the RandomForestRegressor and
appeared nowhere in the manifest, so a model saved with it could not be told
apart from one trained with the default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_basic import engine  # noqa: E402
from servers.ml_basic._basic_helpers import (  # noqa: E402
    ALLOWED_CLASSIFIERS,
    ALLOWED_REGRESSORS,
    CLASSIFIER_ARG_MODELS,
    REGRESSOR_ARG_MODELS,
)


@pytest.fixture
def csv(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    f = tmp_path / "t.csv"
    pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "y": rng.integers(0, 2, n),
            "v": rng.normal(size=n) * 5,
        }
    ).to_csv(f, index=False)
    return f


# --- the tables cover the models the tools advertise ------------------------


class TestTheTableAndTheModelListAgree:
    def test_every_named_reader_is_a_real_classifier(self):
        for arg, models in CLASSIFIER_ARG_MODELS.items():
            unknown = models - set(ALLOWED_CLASSIFIERS)
            assert not unknown, f"{arg} claims to be read by {unknown}"

    def test_every_named_reader_is_a_real_regressor(self):
        for arg, models in REGRESSOR_ARG_MODELS.items():
            unknown = models - set(ALLOWED_REGRESSORS)
            assert not unknown, f"{arg} claims to be read by {unknown}"

    def test_no_argument_claims_every_model(self):
        """If it were read by all of them there would be nothing to refuse."""
        for arg, models in REGRESSOR_ARG_MODELS.items():
            assert models != set(ALLOWED_REGRESSORS), arg


# --- an argument the model does not read is refused, not dropped ------------


class TestTheClassifierRefusesWhatItWouldIgnore:
    @pytest.mark.parametrize("model", ["knn", "nb", "xgb"])
    def test_class_weight_on_a_model_without_one(self, csv, model):
        r = engine.train_classifier(str(csv), "y", model, class_weight="balanced")
        assert r["success"] is False
        assert "does not read class_weight" in r["error"]
        assert f"model='{model}'" in r["error"]

    def test_the_refusal_names_the_models_that_do(self, csv):
        r = engine.train_classifier(str(csv), "y", "nb", class_weight="balanced")
        for name in ("lr", "svm", "rf", "dtc"):
            assert name in r["hint"], r["hint"]

    @pytest.mark.parametrize("model", ["knn", "nb"])
    def test_those_models_still_train_without_it(self, csv, model):
        r = engine.train_classifier(str(csv), "y", model)
        assert r["success"] is True, r.get("error")

    @pytest.mark.parametrize("model", ["lr", "rf", "dtc"])
    def test_the_models_that_read_it_are_untouched(self, csv, model):
        r = engine.train_classifier(str(csv), "y", model, class_weight="balanced")
        assert r["success"] is True, r.get("error")


class TestAnUnknownClassWeightIsNotSilentlyTheDefault:
    @pytest.mark.parametrize("value", ["balance", "auto", "class_weight"])
    def test_a_misspelled_weight_is_refused(self, csv, value):
        r = engine.train_classifier(str(csv), "y", "lr", class_weight=value)
        assert r["success"] is False
        assert value in r["error"]
        assert "balanced" in r["hint"]

    def test_the_refusal_happens_before_the_model_is_chosen(self, csv):
        """Otherwise the misspelling is reported as a model mismatch instead."""
        r = engine.train_classifier(str(csv), "y", "nb", class_weight="balance")
        assert "Unknown class_weight" in r["error"]

    def test_omitting_it_is_still_fine(self, csv):
        assert engine.train_classifier(str(csv), "y", "lr")["success"] is True

    def test_case_and_spacing_are_forgiven(self, csv):
        r = engine.train_classifier(str(csv), "y", "lr", class_weight=" Balanced ")
        assert r["success"] is True, r.get("error")


class TestTheRegressorRefusesWhatItWouldIgnore:
    @pytest.mark.parametrize("model", ["lir", "lar", "rr", "dtr", "rfr"])
    def test_degree_outside_the_polynomial_model(self, csv, model):
        r = engine.train_regressor(str(csv), "v", model, degree=3)
        assert r["success"] is False
        assert "does not read degree" in r["error"]
        assert "pr" in r["hint"]

    @pytest.mark.parametrize("model", ["lir", "pr", "dtr", "rfr"])
    def test_alpha_outside_the_penalised_models(self, csv, model):
        r = engine.train_regressor(str(csv), "v", model, alpha=0.5)
        assert r["success"] is False
        assert "does not read alpha" in r["error"]

    @pytest.mark.parametrize("model", ["lir", "pr", "lar", "rr", "dtr"])
    def test_n_estimators_outside_the_ensembles(self, csv, model):
        r = engine.train_regressor(str(csv), "v", model, n_estimators=15)
        assert r["success"] is False
        assert "does not read n_estimators" in r["error"]

    def test_several_ignored_arguments_are_all_named(self, csv):
        r = engine.train_regressor(str(csv), "v", "lir", degree=3, alpha=0.5)
        assert "alpha" in r["error"] and "degree" in r["error"]

    def test_defaults_are_never_treated_as_a_request(self, csv):
        """Passing nothing must not trip a refusal on any model."""
        for model in ALLOWED_REGRESSORS:
            r = engine.train_regressor(str(csv), "v", model)
            assert r["success"] is True, f"{model}: {r.get('error')}"


class TestTheModelsThatReadThemStillDo:
    def test_pr_reads_degree(self, csv, tmp_path):
        r = engine.train_regressor(str(csv), "v", "pr", degree=2, output_path=str(tmp_path / "p.pkl"))
        assert r["success"] is True, r.get("error")
        assert r["hyperparameters"] == {"degree": 2}

    @pytest.mark.parametrize("model", ["lar", "rr"])
    def test_the_penalised_models_read_alpha(self, csv, model):
        r = engine.train_regressor(str(csv), "v", model, alpha=0.5)
        assert r["success"] is True, r.get("error")
        assert r["hyperparameters"] == {"alpha": 0.5}

    def test_rfr_reads_n_estimators(self, csv):
        r = engine.train_regressor(str(csv), "v", "rfr", n_estimators=15)
        assert r["success"] is True, r.get("error")
        assert r["hyperparameters"] == {"n_estimators": 15}

    def test_xgb_now_reads_n_estimators_too(self, csv):
        """A boosted tree's n_estimators is its number of rounds."""
        r = engine.train_regressor(str(csv), "v", "xgb", n_estimators=25)
        assert r["success"] is True, r.get("error")
        assert r["hyperparameters"] == {"n_estimators": 25}

    def test_more_rounds_is_a_different_model(self, csv, tmp_path):
        few = engine.train_regressor(str(csv), "v", "xgb", n_estimators=2, output_path=str(tmp_path / "a.pkl"))
        many = engine.train_regressor(str(csv), "v", "xgb", n_estimators=40, output_path=str(tmp_path / "b.pkl"))
        assert few["metrics"] != many["metrics"], "the argument reached the trainer"


# --- what was used is written down ------------------------------------------


class TestTheManifestRecordsWhatWasUsed:
    def test_the_regressor_manifest_carries_them(self, csv, tmp_path):
        out = tmp_path / "r.pkl"
        engine.train_regressor(str(csv), "v", "rfr", n_estimators=15, output_path=str(out))
        manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert manifest["hyperparameters"] == {"n_estimators": 15}

    def test_the_classifier_manifest_carries_them(self, csv, tmp_path):
        out = tmp_path / "c.pkl"
        engine.train_classifier(str(csv), "y", "lr", class_weight="balanced", output_path=str(out))
        manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert manifest["hyperparameters"] == {"class_weight": "balanced"}

    def test_an_unweighted_model_says_so_by_omission(self, csv, tmp_path):
        out = tmp_path / "c2.pkl"
        engine.train_classifier(str(csv), "y", "lr", output_path=str(out))
        manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert manifest["hyperparameters"] == {}

    def test_a_defaulted_regressor_records_its_defaults(self, csv, tmp_path):
        out = tmp_path / "r2.pkl"
        engine.train_regressor(str(csv), "v", "rfr", output_path=str(out))
        manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert manifest["hyperparameters"] == {"n_estimators": 10}


# --- the third of three things the docstring promises -----------------------


class TestAProfilePromisesTopValues:
    """ "Profile one column. Returns stats, null count, top values."

    Only the categorical branch produced the third one.
    """

    @pytest.fixture
    def shaped(self, tmp_path) -> Path:
        f = tmp_path / "shaped.csv"
        pd.DataFrame(
            {
                "n": [0, 0, 0, 0, 1, 2, 450],
                "c": ["a", "a", "b", "c", "c", "c", "d"],
                "flag": [True, False, True, True, False, True, True],
            }
        ).to_csv(f, index=False)
        return f

    def test_a_numeric_column_gets_them(self, shaped):
        p = engine.read_column_profile(str(shaped), "n")["profile"]
        assert p["kind"] == "numeric"
        assert p["top_values"]["0"] == 4, "the shape median/q25/q75 all report as 0.0"

    def test_a_boolean_column_gets_them(self, shaped):
        p = engine.read_column_profile(str(shaped), "flag")["profile"]
        assert p["kind"] == "boolean"
        assert p["top_values"] == {"True": 5, "False": 2}

    def test_a_categorical_column_is_unchanged(self, shaped):
        p = engine.read_column_profile(str(shaped), "c")["profile"]
        assert p["kind"] == "categorical"
        assert p["top_values"] == {"c": 3, "a": 2, "b": 1, "d": 1}

    def test_the_list_is_bounded(self, tmp_path):
        f = tmp_path / "wide.csv"
        pd.DataFrame({"n": list(range(500))}).to_csv(f, index=False)
        p = engine.read_column_profile(str(f), "n")["profile"]
        assert len(p["top_values"]) == 10

    def test_the_stats_still_come_too(self, shaped):
        p = engine.read_column_profile(str(shaped), "n")["profile"]
        assert p["median"] == 0.0
        assert p["null_count"] == 0
