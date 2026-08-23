"""Only two of five model writers recorded the panel's starting values.

generate_training_report embeds the model as a scoring function the page can
run, with a form of inputs so a reader can try it. The starting values come
from `feature_defaults` in the model's manifest -- a typical row, median for
numeric columns and the most common label for categorical ones.

Two sites wrote it, both in ml_basic (train_classifier, train_regressor).
tune_hyperparameters, train_with_cv and compare_models did not. A report built
on any of their models opened like this:

    campaign_platform  [Facebook Ads]      <- first option, not the modal one
    spends             [            ]      <- empty
    impressions        [            ]      <- empty
    clicks             [            ]      <- empty
    link_clicks        [            ]      <- empty
    CAMPAIGN_TYPE      Search  68.0%       <- a confident answer anyway

Measured from the DOM, not the screenshot: eleven selects carrying a value,
four number inputs carrying "". The page still says "Change any value to see
the model's answer update", so the prediction on screen reads as if it came
from the values shown, and none of the numeric ones were shown.

Same shape as leakage_warning(), which was also called only by ml_basic while
three other trainers returned a leaked 1.000 in silence. A good thing added to
one tier is not added until every sibling has it.

Found by rendering a round-7 sweep artifact and reading its input values.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from servers.ml_advanced.engine import tune_hyperparameters
from servers.ml_basic.engine import train_classifier
from servers.ml_medium.engine import compare_models, train_with_cv

CATEGORICAL_LABELS = ("Google Ads", "Facebook Ads")


def _manifest_for(tmp_path: Path, marker: str) -> dict:
    hits = [p for p in glob.glob(str(tmp_path / "**" / "*.manifest.json"), recursive=True) if marker in p]
    assert hits, f"no manifest matching {marker!r} under {tmp_path}"
    return json.loads(Path(sorted(hits)[-1]).read_text(encoding="utf-8"))


@pytest.fixture()
def trained_by_tuning(ad_data_full_with_ctr: Path, tmp_path: Path) -> dict:
    r = tune_hyperparameters(str(ad_data_full_with_ctr), "campaign_type", model="rf", task="classification")
    assert r["success"] is True, r.get("error")
    return _manifest_for(tmp_path, "tuned")


@pytest.fixture()
def trained_by_cv(ad_data_full_with_ctr: Path, tmp_path: Path) -> dict:
    r = train_with_cv(str(ad_data_full_with_ctr), "campaign_type", model="rf", task="classification")
    assert r["success"] is True, r.get("error")
    return _manifest_for(tmp_path, "_cv_")


@pytest.fixture()
def trained_by_comparison(ad_data_full_with_ctr: Path, tmp_path: Path) -> dict:
    r = compare_models(str(ad_data_full_with_ctr), "spends", task="regression", models=["rfr", "lir"])
    assert r["success"] is True, r.get("error")
    return _manifest_for(tmp_path, "best")


class TestEveryWriterRecordsThem:
    def test_tune_hyperparameters_does(self, trained_by_tuning: dict):
        assert trained_by_tuning.get("feature_defaults"), trained_by_tuning.keys()

    def test_train_with_cv_does(self, trained_by_cv: dict):
        assert trained_by_cv.get("feature_defaults"), trained_by_cv.keys()

    def test_compare_models_does(self, trained_by_comparison: dict):
        assert trained_by_comparison.get("feature_defaults"), trained_by_comparison.keys()

    def test_the_basic_trainer_still_does(self, ad_data_full_with_ctr: Path, tmp_path: Path):
        r = train_classifier(str(ad_data_full_with_ctr), "campaign_type", model="rf")
        assert r["success"] is True, r.get("error")
        assert _manifest_for(tmp_path, "rf").get("feature_defaults")


class TestTheValuesAreUsableAsPanelInputs:
    def test_the_numeric_columns_are_present_and_numeric(self, trained_by_tuning: dict):
        """The four empty boxes were spends, impressions, clicks, link_clicks."""
        defaults = trained_by_tuning["feature_defaults"]
        for column in ("spends", "impressions", "clicks"):
            assert column in defaults, defaults.keys()
            assert isinstance(defaults[column], int | float), (column, defaults[column])

    def test_a_categorical_column_keeps_its_own_label(self, trained_by_tuning: dict):
        """Not the integer _auto_preprocess encoded it to -- the panel offers
        labels, so an encoded value would not match any option."""
        platform = trained_by_tuning["feature_defaults"].get("campaign_platform")
        assert platform in CATEGORICAL_LABELS, platform

    def test_no_default_is_an_empty_string(self, trained_by_tuning: dict):
        empties = [k for k, v in trained_by_tuning["feature_defaults"].items() if v == ""]
        assert not empties, empties

    def test_it_covers_the_features_the_model_was_given(self, trained_by_tuning: dict):
        defaults = set(trained_by_tuning["feature_defaults"])
        features = set(trained_by_tuning["feature_columns"])
        assert defaults, "no defaults at all"
        assert defaults <= features, defaults - features

    def test_the_target_is_not_offered_as_an_input(self, trained_by_tuning: dict):
        assert trained_by_tuning["target_column"] not in trained_by_tuning["feature_defaults"]


class TestTheRawFrameSurvivesEncoding:
    def test_two_writers_in_one_run_agree(self, ad_data_full_with_ctr: Path, tmp_path: Path):
        """compare_models refits the winner after encoding; if df_raw had been
        overwritten by _auto_preprocess the labels would come back as ints."""
        r = compare_models(str(ad_data_full_with_ctr), "spends", task="regression", models=["rfr", "lir"])
        assert r["success"] is True, r.get("error")
        platform = _manifest_for(tmp_path, "best")["feature_defaults"].get("campaign_platform")
        assert platform in CATEGORICAL_LABELS, platform
