"""`evaluate_model` is where a number gets quoted, and it said nothing.

Every trainer in this repo already flags leakage -- `_basic_train.py`,
`_medium_train.py`, and as of today `check_data_quality`. `evaluate_model` takes
a `target_column`, loads a labelled test frame, and reports an accuracy, and it
was the one target-taking tool in the fleet that never ran the check.

It matters because of the path that skips every warning already in place: a
model is trained somewhere, `export_model` ships it, and a different caller
evaluates it on a fresh test file. Nothing in that sequence shows the training
warning, and the score they get back is the one they will quote. If the test
frame carries post-outcome columns, that number is inflated at exactly the
moment somebody believes it.

**Measured before the encoding loop, on purpose.** `evaluate_model` maps
categoricals through the model's stored encoding and calls `.fillna(-1)`, which
turns a null into the number -1. One of the three signals is that a column's
*missingness* tracks the target -- `last_payment_date` is null exactly when
nothing was ever repaid -- and after that loop the signal can never fire. Run in
the wrong order the check would lose a third of its reach and still report a
result, which is worse than not running it.

**Only the model's own features are examined.** The question is not "is anything
in this file suspicious" but "did this score come from a feature that already
knows the answer", so the candidate list is `feature_columns` from the manifest.

The note carries the score it is casting doubt on -- "Score 0.9628 may not be
real" -- which is the review's sentence, and it only exists once the score does.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from servers.ml_basic.engine import train_classifier
from servers.ml_medium._medium_data import evaluate_model

ROWS = 400


LEAKY_HEADER = "annual_income,total_payment,last_payment_date,loan_status"
# Honest names as well as honest data. The first pass at this file kept
# `total_payment` on the clean fixture and the name-based hint fired -- correctly,
# since that hint is about the name and says so. "Clean" has to mean both, or the
# fixture is testing the wrong thing.
CLEAN_HEADER = "annual_income,monthly_spend,last_contact_date,loan_status"


def _rows(seed: int, leaky: bool, header: str = "") -> list[str]:
    random.seed(seed)
    out = [header or (LEAKY_HEADER if leaky else CLEAN_HEADER)]
    for i in range(ROWS):
        off = i % 4 == 0
        income = random.gauss(60_000, 15_000)
        if leaky:
            paid = random.uniform(0, 900) if off else random.uniform(4_000, 30_000)
            last = "" if off else "2024-03-01"
        else:
            paid = random.gauss(500, 120)
            last = "2024-03-01"
        out.append(f"{income:.2f},{paid:.2f},{last},{'Charged Off' if off else 'Fully Paid'}")
    return out


@pytest.fixture()
def leaky_model(tmp_path: Path):
    """A model trained on the review's shape, plus a test file of the same."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("\n".join(_rows(7, leaky=True)), encoding="utf-8")
    model = tmp_path / "m.pkl"
    result = train_classifier(str(train_csv), "loan_status", model="rf", output_path=str(model))
    assert result["success"] is True, result.get("error")

    test_csv = tmp_path / "test.csv"
    test_csv.write_text("\n".join(_rows(11, leaky=True)), encoding="utf-8")
    return str(model), str(test_csv)


@pytest.fixture()
def clean_model(tmp_path: Path):
    train_csv = tmp_path / "train_clean.csv"
    train_csv.write_text("\n".join(_rows(3, leaky=False)), encoding="utf-8")
    model = tmp_path / "clean.pkl"
    result = train_classifier(str(train_csv), "loan_status", model="rf", output_path=str(model))
    assert result["success"] is True, result.get("error")

    test_csv = tmp_path / "test_clean.csv"
    test_csv.write_text("\n".join(_rows(5, leaky=False)), encoding="utf-8")
    return str(model), str(test_csv)


class TestTheFixturesAreWhatTheAssertionsAssume:
    def test_the_leaky_model_evaluates(self, leaky_model):
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_status")
        assert result["success"] is True, result.get("error")
        assert result["test_rows"] == ROWS

    def test_the_clean_model_evaluates(self, clean_model):
        model, test = clean_model
        result = evaluate_model(model, test, "loan_status")
        assert result["success"] is True, result.get("error")

    def test_the_model_records_the_features_the_check_will_use(self, leaky_model):
        model, _test = leaky_model
        manifest = json.loads(Path(model).with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert "total_payment" in manifest["feature_columns"]


class TestTheScoreArrivesWithItsCaveat:
    def test_the_post_outcome_feature_is_named(self, leaky_model):
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_status")
        assert "total_payment" in {s["feature"] for s in result["leakage_suspects"]}

    def test_the_missingness_signal_survives_the_encoding_loop(self, leaky_model):
        """The ordering guard. Encoding fills nulls with -1 and erases this."""
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_status")
        reasons = {sig["reason"] for s in result["leakage_suspects"] for sig in s["signals"]}
        assert "missingness_tracks_target" in reasons, result["leakage_suspects"]

    def test_the_note_quotes_the_score_it_doubts(self, leaky_model):
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_status")
        assert "may not be real" in result["leakage_note"]
        assert str(result["metrics"]["accuracy"]) in result["leakage_note"]

    def test_the_count_rides_on_the_context_line(self, leaky_model):
        """A handover reading only `context` must not get the clean half."""
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_status")
        assert "leakage" in str(result["context"]).lower()

    def test_progress_says_it_too(self, leaky_model):
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_status")
        assert any("leakage" in str(p).lower() for p in result["progress"])


class TestItCanAlsoSayNo:
    def test_a_clean_model_gets_a_stated_result(self, clean_model):
        model, test = clean_model
        result = evaluate_model(model, test, "loan_status")
        assert result["leakage_count"] == 0
        assert "No feature this model uses" in result["leakage_note"]

    def test_a_clean_model_names_no_suspects(self, clean_model):
        model, test = clean_model
        result = evaluate_model(model, test, "loan_status")
        assert result["leakage_suspects"] == []


class TestASuspiciousNameOverHonestDataIsOnlyEverAHint:
    """The distinction the first draft of this file got wrong.

    A column called `total_payment` might be a budget rather than a settlement.
    The name earns a hint and nothing more; if it also separated the classes or
    tracked the target through its nulls, that would be a measurement, and the
    confidence would say so.
    """

    @pytest.fixture()
    def honest_data_suspicious_names(self, tmp_path: Path):
        train_csv = tmp_path / "train_named.csv"
        train_csv.write_text("\n".join(_rows(3, leaky=False, header=LEAKY_HEADER)), encoding="utf-8")
        model = tmp_path / "named.pkl"
        result = train_classifier(str(train_csv), "loan_status", model="rf", output_path=str(model))
        assert result["success"] is True, result.get("error")
        test_csv = tmp_path / "test_named.csv"
        test_csv.write_text("\n".join(_rows(5, leaky=False, header=LEAKY_HEADER)), encoding="utf-8")
        return str(model), str(test_csv)

    def test_it_is_still_raised(self, honest_data_suspicious_names):
        model, test = honest_data_suspicious_names
        result = evaluate_model(model, test, "loan_status")
        assert "total_payment" in {s["feature"] for s in result["leakage_suspects"]}

    def test_but_only_as_a_hint(self, honest_data_suspicious_names):
        model, test = honest_data_suspicious_names
        result = evaluate_model(model, test, "loan_status")
        assert {s["confidence"] for s in result["leakage_suspects"]} == {"hint"}

    def test_and_the_note_says_nothing_was_measured(self, honest_data_suspicious_names):
        model, test = honest_data_suspicious_names
        note = evaluate_model(model, test, "loan_status")["leakage_note"]
        assert "Nothing was measured" in note


class TestTheMetricsAreUntouched:
    def test_the_check_does_not_change_the_score(self, leaky_model):
        """Two runs of the same evaluation must agree exactly."""
        model, test = leaky_model
        first = evaluate_model(model, test, "loan_status")
        second = evaluate_model(model, test, "loan_status")
        assert first["metrics"] == second["metrics"]

    def test_the_metrics_still_carry_what_they_always_did(self, leaky_model):
        model, test = leaky_model
        metrics = evaluate_model(model, test, "loan_status")["metrics"]
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "confusion_matrix" in metrics


class TestTheTrainingWarningTravelsWithTheModel:
    def test_a_flagged_model_says_so_when_evaluated_elsewhere(self, leaky_model, tmp_path):
        """The path that skips every warning: train here, evaluate there."""
        model, test = leaky_model
        manifest = Path(model).with_suffix(".manifest.json")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not data.get("leakage_warning"):
            pytest.skip("this fixture did not trip the trainer's own warning")
        result = evaluate_model(model, test, "loan_status")
        assert result["training_leakage_warning"]

    def test_an_unflagged_model_adds_no_such_key(self, clean_model):
        model, test = clean_model
        manifest = Path(model).with_suffix(".manifest.json")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("leakage_warning"):
            pytest.skip("this fixture did trip the trainer's warning")
        assert "training_leakage_warning" not in evaluate_model(model, test, "loan_status")


class TestTheFailurePathsAreUnchanged:
    def test_a_missing_target_still_fails_the_same_way(self, leaky_model):
        model, test = leaky_model
        result = evaluate_model(model, test, "loan_stats")
        assert result["success"] is False
        assert "not in test file" in result["error"]

    def test_a_missing_model_still_fails(self, tmp_path, leaky_model):
        _model, test = leaky_model
        result = evaluate_model(str(tmp_path / "nope.pkl"), test, "loan_status")
        assert result["success"] is False
