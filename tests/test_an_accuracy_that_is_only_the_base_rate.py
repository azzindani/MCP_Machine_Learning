"""An accuracy that is really the base rate must not be reported bare.

train_regressor has always explained a *suspiciously perfect* score. The mirror
case had nothing: on a 5.2% positive target a RandomForest scored accuracy=0.950
and f1_weighted=0.9256 -- both exactly correct -- with TP=0 out of 10 positives
and AUC 0.29, which is worse than guessing. The result led with 0.950, the
progress line printed 0.950, and `context.summary` carried 0.950 forward to
whatever ran next.

Both directions get a caveat now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def baseline_warning(*args, **kwargs):
    """Imported lazily so the train_classifier tests collect without it."""
    from shared.ml_utils import baseline_warning as impl

    return impl(*args, **kwargs)


@pytest.fixture
def imbalanced(tmp_path):
    """95/5 target with pure-noise features: nothing to learn, 0.95 available."""
    rng = np.random.default_rng(7)
    n = 1000
    y = (rng.random(n) < 0.05).astype(int)
    df = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=["f1", "f2", "f3", "f4"])
    df["target"] = y
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_a_model_that_never_predicts_the_positive_class_is_called_out(imbalanced):
    from servers.ml_basic._basic_train import train_classifier

    r = train_classifier(imbalanced, target_column="target", model="rf", test_size=0.2, random_state=42)
    assert r["success"] is True

    cm = r["metrics"]["confusion_matrix"]
    # The premise: accuracy looks fine and the model is useless.
    assert r["metrics"]["accuracy"] >= 0.9
    assert cm["TP"] == 0 and cm["FN"] > 0

    warnings = [p for p in r["progress"] if p.get("msg") == "Accuracy overstates this model"]
    assert warnings, "a model catching none of the positives reported its accuracy with no caveat"
    detail = warnings[0]["detail"]
    assert "never predicts" in detail
    assert "most common class" in detail


def test_the_caveat_names_a_concrete_next_step(imbalanced):
    from servers.ml_basic._basic_train import train_classifier

    r = train_classifier(imbalanced, target_column="target", model="rf", test_size=0.2, random_state=42)
    detail = [p for p in r["progress"] if p.get("msg") == "Accuracy overstates this model"][0]["detail"]
    assert "class_weight" in detail or "confusion matrix" in detail


# --- the helper's own boundaries --------------------------------------------


def test_a_genuinely_good_model_gets_no_caveat():
    y_true = np.array([0] * 100 + [1] * 100)
    y_pred = np.array([0] * 95 + [1] * 5 + [1] * 95 + [0] * 5)
    assert baseline_warning(y_true, y_pred, 0.95, auc=0.97) == ""


def test_beating_the_base_rate_on_an_imbalanced_target_is_not_flagged():
    # 90/10, and the model actually finds most of the positives.
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 90 + [1] * 8 + [0] * 2)
    assert baseline_warning(y_true, y_pred, 0.98) == ""


def test_a_majority_class_predictor_is_flagged():
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.zeros(100, dtype=int)
    msg = baseline_warning(y_true, y_pred, 0.95)
    assert "never predicts" in msg
    assert "all 5 of those rows" in msg


def test_an_auc_below_chance_is_flagged():
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5)
    msg = baseline_warning(y_true, y_pred, 0.9, auc=0.29)
    assert "worse than random" in msg


def test_a_single_class_test_set_says_nothing():
    # Nothing to compare a base rate against; must not invent a complaint.
    y = np.zeros(50, dtype=int)
    assert baseline_warning(y, y, 1.0) == ""


def test_an_empty_test_set_says_nothing():
    assert baseline_warning(np.array([]), np.array([]), 0.0) == ""


def test_predictions_are_optional_for_the_base_rate_check():
    # train_with_cv has a mean score and no single fold's predictions.
    y_true = np.array([0] * 95 + [1] * 5)
    msg = baseline_warning(y_true, None, 0.95)
    assert "most common class" in msg
    assert "never predicts" not in msg
