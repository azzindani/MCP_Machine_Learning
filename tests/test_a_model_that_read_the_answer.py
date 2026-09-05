"""0.9628 on a loan outcome, from three columns recorded after it.

A user review trained on a loan book and reported:

    dtc 0.9628; installment 0.379 + total_payment 0.298 + last_payment_date
    0.180. `total_payment`/`installment` are post-outcome; 96.3% accuracy likely
    leaks. Tool is honest about importance -- agent must not ship without
    time-split + drop of `id`/`member_id`/`total_payment`.

Every number `compare_models` returned was correct. It ranked three models,
named the winner, and listed the features it leaned on -- and an agent with no
domain knowledge ships a 96% model that predicts nothing, because those three
columns are filled in *after* the loan resolves.

The existing guard could not catch it. `leakage_warning` fires at
`_NEAR_PERFECT_SCORE = 0.999` and looks for a feature that determines the
target exactly. 0.9628 is nowhere near that, and no single column here
determines the outcome: the leak is statistical, not functional. A check tuned
for "this is obviously impossible" misses "this is quietly meaningless", and the
second is the one that gets shipped.

What is detectable without knowing what a loan is:

* a single feature separating the classes nearly as well as the whole model;
* a feature whose *missingness* predicts the class -- the signature of a field
  populated for one outcome only, which is exactly `last_payment_date`;
* a name from the vocabulary of post-outcome accounting.

The first two are measured. The third is labelled a hint, because a column
called `total_payment` might be a budget rather than a settlement, and a guess
dressed as a finding is the defect this is meant to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.leakage import (
    MISSINGNESS_GAP,
    SINGLE_FEATURE_AUC,
    leakage_note,
    leakage_suspects,
    split_provenance,
)


@pytest.fixture
def loan_book():
    """A small stand-in for the review's file, with the same three shapes.

    `credit_score` is an honest predictor. `total_payment` is post-outcome and
    separates the classes almost perfectly. `last_payment_date` is null exactly
    for the defaults. `noise` is neither.
    """
    rng = np.random.default_rng(7)
    n = 600
    defaulted = rng.random(n) < 0.25
    # Deliberately weak separation, because that is what an honest predictor
    # looks like: a real credit score reaches roughly 0.70 AUC against default
    # on its own. The first version of this fixture used a 2.3-sigma gap, which
    # put credit_score at ~0.95 and made it indistinguishable from the leaks --
    # a bad fixture, not a bad threshold.
    credit_score = np.where(defaulted, rng.normal(645, 90, n), rng.normal(690, 90, n))
    total_payment = np.where(defaulted, rng.normal(500, 120, n), rng.normal(9_000, 800, n))
    last_payment_date = np.where(defaulted, np.nan, rng.normal(100, 5, n))
    return pd.DataFrame(
        {
            "loan_status": np.where(defaulted, "Charged Off", "Fully Paid"),
            "credit_score": credit_score,
            "total_payment": total_payment,
            "last_payment_date": last_payment_date,
            "noise": rng.normal(0, 1, n),
        }
    )


FEATURES = ["credit_score", "total_payment", "last_payment_date", "noise"]


def test_a_post_outcome_column_is_flagged_at_a_score_the_old_guard_ignores(loan_book):
    """The whole point: this fires at 0.96, not only at 0.999."""
    suspects = leakage_suspects(loan_book, "loan_status", FEATURES)
    flagged = {s["feature"] for s in suspects}
    assert "total_payment" in flagged


def test_the_evidence_is_a_measurement_not_an_opinion(loan_book):
    suspects = leakage_suspects(loan_book, "loan_status", FEATURES)
    tp = next(s for s in suspects if s["feature"] == "total_payment")
    auc_signal = next(g for g in tp["signals"] if g["reason"] == "alone_predicts_target")
    assert auc_signal["auc"] >= SINGLE_FEATURE_AUC
    assert "AUC" in auc_signal["evidence"]


def test_a_field_only_filled_in_for_one_outcome_is_caught_by_its_nulls(loan_book):
    """`last_payment_date` is null exactly when nothing was ever repaid."""
    suspects = leakage_suspects(loan_book, "loan_status", FEATURES)
    lpd = next(s for s in suspects if s["feature"] == "last_payment_date")
    gap = next(g for g in lpd["signals"] if g["reason"] == "missingness_tracks_target")
    assert gap["missingness_gap"] >= MISSINGNESS_GAP
    assert "recorded after that outcome is known" in gap["evidence"]


def test_an_honest_predictor_is_not_flagged(loan_book):
    """A guard that flags everything is a guard nobody reads."""
    suspects = leakage_suspects(loan_book, "loan_status", FEATURES)
    flagged = {s["feature"] for s in suspects}
    assert "noise" not in flagged
    assert "credit_score" not in flagged, "a real predictor must survive the check"


def test_a_name_only_match_is_labelled_a_hint():
    """Nothing is measured, so nothing may be claimed."""
    df = pd.DataFrame({"y": [0, 1] * 40, "settlement_amount": range(80)})
    suspects = leakage_suspects(df, "y", ["settlement_amount"])
    assert suspects, "the name alone is worth mentioning"
    hint = suspects[0]
    assert hint["confidence"] == "hint"
    assert "hint from the column name only" in hint["signals"][0]["evidence"]


def test_the_note_says_what_to_do_next(loan_book):
    suspects = leakage_suspects(loan_book, "loan_status", FEATURES)
    note = leakage_note(suspects, 0.9628)
    assert "0.9628" in note
    assert "split on time" in note
    assert "Re-train without them" in note


def test_a_name_only_finding_does_not_claim_a_measurement():
    df = pd.DataFrame({"y": [0, 1] * 40, "final_balance": range(80)})
    note = leakage_note(leakage_suspects(df, "y", ["final_balance"]))
    assert "Nothing was measured" in note


def test_nothing_suspect_says_nothing():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.integers(0, 2, 300), "a": rng.normal(size=300)})
    assert leakage_suspects(df, "y", ["a"]) == []
    assert leakage_note([]) == ""


def test_multiclass_is_declined_rather_than_guessed():
    """Rank AUC and a rate gap are two-class statistics."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"y": rng.integers(0, 3, 300), "a": rng.normal(size=300)})
    suspects = leakage_suspects(df, "y", ["a"])
    assert all(s["confidence"] == "hint" for s in suspects), "no measured claim on 3 classes"


def test_a_missing_target_is_not_an_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert leakage_suspects(df, "nope", ["a"]) == []


def test_tiny_groups_are_not_measured():
    """Below MIN_GROUP the statistic is noise wearing a number's clothes."""
    df = pd.DataFrame({"y": [0] * 5 + [1] * 5, "a": range(10)})
    assert leakage_suspects(df, "y", ["a"]) == []


# ---------------------------------------------------------------------------
# the manifest half of A1
# ---------------------------------------------------------------------------


def test_the_split_is_recorded_with_the_score():
    prov = split_provenance(test_size=0.2, random_state=42)
    assert prov["test_size"] == 0.2
    assert prov["random_state"] == 42
    assert prov["time_ordered_split"] is False


def test_a_random_split_says_what_it_assumes():
    """A 0.9628 from a random split of time-ordered rows is a different claim."""
    prov = split_provenance(test_size=0.2, random_state=0)
    assert "learn from the future" in prov["split_note"]


def test_a_time_ordered_split_needs_no_caveat():
    prov = split_provenance(test_size=0.2, random_state=0, time_ordered=True)
    assert "split_note" not in prov
