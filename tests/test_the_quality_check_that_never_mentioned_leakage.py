"""A data-quality pass that could not tell you the model would be worthless.

A user review trained on a 38,576-row loan book and got 0.9628. The top three
features by importance were `installment` (0.379), `total_payment` (0.298) and
`last_payment_date` (0.180) -- every one of them recorded after the loan
resolved. The model was reading repayment history, not predicting default. The
review's request was specific about where the warning belonged:

    Suggest check_data_quality add a "possible leakage: post-outcome column"
    hint when target is loan_status.

`shared/leakage.py` had already been built for the training path, so by the time
anything said "this score may not be real" three models had been fit and ranked.
The check is cheap, needs no model, and belongs in the pass a caller runs
*before* training. It was simply never wired to it: `grep -rn leak` over this
server's data tools returned nothing.

**Two decisions worth stating, because both could reasonably have gone the other
way.**

Leakage suspects are not added to `alerts` and do not move `quality_score`. If
they did, the same file would score two ways depending on whether the caller
passed `target_column` -- a number that moves with the question rather than the
data. That is precisely the defect that had `run_eda` reporting 77 and this tool
53 for one file, and it is not worth reintroducing from the other side. It is
also a different kind of finding: a null column is wrong with the data, while a
post-outcome feature is good data that is wrong for *this target*, and would
still be wrong after every alert here was cleared.

And with no `target_column` the response says the check did not run, rather than
omitting it. A quality report that scores 96 and mentions no leakage reads as
"no leakage found". The review's verdict on the receipt log -- a record that
silently holds a subset cannot be trusted as an audit log -- is the same
sentence about a different file.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from servers.ml_medium._medium_data import check_data_quality

ROWS = 400


def _write(path: Path, header: str, rows: list[str]) -> str:
    path.write_text("\n".join([header, *rows]), encoding="utf-8")
    return str(path)


@pytest.fixture()
def leaky(tmp_path: Path) -> str:
    """A loan book in miniature, with the review's own shape of leak.

    `total_payment` is drawn from a different distribution per outcome, so it
    separates the classes on its own. `last_payment_date` is null exactly when
    nothing was ever repaid. `annual_income` is honest noise and must stay
    unflagged, or the check is just a column lister.
    """
    random.seed(7)
    rows = []
    for i in range(ROWS):
        charged_off = i % 4 == 0
        income = random.gauss(60_000, 15_000)
        if charged_off:
            total_payment = random.uniform(0, 900)
            last_payment = ""
        else:
            total_payment = random.uniform(4_000, 30_000)
            last_payment = "2024-03-01"
        status = "Charged Off" if charged_off else "Fully Paid"
        rows.append(f"{income:.2f},{total_payment:.2f},{last_payment},{status}")
    return _write(tmp_path / "loans.csv", "annual_income,total_payment,last_payment_date,loan_status", rows)


@pytest.fixture()
def clean(tmp_path: Path) -> str:
    """Same shape, no leak: every feature is noise the target cannot be read from."""
    random.seed(9)
    rows = []
    for i in range(ROWS):
        rows.append(
            f"{random.gauss(60_000, 15_000):.2f},{random.gauss(500, 120):.2f},"
            f"{random.choice('ABC')},{'Charged Off' if i % 4 == 0 else 'Fully Paid'}"
        )
    return _write(tmp_path / "clean.csv", "annual_income,monthly_spend,grade,loan_status", rows)


class TestTheFixturesAreWhatTheAssertionsAssume:
    """Without these, every test below could pass against an empty result."""

    def test_the_leaky_file_scores_and_loads(self, leaky):
        result = check_data_quality(leaky)
        assert result["success"] is True, result.get("error")
        assert result["row_count"] == ROWS

    def test_the_clean_file_scores_and_loads(self, clean):
        result = check_data_quality(clean)
        assert result["success"] is True, result.get("error")
        assert result["row_count"] == ROWS


class TestItNamesTheColumnThatAlreadyKnowsTheAnswer:
    def test_a_post_outcome_feature_is_flagged(self, leaky):
        result = check_data_quality(leaky, target_column="loan_status")
        flagged = {s["feature"] for s in result["leakage_suspects"]}
        assert "total_payment" in flagged, result["leakage_suspects"]

    def test_a_field_only_filled_in_for_one_outcome_is_flagged(self, leaky):
        result = check_data_quality(leaky, target_column="loan_status")
        flagged = {s["feature"] for s in result["leakage_suspects"]}
        assert "last_payment_date" in flagged, result["leakage_suspects"]

    def test_an_honest_feature_is_left_alone(self, leaky):
        """The check has to be able to say no, or it says nothing."""
        result = check_data_quality(leaky, target_column="loan_status")
        assert "annual_income" not in {s["feature"] for s in result["leakage_suspects"]}

    def test_the_target_is_never_a_suspect_of_itself(self, leaky):
        result = check_data_quality(leaky, target_column="loan_status")
        assert "loan_status" not in {s["feature"] for s in result["leakage_suspects"]}

    def test_every_suspect_carries_its_evidence(self, leaky):
        """A verdict with no measurement behind it is the thing to avoid."""
        for suspect in check_data_quality(leaky, target_column="loan_status")["leakage_suspects"]:
            assert suspect["signals"], suspect
            for signal in suspect["signals"]:
                assert signal["reason"]
                assert signal["evidence"].strip()
                assert signal["confidence"] in {"high", "medium", "hint"}

    def test_the_note_reads_as_a_suspicion_not_a_verdict(self, leaky):
        note = check_data_quality(leaky, target_column="loan_status")["leakage_note"]
        assert "may not be real" in note or "Possible target leakage" in note

    def test_a_clean_file_says_so_rather_than_going_quiet(self, clean):
        result = check_data_quality(clean, target_column="loan_status")
        assert result["leakage_suspects"] == []
        assert result["leakage_count"] == 0
        assert "No feature looks like" in result["leakage_note"]


class TestTheScoreDoesNotMoveWithTheQuestionAsked:
    def test_naming_a_target_changes_no_score(self, leaky):
        without = check_data_quality(leaky)
        with_target = check_data_quality(leaky, target_column="loan_status")
        assert without["quality_score"] == with_target["quality_score"]

    def test_naming_a_target_adds_no_alerts(self, leaky):
        without = check_data_quality(leaky)
        with_target = check_data_quality(leaky, target_column="loan_status")
        assert without["alerts_count"] == with_target["alerts_count"]
        assert without["alerts"] == with_target["alerts"]

    def test_leakage_is_reported_outside_the_alert_list(self, leaky):
        result = check_data_quality(leaky, target_column="loan_status")
        assert result["leakage_suspects"]
        assert not [a for a in result["alerts"] if "leak" in str(a).lower()]


class TestSilenceIsNotACleanBillOfHealth:
    def test_without_a_target_it_says_the_check_did_not_run(self, leaky):
        result = check_data_quality(leaky)
        assert "not run" in result["leakage_check"]
        assert "target_column" in result["leakage_check"]

    def test_without_a_target_it_claims_nothing_about_leakage(self, leaky):
        result = check_data_quality(leaky)
        assert "leakage_suspects" not in result
        assert "leakage_note" not in result

    def test_with_a_target_the_did_not_run_line_is_gone(self, leaky):
        assert "leakage_check" not in check_data_quality(leaky, target_column="loan_status")

    def test_the_handover_context_carries_the_count(self, leaky):
        """A next step that reads only `context` must not get the clean half."""
        result = check_data_quality(leaky, target_column="loan_status")
        assert "leakage" in str(result["context"]).lower()


class TestAWrongTargetIsRefusedNotIgnored:
    def test_an_unknown_column_fails(self, leaky):
        result = check_data_quality(leaky, target_column="loan_stats")
        assert result["success"] is False
        assert "loan_stats" in result["error"]

    def test_the_refusal_lists_what_it_could_have_been(self, leaky):
        result = check_data_quality(leaky, target_column="loan_stats")
        assert "loan_status" in result["hint"]

    def test_it_does_not_score_the_file_anyway(self, leaky):
        """Answering the question that was not asked is the failure mode."""
        assert "quality_score" not in check_data_quality(leaky, target_column="nope")


class TestItSurvivesTheShapesThatAreNotALoanBook:
    def test_a_multiclass_target_does_not_crash(self, tmp_path):
        random.seed(3)
        rows = [f"{random.random():.4f},{random.choice('xyz')}" for _ in range(120)]
        path = _write(tmp_path / "multi.csv", "feature,target", rows)
        result = check_data_quality(path, target_column="target")
        assert result["success"] is True, result.get("error")

    def test_a_single_column_file_has_no_features_to_suspect(self, tmp_path):
        rows = [f"{'a' if i % 2 else 'b'}" for i in range(80)]
        path = _write(tmp_path / "one.csv", "target", rows)
        result = check_data_quality(path, target_column="target")
        assert result["success"] is True, result.get("error")
        assert result["leakage_suspects"] == []

    def test_too_few_rows_to_measure_still_answers(self, tmp_path):
        """Below the group minimum nothing is measurable; that is not an error."""
        rows = ["1.0,10.0,Fully Paid", "2.0,20.0,Charged Off", "3.0,30.0,Fully Paid"]
        path = _write(tmp_path / "tiny.csv", "a,total_payment,loan_status", rows)
        result = check_data_quality(path, target_column="loan_status")
        assert result["success"] is True, result.get("error")
        # The name-based hint can still fire; nothing measured may accompany it.
        for suspect in result["leakage_suspects"]:
            assert suspect["confidence"] == "hint"
