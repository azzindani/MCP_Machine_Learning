"""Round 27: the leak the check could not see, and the caller could not remove.

Two findings that compound into one failure, found while doing an ordinary task
-- predict `clicks` on the fleet's ad dataset.

**The check was blind here.** `train_regressor` scored r2 0.983 with
`link_clicks` in the feature set. `link_clicks <= clicks` in 100.0% of the
16,288 rows that have it, because a link click *is* a click: the model was
reading part of its own answer. The manifest said `"leakage_warning": ""`.

Two reasons, both fixed here. `ml_basic` ran only `leakage_warning`, the 0.999
exact-determination check that `shared/leakage.py` was written to supersede --
never `leakage_suspects`, which had lived in `ml_medium` since the credit-risk
review. And `leakage_suspects` itself measured nothing on a continuous target:
rank AUC and the missingness gap are two-class statistics, so `_as_binary`
returned None and only the post-outcome name regex could fire. That regex is
credit vocabulary (`total_payment`, `chargeoff`, `settlement`); no ad, traffic
or revenue column will ever match it.

**And the caller could not act.** Having found the leak by hand, the obvious
next call was

    train_regressor(target_column="clicks", feature_columns=[...without it...])

which returned `success: true` with metrics identical to four decimal places,
because `feature_columns` was not a parameter and the bundled FastMCP dropped it
in silence. The review's instruction was "drop id/member_id/total_payment"; the
drop was never expressible.

**Calibration, measured rather than guessed.** The first draft of the
continuous signal used Spearman >= 0.90, mirroring SINGLE_FEATURE_AUC. On this
data that flagged `spends` (rho 0.923) -- an honest causal predictor of clicks --
and missed `link_clicks` entirely, whose 89.7% zeros crush its Spearman to
0.261 while its Pearson is 0.926. Both thresholds moved, and both directions
are asserted below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.feature_select import select_features  # noqa: E402
from shared.leakage import SINGLE_FEATURE_RHO, leakage_note, leakage_suspects  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def ads() -> pd.DataFrame:
    """The real 16,834-row Ad_Data.csv, not a synthetic stand-in.

    A synthetic fixture is what got the thresholds wrong the first time: built
    with `spends = clicks * 3 + noise` it correlates at rho 0.999, which is a
    property of the generator rather than of advertising, and every threshold
    tuned against it would be meaningless. The real file has spends at 0.923
    and link_clicks at 0.261 rank / 0.926 linear -- the exact spread that
    decides whether this check is useful or noisy.
    """
    return pd.read_csv(FIXTURES / "ad_data_full.csv")


FEATURES = ["spends", "impressions", "link_clicks", "campaign_platform", "audience_type", "device", "age"]


class TestAComponentOfTheTargetIsFound:
    def test_link_clicks_is_flagged(self, ads):
        suspects = leakage_suspects(ads, "clicks", FEATURES)
        assert "link_clicks" in {s["feature"] for s in suspects}

    def test_and_the_reason_is_containment_not_a_guess(self, ads):
        suspects = leakage_suspects(ads, "clicks", FEATURES)
        found = next(s for s in suspects if s["feature"] == "link_clicks")
        assert found["reason"] == "component_of_target"
        assert found["confidence"] == "high"
        assert found["signals"][0]["containment"] == 1.0

    def test_the_evidence_names_the_measurement(self, ads):
        suspects = leakage_suspects(ads, "clicks", FEATURES)
        evidence = next(s for s in suspects if s["feature"] == "link_clicks")["signals"][0]["evidence"]
        assert "100.0% of rows" in evidence

    def test_a_zero_inflated_component_is_not_hidden_by_rank_ties(self, ads):
        """The calibration that mattered: ranks say 0.26, the truth is containment."""
        from shared.leakage import _as_continuous, _rank_rho

        rho = _rank_rho(ads["link_clicks"], _as_continuous(ads["clicks"]))
        assert rho < 0.80, "precondition: ties crush the rank correlation"
        suspects = leakage_suspects(ads, "clicks", ["link_clicks"])
        assert suspects, "a rank-only gate would have missed this"

    def test_the_note_tells_the_caller_what_to_do(self, ads):
        suspects = leakage_suspects(ads, "clicks", FEATURES)
        note = leakage_note(suspects, 0.983)
        assert "link_clicks" in note and "Re-train without them" in note


class TestAnHonestPredictorIsNotAccused:
    def test_spend_is_not_leakage(self, ads):
        suspects = leakage_suspects(ads, "clicks", FEATURES)
        assert "spends" not in {s["feature"] for s in suspects}

    def test_impressions_is_not_leakage(self, ads):
        suspects = leakage_suspects(ads, "clicks", FEATURES)
        assert "impressions" not in {s["feature"] for s in suspects}

    def test_the_threshold_is_far_above_a_strong_correlation(self):
        """0.90 is where an AUC becomes an encoding; a correlation of 0.90 is not."""
        assert SINGLE_FEATURE_RHO >= 0.95

    def test_dropping_the_component_leaves_nothing_suspect(self, ads):
        clean = ads.drop(columns=["link_clicks"])
        assert leakage_suspects(clean, "clicks", [c for c in FEATURES if c != "link_clicks"]) == []


class TestTheBinaryPathIsUnchanged:
    def test_a_two_class_target_still_uses_auc(self, ads):
        suspects = leakage_suspects(ads, "campaign_platform", [c for c in FEATURES if c != "campaign_platform"])
        assert isinstance(suspects, list)

    def test_a_perfect_separator_is_still_caught(self):
        frame = pd.DataFrame(
            {
                "outcome": ["yes"] * 100 + ["no"] * 100,
                "tell": list(range(100)) + list(range(500, 600)),
                "noise": [i % 7 for i in range(200)],
            }
        )
        suspects = leakage_suspects(frame, "outcome", ["tell", "noise"])
        assert "tell" in {s["feature"] for s in suspects}
        assert "noise" not in {s["feature"] for s in suspects}

    def test_a_small_ordinal_code_is_not_read_as_a_measurement(self):
        """A 0-4 severity column is a label, not a quantity; no rank rho on it."""
        frame = pd.DataFrame({"severity": [i % 5 for i in range(200)], "other": list(range(200))})
        assert leakage_suspects(frame, "severity", ["other"]) == []


class TestTheCallerCanDropIt:
    def test_feature_columns_narrows_the_set(self, ads):
        cols, note, err = select_features(ads, "clicks", ["spends", "impressions"], None)
        assert err is None
        assert cols == ["spends", "impressions"]
        assert note.startswith("2 of ")

    def test_exclude_columns_removes_from_the_default(self, ads):
        cols, note, err = select_features(ads, "clicks", None, ["link_clicks"])
        assert err is None
        assert "link_clicks" not in cols
        assert "spends" in cols and "impressions" in cols

    def test_the_default_is_still_every_column_but_the_target(self, ads):
        cols, note, err = select_features(ads, "clicks", None, None)
        assert err is None and note == ""
        assert set(cols) == set(ads.columns) - {"clicks"}

    def test_an_unknown_column_is_refused_with_the_real_ones(self, ads):
        _, _, err = select_features(ads, "clicks", ["spends", "spendz"], None)
        assert err is not None
        assert "spendz" in err["error"] and "spends" in err["hint"]

    def test_both_at_once_is_refused_rather_than_half_honoured(self, ads):
        _, _, err = select_features(ads, "clicks", ["spends"], ["link_clicks"])
        assert err is not None
        assert "not both" in err["error"]

    def test_the_target_cannot_be_its_own_feature(self, ads):
        _, _, err = select_features(ads, "clicks", ["clicks", "spends"], None)
        assert err is not None
        assert "target" in err["error"]

    def test_excluding_everything_is_refused(self, ads):
        _, _, err = select_features(ads, "clicks", None, [c for c in ads.columns if c != "clicks"])
        assert err is not None
        assert "every feature" in err["error"]

    @pytest.mark.parametrize("tool", ["train_regressor", "train_classifier"])
    def test_both_trainers_declare_the_parameters(self, tool):
        """The finding was that the argument existed nowhere; the schema must show it."""
        import inspect

        import servers.ml_basic.server as server

        params = inspect.signature(getattr(server, tool)).parameters
        assert "feature_columns" in params
        assert "exclude_columns" in params

    @pytest.mark.parametrize("tool", ["train_with_cv", "compare_models"])
    def test_the_medium_trainers_declare_them_too(self, tool):
        import inspect

        import servers.ml_medium.server as server

        params = inspect.signature(getattr(server, tool)).parameters
        assert "feature_columns" in params
        assert "exclude_columns" in params
