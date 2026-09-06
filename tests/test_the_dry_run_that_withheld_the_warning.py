"""The one mode a careful caller uses was the only one that said nothing.

Round 27 built the leakage check: `leakage_suspects` names a feature that is a
component of the target, and `leakage_note` says the score may not be real.
Round 28 confirmed it works -- on Ad_Data.csv with `target_column="clicks"` it
names `link_clicks` at containment 1.0 and calls the 0.983 into question.

Then it asked for the same thing with `dry_run=True` and got:

    feature_columns: [... "spends", "impressions", "link_clicks"]
    would_train: true
    (no leakage_suspects, no leakage_note)

`servers/ml_basic/_basic_train.py` returned from the dry-run branch at line 169
(classifier) and 497 (regressor); `leakage_suspects` was not called until 304
and 591. So the mode whose entire purpose is "tell me what this would do before
I commit" was the one mode that withheld the warning -- and it printed the
leaking column among `feature_columns` beside `would_train: true`, which reads
as approval.

Nothing needed computing that was not already in hand: `df`, the target and
`feature_cols` are all decided before the dry-run return, and `feature_cols` is
the very list the dry run prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "servers" / "ml_basic"), str(ROOT / "servers" / "ml_medium")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from servers.ml_basic._basic_train import train_classifier, train_regressor  # noqa: E402


@pytest.fixture
def leaky(tmp_path):
    """`part` is bounded by `whole` and moves with it -- a component of the target."""
    rows = 400
    whole = [float(i % 37 + 1) for i in range(rows)]
    part = [w * 0.6 for w in whole]
    noise = [float((i * 7) % 11) for i in range(rows)]
    path = tmp_path / "leaky.csv"
    pd.DataFrame({"whole": whole, "part": part, "noise": noise}).to_csv(path, index=False)
    return str(path)


class TestTheDryRunCarriesTheWarning:
    def test_a_live_regressor_names_the_component(self, leaky, tmp_path):
        r = train_regressor(leaky, "whole", "lir", output_path=str(tmp_path / "m.pkl"), random_state=42)
        assert r["success"] is True, r.get("error")
        named = [s["feature"] for s in r.get("leakage_suspects", [])]
        assert "part" in named, r.get("leakage_suspects")

    def test_the_dry_run_names_it_too(self, leaky):
        r = train_regressor(leaky, "whole", "lir", dry_run=True, random_state=42)
        assert r["success"] is True and r["dry_run"] is True
        named = [s["feature"] for s in r.get("leakage_suspects", [])]
        assert "part" in named, (
            "dry_run listed the leaking feature under feature_columns and said would_train: true, "
            "without saying it leaks"
        )

    def test_the_dry_run_carries_the_note_as_well(self, leaky):
        r = train_regressor(leaky, "whole", "lir", dry_run=True, random_state=42)
        assert r.get("leakage_note"), r
        assert "part" in r["leakage_note"]

    def test_the_classifier_dry_run_does_the_same(self, leaky, tmp_path):
        frame = pd.read_csv(leaky)
        frame["label"] = (frame["whole"] > frame["whole"].median()).astype(int)
        frame["label_copy"] = frame["label"]
        path = tmp_path / "clf.csv"
        frame.to_csv(path, index=False)

        r = train_classifier(str(path), "label", "lr", dry_run=True, random_state=42)
        assert r["success"] is True and r["dry_run"] is True
        named = [s["feature"] for s in r.get("leakage_suspects", [])]
        assert "label_copy" in named, r.get("leakage_suspects")

    def test_a_clean_dry_run_reports_no_suspects_rather_than_omitting_the_key(self, tmp_path):
        """Absent and empty must not be the same thing to a caller reading this."""
        rows = 300
        path = tmp_path / "clean.csv"
        pd.DataFrame(
            {
                "y": [float(i % 23) for i in range(rows)],
                "a": [float((i * 13) % 17) for i in range(rows)],
                "b": [float((i * 5) % 7) for i in range(rows)],
            }
        ).to_csv(path, index=False)

        r = train_regressor(str(path), "y", "lir", dry_run=True, random_state=42)
        assert r["success"] is True
        assert r["leakage_suspects"] == []
        assert "leakage_note" not in r

    def test_the_dry_run_still_writes_nothing(self, leaky, tmp_path):
        out = tmp_path / "should_not_exist.pkl"
        r = train_regressor(leaky, "whole", "lir", output_path=str(out), dry_run=True, random_state=42)
        assert r["success"] is True
        assert not out.exists(), "a dry run wrote a model"


class TestPlotLearningCurveChecksItsTask:
    """`task` decides r2 or accuracy, and this was the one tool that never looked."""

    def test_an_unknown_task_is_refused(self, leaky, tmp_path):
        from servers.ml_advanced._adv_viz import plot_learning_curve

        r = plot_learning_curve(leaky, "whole", "lir", "definitely_not_a_task", output_path=str(tmp_path / "lc.html"))
        assert r["success"] is False
        assert "classification" in r["hint"] and "regression" in r["hint"]

    def test_it_matches_what_its_siblings_say(self, leaky, tmp_path):
        from servers.ml_advanced._adv_viz import plot_learning_curve
        from servers.ml_advanced.engine import tune_hyperparameters

        a = plot_learning_curve(leaky, "whole", "lir", "typo", output_path=str(tmp_path / "lc.html"))
        b = tune_hyperparameters(leaky, "whole", "lir", "typo", output_path=str(tmp_path / "t.pkl"))
        assert a["error"] == b["error"], (a["error"], b["error"])


class TestDropColumnSpeaksBothDialects:
    """ml-medium spells it `column`; the two Data_Analyst tools spell it `columns`."""

    def test_the_native_spelling_works(self, leaky, tmp_path):
        from servers.ml_medium._medium_preprocess import run_preprocessing

        r = run_preprocessing(leaky, [{"op": "drop_column", "column": "noise"}], output_path=str(tmp_path / "a.csv"))
        assert r["success"] is True, r.get("error")
        assert "noise" not in pd.read_csv(tmp_path / "a.csv").columns

    def test_the_data_analyst_spelling_works_too(self, leaky, tmp_path):
        from servers.ml_medium._medium_preprocess import run_preprocessing

        r = run_preprocessing(leaky, [{"op": "drop_column", "columns": ["noise"]}], output_path=str(tmp_path / "b.csv"))
        assert r["success"] is True, r.get("error")
        assert "noise" not in pd.read_csv(tmp_path / "b.csv").columns

    def test_a_list_of_several_drops_all_of_them(self, leaky, tmp_path):
        from servers.ml_medium._medium_preprocess import run_preprocessing

        r = run_preprocessing(
            leaky, [{"op": "drop_column", "columns": ["noise", "part"]}], output_path=str(tmp_path / "c.csv")
        )
        assert r["success"] is True, r.get("error")
        assert list(pd.read_csv(tmp_path / "c.csv").columns) == ["whole"]

    def test_one_missing_column_drops_none_of_them(self, leaky, tmp_path):
        """All or nothing: a partial drop reported as success is the silent write."""
        from servers.ml_medium._medium_preprocess import run_preprocessing

        out = tmp_path / "d.csv"
        run_preprocessing(leaky, [{"op": "drop_column", "columns": ["noise", "not_a_column"]}], output_path=str(out))
        if out.exists():
            assert "noise" in pd.read_csv(out).columns
