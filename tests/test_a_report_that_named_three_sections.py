"""A report promising three sections, delivering one, and saying nothing.

    Generate HTML report: metrics, confusion matrix, feature importance.

From the round-14 sweep, phase 10, on a model saved by `tune_hyperparameters`
(which records `best_score` and nothing else):

    generate_training_report(model_path=...)   success: true, 48 KB of HTML
    grep -c confusion  0
    grep -c importance 0
    grep -c accuracy   0

An Evaluation Metrics table with a single row, and two of the three sections the
docstring names simply absent -- not empty, not explained, gone. Both were
conditional on what the manifest and the loaded object happened to carry, and
neither branch had an else.

`read_model_report`, in this same file, already answers an absent confusion
matrix with a note naming `evaluate_model()` as the tool that can produce one.
This was the sibling that did not -- the fourth time this round that a fix had
stopped at one of a pair.

The absence is now stated in three places, because a report is read as a file
and an omission is invisible unless something writes it down: `sections_omitted`
in the response, a warn in the progress log, and a "Not in this report" section
on the page itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_advanced import engine as adv  # noqa: E402
from servers.ml_basic import engine as basic  # noqa: E402


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_ML_SIGNING_KEY_FILE", str(tmp_path / "keys" / "signing.key"))
    rng = np.random.default_rng(0)
    csv = tmp_path / "t.csv"
    pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "y": rng.integers(0, 2, 200)}).to_csv(
        csv, index=False
    )
    return tmp_path, csv


def report(tmp_path, model, name):
    r = adv.generate_training_report(str(model), open_after=False, output_path=str(tmp_path / f"{name}.html"))
    assert r["success"] is True, r.get("error")
    return r, (tmp_path / f"{name}.html").read_text(encoding="utf-8")


class TestAMissingSectionIsReported:
    def test_a_tuned_model_omits_both(self, bench):
        """The exact call the sweep made."""
        tmp_path, csv = bench
        t = adv.tune_hyperparameters(
            str(csv),
            "y",
            "lr",
            "classification",
            param_grid='{"C": [0.1, 1]}',
            cv=3,
            output_path=str(tmp_path / "tuned.pkl"),
        )
        assert t["success"] is True, t.get("error")
        r, _ = report(tmp_path, tmp_path / "tuned.pkl", "tuned")
        assert sorted(r["sections_omitted"]) == ["confusion", "importance"]

    def test_the_reason_names_a_tool_that_can_produce_it(self, bench):
        tmp_path, csv = bench
        adv.tune_hyperparameters(
            str(csv),
            "y",
            "lr",
            "classification",
            param_grid='{"C": [0.1, 1]}',
            cv=3,
            output_path=str(tmp_path / "tuned.pkl"),
        )
        r, _ = report(tmp_path, tmp_path / "tuned.pkl", "tuned")
        assert "evaluate_model()" in r["sections_omitted"]["confusion"]
        assert "train_classifier()" in r["sections_omitted"]["confusion"]

    def test_the_importance_reason_names_a_model_that_has_them(self, bench):
        tmp_path, csv = bench
        basic.train_classifier(str(csv), "y", "lr", output_path=str(tmp_path / "lr.pkl"))
        r, _ = report(tmp_path, tmp_path / "lr.pkl", "lr")
        why = r["sections_omitted"]["importance"]
        assert "LogisticRegression" in why, "say which model type this is"
        assert "rf" in why and "dtr" in why

    def test_it_warns_rather_than_only_recording(self, bench):
        tmp_path, csv = bench
        basic.train_classifier(str(csv), "y", "lr", output_path=str(tmp_path / "lr.pkl"))
        r, _ = report(tmp_path, tmp_path / "lr.pkl", "lr")
        warnings = [p for p in r["progress"] if p.get("icon") not in ("✔", "ℹ")]
        assert any("promised section" in p["msg"] for p in warnings), r["progress"]

    def test_the_page_itself_says_so(self, bench):
        """Read the artifact, not the response."""
        tmp_path, csv = bench
        basic.train_classifier(str(csv), "y", "lr", output_path=str(tmp_path / "lr.pkl"))
        _, html = report(tmp_path, tmp_path / "lr.pkl", "lr")
        assert "Not in this report" in html
        assert "Feature Importance" in html


class TestAFullReportIsUnchanged:
    def test_a_forest_gets_both_sections(self, bench):
        tmp_path, csv = bench
        basic.train_classifier(str(csv), "y", "rf", output_path=str(tmp_path / "rf.pkl"))
        r, html = report(tmp_path, tmp_path / "rf.pkl", "rf")
        assert r["sections_omitted"] == {}
        assert "confusion" in r["sections_generated"]
        assert "importance" in r["sections_generated"]
        assert "Not in this report" not in html

    def test_the_sections_listed_are_the_sections_rendered(self, bench):
        tmp_path, csv = bench
        basic.train_classifier(str(csv), "y", "rf", output_path=str(tmp_path / "rf.pkl"))
        r, html = report(tmp_path, tmp_path / "rf.pkl", "rf")
        for heading in ("Model Overview", "Evaluation Metrics", "Confusion Matrix"):
            assert heading in html, heading
        assert "overview" in r["sections_generated"]

    def test_a_regressor_is_not_asked_for_a_confusion_matrix(self, bench):
        tmp_path, csv = bench
        basic.train_regressor(str(csv), "a", "rfr", output_path=str(tmp_path / "reg.pkl"))
        r, _ = report(tmp_path, tmp_path / "reg.pkl", "reg")
        assert "confusion" not in r["sections_omitted"], "regression has no classes to confuse"
        assert "importance" in r["sections_generated"]
