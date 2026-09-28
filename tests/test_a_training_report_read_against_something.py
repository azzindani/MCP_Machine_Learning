"""A training report reads the model against a baseline, per class, and by its coefficients.

The sweep's report for a LogisticRegression printed accuracy 0.915 with
nothing to read it against, listed the confusion matrix as TP/FP/FN/TN rows,
had no per-class precision or recall, and said to "read the coefficients from
the model itself". It now carries what always answering the majority class
scores on the same test rows, a per-class table, the matrix as a labelled
heatmap, and a linear model's coefficients -- odds ratios for a logistic one --
each text value named against its baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from servers.ml_advanced.engine import generate_training_report
from servers.ml_basic._basic_train import train_classifier, train_regressor
from shared.model_signing import load_signed


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(11)
    n = 400
    region = rng.choice(["APAC", "EMEA", "LATAM"], n)
    spend = rng.uniform(0, 100, n)
    score = 0.04 * spend + np.where(region == "EMEA", 1.5, 0.0) - 3.2 + rng.normal(0, 0.6, n)
    frame = pd.DataFrame({"region": region, "spend": spend.round(2), "churned": np.where(score > 0, "yes", "no")})
    frame["revenue"] = (3 * spend + np.where(region == "EMEA", 40, 0) + rng.normal(0, 2, n)).round(2)
    frame.to_csv(tmp_path / "c.csv", index=False)
    return tmp_path


def _report(model_path: str, home) -> tuple[dict, str]:
    r = generate_training_report(model_path, output_path=str(home / "report.html"), open_after=False)
    assert r["success"] is True, r
    return r, (home / "report.html").read_text(encoding="utf-8")


@pytest.fixture
def classifier(home):
    r = train_classifier(str(home / "c.csv"), "churned", model="lr", exclude_columns=["revenue"])
    assert r["success"] is True, r
    return r


def test_the_baseline_is_the_majority_share_of_the_test_rows(home, classifier):
    r, page = _report(classifier["model_path"], home)
    frame = pd.read_csv(home / "c.csv")
    _, test = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame["churned"])
    majority = test["churned"].value_counts(normalize=True)
    assert r["baseline"]["baseline"] == pytest.approx(majority.max(), abs=1e-4)
    assert r["baseline"]["rule"] == f"always answer {majority.idxmax()!r}"
    assert "Against a baseline" in page


def test_per_class_results_are_sklearns_on_the_same_split(home, classifier):
    with open(classifier["model_path"], "rb") as fh:
        saved = load_signed(fh)
    meta = saved["metadata"]
    frame = pd.read_csv(home / "c.csv")
    codes = meta["encoding_map"]["region"]
    x = np.column_stack([frame["region"].map(codes), frame["spend"]]).astype(float)
    y = frame["churned"].map(meta["encoding_map"]["__target__churned"]).to_numpy()
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    oracle = classification_report(y_test, saved["model"].predict(x_test), output_dict=True)
    _, page = _report(classifier["model_path"], home)
    for label, code in meta["encoding_map"]["__target__churned"].items():
        expected = oracle[str(code)]
        row = f"<tr><td>{label}</td><td>{round(expected['precision'], 4)}</td><td>{round(expected['recall'], 4)}</td>"
        assert row in page, f"{label}: {row}"


def test_a_logistic_models_odds_ratios_name_each_value(home, classifier):
    with open(classifier["model_path"], "rb") as fh:
        model = load_signed(fh)["model"]
    coef = model.named_steps["model"].coef_[0]
    _, page = _report(classifier["model_path"], home)
    assert "region = EMEA (vs APAC)" in page and "region = LATAM (vs APAC)" in page
    assert f"<td>{round(float(np.exp(coef[0])), 6)}</td>" in page
    assert "Odds ratio" in page


def test_the_confusion_matrix_is_a_labelled_heatmap(home, classifier):
    _, page = _report(classifier["model_path"], home)
    assert '"type":"heatmap"' in page.replace(" ", "")
    assert "predicted yes" in page and "actual no" in page


def test_a_regression_reads_against_the_mean(home):
    t = train_regressor(str(home / "c.csv"), "revenue", model="lir", exclude_columns=["churned"])
    r, page = _report(t["model_path"], home)
    assert r["baseline"] == {
        "rule": "predict the mean of the test rows",
        "metric": "r2",
        "baseline": 0.0,
        "model": t["metrics"]["r2"],
    }
    assert "Coefficients" in page and "Odds ratio" not in page


def test_a_tree_has_importances_and_no_coefficients(home):
    t = train_classifier(str(home / "c.csv"), "churned", model="rf", exclude_columns=["revenue"])
    r, page = _report(t["model_path"], home)
    assert "importance" in r["sections_generated"] and "coefficients" not in r["sections_generated"]
