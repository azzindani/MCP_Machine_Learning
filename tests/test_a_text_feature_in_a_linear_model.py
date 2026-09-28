"""A linear model reads a text feature one-hot, and says how it read it.

Training label-encodes a text column alphabetically -- APAC 0, EMEA 1, LATAM 2 --
and every model got those codes. A tree only splits on them; a coefficient
reads them as a quantity, so LATAM counted as twice EMEA. The sweep's housing
regression (location, income_level) reproduced exactly with integer codes, and
nothing in the response said how text had been encoded. Linear and logistic
models now read one indicator per value, the prediction paths and the in-page
scorer read the same model, and each trainer says which encoding it used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from servers.ml_advanced.engine import tune_hyperparameters
from servers.ml_basic._basic_predict import get_predictions, predict_single
from servers.ml_basic._basic_train import train_classifier, train_regressor
from servers.ml_medium._medium_train import compare_models
from shared.model_js import build_payload
from shared.model_signing import load_signed
from tests.test_model_js import NODE, _js_predict

# EMEA is the high one: in alphabetical code order (APAC 0, EMEA 1, LATAM 2) no
# single slope can fit it, and one indicator per region fits it exactly.
EFFECT = {"APAC": 0.0, "EMEA": 30.0, "LATAM": 5.0}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(7)
    n = 240
    region = rng.choice(list(EFFECT), n)
    spend = rng.uniform(0, 100, n).round(2)
    frame = pd.DataFrame({"region": region, "spend": spend})
    frame["sales"] = (2.0 * spend + frame["region"].map(EFFECT) + rng.normal(0, 1, n)).round(3)
    frame["won"] = (frame["sales"] > frame["sales"].median()).astype(int)
    frame.to_csv(tmp_path / "sales.csv", index=False)
    return tmp_path


def _model(path: str):
    with open(path, "rb") as fh:
        return load_signed(fh)


def test_the_score_is_the_one_hot_score(home):
    r = train_regressor(str(home / "sales.csv"), "sales", model="lir", exclude_columns=["won"], random_state=42)
    assert r["success"] is True, r

    frame = pd.read_csv(home / "sales.csv")
    x = pd.get_dummies(frame[["region", "spend"]], columns=["region"], drop_first=True).astype(float)
    x_tr, x_te, y_tr, y_te = train_test_split(x, frame["sales"], test_size=0.2, random_state=42)
    oracle = r2_score(y_te, LinearRegression().fit(x_tr, y_tr).predict(x_te))
    assert r["metrics"]["r2"] == pytest.approx(oracle, abs=1e-4)
    assert r["feature_encoding"]["one_hot"] == ["region"]


def test_a_tree_says_it_read_codes(home):
    r = train_regressor(str(home / "sales.csv"), "sales", model="dtr", exclude_columns=["won"])
    assert r["feature_encoding"]["label_codes"] == ["region"]
    assert "one_hot" not in r["feature_encoding"]


def test_the_prediction_paths_read_the_same_model(home):
    r = train_regressor(str(home / "sales.csv"), "sales", model="lir", exclude_columns=["won"])
    (home / "new.csv").write_text("region,spend\nEMEA,10\nAPAC,10\nMars,10\n", encoding="utf-8")
    preds = [p["prediction"] for p in get_predictions(r["model_path"], str(home / "new.csv"))["predictions"]]
    assert preds[0] - preds[1] == pytest.approx(EFFECT["EMEA"], abs=1.0)
    # A value never seen in training scores as the baseline region, not as a code.
    assert preds[2] == pytest.approx(preds[1], abs=1e-6)
    single = predict_single(r["model_path"], {"region": "EMEA", "spend": 10})
    assert single["prediction"] == pytest.approx(preds[0], abs=1e-6)


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("model", ["lir", "lr"])
def test_the_in_page_scorer_agrees_with_the_model(home, model):
    if model == "lir":
        r = train_regressor(str(home / "sales.csv"), "sales", model="lir", exclude_columns=["won"])
    else:
        r = train_classifier(str(home / "sales.csv"), "won", model="lr", exclude_columns=["sales"])
    saved = _model(r["model_path"])
    payload = build_payload(saved["model"], saved["metadata"])
    assert payload["model"]["levels"], "the one-hot weights were not carried to the page"
    rows = [["APAC", 12.5], ["EMEA", 12.5], ["LATAM", 80.0]]
    codes = saved["metadata"]["encoding_map"]["region"]
    x = np.array([[codes[region], spend] for region, spend in rows], dtype=float)
    js = _js_predict(payload, rows)
    if model == "lir":
        assert [row["value"] for row in js] == pytest.approx(list(saved["model"].predict(x)), abs=1e-6)
    else:
        assert [row["scores"][1] for row in js] == pytest.approx(list(saved["model"].predict_proba(x)[:, 1]), abs=1e-6)


def test_tuning_names_the_parameters_as_given(home):
    r = tune_hyperparameters(str(home / "sales.csv"), "won", "lr", "classification", param_grid={"C": [0.1, 1.0]}, cv=3)
    assert r["success"] is True, r
    assert set(r["best_params"]) == {"C"}
    assert all(set(row["params"]) == {"C"} for row in r["top_results"])
    assert r["feature_encoding"]["one_hot"] == ["region"]


def test_a_comparison_says_which_side_read_what(home):
    r = compare_models(str(home / "sales.csv"), "sales", "regression", models=["lir", "dtr"], exclude_columns=["won"])
    assert r["success"] is True, r
    assert r["feature_encoding"] == {"one_hot_for": ["lir"], "label_codes_for": ["dtr"], "text_columns": ["region"]}
