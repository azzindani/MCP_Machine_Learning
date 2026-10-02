"""A null meets the same preparation at prediction as it met at training.

Training label-encodes text with a null as its own "nan" class and fills a
number's null with that column's median. Every prediction path answered with 0
for a numeric null and -1 for a text one. A model trained on median-filled rows
was therefore asked about a different table than the one it was measured on: a
loan model scored 100% accuracy on its test split and then predicted one class,
"no default", for every one of the 118,936 rows it had been trained on -- the rows
with missing values, which are the rows the model had learned from.

The medians now travel with the model (`encoding_map["__fill__"]`) and one
function, `prepare_features`, applies them in get_predictions, predict_single,
batch_predict, evaluate_model, the ROC and fit charts and the model dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from servers.ml_basic import server as _basic
from servers.ml_basic.engine import train_classifier
from servers.ml_medium import server as _medium
from shared.ml_utils import FILL_KEY, model_matrix, prepare_features

predict_single = _basic.mcp._tool_manager._tools["predict_single"].fn
batch_predict = _medium.mcp._tool_manager._tools["batch_predict"].fn
evaluate_model = _medium.mcp._tool_manager._tools["evaluate_model"].fn


def _numeric_nulls(seed: int = 5) -> pd.DataFrame:
    """Class 0 sits near x=100, class 1 near x=0; the median of x is about 100, and
    twenty class-0 rows have no x at all, so a null means "the typical value"."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(100, 1, 100), rng.normal(0, 1, 50)])
    y = [0] * 100 + [1] * 50
    frame = pd.DataFrame({"x": x, "y": y})
    nulled = pd.DataFrame({"x": [np.nan] * 20, "y": [0] * 20})
    return pd.concat([frame, nulled], ignore_index=True)


def _text_nulls(seed: int = 7) -> pd.DataFrame:
    """Rows with no `kind` are always class 1; 'a' and 'b' are class 0."""
    rng = np.random.default_rng(seed)
    kind = ["a"] * 40 + ["b"] * 40 + [None] * 40
    return pd.DataFrame({"kind": kind, "noise": rng.normal(0, 1, 120), "y": [0] * 80 + [1] * 40})


@pytest.fixture
def numeric_model(tmp_path):
    path = tmp_path / "numeric.csv"
    _numeric_nulls().to_csv(path, index=False)
    result = train_classifier(str(path), "y", "dtc")
    assert result["success"] is True, result.get("error")
    return result["model_path"], str(path)


@pytest.fixture
def text_model(tmp_path):
    path = tmp_path / "text.csv"
    _text_nulls().to_csv(path, index=False)
    result = train_classifier(str(path), "y", "rf")
    assert result["success"] is True, result.get("error")
    return result["model_path"], str(path)


class TestANumberNull:
    def test_training_keeps_the_median_it_filled_with(self, numeric_model):
        from shared.model_signing import load_signed

        with open(numeric_model[0], "rb") as f:
            metadata = load_signed(f)["metadata"]
        fill = metadata["encoding_map"][FILL_KEY]
        assert fill["x"] == pytest.approx(float(_numeric_nulls()["x"].median()))

    def test_predict_single_scores_a_null_as_the_typical_value(self, numeric_model):
        result = predict_single(numeric_model[0], {"x": None})
        assert result["success"] is True, result.get("error")
        assert result["prediction"] == 0  # filled with ~100, class 0 -- not 0.0, which is class 1

    def test_the_response_says_what_it_filled_with(self, numeric_model):
        result = predict_single(numeric_model[0], {"x": None})
        text = " ".join(f"{p['msg']} {p['detail']}" for p in result["progress"])
        assert "training medians" in text and "x:" in text

    def test_batch_predict_matches_training_on_the_rows_it_trained_on(self, numeric_model):
        model, data = numeric_model
        result = batch_predict(model, data, output_path=data.replace(".csv", "_pred.csv"))
        assert result["success"] is True, result.get("error")
        scored = pd.read_csv(result["output_path"])
        truth = _numeric_nulls()["y"].to_numpy()
        assert (scored["prediction"].to_numpy() == truth).mean() >= 0.97
        assert (scored.loc[scored["x"].isna(), "prediction"] == 0).all()

    def test_evaluate_model_agrees_with_the_training_score(self, numeric_model):
        model, data = numeric_model
        result = evaluate_model(model, test_file_path=data, target_column="y")
        assert result["success"] is True, result.get("error")
        assert result["metrics"]["accuracy"] >= 0.97


class TestATextNull:
    def test_a_null_is_the_class_training_gave_it(self, text_model):
        result = predict_single(text_model[0], {"kind": None, "noise": 0.1})
        assert result["success"] is True, result.get("error")
        assert result["prediction"] == 1  # the "nan" class, not -1 (which reads as 'a')
        assert result["unseen_categories"] == {}

    def test_batch_predict_reports_no_unseen_value_for_a_null(self, text_model):
        model, data = text_model
        result = batch_predict(model, data, output_path=data.replace(".csv", "_pred.csv"))
        assert result["unmapped_categories"] == {}
        scored = pd.read_csv(result["output_path"])
        assert (scored.loc[scored["kind"].isna(), "prediction"] == 1).all()

    def test_a_value_it_never_saw_is_named_not_swallowed(self, text_model):
        result = predict_single(text_model[0], {"kind": "zzz", "noise": 0.1})
        assert result["success"] is True
        assert result["unseen_categories"] == {"kind": "zzz"}
        assert any("never saw" in p["msg"] for p in result["progress"])


class TestAModelSavedBeforeTheMediansWere:
    def test_numbers_fill_with_zero_and_the_report_says_so(self):
        metadata = {"feature_columns": ["x"], "encoding_map": {}}
        frame, report = prepare_features(pd.DataFrame({"x": [1.0, np.nan]}), metadata)
        assert frame["x"].tolist() == [1.0, 0.0]
        assert "saved before" in report["null_fill"]

    def test_infinity_is_a_null_too(self):
        metadata = {"feature_columns": ["x"], "encoding_map": {FILL_KEY: {"x": 7.0}}}
        frame, report = prepare_features(pd.DataFrame({"x": [np.inf, -np.inf, 2.0]}), metadata)
        assert frame["x"].tolist() == [7.0, 7.0, 2.0]
        assert report["null_filled"] == {"x": 2}


class TestTheMatrixIsTheOneTheModelWasFittedBehind:
    def test_scaler_and_poly_are_applied_in_order(self):
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        scaler = StandardScaler().fit(frame[["x"]])
        poly = PolynomialFeatures(2, include_bias=False).fit(scaler.transform(frame[["x"]]))
        metadata = {"feature_columns": ["x"], "encoding_map": {}, "scaler": scaler, "poly": poly}
        x, _ = model_matrix(frame, metadata)
        assert x.shape == (4, 2)
        np.testing.assert_allclose(x[:, 1], x[:, 0] ** 2)

    def test_the_target_column_is_never_encoded_as_a_feature(self):
        metadata = {"feature_columns": ["a"], "encoding_map": {"a": {"p": 0, "q": 1}, "y": {"u": 0, "v": 1}}}
        frame, _ = prepare_features(pd.DataFrame({"a": ["q", "p"], "y": ["u", "v"]}), metadata, ["a"], target="y")
        assert frame["a"].tolist() == [1.0, 0.0]


class TestTheInPagePanel:
    def test_a_blank_number_box_carries_the_training_median(self, numeric_model):
        from shared.model_js import build_payload
        from shared.model_signing import load_signed

        with open(numeric_model[0], "rb") as f:
            payload = load_signed(f)
        built = build_payload(payload["model"], payload["metadata"])
        assert built["fill"]["x"] == pytest.approx(float(_numeric_nulls()["x"].median()))
