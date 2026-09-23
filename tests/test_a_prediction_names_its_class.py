"""A classifier's prediction names the class it predicted.

Training label-encodes a text target -- 'Facebook Ads', 'Google Ads' become 0
and 1 -- and saves that map with the model. Prediction never read it back: a
model trained on campaign_platform answered `prediction: 1` and
`probabilities: {'0': 0.0, '1': 1.0}` in predict_single, get_predictions and
batch_predict's distribution alike, so the caller could not say which platform
was predicted. Each now answers with the class name, the code beside it, and
`class_labels` -- position i is class i -- so a probability row can be read.

evaluate_model had the same map missing the other way round: it fitted a new
encoder to the evaluation file's own classes. On rows of one class, that class
became 0 whatever the model calls it, and every metric compared codes that did
not match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from servers.ml_basic import server as _basic
from servers.ml_basic.engine import train_classifier
from servers.ml_medium import server as _medium

get_predictions = _basic.mcp._tool_manager._tools["get_predictions"].fn
predict_single = _basic.mcp._tool_manager._tools["predict_single"].fn
batch_predict = _medium.mcp._tool_manager._tools["batch_predict"].fn
evaluate_model = _medium.mcp._tool_manager._tools["evaluate_model"].fn

LABELS = ["Facebook Ads", "Google Ads"]  # a LabelEncoder numbers them in sorted order


def _frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = np.concatenate([rng.normal(0, 0.3, n // 2), rng.normal(5, 0.3, n // 2)])
    x2 = np.concatenate([rng.normal(1, 0.3, n // 2), rng.normal(-4, 0.3, n // 2)])
    platform = ["Google Ads"] * (n // 2) + ["Facebook Ads"] * (n // 2)
    return pd.DataFrame({"x1": x1, "x2": x2, "platform": platform})


@pytest.fixture
def data(tmp_path) -> str:
    path = tmp_path / "ads.csv"
    _frame().to_csv(path, index=False)
    return str(path)


@pytest.fixture(params=["rf", "xgb"])
def model(request, data) -> str:
    result = train_classifier(data, "platform", request.param)
    assert result["success"] is True, result.get("error")
    return result["model_path"]


class TestEachToolNamesTheClass:
    def test_get_predictions(self, model, data):
        result = get_predictions(model, data, max_rows=120, return_proba=True)
        assert result["success"] is True, result.get("error")
        assert result["class_labels"] == LABELS
        for entry in result["predictions"]:
            assert entry["prediction"] == LABELS[entry["class_code"]]
            assert len(entry["probabilities"]) == len(LABELS)
        truth = _frame()["platform"].tolist()
        right = sum(e["prediction"] == truth[e["row"]] for e in result["predictions"])
        assert right >= 0.9 * len(result["predictions"]) > 0

    def test_predict_single(self, model):
        result = predict_single(model, {"x1": 5.0, "x2": -4.0})
        assert result["success"] is True, result.get("error")
        assert result["prediction"] == "Facebook Ads"
        assert result["class_code"] == 0 and result["class_labels"] == LABELS
        assert set(result["probabilities"]) == set(LABELS)
        assert result["probabilities"]["Facebook Ads"] > 0.5

    def test_batch_predict(self, model, data, tmp_path):
        out = tmp_path / "preds.csv"
        result = batch_predict(model, data, output_path=str(out))
        assert result["success"] is True, result.get("error")
        assert set(result["prediction_distribution"]) == set(LABELS)
        assert result["class_labels"] == LABELS
        assert set(pd.read_csv(out)["prediction"]) == set(LABELS)


class TestEvaluationUsesTheModelsOwnCodes:
    def test_rows_of_one_class_are_scored_against_the_right_code(self, model, tmp_path):
        only_google = _frame()
        only_google = only_google[only_google["platform"] == "Google Ads"]
        path = tmp_path / "google_only.csv"
        only_google.to_csv(path, index=False)
        result = evaluate_model(model, test_file_path=str(path), target_column="platform")
        assert result["success"] is True, result.get("error")
        assert result["metrics"]["accuracy"] >= 0.95, result["metrics"]


class TestANumericTargetIsUnchanged:
    def test_its_prediction_is_still_the_number(self, tmp_path):
        frame = _frame()
        frame["platform"] = (frame["platform"] == "Facebook Ads").astype(int)
        path = tmp_path / "numeric.csv"
        frame.to_csv(path, index=False)
        trained = train_classifier(str(path), "platform", "rf")
        assert trained["success"] is True, trained.get("error")
        result = predict_single(trained["model_path"], {"x1": 5.0, "x2": -4.0})
        assert result["prediction"] == 1 and "class_labels" not in result
