"""train_with_cv and compare_models wrote model files with no model in them.

Both hardcoded `payload = {"model": None, "metadata": metadata}`, then reported
"Saved best model" and returned a model_path. The file was written, signed,
loadable, and had a matching manifest -- it simply contained None where the
estimator should be. Every downstream use died on

    'NoneType' object has no attribute 'predict'

which is how a coverage sweep surfaced it: evaluate_model and batch_predict
both failed, and the error pointed at the consumer rather than at the two tools
that had produced an empty file.

The cause is structural, not a typo. Cross-validation fits a throwaway
estimator per fold and keeps only its predictions, and compare_models fits each
candidate only to score it, so in both cases nothing was in scope by the time
the save ran. The fix refits the chosen model once on the full dataset, which is
what the shipped model should be anyway once CV has estimated how well it
generalises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from servers.ml_medium.engine import compare_models, train_with_cv
from shared.model_signing import load_signed


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    path = tmp_path / "d.csv"
    pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "target": rng.integers(0, 2, n)}).to_csv(
        path, index=False
    )
    return path


def _stored_model(model_path: str):
    with open(model_path, "rb") as fh:
        return load_signed(fh)


class TestTrainWithCv:
    def test_the_saved_file_contains_an_estimator(self, dataset):
        result = train_with_cv(str(dataset), "target", "rf", "classification", n_splits=3)
        assert result["success"] is True

        payload = _stored_model(result["model_path"])
        assert payload["model"] is not None, "wrote a model file with no model in it"

    def test_the_saved_model_can_actually_predict(self, dataset):
        """Loading it is not enough -- the failure was at .predict() time."""
        result = train_with_cv(str(dataset), "target", "rf", "classification", n_splits=3)
        model = _stored_model(result["model_path"])["model"]
        assert model.predict(np.array([[0.1, 0.2]])).shape == (1,)

    def test_the_metadata_records_that_it_was_refit(self, dataset):
        result = train_with_cv(str(dataset), "target", "rf", "classification", n_splits=3)
        payload = _stored_model(result["model_path"])
        assert payload["metadata"]["refit_on_full_data"] is True

    def test_regression_task_too(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 200
        path = tmp_path / "r.csv"
        a = rng.normal(size=n)
        pd.DataFrame({"a": a, "b": rng.normal(size=n), "target": 2 * a + 1}).to_csv(path, index=False)

        result = train_with_cv(str(path), "target", "rfr", "regression", n_splits=3)
        assert result["success"] is True
        model = _stored_model(result["model_path"])["model"]
        assert model is not None
        assert model.predict(np.array([[0.1, 0.2]])).shape == (1,)


class TestCompareModels:
    def test_the_best_model_file_contains_the_best_model(self, dataset):
        result = compare_models(str(dataset), "target", "classification", ["lr", "rf"])
        assert result["success"] is True

        payload = _stored_model(result["best_model_path"])
        assert payload["model"] is not None, 'wrote a file named "_best" with no model in it'
        assert payload["model"].predict(np.array([[0.1, 0.2]])).shape == (1,)

    def test_the_stored_model_matches_the_winner_it_reported(self, dataset):
        """A file named for one algorithm must not contain another."""
        result = compare_models(str(dataset), "target", "classification", ["lr", "rf"])
        payload = _stored_model(result["best_model_path"])

        expected = {"lr": "LogisticRegression", "rf": "RandomForestClassifier"}[result["best_model"]]
        assert type(payload["model"]).__name__ == expected


class TestScaledModelsKeepTheirScaler:
    def test_a_scaled_model_is_saved_as_a_pipeline(self, dataset):
        """knn and svm are built as (name, scaler, estimator) tuples. Saving the
        bare estimator would drop the scaling it was fitted with and predict
        against differently-scaled input."""
        result = train_with_cv(str(dataset), "target", "knn", "classification", n_splits=3)
        model = _stored_model(result["model_path"])["model"]

        assert model is not None
        assert hasattr(model, "steps"), "scaler was dropped from the saved model"
        assert model.predict(np.array([[0.1, 0.2]])).shape == (1,)
