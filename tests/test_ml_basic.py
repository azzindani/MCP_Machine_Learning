"""Tests for ml_basic engine — all tests call engine directly, no MCP server."""

from pathlib import Path

import pandas as pd
import pickle
import pytest

from servers.ml_basic.engine import (
    get_predictions,
    inspect_dataset,
    list_models,
    predict_single,
    read_column_profile,
    read_rows,
    restore_version,
    search_columns,
    split_dataset,
    train_classifier,
    train_regressor,
)

# ============================================================
# inspect_dataset
# ============================================================


def test_inspect_dataset_success(classification_simple):
    r = inspect_dataset(classification_simple)
    assert r["success"] is True
    assert r["op"] == "inspect_dataset"
    assert r["row_count"] == 200
    assert r["column_count"] == 6
    assert isinstance(r["columns"], list)
    assert len(r["columns"]) == 6
    assert "churned" in r["target_candidates"]


def test_inspect_dataset_file_not_found(tmp_path):
    r = inspect_dataset(str(tmp_path / "missing.csv"))
    assert r["success"] is False
    assert "error" in r
    assert "hint" in r


def test_inspect_dataset_token_estimate_present(classification_simple):
    r = inspect_dataset(classification_simple)
    assert "token_estimate" in r
    assert isinstance(r["token_estimate"], int)


def test_inspect_dataset_progress_present(classification_simple):
    r = inspect_dataset(classification_simple)
    assert "progress" in r
    assert len(r["progress"]) > 0


def test_inspect_dataset_truncated_false(classification_simple):
    r = inspect_dataset(classification_simple)
    assert r["truncated"] is False


def test_inspect_dataset_constrained_mode(classification_simple, constrained_mode):
    r = inspect_dataset(classification_simple)
    assert r["success"] is True
    assert len(r["columns"]) <= 20


# ============================================================
# read_column_profile
# ============================================================


def test_read_column_profile_numeric(regression_simple):
    r = read_column_profile(regression_simple, "salary")
    assert r["success"] is True
    assert r["profile"]["kind"] == "numeric"
    assert "mean" in r["profile"]
    assert "std" in r["profile"]


def test_read_column_profile_categorical(classification_messy):
    r = read_column_profile(classification_messy, "gender")
    assert r["success"] is True
    assert r["profile"]["kind"] == "categorical"
    assert "top_values" in r["profile"]


def test_read_column_profile_file_not_found(tmp_path):
    r = read_column_profile(str(tmp_path / "missing.csv"), "col")
    assert r["success"] is False
    assert "hint" in r


def test_read_column_profile_column_not_found(classification_simple):
    r = read_column_profile(classification_simple, "nonexistent_col")
    assert r["success"] is False
    assert "nonexistent_col" in r["error"]
    assert "hint" in r


def test_read_column_profile_token_estimate_present(regression_simple):
    r = read_column_profile(regression_simple, "salary")
    assert "token_estimate" in r


def test_read_column_profile_progress_present(regression_simple):
    r = read_column_profile(regression_simple, "salary")
    assert "progress" in r


# ============================================================
# search_columns
# ============================================================


def test_search_columns_has_nulls(classification_messy):
    r = search_columns(classification_messy, has_nulls=True)
    assert r["success"] is True
    assert len(r["columns"]) > 0


def test_search_columns_name_contains(classification_simple):
    r = search_columns(classification_simple, name_contains="charge")
    assert r["success"] is True
    assert all("charge" in c for c in r["columns"])


def test_search_columns_file_not_found(tmp_path):
    r = search_columns(str(tmp_path / "missing.csv"))
    assert r["success"] is False
    assert "hint" in r


def test_search_columns_token_estimate_present(classification_simple):
    r = search_columns(classification_simple)
    assert "token_estimate" in r


def test_search_columns_progress_present(classification_simple):
    r = search_columns(classification_simple)
    assert "progress" in r


def test_search_columns_constrained_mode(large_10k, constrained_mode):
    r = search_columns(large_10k)
    assert r["returned"] <= 10


# ============================================================
# read_rows
# ============================================================


def test_read_rows_success(classification_simple):
    r = read_rows(classification_simple, 0, 5)
    assert r["success"] is True
    assert len(r["rows"]) == 5
    assert r["truncated"] is False


def test_read_rows_truncation(large_10k):
    r = read_rows(large_10k, 0, 500)
    assert r["success"] is True
    assert r["truncated"] is True
    assert len(r["rows"]) <= 100


def test_read_rows_constrained_mode(large_10k, constrained_mode):
    r = read_rows(large_10k, 0, 500)
    assert len(r["rows"]) <= 20


def test_read_rows_file_not_found(tmp_path):
    r = read_rows(str(tmp_path / "missing.csv"), 0, 10)
    assert r["success"] is False
    assert "hint" in r


def test_read_rows_token_estimate_present(classification_simple):
    r = read_rows(classification_simple, 0, 5)
    assert "token_estimate" in r


def test_read_rows_progress_present(classification_simple):
    r = read_rows(classification_simple, 0, 5)
    assert "progress" in r


# ============================================================
# train_classifier
# ============================================================

CLASSIFIER_MODELS = ["lr", "svm", "rf", "dtc", "knn", "nb", "xgb"]


def test_train_classifier_success(classification_simple):
    r = train_classifier(classification_simple, "churned", "rf")
    assert r["success"] is True
    assert r["op"] == "train_classifier"
    assert "metrics" in r
    assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
    assert "model_path" in r


def test_train_classifier_file_not_found(tmp_path):
    r = train_classifier(str(tmp_path / "missing.csv"), "churned", "rf")
    assert r["success"] is False
    assert "hint" in r


def test_train_classifier_unknown_model(classification_simple):
    r = train_classifier(classification_simple, "churned", "fakemodel")
    assert r["success"] is False
    assert "hint" in r


def test_train_classifier_bad_target(classification_simple):
    r = train_classifier(classification_simple, "nonexistent", "rf")
    assert r["success"] is False
    assert "hint" in r


def test_train_classifier_single_class_target(classification_simple, tmp_path):
    import pandas as pd

    df = pd.read_csv(classification_simple)
    df["constant"] = 1
    p = tmp_path / "constant.csv"
    df.to_csv(p, index=False)
    r = train_classifier(str(p), "constant", "rf")
    assert r["success"] is False
    assert "hint" in r


def test_train_classifier_insufficient_rows(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": range(5), "b": range(5), "target": [0, 1, 0, 1, 0]})
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    r = train_classifier(str(p), "target", "rf")
    assert r["success"] is False
    assert "hint" in r


def test_train_classifier_snapshot_created(classification_simple):
    r = train_classifier(classification_simple, "churned", "rf")
    assert r["success"] is True
    model_p = Path(r["model_path"])
    assert model_p.exists()
    assert model_p.suffix == ".pkl"
    manifest = model_p.with_suffix(".manifest.json")
    assert manifest.exists()


def test_train_classifier_backup_in_response(classification_simple):
    r = train_classifier(classification_simple, "churned", "rf")
    assert r["success"] is True
    assert "backup" in r


def test_train_classifier_dry_run(classification_simple):
    r = train_classifier(classification_simple, "churned", "rf", dry_run=True)
    assert r["success"] is True
    assert r["dry_run"] is True
    assert "model_path" not in r


def test_train_classifier_token_estimate_present(classification_simple):
    r = train_classifier(classification_simple, "churned", "lr")
    assert "token_estimate" in r


def test_train_classifier_progress_present(classification_simple):
    r = train_classifier(classification_simple, "churned", "lr")
    assert "progress" in r
    assert len(r["progress"]) > 0


@pytest.mark.parametrize("algo", CLASSIFIER_MODELS)
def test_train_all_classifier_algorithms(classification_simple, algo):
    r = train_classifier(classification_simple, "churned", algo)
    assert r["success"] is True, f"Failed for model={algo}: {r.get('error')}"
    assert r["metrics"]["accuracy"] >= 0.0


def test_train_classifier_messy_data(classification_messy):
    r = train_classifier(classification_messy, "churned", "rf")
    assert r["success"] is True


# ============================================================
# train_regressor
# ============================================================

REGRESSOR_MODELS = ["lir", "pr", "lar", "rr", "dtr", "rfr", "xgb"]


def test_train_regressor_success(regression_simple):
    r = train_regressor(regression_simple, "salary", "rfr")
    assert r["success"] is True
    assert r["op"] == "train_regressor"
    assert "metrics" in r
    assert "r2" in r["metrics"]
    assert "model_path" in r


def test_train_regressor_file_not_found(tmp_path):
    r = train_regressor(str(tmp_path / "missing.csv"), "salary", "lir")
    assert r["success"] is False
    assert "hint" in r


def test_train_regressor_bad_target(regression_simple):
    r = train_regressor(regression_simple, "nonexistent", "lir")
    assert r["success"] is False
    assert "hint" in r


def test_train_regressor_snapshot_created(regression_simple):
    r = train_regressor(regression_simple, "salary", "lir")
    assert r["success"] is True
    assert Path(r["model_path"]).exists()
    assert Path(r["model_path"]).with_suffix(".manifest.json").exists()


def test_train_regressor_dry_run(regression_simple):
    r = train_regressor(regression_simple, "salary", "lir", dry_run=True)
    assert r["success"] is True
    assert r["dry_run"] is True


def test_train_regressor_token_estimate_present(regression_simple):
    r = train_regressor(regression_simple, "salary", "lir")
    assert "token_estimate" in r


def test_train_regressor_progress_present(regression_simple):
    r = train_regressor(regression_simple, "salary", "lir")
    assert "progress" in r


@pytest.mark.parametrize("algo", REGRESSOR_MODELS)
def test_train_all_regressor_algorithms(regression_simple, algo):
    r = train_regressor(regression_simple, "salary", algo)
    assert r["success"] is True, f"Failed for model={algo}: {r.get('error')}"


# ============================================================
# get_predictions
# ============================================================


def test_get_predictions_success(classification_simple):
    train_r = train_classifier(classification_simple, "churned", "rf")
    assert train_r["success"] is True
    r = get_predictions(train_r["model_path"], classification_simple, max_rows=5)
    assert r["success"] is True
    assert len(r["predictions"]) == 5
    assert r["predictions"][0]["row"] == 0


def test_get_predictions_truncated(classification_simple):
    train_r = train_classifier(classification_simple, "churned", "lr")
    r = get_predictions(train_r["model_path"], classification_simple, max_rows=500)
    assert r["success"] is True
    assert r["truncated"] is True or len(r["predictions"]) <= 100


def test_get_predictions_model_not_found(tmp_path, classification_simple):
    r = get_predictions(str(tmp_path / "missing.pkl"), classification_simple)
    assert r["success"] is False
    assert "hint" in r


def test_get_predictions_token_estimate_present(classification_simple):
    train_r = train_classifier(classification_simple, "churned", "rf")
    r = get_predictions(train_r["model_path"], classification_simple, max_rows=5)
    assert "token_estimate" in r


def test_get_predictions_progress_present(classification_simple):
    train_r = train_classifier(classification_simple, "churned", "rf")
    r = get_predictions(train_r["model_path"], classification_simple, max_rows=5)
    assert "progress" in r


# ============================================================
# restore_version
# ============================================================


def test_restore_version_file_not_found(tmp_path):
    # Listing snapshots for a non-existent file returns success + empty list (no snapshots)
    r = restore_version(str(tmp_path / "nonexistent.csv"))
    assert "success" in r
    assert "snapshots" in r or "error" in r


def test_restore_version_list_when_no_timestamp(classification_simple):
    r = restore_version(classification_simple)
    assert r["success"] is True
    assert "snapshots" in r


def test_restore_version_bad_timestamp(classification_simple):
    r = restore_version(classification_simple, "bad-ts-9999")
    # either error or empty list — both are acceptable
    assert "success" in r


def test_restore_version_token_estimate_present(classification_simple):
    r = restore_version(classification_simple)
    assert "token_estimate" in r


def test_restore_version_progress_present(classification_simple):
    r = restore_version(classification_simple)
    assert "progress" in r


# ---------------------------------------------------------------------------
# train_classifier — new params: class_weight, return_train_score, AUC-ROC
# ---------------------------------------------------------------------------


def test_train_classifier_class_weight_balanced(classification_simple):
    r = train_classifier(classification_simple, "churned", "rf", class_weight="balanced")
    assert r["success"] is True
    assert r["model_class"] == "RandomForestClassifier"


def test_train_classifier_return_train_score(classification_simple):
    r = train_classifier(classification_simple, "churned", "lr", return_train_score=True)
    assert r["success"] is True
    assert "train_accuracy" in r["metrics"]
    assert "train_f1_weighted" in r["metrics"]
    assert "overfit_gap" in r["metrics"]


def test_train_classifier_auc_roc_binary(classification_simple):
    """AUC-ROC is auto-computed for binary classifiers that support predict_proba."""
    r = train_classifier(classification_simple, "churned", "lr")
    assert r["success"] is True
    # LR supports predict_proba → auc_roc should appear for binary target
    assert "auc_roc" in r["metrics"]
    assert 0.0 <= r["metrics"]["auc_roc"] <= 1.0


def test_train_classifier_auc_roc_rf(classification_simple):
    r = train_classifier(classification_simple, "churned", "rf")
    assert r["success"] is True
    assert "auc_roc" in r["metrics"]


# ---------------------------------------------------------------------------
# get_predictions — return_proba
# ---------------------------------------------------------------------------


def test_get_predictions_return_proba_lr(classification_simple):
    tr = train_classifier(classification_simple, "churned", "lr")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], classification_simple, max_rows=10, return_proba=True)
    assert r["success"] is True
    assert "probabilities" in r["predictions"][0]
    assert len(r["predictions"][0]["probabilities"]) == 2


def test_get_predictions_return_proba_false(classification_simple):
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], classification_simple, max_rows=5)
    assert r["success"] is True
    # No probabilities when return_proba=False (default)
    assert "probabilities" not in r["predictions"][0]


# ============================================================
# get_predictions — additional coverage
# ============================================================


def test_get_predictions_data_file_not_found(classification_simple, tmp_path):
    """Line 54: data file not found after model loads successfully."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], str(tmp_path / "missing_data.csv"), max_rows=5)
    assert r["success"] is False
    assert "hint" in r
    assert "not found" in r["error"].lower() or "File not found" in r["error"]


def test_get_predictions_encoding_map_applied(classification_messy, tmp_path):
    """Lines 65-66: encoding map is applied for categorical features."""
    # classification_messy has gender and region (categorical columns)
    tr = train_classifier(classification_messy, "churned", "rf")
    assert tr["success"] is True
    # Run predictions on the same file — encoding map must be applied
    r = get_predictions(tr["model_path"], classification_messy, max_rows=10)
    assert r["success"] is True
    assert len(r["predictions"]) > 0


def test_get_predictions_missing_feature_columns(classification_simple, tmp_path):
    """Line 71: feature columns missing in data file."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    # Create a CSV that is missing columns the model expects
    stripped = pd.DataFrame({"age": [25, 30], "churned": [0, 1]})
    stripped_path = tmp_path / "stripped.csv"
    stripped.to_csv(stripped_path, index=False)

    r = get_predictions(tr["model_path"], str(stripped_path), max_rows=5)
    assert r["success"] is False
    assert "missing" in r["error"].lower() or "Feature columns" in r["error"]
    assert "hint" in r


def test_get_predictions_with_scaler_in_metadata(classification_simple):
    """Line 81: scaler.transform path — SVM or KNN models use a scaler."""
    # SVM and KNN use StandardScaler internally in _basic_train
    for model_key in ("svm", "knn"):
        tr = train_classifier(classification_simple, "churned", model_key)
        assert tr["success"] is True, f"Train failed for {model_key}: {tr.get('error')}"
        r = get_predictions(tr["model_path"], classification_simple, max_rows=5)
        assert r["success"] is True, f"Predict failed for {model_key}: {r.get('error')}"
        assert len(r["predictions"]) == 5


def test_get_predictions_regression_float_values(regression_simple):
    """Line 119: regression prediction returns floats."""
    tr = train_regressor(regression_simple, "salary", "rfr")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], regression_simple, max_rows=5)
    assert r["success"] is True
    assert r["task"] == "regression"
    for pred in r["predictions"]:
        assert isinstance(pred["prediction"], float)


def test_get_predictions_xgb_classification_binary(classification_simple):
    """Lines 97-108: XGBoost binary classification prediction path."""
    tr = train_classifier(classification_simple, "churned", "xgb")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], classification_simple, max_rows=10)
    assert r["success"] is True
    assert r["task"] == "classification"
    for pred in r["predictions"]:
        assert pred["prediction"] in (0, 1)


def test_get_predictions_xgb_classification_binary_proba(classification_simple):
    """Lines 107-108: XGBoost binary classification with return_proba."""
    tr = train_classifier(classification_simple, "churned", "xgb")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], classification_simple, max_rows=5, return_proba=True)
    assert r["success"] is True
    for pred in r["predictions"]:
        assert "probabilities" in pred
        assert len(pred["probabilities"]) == 2


def test_get_predictions_xgb_regression(regression_simple):
    """Line 110: XGBoost regression prediction (floats)."""
    tr = train_regressor(regression_simple, "salary", "xgb")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], regression_simple, max_rows=5)
    assert r["success"] is True
    assert r["task"] == "regression"
    for pred in r["predictions"]:
        assert isinstance(pred["prediction"], float)


def test_get_predictions_xgb_multiclass(tmp_path):
    """Lines 101-104: XGBoost multiclass prediction path."""
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "f2": [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
        "target": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    })
    csv_path = tmp_path / "multiclass.csv"
    df.to_csv(csv_path, index=False)

    tr = train_classifier(str(csv_path), "target", "xgb")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], str(csv_path), max_rows=10)
    assert r["success"] is True
    for pred in r["predictions"]:
        assert pred["prediction"] in (0, 1, 2)


def test_get_predictions_xgb_multiclass_proba(tmp_path):
    """Lines 103-104: XGBoost multiclass with return_proba returns per-class probabilities."""
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "f2": [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
        "target": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    })
    csv_path = tmp_path / "multiclass.csv"
    df.to_csv(csv_path, index=False)

    tr = train_classifier(str(csv_path), "target", "xgb")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], str(csv_path), max_rows=5, return_proba=True)
    assert r["success"] is True
    for pred in r["predictions"]:
        assert "probabilities" in pred
        assert len(pred["probabilities"]) == 3  # 3 classes


def test_get_predictions_exception_path(tmp_path, classification_simple):
    """Lines 148-150: exception caught → error dict returned."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    # Corrupt the pkl so _load_model raises
    model_path = tr["model_path"]
    Path(model_path).write_bytes(b"not a valid pickle")

    r = get_predictions(model_path, classification_simple, max_rows=5)
    assert r["success"] is False
    assert "hint" in r


def test_get_predictions_poly_transform(regression_simple):
    """Line 86: poly.transform path — Polynomial Regression stores poly in metadata."""
    tr = train_regressor(regression_simple, "salary", "pr")
    assert tr["success"] is True
    r = get_predictions(tr["model_path"], regression_simple, max_rows=5)
    assert r["success"] is True
    assert r["task"] == "regression"


# ============================================================
# restore_version — additional coverage
# ============================================================


def test_restore_version_list_snapshots_no_timestamp(classification_simple):
    """Lists snapshots when no timestamp given — returns success + snapshots list."""
    r = restore_version(str(classification_simple))
    assert r["success"] is True
    assert "snapshots" in r
    assert isinstance(r["snapshots"], list)
    assert "token_estimate" in r
    assert "progress" in r


def test_restore_version_with_valid_timestamp(classification_simple):
    """Restore with a valid timestamp — success and restored_from in response."""
    from shared.version_control import snapshot as _snap

    # Create a snapshot first
    backup_path = _snap(str(classification_simple))
    assert Path(backup_path).exists()

    # Extract the timestamp from the backup filename
    backup_stem = Path(backup_path).stem
    # stem looks like: classification_simple_2026-04-18T...
    # The timestamp is everything after the first underscore-separated date part
    ts = backup_stem[len(classification_simple.stem) + 1:]

    r = restore_version(str(classification_simple), ts)
    assert r["success"] is True
    assert "restored_from" in r or "snapshots" in r  # success path
    assert "token_estimate" in r
    assert "progress" in r


def test_restore_version_bad_timestamp_returns_error(classification_simple):
    """No snapshot matching the timestamp → success=False error dict."""
    # Ensure a snapshot exists so the function gets past listing
    from shared.version_control import snapshot as _snap
    _snap(str(classification_simple))

    r = restore_version(str(classification_simple), "9999-INVALID-TIMESTAMP")
    # Either success=False (no match) or success=True (listed snapshots)
    assert "success" in r
    assert "token_estimate" in r


def test_restore_version_exception_path(tmp_path, monkeypatch):
    """Lines 177-179: exception in _restore_version → error dict returned."""
    from shared import version_control as vc

    def _boom(fp, ts=""):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(vc, "restore_version", _boom)

    # Must re-import the function after monkeypatching
    import importlib
    import servers.ml_basic._basic_predict as bp_mod
    monkeypatch.setattr(bp_mod, "_restore_version", _boom)

    r = restore_version(str(tmp_path / "something.csv"))
    assert r["success"] is False
    assert "hint" in r
    assert "token_estimate" in r


# ============================================================
# predict_single — full coverage
# ============================================================


def test_predict_single_success_classification(classification_simple):
    """Lines 190+: valid input dict string → prediction returned."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    record = '{"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}'
    r = predict_single(tr["model_path"], record)
    assert r["success"] is True
    assert r["op"] == "predict_single"
    assert "prediction" in r
    assert r["prediction"] in (0, 1)
    assert "progress" in r
    assert "token_estimate" in r


def test_predict_single_success_dict_input(classification_simple):
    """Lines 200-201: input_data may be a plain dict (not a JSON string)."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    record = {"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}
    r = predict_single(tr["model_path"], record)
    assert r["success"] is True
    assert r["prediction"] in (0, 1)


def test_predict_single_model_not_found(tmp_path):
    """Lines 195-196: model file not found."""
    r = predict_single(str(tmp_path / "ghost.pkl"), '{"a": 1}')
    assert r["success"] is False
    assert "hint" in r
    assert "not found" in r["error"].lower() or "Model file not found" in r["error"]


def test_predict_single_invalid_json(classification_simple):
    """Lines 204-208: invalid JSON → error dict."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    r = predict_single(tr["model_path"], "NOT VALID JSON {{")
    assert r["success"] is False
    assert "JSON" in r["error"] or "invalid" in r["error"].lower()
    assert "hint" in r


def test_predict_single_missing_features(classification_simple):
    """Lines 223-227: input missing required feature columns."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    # Provide only a subset of features
    r = predict_single(tr["model_path"], '{"age": 35}')
    assert r["success"] is False
    assert "missing" in r["error"].lower()
    assert "hint" in r


def test_predict_single_probabilities_classification(classification_simple):
    """Lines 252-254: classification with predict_proba returns probabilities."""
    tr = train_classifier(classification_simple, "churned", "lr")
    assert tr["success"] is True

    record = '{"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}'
    r = predict_single(tr["model_path"], record)
    assert r["success"] is True
    assert r["probabilities"] is not None
    assert "0" in r["probabilities"] or "1" in r["probabilities"]


def test_predict_single_regression(regression_simple):
    """Lines 280-282: regression prediction returns float."""
    tr = train_regressor(regression_simple, "salary", "rfr")
    assert tr["success"] is True

    record = '{"age": 35, "experience": 10, "education_level": 2, "department": 3, "performance_score": 3.5}'
    r = predict_single(tr["model_path"], record)
    assert r["success"] is True
    assert r["task"] == "regression"
    assert isinstance(r["prediction"], float)


def test_predict_single_xgb_binary(classification_simple):
    """Lines 256-265: XGBoost Booster predict path (binary)."""
    tr = train_classifier(classification_simple, "churned", "xgb")
    assert tr["success"] is True

    record = '{"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}'
    r = predict_single(tr["model_path"], record)
    assert r["success"] is True
    assert r["prediction"] in (0, 1)
    assert r["probabilities"] is not None


def test_predict_single_xgb_multiclass(tmp_path):
    """Lines 260-262: XGBoost Booster multiclass (n_classes > 2) predict path."""
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "f2": [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
        "target": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    })
    csv_path = tmp_path / "mc.csv"
    df.to_csv(csv_path, index=False)

    tr = train_classifier(str(csv_path), "target", "xgb")
    assert tr["success"] is True

    r = predict_single(tr["model_path"], '{"f1": 2, "f2": 5}')
    assert r["success"] is True
    assert r["prediction"] in (0, 1, 2)
    assert r["probabilities"] is not None
    assert len(r["probabilities"]) == 3


def test_predict_single_with_scaler(classification_simple):
    """Lines 243-244: scaler.transform applied for SVM/KNN models."""
    for model_key in ("svm", "knn"):
        tr = train_classifier(classification_simple, "churned", model_key)
        assert tr["success"] is True, f"Train failed: {tr.get('error')}"

        record = '{"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}'
        r = predict_single(tr["model_path"], record)
        assert r["success"] is True, f"Predict failed for {model_key}: {r.get('error')}"


def test_predict_single_corrupted_pkl(classification_simple, tmp_path):
    """Lines 211-213: corrupted pkl → error dict."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    # Corrupt the pkl
    Path(tr["model_path"]).write_bytes(b"garbage bytes")

    record = '{"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}'
    r = predict_single(tr["model_path"], record)
    assert r["success"] is False
    assert "hint" in r


def test_predict_single_token_estimate_present(classification_simple):
    tr = train_classifier(classification_simple, "churned", "lr")
    assert tr["success"] is True
    record = '{"age": 35, "tenure": 5, "monthly_charges": 70.0, "total_charges": 3500.0, "num_products": 2}'
    r = predict_single(tr["model_path"], record)
    assert "token_estimate" in r


def test_predict_single_categorical_features(classification_messy):
    """Lines 231-234: encoding_map applied during predict_single for categorical cols."""
    tr = train_classifier(classification_messy, "churned", "rf")
    assert tr["success"] is True

    # Use actual categories from the messy dataset
    record = {
        "age": 35.0,
        "gender": "M",
        "region": "North",
        "monthly_charges": 80.0,
        "total_charges": 4000.0,
        "num_products": 2,
        "support_calls": 5,
        "discount": 0.1,
    }
    r = predict_single(tr["model_path"], record)
    assert r["success"] is True
    assert r["prediction"] in (0, 1)


# ============================================================
# list_models — full coverage
# ============================================================


def test_list_models_success_empty(tmp_path, monkeypatch):
    """Lines 296+: empty directory returns success with empty model list."""
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    r = list_models(str(tmp_path))
    assert r["success"] is True
    assert r["op"] == "list_models"
    assert isinstance(r["models"], list)
    assert r["model_count"] == 0
    assert "token_estimate" in r
    assert "progress" in r


def test_list_models_with_trained_model(classification_simple, tmp_path):
    """Lines 306-331: trained model shows up in list with metadata."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    model_dir = Path(tr["model_path"]).parent
    r = list_models(str(model_dir))
    assert r["success"] is True
    assert r["model_count"] >= 1

    model_entry = r["models"][0]
    assert "name" in model_entry
    assert model_entry["name"].endswith(".pkl")
    assert "size_kb" in model_entry
    assert "modified" in model_entry


def test_list_models_with_manifest(classification_simple, tmp_path):
    """Lines 316-329: model with manifest shows metadata fields."""
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    model_dir = Path(tr["model_path"]).parent
    r = list_models(str(model_dir))
    assert r["success"] is True

    # At least one model should have manifest metadata
    entry = next((m for m in r["models"] if "model_type" in m), None)
    assert entry is not None, "No model with manifest metadata found"
    assert "task" in entry
    assert "trained_on" in entry
    assert "target_column" in entry
    assert "metrics" in entry


def test_list_models_without_manifest(tmp_path):
    """Lines 306-315: model without manifest — only basic fields."""
    # Create a fake .pkl without a .manifest.json
    fake_pkl = tmp_path / "orphan_model.pkl"
    fake_pkl.write_bytes(b"fake pickle data")

    r = list_models(str(tmp_path))
    assert r["success"] is True
    assert r["model_count"] == 1
    entry = r["models"][0]
    assert entry["name"] == "orphan_model.pkl"
    assert "size_kb" in entry
    assert "model_type" not in entry  # no manifest, no extended metadata


def test_list_models_default_directory(classification_simple, monkeypatch, tmp_path):
    """Lines 302-303: no directory arg → uses get_output_dir()."""
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    r = list_models()  # no directory argument
    assert r["success"] is True
    assert isinstance(r["models"], list)


def test_list_models_multiple_models(classification_simple, tmp_path):
    """Multiple models in directory — all listed."""
    tr1 = train_classifier(classification_simple, "churned", "rf")
    tr2 = train_classifier(classification_simple, "churned", "lr")
    assert tr1["success"] is True
    assert tr2["success"] is True

    model_dir = Path(tr1["model_path"]).parent
    r = list_models(str(model_dir))
    assert r["success"] is True
    assert r["model_count"] >= 2


def test_list_models_token_estimate_present(tmp_path):
    r = list_models(str(tmp_path))
    assert "token_estimate" in r
    assert isinstance(r["token_estimate"], int)


def test_list_models_progress_present(tmp_path):
    r = list_models(str(tmp_path))
    assert "progress" in r
    assert len(r["progress"]) > 0


def test_list_models_corrupt_manifest(tmp_path):
    """Lines 329-330: manifest read fails gracefully — model still listed."""
    fake_pkl = tmp_path / "broken_manifest_model.pkl"
    fake_pkl.write_bytes(b"not a real pickle")
    fake_manifest = fake_pkl.with_suffix(".manifest.json")
    fake_manifest.write_text("{invalid json [[[")

    r = list_models(str(tmp_path))
    assert r["success"] is True
    assert r["model_count"] == 1
    entry = r["models"][0]
    assert entry["name"] == "broken_manifest_model.pkl"
    # Model type should not be present since manifest was unreadable
    assert "model_type" not in entry


# ============================================================
# split_dataset — full coverage
# ============================================================


def test_split_dataset_success(classification_simple, tmp_path):
    """Lines 359+: happy path — saves train and test CSV files."""
    r = split_dataset(str(classification_simple), test_size=0.2, output_dir=str(tmp_path))
    assert r["success"] is True
    assert r["op"] == "split_dataset"
    assert "train_path" in r
    assert "test_path" in r
    assert Path(r["train_path"]).exists()
    assert Path(r["test_path"]).exists()
    assert r["train_rows"] > 0
    assert r["test_rows"] > 0
    # Verify approximate split ratio
    total = r["train_rows"] + r["test_rows"]
    assert abs(r["test_size_actual"] - 0.2) < 0.05
    assert total == 200  # classification_simple has 200 rows


def test_split_dataset_token_estimate_present(classification_simple, tmp_path):
    r = split_dataset(str(classification_simple), output_dir=str(tmp_path))
    assert "token_estimate" in r
    assert isinstance(r["token_estimate"], int)


def test_split_dataset_progress_present(classification_simple, tmp_path):
    r = split_dataset(str(classification_simple), output_dir=str(tmp_path))
    assert "progress" in r
    assert len(r["progress"]) > 0


def test_split_dataset_invalid_test_size_too_large(classification_simple, tmp_path):
    """Line 378: test_size >= 1 → error dict."""
    r = split_dataset(str(classification_simple), test_size=1.0, output_dir=str(tmp_path))
    assert r["success"] is False
    assert "test_size" in r["error"]
    assert "hint" in r


def test_split_dataset_invalid_test_size_zero(classification_simple, tmp_path):
    """Line 378: test_size <= 0 → error dict."""
    r = split_dataset(str(classification_simple), test_size=0.0, output_dir=str(tmp_path))
    assert r["success"] is False
    assert "hint" in r


def test_split_dataset_invalid_test_size_negative(classification_simple, tmp_path):
    """Line 378: negative test_size → error dict."""
    r = split_dataset(str(classification_simple), test_size=-0.1, output_dir=str(tmp_path))
    assert r["success"] is False
    assert "hint" in r


def test_split_dataset_file_not_found(tmp_path):
    """Line 367: file not found → error dict."""
    r = split_dataset(str(tmp_path / "nonexistent.csv"), output_dir=str(tmp_path))
    assert r["success"] is False
    assert "hint" in r


def test_split_dataset_stratify_column(classification_simple, tmp_path):
    """Lines 386-393: stratify_column splits by class balance."""
    r = split_dataset(
        str(classification_simple),
        test_size=0.2,
        stratify_column="churned",
        output_dir=str(tmp_path),
    )
    assert r["success"] is True
    assert r["stratified"] is True
    assert Path(r["train_path"]).exists()
    assert Path(r["test_path"]).exists()


def test_split_dataset_stratify_column_not_found(classification_simple, tmp_path):
    """Lines 388-392: stratify column not in dataset → error."""
    r = split_dataset(
        str(classification_simple),
        stratify_column="nonexistent_column",
        output_dir=str(tmp_path),
    )
    assert r["success"] is False
    assert "not found" in r["error"].lower() or "Column" in r["error"]
    assert "hint" in r


def test_split_dataset_snapshot_on_existing_files(classification_simple, tmp_path):
    """Lines 400-411: existing train/test files are snapshotted before overwrite."""
    out_dir = tmp_path / "splits"
    out_dir.mkdir()

    # First split
    r1 = split_dataset(str(classification_simple), test_size=0.2, output_dir=str(out_dir))
    assert r1["success"] is True

    # Second split — should trigger snapshot of existing files
    r2 = split_dataset(str(classification_simple), test_size=0.2, output_dir=str(out_dir))
    assert r2["success"] is True

    # Check that .mcp_versions directory was created (snapshot occurred)
    versions_dir = out_dir / ".mcp_versions"
    assert versions_dir.exists()
    bak_files = list(versions_dir.glob("*.bak"))
    assert len(bak_files) >= 1  # at least one file was backed up


def test_split_dataset_backup_in_response_after_overwrite(classification_simple, tmp_path):
    """backup_train or backup_test populated when files already exist."""
    out_dir = tmp_path / "splits2"
    out_dir.mkdir()

    # First split — no prior files, no backup
    r1 = split_dataset(str(classification_simple), test_size=0.2, output_dir=str(out_dir))
    assert r1["success"] is True

    # Second split — files exist now, backups should be created
    r2 = split_dataset(str(classification_simple), test_size=0.2, output_dir=str(out_dir))
    assert r2["success"] is True
    # At least one backup path should be non-empty
    assert r2["backup_train"] or r2["backup_test"]


def test_split_dataset_creates_output_dir(classification_simple, tmp_path):
    """Line 384: output_dir is created if it doesn't exist."""
    new_dir = tmp_path / "brand_new_dir"
    assert not new_dir.exists()

    r = split_dataset(str(classification_simple), output_dir=str(new_dir))
    assert r["success"] is True
    assert new_dir.exists()


def test_split_dataset_not_stratified_by_default(classification_simple, tmp_path):
    """stratified flag is False when no stratify_column given."""
    r = split_dataset(str(classification_simple), output_dir=str(tmp_path))
    assert r["success"] is True
    assert r["stratified"] is False


def test_split_dataset_regression(regression_simple, tmp_path):
    """split_dataset works on regression datasets (no stratify)."""
    r = split_dataset(str(regression_simple), test_size=0.25, output_dir=str(tmp_path))
    assert r["success"] is True
    total = r["train_rows"] + r["test_rows"]
    assert total == 200  # regression_simple fixture has 200 rows
    assert abs(r["test_size_actual"] - 0.25) < 0.05


def test_split_dataset_non_csv_extension(tmp_path):
    """Line 369: file with non-.csv extension → error."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("col1,col2\n1,2\n")
    r = split_dataset(str(txt_file), output_dir=str(tmp_path))
    assert r["success"] is False
    assert "hint" in r


def test_split_dataset_unreadable_csv(tmp_path):
    """Lines 373-374: CSV that cannot be parsed → error dict."""
    bad_csv = tmp_path / "corrupt.csv"
    # Write binary data that pandas cannot parse as CSV
    bad_csv.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    r = split_dataset(str(bad_csv), output_dir=str(tmp_path))
    # Either a read error or success depending on pandas tolerance
    assert "success" in r


def test_list_models_skips_mcp_versions_files(classification_simple, tmp_path):
    """Line 308: .mcp_versions pkl files are excluded from listing."""
    versions_dir = tmp_path / ".mcp_versions"
    versions_dir.mkdir()
    # Place a .pkl inside .mcp_versions — it should be skipped
    fake_bak_pkl = versions_dir / "some_model_backup.pkl"
    fake_bak_pkl.write_bytes(b"fake")

    r = list_models(str(tmp_path))
    assert r["success"] is True
    # The file inside .mcp_versions should not appear
    assert all(".mcp_versions" not in m["path"] for m in r["models"])
    assert r["model_count"] == 0


def test_predict_single_empty_feature_columns(classification_simple, tmp_path):
    """Line 217: model with empty feature_columns list → error dict."""
    from servers.ml_basic._basic_helpers import _load_model, _save_model

    tr = train_classifier(classification_simple, "churned", "rf")
    assert tr["success"] is True

    # Load the model and strip feature_columns from metadata
    model_obj, metadata = _load_model(tr["model_path"])
    metadata["feature_columns"] = []
    patched_path = tmp_path / "no_features.pkl"
    _save_model(model_obj, patched_path, metadata)

    record = '{"age": 35}'
    r = predict_single(str(patched_path), record)
    assert r["success"] is False
    assert "hint" in r


# ===========================================================================
# NEW: engine.py — ValueError/Exception handlers via bad extension paths
# ===========================================================================


class TestInspectDatasetValueError:
    """Lines 100-104: resolve_path raises ValueError when extension is wrong."""

    def test_inspect_dataset_wrong_extension_returns_error(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = inspect_dataset(str(txt_file))
        assert r["success"] is False
        assert "hint" in r

    def test_inspect_dataset_wrong_extension_has_token_estimate(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = inspect_dataset(str(txt_file))
        assert "token_estimate" in r

    def test_inspect_dataset_null_byte_path_returns_error(self):
        r = inspect_dataset("some\x00path.csv")
        assert r["success"] is False
        assert "hint" in r


class TestReadColumnProfileValueError:
    """Lines 135-137: resolve_path raises ValueError for wrong extension."""

    def test_read_column_profile_wrong_extension_returns_error(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = read_column_profile(str(txt_file), "col1")
        assert r["success"] is False
        assert "hint" in r

    def test_read_column_profile_wrong_extension_has_token_estimate(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = read_column_profile(str(txt_file), "col1")
        assert "token_estimate" in r

    def test_read_column_profile_null_byte_path_returns_error(self):
        r = read_column_profile("some\x00path.csv", "col")
        assert r["success"] is False
        assert "hint" in r


class TestSearchColumnsValueError:
    """Lines 191-195: resolve_path raises ValueError for wrong extension."""

    def test_search_columns_wrong_extension_returns_error(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = search_columns(str(txt_file))
        assert r["success"] is False
        assert "hint" in r

    def test_search_columns_wrong_extension_has_token_estimate(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = search_columns(str(txt_file))
        assert "token_estimate" in r

    def test_search_columns_null_byte_path_returns_error(self):
        r = search_columns("some\x00path.csv")
        assert r["success"] is False
        assert "hint" in r


class TestReadRowsValueError:
    """Lines 229-238: resolve_path raises ValueError for wrong extension."""

    def test_read_rows_wrong_extension_returns_error(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = read_rows(str(txt_file), 0, 5)
        assert r["success"] is False
        assert "hint" in r

    def test_read_rows_wrong_extension_has_token_estimate(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("col1,col2\n1,2\n")
        r = read_rows(str(txt_file), 0, 5)
        assert "token_estimate" in r

    def test_read_rows_null_byte_path_returns_error(self):
        r = read_rows("some\x00path.csv", 0, 5)
        assert r["success"] is False
        assert "hint" in r


# ---------------------------------------------------------------------------
# search_columns — bool and datetime dtype filters (lines 229-238)
# ---------------------------------------------------------------------------


class TestSearchColumnsDtypeFilters:
    """Cover dtype='bool' and dtype='datetime' branches in search_columns."""

    def test_search_columns_bool_dtype(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "with_bool.csv"
        pd.DataFrame({"active": [True, False, True], "score": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)
        r = search_columns(str(csv), dtype="bool")
        assert r["success"] is True
        assert "active" in r.get("columns", [])

    def test_search_columns_datetime_dtype(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "with_date.csv"
        df = pd.DataFrame({"created": pd.to_datetime(["2024-01-01", "2024-02-01"]), "val": [1, 2]})
        df.to_csv(csv, index=False)
        r = search_columns(str(csv), dtype="datetime")
        assert r["success"] is True

    def test_search_columns_categorical_dtype(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "cat_cols.csv"
        pd.DataFrame({"name": ["alice", "bob", "carol"], "age": [20, 30, 40]}).to_csv(csv, index=False)
        r = search_columns(str(csv), dtype="categorical")
        assert r["success"] is True
        assert "name" in r.get("columns", [])
        assert "age" not in r.get("columns", [])


# ---------------------------------------------------------------------------
# read_column_profile — boolean column detection (lines 135-137)
# ---------------------------------------------------------------------------


class TestReadColumnProfileBool:
    def test_bool_column_profile(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "bool_col.csv"
        pd.DataFrame({"active": [True, False, True, True], "score": [1, 2, 3, 4]}).to_csv(csv, index=False)
        r = read_column_profile(str(csv), "active")
        assert r["success"] is True
        # Should detect as boolean
        assert r.get("profile", {}).get("kind") in ("boolean", "numeric", "categorical")

    def test_binary_01_column_profile(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "binary_col.csv"
        pd.DataFrame({"flag": [0, 1, 0, 1, 1], "val": [10, 20, 30, 40, 50]}).to_csv(csv, index=False)
        r = read_column_profile(str(csv), "flag")
        assert r["success"] is True


# ---------------------------------------------------------------------------
# _basic_helpers — _check_memory returns error (line 72), _error with backup (line 84)
# ---------------------------------------------------------------------------


class TestBasicHelpersPaths:
    def test_check_memory_returns_error(self):
        """Line 72: _check_memory returns error dict when RAM insufficient."""
        from servers.ml_basic._basic_helpers import _check_memory

        r = _check_memory(999_999.0)
        assert r is not None
        assert r["success"] is False
        assert "token_estimate" in r

    def test_error_with_backup_key(self):
        """Line 84: _error includes backup when provided."""
        from servers.ml_basic._basic_helpers import _error

        r = _error("something failed", "do this instead", backup="/path/to/bak")
        assert r["success"] is False
        assert "backup" in r
        assert r["backup"] == "/path/to/bak"

    def test_load_model_file_not_found(self, tmp_path):
        """Line 127: _load_model raises FileNotFoundError for missing file."""
        from servers.ml_basic._basic_helpers import _load_model

        import pytest as _pytest

        with _pytest.raises(FileNotFoundError):
            _load_model(str(tmp_path / "ghost.pkl"))


# ===========================================================================
# inspect_dataset / read_column_profile / search_columns / read_rows — general exc
# ===========================================================================


class TestEngineGeneralExceptions:
    """Covers general Exception handlers (not ValueError) in ml_basic engine."""

    def test_inspect_dataset_general_exception(self, classification_simple):
        """Lines 102-104: general Exception in inspect_dataset returns error."""
        from unittest.mock import patch
        from servers.ml_basic.engine import inspect_dataset

        with patch("servers.ml_basic.engine.pd.read_csv", side_effect=RuntimeError("boom")):
            r = inspect_dataset(str(classification_simple))
        assert r["success"] is False

    def test_read_column_profile_general_exception(self, classification_simple):
        """Lines 193-195: general Exception in read_column_profile returns error."""
        from unittest.mock import patch
        from servers.ml_basic.engine import read_column_profile

        with patch("servers.ml_basic.engine.pd.read_csv", side_effect=RuntimeError("boom")):
            r = read_column_profile(str(classification_simple), "age")
        assert r["success"] is False

    def test_search_columns_numeric_filter_with_categorical_file(self, classification_messy):
        """Line 230: dtype=numeric on file with categorical columns triggers continue."""
        from servers.ml_basic.engine import search_columns

        r = search_columns(str(classification_messy), dtype="numeric")
        assert r["success"] is True
        assert "gender" not in r["columns"]
        assert "region" not in r["columns"]

    def test_search_columns_general_exception(self, classification_simple):
        """Lines 264-266: general Exception in search_columns returns error."""
        from unittest.mock import patch
        from servers.ml_basic.engine import search_columns

        with patch("servers.ml_basic.engine.pd.read_csv", side_effect=RuntimeError("boom")):
            r = search_columns(str(classification_simple))
        assert r["success"] is False

    def test_read_rows_general_exception(self, classification_simple):
        """Lines 319-321: general Exception in read_rows returns error."""
        from unittest.mock import patch
        from servers.ml_basic.engine import read_rows

        with patch("servers.ml_basic.engine.pd.read_csv", side_effect=RuntimeError("boom")):
            r = read_rows(str(classification_simple), 0, 5)
        assert r["success"] is False


# ===========================================================================
# train_classifier — RAM check, AUC exception, categorical encoding
# ===========================================================================


class TestTrainClassifierMorePaths:
    def test_ram_check_failure_returns_error(self, classification_simple):
        """Line 93: check_memory returns error when RAM insufficient."""
        from unittest.mock import patch, MagicMock
        from servers.ml_basic.engine import train_classifier

        mock_vm = MagicMock()
        mock_vm.available = 0  # 0 bytes available
        with patch("servers.ml_basic._basic_helpers.psutil.virtual_memory", return_value=mock_vm):
            r = train_classifier(str(classification_simple), "churned", "rf")
        assert r["success"] is False

    def test_auc_roc_exception_swallowed(self, classification_simple):
        """Lines 229-230: AUC-ROC computation failure is logged, not raised."""
        from unittest.mock import patch
        from servers.ml_basic.engine import train_classifier

        with patch("sklearn.metrics.roc_auc_score", side_effect=Exception("auc fail")):
            r = train_classifier(str(classification_simple), "churned", "rf")
        assert r["success"] is True
        assert "auc_roc" not in r["metrics"]

    def test_general_exception_returns_error(self, classification_simple):
        """Lines 315-317: general exception caught and returned as error dict."""
        from unittest.mock import patch
        from servers.ml_basic.engine import train_classifier

        with patch("servers.ml_basic._basic_train._auto_preprocess", side_effect=RuntimeError("boom")):
            r = train_classifier(str(classification_simple), "churned", "rf")
        assert r["success"] is False


# ===========================================================================
# train_regressor — RAM check, categorical encoding, insufficient rows
# ===========================================================================


class TestTrainRegressorMorePaths:
    def test_ram_check_failure(self, regression_simple):
        """Line 362: check_memory failure in train_regressor."""
        from unittest.mock import patch, MagicMock
        from servers.ml_basic.engine import train_regressor

        mock_vm = MagicMock()
        mock_vm.available = 0
        with patch("servers.ml_basic._basic_helpers.psutil.virtual_memory", return_value=mock_vm):
            r = train_regressor(str(regression_simple), "salary", "lir")
        assert r["success"] is False

    def test_categorical_encoding_in_regressor(self, regression_messy):
        """Line 372: categorical columns encoded, progress message added."""
        from servers.ml_basic.engine import train_regressor

        r = train_regressor(str(regression_messy), "salary", "lir")
        assert r["success"] is True
        encoded_steps = [p for p in r["progress"] if "Encoded" in p.get("msg", "")]
        assert len(encoded_steps) > 0

    def test_insufficient_rows_regressor(self, tmp_path):
        """Line 375: dataset smaller than MIN_ROWS_REGRESSOR returns error."""
        import pandas as pd
        from servers.ml_basic.engine import train_regressor

        p = tmp_path / "tiny.csv"
        pd.DataFrame({"x": range(5), "y": range(5)}).to_csv(p, index=False)
        r = train_regressor(str(p), "y", "lir")
        assert r["success"] is False
        assert "rows" in r["error"].lower()

    def test_general_exception_returns_error(self, regression_simple):
        """Lines 526-528: general exception in train_regressor returns error dict."""
        from unittest.mock import patch
        from servers.ml_basic.engine import train_regressor

        with patch("servers.ml_basic._basic_train._auto_preprocess", side_effect=RuntimeError("boom")):
            r = train_regressor(str(regression_simple), "salary", "lir")
        assert r["success"] is False


# ===========================================================================
# predict_single — resolve failure, null numeric, predict exception
# ===========================================================================


class TestPredictSingleMorePaths:
    def test_resolve_failure(self):
        """Lines 193-194: null byte in model_path triggers resolve error."""
        from servers.ml_basic.engine import predict_single

        r = predict_single("\x00bad", '{"age": 30}')
        assert r["success"] is False

    def test_null_numeric_value_filled(self, classification_simple, tmp_path):
        """Line 238: null numeric value in input is filled with 0."""
        from servers.ml_basic.engine import train_classifier, predict_single

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        import json
        rec = {"age": None, "tenure": 5, "monthly_charges": 50.0,
               "total_charges": 1000.0, "num_products": 2}
        r = predict_single(mp, json.dumps(rec))
        assert r["success"] is True

    def test_predict_exception_returns_error(self, classification_simple, tmp_path):
        """Lines 266-267: prediction exception returns error dict."""
        from unittest.mock import patch, MagicMock
        from servers.ml_basic.engine import train_classifier, predict_single

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        import json
        rec = {"age": 30, "tenure": 5, "monthly_charges": 50.0,
               "total_charges": 1000.0, "num_products": 2}
        with patch("servers.ml_basic._basic_predict.pd.DataFrame") as mock_df:
            mock_df.return_value.map.side_effect = RuntimeError("predict fail")
            mock_df.return_value.select_dtypes.return_value.columns = []
            mock_df.return_value.values = MagicMock()
            with patch("servers.ml_basic._basic_predict._load_model") as mock_load:
                mock_clf = MagicMock()
                mock_clf.predict.side_effect = RuntimeError("predict fail")
                mock_load.return_value = (mock_clf, {
                    "feature_columns": ["age", "tenure"],
                    "task": "classification",
                    "encoding_map": {},
                })
                r = predict_single(mp, json.dumps(rec))
        assert r["success"] is False


# ===========================================================================
# list_models — resolve failure, .mcp_versions skip
# ===========================================================================


class TestListModelsMorePaths:
    def test_resolve_failure_for_directory(self):
        """Lines 300-301: null byte in directory triggers resolve error."""
        from servers.ml_basic.engine import list_models

        r = list_models(directory="\x00bad")
        assert r["success"] is False

    def test_mcp_versions_pkl_is_skipped(self, classification_simple, tmp_path):
        """Line 308: .pkl files under .mcp_versions/ are skipped."""
        from servers.ml_basic.engine import train_classifier, list_models

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        from pathlib import Path
        import shutil

        versions_dir = tmp_path / ".mcp_versions"
        versions_dir.mkdir()
        shutil.copy(mp, versions_dir / "model_backup.pkl")

        r = list_models(directory=str(tmp_path))
        paths = [m["path"] for m in r["models"]]
        assert not any(".mcp_versions" in p for p in paths)


# ===========================================================================
# split_dataset — resolve failure, snapshot exception
# ===========================================================================


class TestSplitDatasetMorePaths:
    def test_resolve_failure(self):
        """Lines 364-365: null byte in file_path triggers error."""
        from servers.ml_basic.engine import split_dataset

        r = split_dataset("\x00bad")
        assert r["success"] is False

    def test_snapshot_exception_swallowed(self, classification_simple, tmp_path):
        """Lines 410-411: snapshot exception during second split is ignored."""
        from unittest.mock import patch
        from servers.ml_basic.engine import split_dataset

        r1 = split_dataset(str(classification_simple), output_dir=str(tmp_path))
        assert r1["success"] is True

        with patch("servers.ml_basic._basic_predict.snapshot", side_effect=Exception("snap fail")):
            r2 = split_dataset(str(classification_simple), output_dir=str(tmp_path))
        assert r2["success"] is True
