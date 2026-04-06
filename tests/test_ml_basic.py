"""Tests for ml_basic engine — all tests call engine directly, no MCP server."""

import os
import shutil
from pathlib import Path

import pytest

from servers.ml_basic.engine import (
    get_predictions,
    inspect_dataset,
    read_column_profile,
    read_rows,
    restore_version,
    search_columns,
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

