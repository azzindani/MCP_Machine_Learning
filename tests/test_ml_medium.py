"""Tests for ml_medium engine (Tier 2).

Fixture column names:
  classification_simple: age, tenure, monthly_charges, total_charges, num_products, churned
  classification_messy:  age, gender, region, monthly_charges, total_charges,
                         num_products, support_calls, discount, churned
  regression_simple:     age, experience, education_level, department, performance_score, salary
  regression_messy:      age, experience, city, education_level, performance_score, salary
  clustering_simple:     x, y
"""

from pathlib import Path

import pandas as pd
import pytest

from servers.ml_medium.engine import (
    compare_models,
    detect_outliers,
    read_receipt,
    run_clustering,
    run_preprocessing,
    train_with_cv,
)

# ---------------------------------------------------------------------------
# run_preprocessing
# ---------------------------------------------------------------------------


class TestRunPreprocessing:
    def test_success_fill_nulls(self, classification_messy):
        r = run_preprocessing(
            classification_messy,
            [{"op": "fill_nulls", "column": "age", "strategy": "median"}],
        )
        assert r["success"] is True
        assert r["op"] == "run_preprocessing"
        assert r["applied"] == 1

    def test_file_not_found(self, home_tmp):
        path = str(home_tmp / "nonexistent.csv")
        r = run_preprocessing(path, [])
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple):
        r = run_preprocessing(classification_simple, [])
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = run_preprocessing(classification_simple, [])
        assert "progress" in r
        assert isinstance(r["progress"], list)

    def test_snapshot_created(self, classification_simple):
        r = run_preprocessing(
            classification_simple,
            [{"op": "fill_nulls", "column": "age", "strategy": "mean"}],
        )
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple):
        r = run_preprocessing(
            classification_simple,
            [{"op": "fill_nulls", "column": "age", "strategy": "mean"}],
        )
        assert "backup" in r

    def test_dry_run(self, classification_simple):
        orig_df = pd.read_csv(classification_simple)
        r = run_preprocessing(
            classification_simple,
            [{"op": "drop_column", "column": "age"}],
            dry_run=True,
        )
        assert r["success"] is True
        assert r.get("dry_run") is True
        # File should not be modified
        df = pd.read_csv(classification_simple)
        assert "age" in df.columns

    def test_constrained_mode(self, classification_simple, constrained_mode):
        r = run_preprocessing(classification_simple, [])
        assert r["success"] is True

    def test_invalid_op(self, classification_simple):
        r = run_preprocessing(classification_simple, [{"op": "invalid_op_xyz"}])
        assert r["success"] is False
        assert "hint" in r

    def test_invalid_fill_strategy(self, classification_simple):
        r = run_preprocessing(
            classification_simple,
            [{"op": "fill_nulls", "column": "age", "strategy": "badstrat"}],
        )
        assert r["success"] is False

    def test_label_encode(self, classification_messy):
        r = run_preprocessing(
            classification_messy,
            [{"op": "label_encode", "column": "gender"}],
        )
        assert r["success"] is True

    def test_scale_standard(self, classification_simple):
        r = run_preprocessing(
            classification_simple,
            [{"op": "scale", "columns": ["age", "tenure"], "method": "standard"}],
        )
        assert r["success"] is True

    def test_drop_duplicates(self, classification_messy):
        r = run_preprocessing(
            classification_messy,
            [{"op": "drop_duplicates"}],
        )
        assert r["success"] is True

    def test_output_path(self, classification_simple, home_tmp):
        out = str(home_tmp / "out_preproc.csv")
        r = run_preprocessing(
            classification_simple,
            [{"op": "fill_nulls", "column": "age", "strategy": "median"}],
            output_path=out,
        )
        assert r["success"] is True
        assert Path(out).exists()

    def test_too_many_ops(self, classification_simple):
        ops = [{"op": "fill_nulls", "column": "age", "strategy": "median"}] * 51
        r = run_preprocessing(classification_simple, ops)
        assert r["success"] is False


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    def test_success_iqr(self, regression_messy):
        r = detect_outliers(regression_messy, ["age"], method="iqr")
        assert r["success"] is True
        assert r["op"] == "detect_outliers"
        assert len(r["results"]) == 1

    def test_success_std(self, regression_messy):
        r = detect_outliers(regression_messy, ["age"], method="std")
        assert r["success"] is True

    def test_file_not_found(self, home_tmp):
        r = detect_outliers(str(home_tmp / "nope.csv"), ["col"])
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, regression_simple):
        r = detect_outliers(regression_simple, ["age"])
        assert "token_estimate" in r

    def test_progress_present(self, regression_simple):
        r = detect_outliers(regression_simple, ["age"])
        assert "progress" in r

    def test_invalid_method(self, regression_simple):
        r = detect_outliers(regression_simple, ["age"], method="badmethod")
        assert r["success"] is False

    def test_missing_column(self, regression_simple):
        r = detect_outliers(regression_simple, ["nonexistent_col"])
        assert r["success"] is False

    def test_outlier_count_field(self, regression_messy):
        r = detect_outliers(regression_messy, ["age"])
        assert "outlier_count" in r["results"][0]

    def test_sample_outliers_bounded(self, regression_messy):
        r = detect_outliers(regression_messy, ["age"])
        assert len(r["results"][0]["sample_outliers"]) <= 5

    def test_multiple_columns(self, regression_messy):
        r = detect_outliers(regression_messy, ["age", "experience"])
        assert len(r["results"]) == 2


# ---------------------------------------------------------------------------
# train_with_cv
# ---------------------------------------------------------------------------


class TestTrainWithCV:
    def test_success_classification(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "rf", "classification", n_splits=3)
        assert r["success"] is True
        assert r["op"] == "train_with_cv"
        assert "mean_metrics" in r
        assert "fold_scores" in r

    def test_success_regression(self, regression_simple):
        r = train_with_cv(regression_simple, "salary", "rfr", "regression", n_splits=3)
        assert r["success"] is True
        assert "r2_mean" in r["mean_metrics"]

    def test_file_not_found(self, home_tmp):
        r = train_with_cv(str(home_tmp / "nope.csv"), "target", "rf", "classification")
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "lr", "classification", n_splits=3)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "lr", "classification", n_splits=3)
        assert "progress" in r

    def test_snapshot_created(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "dtc", "classification", n_splits=3)
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "nb", "classification", n_splits=3)
        assert "backup" in r

    def test_dry_run(self, classification_simple):
        r = train_with_cv(
            classification_simple, "churned", "rf", "classification", n_splits=3, dry_run=True
        )
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_constrained_mode(self, classification_simple, constrained_mode):
        r = train_with_cv(classification_simple, "churned", "lr", "classification", n_splits=5)
        assert r.get("n_splits", 5) <= 3

    def test_invalid_task(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "rf", "badtask")
        assert r["success"] is False

    def test_invalid_model(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "zzz", "classification")
        assert r["success"] is False

    def test_column_not_found(self, classification_simple):
        r = train_with_cv(classification_simple, "no_such_col", "rf", "classification")
        assert r["success"] is False


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_success_classification(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "rf"])
        assert r["success"] is True
        assert r["op"] == "compare_models"
        assert len(r["results"]) >= 1

    def test_success_regression(self, regression_simple):
        r = compare_models(regression_simple, "salary", "regression", ["lir", "rfr"])
        assert r["success"] is True
        assert "best_model" in r

    def test_file_not_found(self, home_tmp):
        r = compare_models(str(home_tmp / "nope.csv"), "churned", "classification", ["lr"])
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr"])
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr"])
        assert "progress" in r

    def test_snapshot_created(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "dtc"])
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr"])
        assert "backup" in r

    def test_dry_run(self, classification_simple):
        r = compare_models(
            classification_simple, "churned", "classification", ["lr", "rf"], dry_run=True
        )
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_constrained_mode(self, classification_simple, constrained_mode):
        many = ["lr", "svm", "rf", "dtc", "knn", "nb", "xgb"]
        r = compare_models(classification_simple, "churned", "classification", many)
        assert len(r.get("results", [])) <= 3

    def test_sorted_by_f1(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "rf", "dtc"])
        assert r["success"] is True
        f1s = [res["f1_weighted"] for res in r["results"] if "f1_weighted" in res]
        assert f1s == sorted(f1s, reverse=True) or len(f1s) <= 1

    def test_invalid_model(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "zzz"])
        assert r["success"] is False

    def test_invalid_task(self, classification_simple):
        r = compare_models(classification_simple, "churned", "badtask", ["lr"])
        assert r["success"] is False

    def test_rank_field_present(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "dtc"])
        assert r["success"] is True
        for res in r["results"]:
            assert "rank" in res


# ---------------------------------------------------------------------------
# run_clustering
# ---------------------------------------------------------------------------


class TestRunClustering:
    def test_success_kmeans(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", n_clusters=3)
        assert r["success"] is True
        assert r["op"] == "run_clustering"
        assert "label_counts" in r

    def test_success_dbscan(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "dbscan", eps=1.0, min_samples=5)
        assert r["success"] is True

    def test_success_meanshift(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "meanshift")
        assert r["success"] is True

    def test_file_not_found(self, home_tmp):
        r = run_clustering(str(home_tmp / "nope.csv"), ["x"], "kmeans")
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans")
        assert "token_estimate" in r

    def test_progress_present(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans")
        assert "progress" in r

    def test_snapshot_when_save_labels(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", save_labels=True)
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", save_labels=True)
        assert "backup" in r

    def test_dry_run(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_constrained_mode(self, clustering_simple, constrained_mode):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans")
        assert r["success"] is True

    def test_invalid_algorithm(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "badalgos")
        assert r["success"] is False

    def test_pca_reduction(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", reduce_dims="pca")
        assert r["success"] is True

    def test_missing_column(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "nonexistent"], "kmeans")
        assert r["success"] is False

    def test_inertia_in_kmeans_result(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans")
        assert "inertia" in r


# ---------------------------------------------------------------------------
# read_receipt
# ---------------------------------------------------------------------------


class TestReadReceipt:
    def test_success_empty(self, classification_simple):
        r = read_receipt(classification_simple)
        assert r["success"] is True
        assert r["op"] == "read_receipt"
        assert isinstance(r["entries"], list)

    def test_token_estimate_present(self, classification_simple):
        r = read_receipt(classification_simple)
        assert "token_estimate" in r

    def test_entries_after_operation(self, classification_simple):
        run_preprocessing(
            classification_simple,
            [{"op": "fill_nulls", "column": "age", "strategy": "median"}],
        )
        r = read_receipt(classification_simple)
        assert r["success"] is True
        assert r["entry_count"] >= 1

    def test_path_outside_home(self, tmp_path):
        r = read_receipt(str(tmp_path / "outside.csv"))
        assert r["success"] is False
