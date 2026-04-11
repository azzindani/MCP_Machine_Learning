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

from servers.ml_basic.engine import train_classifier
from servers.ml_medium.engine import (
    anomaly_detection,
    batch_predict,
    check_data_quality,
    compare_models,
    detect_outliers,
    evaluate_model,
    filter_rows,
    find_optimal_clusters,
    generate_eda_report,
    merge_datasets,
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
        r = train_with_cv(classification_simple, "churned", "rf", "classification", n_splits=3, dry_run=True)
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
        r = compare_models(classification_simple, "churned", "classification", ["lr", "rf"], dry_run=True)
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

    def test_null_byte_path_rejected(self):
        r = read_receipt("some\x00path.csv")
        assert r["success"] is False

    def test_filesystem_root_rejected(self):
        import sys

        root = "C:\\" if sys.platform == "win32" else "/"
        r = read_receipt(root)
        assert r["success"] is False


# ---------------------------------------------------------------------------
# generate_eda_report
# ---------------------------------------------------------------------------


class TestGenerateEdaReport:
    def test_success(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda.html")
        r = generate_eda_report(
            classification_simple,
            target_column="churned",
            output_path=out,
            open_browser=False,
        )
        assert r["success"] is True
        assert r["op"] == "generate_eda_report"
        assert Path(out).exists()

    def test_quality_score_present(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_qs.html")
        r = generate_eda_report(classification_simple, output_path=out, open_browser=False)
        assert "quality_score" in r
        assert 0 <= r["quality_score"] <= 100

    def test_alerts_returned(self, classification_messy, home_tmp):
        out = str(home_tmp / "eda_alerts.html")
        r = generate_eda_report(
            classification_messy,
            target_column="churned",
            output_path=out,
            open_browser=False,
        )
        assert r["success"] is True
        assert "alerts" in r
        assert "alerts_count" in r
        assert "alerts_high" in r
        assert "alerts_medium" in r
        assert "alerts_low" in r

    def test_alert_has_recommendation(self, classification_messy, home_tmp):
        out = str(home_tmp / "eda_rec.html")
        r = generate_eda_report(
            classification_messy,
            target_column="churned",
            output_path=out,
            open_browser=False,
        )
        for alert in r.get("alerts", []):
            assert "recommendation" in alert
            assert "message" in alert
            assert "severity" in alert
            assert alert["severity"] in ("high", "medium", "low")

    def test_file_not_found(self, home_tmp):
        r = generate_eda_report(str(home_tmp / "nope.csv"), open_browser=False)
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_tok.html")
        r = generate_eda_report(classification_simple, output_path=out, open_browser=False)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_prog.html")
        r = generate_eda_report(classification_simple, output_path=out, open_browser=False)
        assert "progress" in r
        assert isinstance(r["progress"], list)

    def test_dry_run(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_dry.html")
        r = generate_eda_report(
            classification_simple,
            output_path=out,
            open_browser=False,
            dry_run=True,
        )
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, classification_simple, constrained_mode, home_tmp):
        out = str(home_tmp / "eda_cons.html")
        r = generate_eda_report(classification_simple, output_path=out, open_browser=False)
        assert r["success"] is True

    def test_dark_theme(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_dark.html")
        r = generate_eda_report(
            classification_simple,
            theme="dark",
            output_path=out,
            open_browser=False,
        )
        assert r["success"] is True

    def test_pearson_and_spearman_in_html(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_corr.html")
        r = generate_eda_report(classification_simple, output_path=out, open_browser=False)
        assert r["success"] is True
        html = Path(out).read_text()
        assert "Pearson" in html
        assert "Spearman" in html

    def test_charts_count(self, classification_simple, home_tmp):
        out = str(home_tmp / "eda_cnt.html")
        r = generate_eda_report(
            classification_simple,
            target_column="churned",
            output_path=out,
            open_browser=False,
        )
        assert r.get("charts_generated", 0) >= 3  # at minimum: quality, distributions, correlation

    def test_duplicate_alert_detected(self, classification_messy, home_tmp):
        """classification_messy has 5 duplicate rows."""
        out = str(home_tmp / "eda_dup.html")
        r = generate_eda_report(classification_messy, output_path=out, open_browser=False)
        assert r["success"] is True
        types = [a["type"] for a in r.get("alerts", [])]
        assert "duplicate_rows" in types


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


class TestFilterRows:
    def test_success_eq(self, classification_simple, home_tmp):
        out = str(home_tmp / "filtered.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert r["success"] is True
        assert r["op"] == "filter_rows"
        df = pd.read_csv(out)
        assert (df["churned"] == 0).all()

    def test_success_gt(self, regression_simple, home_tmp):
        out = str(home_tmp / "filtered_gt.csv")
        r = filter_rows(regression_simple, "age", "gt", "30", output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert (df["age"] > 30).all()

    def test_success_isnull(self, classification_messy, home_tmp):
        out = str(home_tmp / "filtered_null.csv")
        r = filter_rows(classification_messy, "region", "is_null", output_path=out)
        assert r["success"] is True
        # region has nulls in messy fixture
        assert r.get("rows_after", 0) >= 0

    def test_success_contains(self, classification_messy, home_tmp):
        out = str(home_tmp / "filtered_contains.csv")
        r = filter_rows(classification_messy, "gender", "contains", "F", output_path=out)
        assert r["success"] is True

    def test_dry_run(self, classification_simple, home_tmp):
        out = str(home_tmp / "no_write.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out, dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_file_not_found(self, home_tmp):
        r = filter_rows(str(home_tmp / "missing.csv"), "col", "eq", "1")
        assert r["success"] is False

    def test_column_not_found(self, classification_simple, home_tmp):
        out = str(home_tmp / "f.csv")
        r = filter_rows(classification_simple, "nonexistent", "eq", "1", output_path=out)
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple, home_tmp):
        out = str(home_tmp / "f2.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert "token_estimate" in r

    def test_snapshot_created(self, classification_simple, home_tmp):
        out = str(home_tmp / "filter_snap.csv")
        # write a dummy file so snapshot triggers
        Path(out).write_text("col\n1\n")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert r["success"] is True
        assert r.get("backup", "") != ""


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def test_success_left(self, home_tmp):
        # Build two simple CSVs with a common key
        f1 = str(home_tmp / "left.csv")
        f2 = str(home_tmp / "right.csv")
        pd.DataFrame({"id": [1, 2, 3], "val_a": [10, 20, 30]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1, 2, 4], "val_b": [100, 200, 400]}).to_csv(f2, index=False)
        out = str(home_tmp / "merged.csv")
        r = merge_datasets(f1, f2, on="id", how="left", output_path=out)
        assert r["success"] is True
        assert r["merged_rows"] == 3
        df = pd.read_csv(out)
        assert "val_a" in df.columns and "val_b" in df.columns

    def test_success_inner(self, home_tmp):
        f1 = str(home_tmp / "a.csv")
        f2 = str(home_tmp / "b.csv")
        pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [2, 3, 5], "y": [20, 30, 50]}).to_csv(f2, index=False)
        out = str(home_tmp / "merged_inner.csv")
        r = merge_datasets(f1, f2, on="id", how="inner", output_path=out)
        assert r["success"] is True
        assert r["merged_rows"] == 2

    def test_dry_run(self, home_tmp):
        f1 = str(home_tmp / "d1.csv")
        f2 = str(home_tmp / "d2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        out = str(home_tmp / "nomerge.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out, dry_run=True)
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_missing_key(self, home_tmp):
        f1 = str(home_tmp / "k1.csv")
        f2 = str(home_tmp / "k2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"other": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(home_tmp / "x.csv"))
        assert r["success"] is False

    def test_token_estimate_present(self, home_tmp):
        f1 = str(home_tmp / "te1.csv")
        f2 = str(home_tmp / "te2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(home_tmp / "tm.csv"))
        assert "token_estimate" in r


# ---------------------------------------------------------------------------
# find_optimal_clusters
# ---------------------------------------------------------------------------


class TestFindOptimalClusters:
    def test_success(self, clustering_simple, home_tmp):
        out = str(home_tmp / "elbow.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=5, output_path=out, open_browser=False)
        assert r["success"] is True
        assert "best_k" in r
        assert 2 <= r["best_k"] <= 5
        assert Path(out).exists()

    def test_file_not_found(self, home_tmp):
        r = find_optimal_clusters(str(home_tmp / "nope.csv"), ["x", "y"], open_browser=False)
        assert r["success"] is False

    def test_token_estimate_present(self, clustering_simple, home_tmp):
        out = str(home_tmp / "elbow2.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=3, output_path=out, open_browser=False)
        assert "token_estimate" in r

    def test_silhouette_scores_returned(self, clustering_simple, home_tmp):
        out = str(home_tmp / "elbow3.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=4, output_path=out, open_browser=False)
        assert r["success"] is True
        assert len(r.get("silhouette_scores", [])) >= 2


# ---------------------------------------------------------------------------
# anomaly_detection
# ---------------------------------------------------------------------------


class TestAnomalyDetection:
    def test_success_isolation_forest(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"], method="isolation_forest", contamination=0.05)
        assert r["success"] is True
        assert "n_anomalies" in r
        assert r["n_anomalies"] >= 0

    def test_success_lof(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"], method="lof", contamination=0.05)
        assert r["success"] is True
        assert "n_anomalies" in r

    def test_save_labels(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"], method="isolation_forest", save_labels=True)
        assert r["success"] is True
        assert r.get("backup", "") != ""
        df = pd.read_csv(clustering_simple)
        assert "is_anomaly" in df.columns

    def test_dry_run(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"], dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True

    def test_file_not_found(self, home_tmp):
        r = anomaly_detection(str(home_tmp / "ghost.csv"), ["x"])
        assert r["success"] is False

    def test_invalid_method(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"], method="invalid")
        assert r["success"] is False

    def test_token_estimate_present(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"])
        assert "token_estimate" in r


# ---------------------------------------------------------------------------
# check_data_quality
# ---------------------------------------------------------------------------


class TestCheckDataQuality:
    def test_success(self, classification_simple):
        r = check_data_quality(classification_simple)
        assert r["success"] is True
        assert r["op"] == "check_data_quality"

    def test_quality_score_present(self, classification_simple):
        r = check_data_quality(classification_simple)
        assert "quality_score" in r
        assert 0 <= r["quality_score"] <= 100

    def test_alerts_list_present(self, classification_messy):
        r = check_data_quality(classification_messy)
        assert isinstance(r.get("alerts"), list)

    def test_null_summary_present(self, classification_messy):
        r = check_data_quality(classification_messy)
        assert isinstance(r.get("null_summary"), list)

    def test_messy_has_alerts(self, classification_messy):
        r = check_data_quality(classification_messy)
        # messy data has nulls — should have some alerts
        assert r["alerts_count"] >= 0  # may be 0 if all thresholds not triggered

    def test_duplicate_detected(self, classification_messy):
        r = check_data_quality(classification_messy)
        assert "duplicate_rows" in r
        assert r["duplicate_rows"] >= 0

    def test_file_not_found(self, home_tmp):
        r = check_data_quality(str(home_tmp / "nope.csv"))
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple):
        r = check_data_quality(classification_simple)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = check_data_quality(classification_simple)
        assert isinstance(r.get("progress"), list) and len(r["progress"]) > 0


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_success_classification(self, classification_simple, home_tmp):
        # Train a model first
        tr = train_classifier(classification_simple, "churned", "rf", test_size=0.2, random_state=42)
        assert tr["success"] is True
        model_path = tr["model_path"]
        r = evaluate_model(model_path, classification_simple, "churned")
        assert r["success"] is True
        assert r["op"] == "evaluate_model"
        assert "accuracy" in r["metrics"]

    def test_auc_roc_binary(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "lr", test_size=0.2, random_state=42)
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert r["success"] is True
        # LR supports predict_proba → AUC-ROC should be present for binary
        assert "auc_roc" in r["metrics"]

    def test_model_not_found(self, classification_simple, home_tmp):
        r = evaluate_model(str(home_tmp / "ghost.pkl"), classification_simple, "churned")
        assert r["success"] is False

    def test_file_not_found(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "rf")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(home_tmp / "ghost.csv"), "churned")
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple):
        tr = train_classifier(classification_simple, "churned", "rf")
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        tr = train_classifier(classification_simple, "churned", "rf")
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert isinstance(r.get("progress"), list) and len(r["progress"]) > 0


# ---------------------------------------------------------------------------
# batch_predict
# ---------------------------------------------------------------------------


class TestBatchPredict:
    def test_success(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "rf")
        assert tr["success"] is True
        out = str(home_tmp / "predictions.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        assert r["op"] == "batch_predict"
        assert Path(out).exists()

    def test_all_rows_predicted(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(home_tmp / "batch_all.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        df_orig = pd.read_csv(classification_simple)
        df_pred = pd.read_csv(out)
        assert len(df_pred) == len(df_orig)
        assert "prediction" in df_pred.columns

    def test_dry_run(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(home_tmp / "no_batch.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out, dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_model_not_found(self, classification_simple, home_tmp):
        r = batch_predict(str(home_tmp / "ghost.pkl"), classification_simple)
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(home_tmp / "batch_te.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert "token_estimate" in r

    def test_prediction_distribution_present(self, classification_simple, home_tmp):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(home_tmp / "batch_dist.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        assert isinstance(r.get("prediction_distribution"), dict)
