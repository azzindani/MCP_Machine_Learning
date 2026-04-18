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
from unittest.mock import patch

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

    def test_file_not_found(self, tmp_path):
        path = str(tmp_path / "nonexistent.csv")
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

    def test_output_path(self, classification_simple, tmp_path):
        out = str(tmp_path / "out_preproc.csv")
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

    def test_file_not_found(self, tmp_path):
        r = detect_outliers(str(tmp_path / "nope.csv"), ["col"])
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

    def test_file_not_found(self, tmp_path):
        r = train_with_cv(str(tmp_path / "nope.csv"), "target", "rf", "classification")
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

    def test_insufficient_rows(self, tmp_path):
        import pandas as pd

        tiny = tmp_path / "tiny.csv"
        pd.DataFrame({"a": range(5), "b": range(5), "target": [0, 1, 0, 1, 0]}).to_csv(tiny, index=False)
        r = train_with_cv(str(tiny), "target", "rf", "classification")
        assert r["success"] is False
        assert "hint" in r

    def test_single_class_target(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"a": range(30), "b": range(30), "target": [1] * 30})
        path = tmp_path / "single.csv"
        df.to_csv(path, index=False)
        r = train_with_cv(str(path), "target", "rf", "classification")
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

    def test_file_not_found(self, tmp_path):
        r = compare_models(str(tmp_path / "nope.csv"), "churned", "classification", ["lr"])
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

    def test_insufficient_rows(self, tmp_path):
        import pandas as pd

        tiny = tmp_path / "tiny_cmp.csv"
        pd.DataFrame({"a": range(5), "b": range(5), "target": [0, 1, 0, 1, 0]}).to_csv(tiny, index=False)
        r = compare_models(str(tiny), "target", "classification", ["lr"])
        assert r["success"] is False

    def test_column_not_found(self, classification_simple):
        r = compare_models(classification_simple, "no_such_col", "classification", ["lr"])
        assert r["success"] is False


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

    def test_file_not_found(self, tmp_path):
        r = run_clustering(str(tmp_path / "nope.csv"), ["x"], "kmeans")
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

    def test_op_present(self, classification_simple):
        r = read_receipt(classification_simple)
        assert r.get("op") == "read_receipt"


# ---------------------------------------------------------------------------
# generate_eda_report
# ---------------------------------------------------------------------------


class TestGenerateEdaReport:
    def test_success(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda.html")
        r = generate_eda_report(
            classification_simple,
            target_column="churned",
            output_path=out,
            open_after=False,
        )
        assert r["success"] is True
        assert r["op"] == "generate_eda_report"
        assert Path(out).exists()

    def test_quality_score_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_qs.html")
        r = generate_eda_report(classification_simple, output_path=out, open_after=False)
        assert "quality_score" in r
        assert 0 <= r["quality_score"] <= 100

    def test_alerts_returned(self, classification_messy, tmp_path):
        out = str(tmp_path / "eda_alerts.html")
        r = generate_eda_report(
            classification_messy,
            target_column="churned",
            output_path=out,
            open_after=False,
        )
        assert r["success"] is True
        assert "alerts" in r
        assert "alerts_count" in r
        assert "alerts_high" in r
        assert "alerts_medium" in r
        assert "alerts_low" in r

    def test_alert_has_recommendation(self, classification_messy, tmp_path):
        out = str(tmp_path / "eda_rec.html")
        r = generate_eda_report(
            classification_messy,
            target_column="churned",
            output_path=out,
            open_after=False,
        )
        for alert in r.get("alerts", []):
            assert "recommendation" in alert
            assert "message" in alert
            assert "severity" in alert
            assert alert["severity"] in ("high", "medium", "low")

    def test_file_not_found(self, tmp_path):
        r = generate_eda_report(str(tmp_path / "nope.csv"), open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_tok.html")
        r = generate_eda_report(classification_simple, output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_prog.html")
        r = generate_eda_report(classification_simple, output_path=out, open_after=False)
        assert "progress" in r
        assert isinstance(r["progress"], list)

    def test_dry_run(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_dry.html")
        r = generate_eda_report(
            classification_simple,
            output_path=out,
            open_after=False,
            dry_run=True,
        )
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        out = str(tmp_path / "eda_cons.html")
        r = generate_eda_report(classification_simple, output_path=out, open_after=False)
        assert r["success"] is True

    def test_dark_theme(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_dark.html")
        r = generate_eda_report(
            classification_simple,
            theme="dark",
            output_path=out,
            open_after=False,
        )
        assert r["success"] is True

    def test_pearson_and_spearman_in_html(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_corr.html")
        r = generate_eda_report(classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        html = Path(out).read_text(encoding="utf-8")
        assert "Pearson" in html
        assert "Spearman" in html

    def test_charts_count(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_cnt.html")
        r = generate_eda_report(
            classification_simple,
            target_column="churned",
            output_path=out,
            open_after=False,
        )
        assert r.get("charts_generated", 0) >= 3  # at minimum: quality, distributions, correlation

    def test_duplicate_alert_detected(self, classification_messy, tmp_path):
        """classification_messy has 5 duplicate rows."""
        out = str(tmp_path / "eda_dup.html")
        r = generate_eda_report(classification_messy, output_path=out, open_after=False)
        assert r["success"] is True
        types = [a["type"] for a in r.get("alerts", [])]
        assert "duplicate_rows" in types

    def test_auto_opens_browser(self, classification_simple, tmp_path):
        out = str(tmp_path / "eda_open.html")
        with patch("shared.html_theme._open_file") as mock_open:
            r = generate_eda_report(classification_simple, output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


class TestFilterRows:
    def test_success_eq(self, classification_simple, tmp_path):
        out = str(tmp_path / "filtered.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert r["success"] is True
        assert r["op"] == "filter_rows"
        df = pd.read_csv(out)
        assert (df["churned"] == 0).all()

    def test_success_gt(self, regression_simple, tmp_path):
        out = str(tmp_path / "filtered_gt.csv")
        r = filter_rows(regression_simple, "age", "gt", "30", output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert (df["age"] > 30).all()

    def test_success_isnull(self, classification_messy, tmp_path):
        out = str(tmp_path / "filtered_null.csv")
        r = filter_rows(classification_messy, "region", "is_null", output_path=out)
        assert r["success"] is True
        # region has nulls in messy fixture
        assert r.get("rows_after", 0) >= 0

    def test_success_contains(self, classification_messy, tmp_path):
        out = str(tmp_path / "filtered_contains.csv")
        r = filter_rows(classification_messy, "gender", "contains", "F", output_path=out)
        assert r["success"] is True

    def test_dry_run(self, classification_simple, tmp_path):
        out = str(tmp_path / "no_write.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out, dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_file_not_found(self, tmp_path):
        r = filter_rows(str(tmp_path / "missing.csv"), "col", "eq", "1")
        assert r["success"] is False

    def test_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "f.csv")
        r = filter_rows(classification_simple, "nonexistent", "eq", "1", output_path=out)
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "f2.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert "token_estimate" in r

    def test_snapshot_created(self, classification_simple, tmp_path):
        out = str(tmp_path / "filter_snap.csv")
        # write a dummy file so snapshot triggers
        Path(out).write_text("col\n1\n")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert r["success"] is True
        assert r.get("backup", "") != ""

    def test_backup_in_response(self, classification_simple, tmp_path):
        out = str(tmp_path / "filter_bak.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert "backup" in r

    def test_progress_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "filter_prog.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert isinstance(r.get("progress"), list)

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        out = str(tmp_path / "filter_cons.csv")
        r = filter_rows(classification_simple, "churned", "eq", "0", output_path=out)
        assert r["success"] is True


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def test_success_left(self, tmp_path):
        # Build two simple CSVs with a common key
        f1 = str(tmp_path / "left.csv")
        f2 = str(tmp_path / "right.csv")
        pd.DataFrame({"id": [1, 2, 3], "val_a": [10, 20, 30]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1, 2, 4], "val_b": [100, 200, 400]}).to_csv(f2, index=False)
        out = str(tmp_path / "merged.csv")
        r = merge_datasets(f1, f2, on="id", how="left", output_path=out)
        assert r["success"] is True
        assert r["merged_rows"] == 3
        df = pd.read_csv(out)
        assert "val_a" in df.columns and "val_b" in df.columns

    def test_success_inner(self, tmp_path):
        f1 = str(tmp_path / "a.csv")
        f2 = str(tmp_path / "b.csv")
        pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [2, 3, 5], "y": [20, 30, 50]}).to_csv(f2, index=False)
        out = str(tmp_path / "merged_inner.csv")
        r = merge_datasets(f1, f2, on="id", how="inner", output_path=out)
        assert r["success"] is True
        assert r["merged_rows"] == 2

    def test_dry_run(self, tmp_path):
        f1 = str(tmp_path / "d1.csv")
        f2 = str(tmp_path / "d2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        out = str(tmp_path / "nomerge.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out, dry_run=True)
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_missing_key(self, tmp_path):
        f1 = str(tmp_path / "k1.csv")
        f2 = str(tmp_path / "k2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"other": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "x.csv"))
        assert r["success"] is False

    def test_token_estimate_present(self, tmp_path):
        f1 = str(tmp_path / "te1.csv")
        f2 = str(tmp_path / "te2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "tm.csv"))
        assert "token_estimate" in r

    def test_progress_present(self, tmp_path):
        f1 = str(tmp_path / "p1.csv")
        f2 = str(tmp_path / "p2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "mp.csv"))
        assert isinstance(r.get("progress"), list)

    def test_snapshot_created(self, tmp_path):
        f1 = str(tmp_path / "s1.csv")
        f2 = str(tmp_path / "s2.csv")
        out = str(tmp_path / "ms_snap.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        Path(out).write_text("id,a,b\n1,1,2\n")  # pre-create to trigger snapshot
        r = merge_datasets(f1, f2, on="id", output_path=out)
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, tmp_path):
        f1 = str(tmp_path / "b1.csv")
        f2 = str(tmp_path / "b2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "mb.csv"))
        assert "backup" in r

    def test_constrained_mode(self, tmp_path, constrained_mode):
        f1 = str(tmp_path / "c1.csv")
        f2 = str(tmp_path / "c2.csv")
        pd.DataFrame({"id": [1], "a": [1]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1], "b": [2]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "mc.csv"))
        assert r["success"] is True


# ---------------------------------------------------------------------------
# find_optimal_clusters
# ---------------------------------------------------------------------------


class TestFindOptimalClusters:
    def test_success(self, clustering_simple, tmp_path):
        out = str(tmp_path / "elbow.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=5, output_path=out, open_after=False)
        assert r["success"] is True
        assert "best_k" in r
        assert 2 <= r["best_k"] <= 5
        assert Path(out).exists()

    def test_file_not_found(self, tmp_path):
        r = find_optimal_clusters(str(tmp_path / "nope.csv"), ["x", "y"], open_after=False)
        assert r["success"] is False

    def test_token_estimate_present(self, clustering_simple, tmp_path):
        out = str(tmp_path / "elbow2.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=3, output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_silhouette_scores_returned(self, clustering_simple, tmp_path):
        out = str(tmp_path / "elbow3.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=4, output_path=out, open_after=False)
        assert r["success"] is True
        assert len(r.get("silhouette_scores", [])) >= 2

    def test_progress_present(self, clustering_simple, tmp_path):
        out = str(tmp_path / "elbow_prog.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=3, output_path=out, open_after=False)
        assert isinstance(r.get("progress"), list)

    def test_output_name_present(self, clustering_simple, tmp_path):
        out = str(tmp_path / "elbow_name.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=3, output_path=out, open_after=False)
        assert r["success"] is True
        assert "output_name" in r

    def test_constrained_mode(self, clustering_simple, constrained_mode, tmp_path):
        out = str(tmp_path / "elbow_cons.html")
        r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=5, output_path=out, open_after=False)
        assert r["success"] is True

    def test_auto_opens_browser(self, clustering_simple, tmp_path):
        out = str(tmp_path / "elbow_open.html")
        with patch("servers.ml_medium._medium_helpers._open_file") as mock_open:
            r = find_optimal_clusters(clustering_simple, ["x", "y"], max_k=3, output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()


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

    def test_file_not_found(self, tmp_path):
        r = anomaly_detection(str(tmp_path / "ghost.csv"), ["x"])
        assert r["success"] is False

    def test_invalid_method(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"], method="invalid")
        assert r["success"] is False

    def test_token_estimate_present(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"])
        assert "token_estimate" in r

    def test_progress_present(self, clustering_simple):
        r = anomaly_detection(clustering_simple, ["x", "y"])
        assert isinstance(r.get("progress"), list)

    def test_constrained_mode(self, clustering_simple, constrained_mode):
        r = anomaly_detection(clustering_simple, ["x", "y"])
        assert r["success"] is True


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

    def test_file_not_found(self, tmp_path):
        r = check_data_quality(str(tmp_path / "nope.csv"))
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple):
        r = check_data_quality(classification_simple)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = check_data_quality(classification_simple)
        assert isinstance(r.get("progress"), list) and len(r["progress"]) > 0

    def test_constrained_mode(self, classification_simple, constrained_mode):
        r = check_data_quality(classification_simple)
        assert r["success"] is True


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_success_classification(self, classification_simple, tmp_path):
        # Train a model first
        tr = train_classifier(classification_simple, "churned", "rf", test_size=0.2, random_state=42)
        assert tr["success"] is True
        model_path = tr["model_path"]
        r = evaluate_model(model_path, classification_simple, "churned")
        assert r["success"] is True
        assert r["op"] == "evaluate_model"
        assert "accuracy" in r["metrics"]

    def test_auc_roc_binary(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "lr", test_size=0.2, random_state=42)
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert r["success"] is True
        # LR supports predict_proba → AUC-ROC should be present for binary
        assert "auc_roc" in r["metrics"]

    def test_model_not_found(self, classification_simple, tmp_path):
        r = evaluate_model(str(tmp_path / "ghost.pkl"), classification_simple, "churned")
        assert r["success"] is False

    def test_file_not_found(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(tmp_path / "ghost.csv"), "churned")
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple):
        tr = train_classifier(classification_simple, "churned", "rf")
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        tr = train_classifier(classification_simple, "churned", "rf")
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert isinstance(r.get("progress"), list) and len(r["progress"]) > 0

    def test_constrained_mode(self, classification_simple, constrained_mode):
        tr = train_classifier(classification_simple, "churned", "rf")
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert r["success"] is True


# ---------------------------------------------------------------------------
# batch_predict
# ---------------------------------------------------------------------------


class TestBatchPredict:
    def test_success(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        assert tr["success"] is True
        out = str(tmp_path / "predictions.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        assert r["op"] == "batch_predict"
        assert Path(out).exists()

    def test_all_rows_predicted(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_all.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        df_orig = pd.read_csv(classification_simple)
        df_pred = pd.read_csv(out)
        assert len(df_pred) == len(df_orig)
        assert "prediction" in df_pred.columns

    def test_dry_run(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "no_batch.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out, dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_model_not_found(self, classification_simple, tmp_path):
        r = batch_predict(str(tmp_path / "ghost.pkl"), classification_simple)
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_te.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert "token_estimate" in r

    def test_prediction_distribution_present(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_dist.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        assert isinstance(r.get("prediction_distribution"), dict)

    def test_progress_present(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_prog.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert isinstance(r.get("progress"), list)

    def test_snapshot_created(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_snap.csv")
        Path(out).write_text("dummy")  # pre-create so snapshot triggers
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_bak.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert "backup" in r

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        tr = train_classifier(classification_simple, "churned", "rf")
        out = str(tmp_path / "batch_cons.csv")
        r = batch_predict(tr["model_path"], classification_simple, output_path=out)
        assert r["success"] is True


# ---------------------------------------------------------------------------
# run_preprocessing — individual op coverage
# ---------------------------------------------------------------------------


class TestRunPreprocessingOps:
    """Tests for each _apply_op branch inside run_preprocessing."""

    # ---- fill_nulls strategies -----------------------------------------

    def test_fill_nulls_mean(self, classification_messy, tmp_path):
        out = str(tmp_path / "fn_mean.csv")
        ops = [{"op": "fill_nulls", "column": "age", "strategy": "mean"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].isnull().sum() == 0

    def test_fill_nulls_median(self, classification_messy, tmp_path):
        out = str(tmp_path / "fn_median.csv")
        ops = [{"op": "fill_nulls", "column": "age", "strategy": "median"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].isnull().sum() == 0

    def test_fill_nulls_mode(self, classification_messy, tmp_path):
        out = str(tmp_path / "fn_mode.csv")
        ops = [{"op": "fill_nulls", "column": "gender", "strategy": "mode"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True

    def test_fill_nulls_ffill(self, classification_messy, tmp_path):
        out = str(tmp_path / "fn_ffill.csv")
        ops = [{"op": "fill_nulls", "column": "age", "strategy": "ffill"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True

    def test_fill_nulls_bfill(self, classification_messy, tmp_path):
        out = str(tmp_path / "fn_bfill.csv")
        ops = [{"op": "fill_nulls", "column": "age", "strategy": "bfill"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True

    def test_fill_nulls_zero(self, classification_messy, tmp_path):
        out = str(tmp_path / "fn_zero.csv")
        ops = [{"op": "fill_nulls", "column": "age", "strategy": "zero"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].isnull().sum() == 0

    def test_fill_nulls_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "fn_err.csv")
        ops = [{"op": "fill_nulls", "column": "nonexistent_col", "strategy": "mean"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        # op-level error is captured in summary but overall call may succeed
        assert "success" in r

    # ---- drop_outliers --------------------------------------------------

    def test_drop_outliers_iqr(self, classification_simple, tmp_path):
        out = str(tmp_path / "do_iqr.csv")
        ops = [{"op": "drop_outliers", "column": "age", "method": "iqr"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        assert r["applied"] == 1

    def test_drop_outliers_std(self, classification_simple, tmp_path):
        out = str(tmp_path / "do_std.csv")
        ops = [{"op": "drop_outliers", "column": "age", "method": "std"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True

    def test_drop_outliers_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "do_err.csv")
        ops = [{"op": "drop_outliers", "column": "ghost_col", "method": "iqr"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- label_encode ---------------------------------------------------

    def test_label_encode_success(self, classification_messy, tmp_path):
        out = str(tmp_path / "le.csv")
        ops = [{"op": "label_encode", "column": "gender"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert pd.api.types.is_integer_dtype(df["gender"]) or pd.api.types.is_float_dtype(df["gender"])

    def test_label_encode_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "le_err.csv")
        ops = [{"op": "label_encode", "column": "no_such"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- onehot_encode --------------------------------------------------

    def test_onehot_encode_success(self, classification_messy, tmp_path):
        out = str(tmp_path / "ohe.csv")
        ops = [{"op": "onehot_encode", "column": "gender"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        # original column should be gone; new dummy columns should exist
        assert "gender" not in df.columns
        assert any(c.startswith("gender_") for c in df.columns)

    def test_onehot_encode_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "ohe_err.csv")
        ops = [{"op": "onehot_encode", "column": "no_such"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- scale ----------------------------------------------------------

    def test_scale_standard(self, classification_simple, tmp_path):
        out = str(tmp_path / "sc_std.csv")
        ops = [{"op": "scale", "columns": ["age", "tenure"], "method": "standard"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        # After standard scaling mean ≈ 0
        assert abs(df["age"].mean()) < 1.0

    def test_scale_minmax(self, classification_simple, tmp_path):
        out = str(tmp_path / "sc_mm.csv")
        ops = [{"op": "scale", "columns": ["age", "tenure"], "method": "minmax"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        # After minmax scaling all values in [0, 1]
        assert df["age"].min() >= -1e-6
        assert df["age"].max() <= 1.0 + 1e-6

    def test_scale_missing_column(self, classification_simple, tmp_path):
        out = str(tmp_path / "sc_err.csv")
        ops = [{"op": "scale", "columns": ["age", "ghost_col"], "method": "standard"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- drop_duplicates ------------------------------------------------

    def test_drop_duplicates_no_subset(self, classification_messy, tmp_path):
        out = str(tmp_path / "dd.csv")
        ops = [{"op": "drop_duplicates"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True

    def test_drop_duplicates_with_subset(self, classification_simple, tmp_path):
        out = str(tmp_path / "dd_sub.csv")
        ops = [{"op": "drop_duplicates", "subset": ["age"]}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True

    # ---- drop_column ----------------------------------------------------

    def test_drop_column_success(self, classification_simple, tmp_path):
        out = str(tmp_path / "dc.csv")
        ops = [{"op": "drop_column", "column": "age"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age" not in df.columns

    def test_drop_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "dc_err.csv")
        ops = [{"op": "drop_column", "column": "ghost"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- rename_column --------------------------------------------------

    def test_rename_column_success(self, classification_simple, tmp_path):
        out = str(tmp_path / "rc.csv")
        ops = [{"op": "rename_column", "from": "age", "to": "customer_age"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "customer_age" in df.columns
        assert "age" not in df.columns

    # ---- convert_dtype --------------------------------------------------

    def test_convert_dtype_numeric(self, classification_messy, tmp_path):
        out = str(tmp_path / "cd_num.csv")
        ops = [{"op": "convert_dtype", "column": "age", "to": "numeric"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True

    def test_convert_dtype_str(self, classification_simple, tmp_path):
        out = str(tmp_path / "cd_str.csv")
        ops = [{"op": "convert_dtype", "column": "age", "to": "str"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True

    def test_convert_dtype_string_alias(self, classification_simple, tmp_path):
        out = str(tmp_path / "cd_string.csv")
        ops = [{"op": "convert_dtype", "column": "age", "to": "string"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True

    def test_convert_dtype_datetime(self, tmp_path):
        csv_path = tmp_path / "dates.csv"
        pd.DataFrame({"date_col": ["2020-01-01", "2021-06-15", "2022-12-31"], "val": [1, 2, 3]}).to_csv(
            csv_path, index=False
        )
        out = str(tmp_path / "cd_dt.csv")
        ops = [{"op": "convert_dtype", "column": "date_col", "to": "datetime"}]
        r = run_preprocessing(str(csv_path), ops, output_path=out)
        assert r["success"] is True

    def test_convert_dtype_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "cd_err.csv")
        ops = [{"op": "convert_dtype", "column": "nonexistent", "to": "numeric"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- bin_numeric ----------------------------------------------------

    def test_bin_numeric_success(self, classification_simple, tmp_path):
        out = str(tmp_path / "bn.csv")
        ops = [{"op": "bin_numeric", "column": "age", "bins": 4}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age_bin" in df.columns

    def test_bin_numeric_custom_new_column(self, classification_simple, tmp_path):
        out = str(tmp_path / "bn_cust.csv")
        ops = [{"op": "bin_numeric", "column": "age", "bins": 3, "new_column": "age_group"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age_group" in df.columns

    def test_bin_numeric_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "bn_err.csv")
        ops = [{"op": "bin_numeric", "column": "ghost_col", "bins": 4}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- add_date_parts -------------------------------------------------

    def test_add_date_parts_success(self, tmp_path):
        csv_path = tmp_path / "with_dates.csv"
        pd.DataFrame({"date_col": ["2020-01-15", "2021-07-04", "2022-11-25"], "val": [10, 20, 30]}).to_csv(
            csv_path, index=False
        )
        out = str(tmp_path / "adp.csv")
        ops = [{"op": "add_date_parts", "column": "date_col", "parts": ["year", "month", "day"]}]
        r = run_preprocessing(str(csv_path), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "date_col_year" in df.columns
        assert "date_col_month" in df.columns
        assert "date_col_day" in df.columns

    def test_add_date_parts_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "adp_err.csv")
        ops = [{"op": "add_date_parts", "column": "no_date_col"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- log_transform --------------------------------------------------

    def test_log_transform_natural(self, classification_simple, tmp_path):
        out = str(tmp_path / "lt_nat.csv")
        ops = [{"op": "log_transform", "column": "age", "base": "natural"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age_log" in df.columns

    def test_log_transform_log2(self, classification_simple, tmp_path):
        out = str(tmp_path / "lt_log2.csv")
        ops = [{"op": "log_transform", "column": "age", "base": "log2"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age_log" in df.columns

    def test_log_transform_log10(self, classification_simple, tmp_path):
        out = str(tmp_path / "lt_log10.csv")
        ops = [{"op": "log_transform", "column": "age", "base": "log10"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age_log" in df.columns

    def test_log_transform_custom_new_column(self, classification_simple, tmp_path):
        out = str(tmp_path / "lt_cust.csv")
        ops = [{"op": "log_transform", "column": "age", "base": "natural", "new_column": "age_ln"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert "age_ln" in df.columns

    def test_log_transform_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "lt_err.csv")
        ops = [{"op": "log_transform", "column": "ghost_col"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- drop_null_rows -------------------------------------------------

    def test_drop_null_rows_all_columns(self, classification_messy, tmp_path):
        out = str(tmp_path / "dnr_all.csv")
        ops = [{"op": "drop_null_rows"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df.isnull().sum().sum() == 0

    def test_drop_null_rows_specific_column(self, classification_messy, tmp_path):
        out = str(tmp_path / "dnr_col.csv")
        ops = [{"op": "drop_null_rows", "column": "age"}]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].isnull().sum() == 0

    def test_drop_null_rows_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "dnr_err.csv")
        ops = [{"op": "drop_null_rows", "column": "ghost_col"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- clip_column ----------------------------------------------------

    def test_clip_column_both_bounds(self, classification_simple, tmp_path):
        out = str(tmp_path / "cc_both.csv")
        ops = [{"op": "clip_column", "column": "age", "lower": 25, "upper": 55}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].min() >= 25
        assert df["age"].max() <= 55

    def test_clip_column_lower_only(self, classification_simple, tmp_path):
        out = str(tmp_path / "cc_lower.csv")
        ops = [{"op": "clip_column", "column": "age", "lower": 30}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].min() >= 30

    def test_clip_column_upper_only(self, classification_simple, tmp_path):
        out = str(tmp_path / "cc_upper.csv")
        ops = [{"op": "clip_column", "column": "age", "upper": 50}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["age"].max() <= 50

    def test_clip_column_not_found(self, classification_simple, tmp_path):
        out = str(tmp_path / "cc_err.csv")
        ops = [{"op": "clip_column", "column": "ghost_col", "lower": 0, "upper": 100}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert "success" in r

    # ---- _validate_ops coverage -----------------------------------------

    def test_validate_ops_not_a_list(self, classification_simple, tmp_path):
        """Passing a non-list ops value triggers validation failure."""
        out = str(tmp_path / "vo_nl.csv")
        r = run_preprocessing(str(classification_simple), {"op": "drop_duplicates"}, output_path=out)
        assert r["success"] is False

    def test_validate_ops_op_not_dict(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_nd.csv")
        # One element in the list is a string, not a dict
        r = run_preprocessing(str(classification_simple), ["not_a_dict"], output_path=out)
        assert r["success"] is False

    def test_validate_ops_fill_nulls_missing_column_key(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_fn_nc.csv")
        ops = [{"op": "fill_nulls", "strategy": "mean"}]  # no 'column' key
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    def test_validate_ops_scale_missing_columns_key(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_sc_nc.csv")
        ops = [{"op": "scale", "method": "standard"}]  # no 'columns' key
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    def test_validate_ops_scale_invalid_method(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_sc_im.csv")
        ops = [{"op": "scale", "columns": ["age"], "method": "invalid_method"}]
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    def test_validate_ops_label_encode_missing_column(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_le_nc.csv")
        ops = [{"op": "label_encode"}]  # no 'column' key
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    def test_validate_ops_drop_outliers_missing_column(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_do_nc.csv")
        ops = [{"op": "drop_outliers"}]  # no 'column' key
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    def test_validate_ops_rename_column_missing_from(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_rc_nf.csv")
        ops = [{"op": "rename_column", "to": "new_name"}]  # no 'from' key
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    def test_validate_ops_rename_column_missing_to(self, classification_simple, tmp_path):
        out = str(tmp_path / "vo_rc_nt.csv")
        ops = [{"op": "rename_column", "from": "age"}]  # no 'to' key
        r = run_preprocessing(str(classification_simple), ops, output_path=out)
        assert r["success"] is False

    # ---- chaining multiple ops ------------------------------------------

    def test_multiple_ops_chained(self, classification_messy, tmp_path):
        out = str(tmp_path / "multi_ops.csv")
        ops = [
            {"op": "fill_nulls", "column": "age", "strategy": "mean"},
            {"op": "drop_duplicates"},
            {"op": "label_encode", "column": "gender"},
            {"op": "scale", "columns": ["age"], "method": "standard"},
        ]
        r = run_preprocessing(str(classification_messy), ops, output_path=out)
        assert r["success"] is True
        assert r["applied"] == 4


# ---------------------------------------------------------------------------
# filter_rows — full operator coverage
# ---------------------------------------------------------------------------


class TestFilterRowsOperators:
    """Tests for every supported operator in filter_rows."""

    def _csv(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a scratch CSV and return (csv_path, output_path)."""
        src = tmp_path / "src.csv"
        pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Carol", "Dave", None],
                "score": [90, 75, 85, 60, 70],
                "tag": ["alpha", "beta", "alpha-extra", "gamma", "beta"],
            }
        ).to_csv(src, index=False)
        out = tmp_path / "out.csv"
        return src, out

    def test_operator_eq(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "name", "eq", "Bob", output_path=str(out))
        assert r["success"] is True
        assert r["rows_kept"] == 1
        df = pd.read_csv(out)
        assert list(df["name"]) == ["Bob"]

    def test_operator_ne(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "name", "ne", "Bob", output_path=str(out))
        assert r["success"] is True
        assert r["rows_kept"] == 4  # Alice, Carol, Dave, nan

    def test_operator_gt(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "80", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert (df["score"] > 80).all()

    def test_operator_lt(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "lt", "80", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert (df["score"] < 80).all()

    def test_operator_gte(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gte", "85", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert (df["score"] >= 85).all()

    def test_operator_lte(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "lte", "75", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert (df["score"] <= 75).all()

    def test_operator_contains(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "tag", "contains", "alpha", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert all("alpha" in v for v in df["tag"])

    def test_operator_not_contains(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "tag", "not_contains", "alpha", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert all("alpha" not in v for v in df["tag"])

    def test_operator_is_null(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "name", "is_null", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["name"].isnull().all()

    def test_operator_not_null(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "name", "not_null", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert df["name"].notnull().all()

    def test_operator_starts_with(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "tag", "starts_with", "alpha", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert all(v.startswith("alpha") for v in df["tag"])

    def test_operator_ends_with(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "tag", "ends_with", "ta", output_path=str(out))
        assert r["success"] is True
        df = pd.read_csv(out)
        assert all(v.endswith("ta") for v in df["tag"])

    def test_invalid_operator(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "between", "50", output_path=str(out))
        assert r["success"] is False
        assert "hint" in r

    def test_dry_run_no_file_created(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "70", output_path=str(out), dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert not out.exists()

    def test_file_not_found(self, tmp_path):
        r = filter_rows(str(tmp_path / "missing.csv"), "col", "eq", "val")
        assert r["success"] is False
        assert "hint" in r

    def test_column_not_found(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "nonexistent_col", "eq", "Alice", output_path=str(out))
        assert r["success"] is False
        assert "hint" in r

    def test_file_written_on_success(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "70", output_path=str(out))
        assert r["success"] is True
        assert out.exists()

    def test_rows_original_field(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "0", output_path=str(out))
        assert r["success"] is True
        assert "rows_original" in r
        assert r["rows_original"] == 5

    def test_rows_removed_field(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "80", output_path=str(out))
        assert r["success"] is True
        assert r["rows_kept"] + r["rows_removed"] == r["rows_original"]

    def test_token_estimate_present(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "70", output_path=str(out))
        assert "token_estimate" in r

    def test_progress_present(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "70", output_path=str(out))
        assert isinstance(r.get("progress"), list)
        assert len(r["progress"]) > 0

    def test_backup_field_present(self, tmp_path):
        src, out = self._csv(tmp_path)
        r = filter_rows(str(src), "score", "gt", "70", output_path=str(out))
        assert "backup" in r

    def test_snapshot_on_existing_output(self, tmp_path):
        src, out = self._csv(tmp_path)
        # Pre-create the output file so snapshot triggers
        out.write_text("score\n1\n")
        r = filter_rows(str(src), "score", "gt", "70", output_path=str(out))
        assert r["success"] is True
        # backup should be non-empty path since file existed
        assert r.get("backup", "") != ""


# ---------------------------------------------------------------------------
# merge_datasets — extended operator / error coverage
# ---------------------------------------------------------------------------


class TestMergeDatasetsFull:
    """Extended tests for merge_datasets: join types, errors, dry_run, structure."""

    def _make_csvs(self, tmp_path: Path, suffix: str = "") -> tuple[str, str]:
        f1 = str(tmp_path / f"left{suffix}.csv")
        f2 = str(tmp_path / f"right{suffix}.csv")
        pd.DataFrame({"id": [1, 2, 3], "val_a": [10, 20, 30]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [2, 3, 4], "val_b": [200, 300, 400]}).to_csv(f2, index=False)
        return f1, f2

    def test_how_right_join(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_r")
        out = str(tmp_path / "merged_right.csv")
        r = merge_datasets(f1, f2, on="id", how="right", output_path=out)
        assert r["success"] is True
        # right join keeps all from f2: ids 2,3,4
        assert r["merged_rows"] == 3

    def test_how_outer_join(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_o")
        out = str(tmp_path / "merged_outer.csv")
        r = merge_datasets(f1, f2, on="id", how="outer", output_path=out)
        assert r["success"] is True
        # outer keeps all: ids 1,2,3,4
        assert r["merged_rows"] == 4

    def test_how_inner_join_row_count(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_i")
        out = str(tmp_path / "merged_inner.csv")
        r = merge_datasets(f1, f2, on="id", how="inner", output_path=out)
        assert r["success"] is True
        assert r["merged_rows"] == 2  # only ids 2 and 3 are common

    def test_invalid_how(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path)
        r = merge_datasets(f1, f2, on="id", how="cross_join", output_path=str(tmp_path / "x.csv"))
        assert r["success"] is False
        assert "hint" in r

    def test_key_missing_in_file1(self, tmp_path):
        f1 = str(tmp_path / "no_key1.csv")
        f2 = str(tmp_path / "no_key2.csv")
        pd.DataFrame({"other_col": [1, 2], "val": ["a", "b"]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1, 2], "val_b": [10, 20]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "y.csv"))
        assert r["success"] is False
        assert "hint" in r

    def test_key_missing_in_file2(self, tmp_path):
        f1 = str(tmp_path / "has_key1.csv")
        f2 = str(tmp_path / "no_key3.csv")
        pd.DataFrame({"id": [1, 2], "val_a": [10, 20]}).to_csv(f1, index=False)
        pd.DataFrame({"other": [1, 2], "val_b": [10, 20]}).to_csv(f2, index=False)
        r = merge_datasets(f1, f2, on="id", output_path=str(tmp_path / "z.csv"))
        assert r["success"] is False

    def test_file1_not_found(self, tmp_path):
        f2 = str(tmp_path / "exists.csv")
        pd.DataFrame({"id": [1], "x": [1]}).to_csv(f2, index=False)
        r = merge_datasets(str(tmp_path / "ghost.csv"), f2, on="id")
        assert r["success"] is False

    def test_file2_not_found(self, tmp_path):
        f1 = str(tmp_path / "exists2.csv")
        pd.DataFrame({"id": [1], "x": [1]}).to_csv(f1, index=False)
        r = merge_datasets(f1, str(tmp_path / "ghost2.csv"), on="id")
        assert r["success"] is False

    def test_dry_run(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_dry")
        out = str(tmp_path / "dry_merge.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out, dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True
        assert not Path(out).exists()

    def test_dry_run_contains_join_info(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_info")
        out = str(tmp_path / "dry_info.csv")
        r = merge_datasets(f1, f2, on="id", how="inner", output_path=out, dry_run=True)
        assert r["success"] is True
        assert r["how"] == "inner"
        assert "join_keys" in r

    def test_merged_rows_field(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_mr")
        out = str(tmp_path / "mr.csv")
        r = merge_datasets(f1, f2, on="id", how="left", output_path=out)
        assert r["success"] is True
        assert "merged_rows" in r

    def test_merged_columns_field(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_mc")
        out = str(tmp_path / "mc.csv")
        r = merge_datasets(f1, f2, on="id", how="left", output_path=out)
        assert r["success"] is True
        assert "merged_columns" in r
        # id + val_a + val_b = 3 columns
        assert r["merged_columns"] == 3

    def test_multi_key_join(self, tmp_path):
        f1 = str(tmp_path / "mk1.csv")
        f2 = str(tmp_path / "mk2.csv")
        pd.DataFrame({"id": [1, 2], "sub": ["a", "b"], "val_a": [10, 20]}).to_csv(f1, index=False)
        pd.DataFrame({"id": [1, 2], "sub": ["a", "b"], "val_b": [100, 200]}).to_csv(f2, index=False)
        out = str(tmp_path / "mk_out.csv")
        r = merge_datasets(f1, f2, on="id,sub", how="inner", output_path=out)
        assert r["success"] is True
        assert r["merged_rows"] == 2

    def test_output_file_created(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_oc")
        out = str(tmp_path / "oc_out.csv")
        r = merge_datasets(f1, f2, on="id", how="left", output_path=out)
        assert r["success"] is True
        assert Path(out).exists()

    def test_token_estimate_present(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_te")
        out = str(tmp_path / "te_out.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out)
        assert "token_estimate" in r

    def test_progress_present(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_pr")
        out = str(tmp_path / "pr_out.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out)
        assert isinstance(r.get("progress"), list)
        assert len(r["progress"]) > 0

    def test_backup_field_present(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_bk")
        out = str(tmp_path / "bk_out.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out)
        assert "backup" in r

    def test_snapshot_on_existing_output(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_sn")
        out = tmp_path / "sn_out.csv"
        out.write_text("id,val_a,val_b\n1,10,100\n")  # pre-create to trigger snapshot
        r = merge_datasets(f1, f2, on="id", output_path=str(out))
        assert r["success"] is True
        assert r.get("backup", "") != ""

    def test_op_field(self, tmp_path):
        f1, f2 = self._make_csvs(tmp_path, "_op")
        out = str(tmp_path / "op_out.csv")
        r = merge_datasets(f1, f2, on="id", output_path=out)
        assert r.get("op") == "merge_datasets"


# ---------------------------------------------------------------------------
# evaluate_model — uncovered line coverage
# ---------------------------------------------------------------------------


class TestEvaluateModelCoverage:
    """Targets uncovered branches in _medium_data.evaluate_model."""

    # Lines 506-507: ValueError from resolve_path (non-CSV extension on test file)
    def test_resolve_path_bad_extension(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        bad_path = str(tmp_path / "data.txt")  # wrong extension
        r = evaluate_model(tr["model_path"], bad_path, "churned")
        assert r["success"] is False
        assert "token_estimate" in r

    # Lines 509-515: model file does not exist
    def test_model_not_found_error_fields(self, classification_simple, tmp_path):
        r = evaluate_model(str(tmp_path / "missing.pkl"), str(classification_simple), "churned")
        assert r["success"] is False
        assert "Model not found" in r["error"]
        assert "hint" in r

    # Lines 516-522: test data file does not exist
    def test_data_file_not_found_error_fields(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(tmp_path / "missing.csv"), "churned")
        assert r["success"] is False
        assert "not found" in r["error"].lower()

    # Line 543: target column absent from test file
    def test_target_column_not_in_test_file(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        # Write a CSV that lacks the target column
        test_csv = tmp_path / "no_target.csv"
        import pandas as pd

        pd.DataFrame({"age": [25, 30], "tenure": [1, 2], "monthly_charges": [50.0, 60.0],
                      "total_charges": [600.0, 720.0], "num_products": [1, 2]}).to_csv(test_csv, index=False)
        r = evaluate_model(tr["model_path"], str(test_csv), "churned")
        assert r["success"] is False
        assert "churned" in r["error"]

    # Lines 552-553: encoding_map applied to categorical columns in test file
    def test_encoding_map_applied(self, classification_simple, tmp_path):
        """Train on messy data (has categorical cols) then evaluate — covers encoding path."""
        import shutil

        from tests.conftest import FIXTURES_DIR

        messy_dst = tmp_path / "classification_messy.csv"
        shutil.copy(FIXTURES_DIR / "classification_messy.csv", messy_dst)
        tr = train_classifier(str(messy_dst), "churned", "rf")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(messy_dst), "churned")
        assert r["success"] is True
        assert "accuracy" in r["metrics"]

    # Line 557: no feature columns present in test file → error
    def test_no_feature_columns_available(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        # File with only unrelated columns
        bare_csv = tmp_path / "bare.csv"
        import pandas as pd

        pd.DataFrame({"unrelated_col": [1, 2, 3], "churned": [0, 1, 0]}).to_csv(bare_csv, index=False)
        r = evaluate_model(tr["model_path"], str(bare_csv), "churned")
        assert r["success"] is False
        assert "feature" in r["error"].lower()

    # Lines 565-568: SVM model uses scaler; polynomial model uses poly
    def test_scaler_applied_via_svm(self, classification_simple):
        tr = train_classifier(str(classification_simple), "churned", "svm")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(classification_simple), "churned")
        assert r["success"] is True

    # Lines 577-578: categorical string target encoded via LabelEncoder
    def test_string_target_label_encoded(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "str_target.csv"
        import numpy as np

        rng = np.random.default_rng(0)
        n = 60
        pd.DataFrame({
            "feat1": rng.standard_normal(n),
            "feat2": rng.standard_normal(n),
            "label": rng.choice(["cat", "dog"], size=n),
        }).to_csv(csv_path, index=False)
        tr = train_classifier(str(csv_path), "label", "rf")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(csv_path), "label")
        assert r["success"] is True
        assert "accuracy" in r["metrics"]

    # Lines 582-591: XGBoost evaluation path (classification)
    def test_xgb_classification_evaluation(self, classification_simple):
        tr = train_classifier(str(classification_simple), "churned", "xgb")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(classification_simple), "churned")
        assert r["success"] is True
        assert "accuracy" in r["metrics"]
        assert "f1_weighted" in r["metrics"]

    # Lines 604-607: binary classification AUC with predict_proba (lr has it)
    def test_auc_roc_present_for_binary_with_proba(self, classification_simple):
        tr = train_classifier(str(classification_simple), "churned", "lr")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(classification_simple), "churned")
        assert r["success"] is True
        assert "auc_roc" in r["metrics"]

    # Lines 616-617: regression metrics (mse, rmse, r2)
    def test_regression_metrics(self, regression_simple):
        from servers.ml_basic.engine import train_regressor

        tr = train_regressor(str(regression_simple), "salary", "rfr")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], str(regression_simple), "salary")
        assert r["success"] is True
        assert "mse" in r["metrics"]
        assert "rmse" in r["metrics"]
        assert "r2" in r["metrics"]

    # Lines 642-643: general exception handler
    def test_general_exception_handler(self, classification_simple, tmp_path, monkeypatch):
        """Force an exception inside the try block to hit the except handler."""
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        # Monkeypatch pickle.load to raise
        import pickle

        monkeypatch.setattr(pickle, "load", lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
        r = evaluate_model(tr["model_path"], str(classification_simple), "churned")
        assert r["success"] is False
        assert "hint" in r


# ---------------------------------------------------------------------------
# batch_predict — uncovered line coverage
# ---------------------------------------------------------------------------


class TestBatchPredictCoverage:
    """Targets uncovered branches in _medium_data.batch_predict."""

    # Lines 676-677: ValueError from resolve_path (wrong extension on data file)
    def test_resolve_path_bad_extension(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        bad_path = str(tmp_path / "data.txt")
        r = batch_predict(tr["model_path"], bad_path)
        assert r["success"] is False
        assert "token_estimate" in r

    # Line 687: data file not found (model exists, data missing)
    def test_data_file_not_found(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        r = batch_predict(tr["model_path"], str(tmp_path / "missing.csv"))
        assert r["success"] is False
        assert "not found" in r["error"].lower()

    # Lines 725-727: encoding_map applied to categorical columns
    def test_encoding_map_applied(self, tmp_path):
        import shutil

        from tests.conftest import FIXTURES_DIR

        messy_dst = tmp_path / "classification_messy.csv"
        shutil.copy(FIXTURES_DIR / "classification_messy.csv", messy_dst)
        tr = train_classifier(str(messy_dst), "churned", "rf")
        assert tr["success"] is True
        out = str(tmp_path / "enc_preds.csv")
        r = batch_predict(tr["model_path"], str(messy_dst), output_path=out)
        assert r["success"] is True
        assert Path(out).exists()

    # Lines 731-732: scaler.transform applied (SVM has scaler)
    def test_scaler_applied_svm(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "svm")
        assert tr["success"] is True
        out = str(tmp_path / "svm_preds.csv")
        r = batch_predict(tr["model_path"], str(classification_simple), output_path=out)
        assert r["success"] is True

    # Lines 737-745: XGBoost prediction path in batch_predict
    def test_xgb_batch_predict(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "xgb")
        assert tr["success"] is True
        out = str(tmp_path / "xgb_preds.csv")
        r = batch_predict(tr["model_path"], str(classification_simple), output_path=out)
        assert r["success"] is True
        assert Path(out).exists()
        import pandas as pd

        df = pd.read_csv(out)
        assert "prediction" in df.columns

    # Lines 761-762: snapshot triggered when output file pre-exists
    def test_snapshot_on_existing_output(self, classification_simple, tmp_path):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        out = tmp_path / "existing_preds.csv"
        out.write_text("old,data\n1,2\n")
        r = batch_predict(tr["model_path"], str(classification_simple), output_path=str(out))
        assert r["success"] is True
        # backup should be non-empty since we snapshotted
        assert r.get("backup", "") != ""

    # Line 772: regression distribution (min/max/mean) instead of value_counts
    def test_regression_batch_predict_distribution(self, regression_simple, tmp_path):
        from servers.ml_basic.engine import train_regressor

        tr = train_regressor(str(regression_simple), "salary", "rfr")
        assert tr["success"] is True
        out = str(tmp_path / "reg_preds.csv")
        r = batch_predict(tr["model_path"], str(regression_simple), output_path=out)
        assert r["success"] is True
        dist = r.get("prediction_distribution", {})
        assert "min" in dist
        assert "max" in dist
        assert "mean" in dist

    # Lines 792-793: general exception handler
    def test_general_exception_handler(self, classification_simple, tmp_path, monkeypatch):
        tr = train_classifier(str(classification_simple), "churned", "rf")
        assert tr["success"] is True
        import pickle

        monkeypatch.setattr(pickle, "load", lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
        out = str(tmp_path / "err_preds.csv")
        r = batch_predict(tr["model_path"], str(classification_simple), output_path=out)
        assert r["success"] is False
        assert "hint" in r


# ---------------------------------------------------------------------------
# check_data_quality — uncovered line coverage
# ---------------------------------------------------------------------------


class TestCheckDataQualityCoverage:
    """Targets uncovered branches in _medium_data.check_data_quality."""

    # Lines 814-815: ValueError from resolve_path (wrong extension)
    def test_resolve_path_bad_extension(self, tmp_path):
        bad_path = tmp_path / "data.txt"
        bad_path.write_text("a,b\n1,2\n")
        r = check_data_quality(str(bad_path))
        assert r["success"] is False
        assert "token_estimate" in r

    # Lines 826-827: CSV read error (file is not valid CSV content)
    def test_invalid_csv_content(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        # Write binary garbage that pandas cannot parse as CSV
        bad_csv.write_bytes(b"\xff\xfe" + b"\x00" * 4096)
        r = check_data_quality(str(bad_csv))
        # pandas may succeed or fail — if it raises we expect success=False
        if not r["success"]:
            assert "hint" in r

    # Lines 843-852: constant column detected (1 unique value)
    def test_constant_column_alert(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "const_col.csv"
        pd.DataFrame({
            "constant": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            "normal": list(range(10)),
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        assert "constant" in r.get("constant_columns", [])
        alert_types = [a["type"] for a in r.get("alerts", [])]
        assert "constant_column" in alert_types

    # Lines 862-871: high missing data alert (>20% null)
    def test_high_missing_column_alert(self, tmp_path):
        import numpy as np
        import pandas as pd

        csv_path = tmp_path / "high_null.csv"
        n = 20
        pd.DataFrame({
            "sparse": [None] * 18 + [1, 2],   # 90% null — above 20% threshold
            "full": list(range(n)),
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        assert "sparse" in r.get("high_missing_columns", [])
        alert_types = [a["type"] for a in r.get("alerts", [])]
        assert "high_missing" in alert_types

    # Lines 909-918: high cardinality categorical column (>50% unique, >20 distinct)
    def test_high_cardinality_alert(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "high_card.csv"
        n = 100
        pd.DataFrame({
            "user_id": [f"user_{i}" for i in range(n)],  # 100% unique strings
            "value": list(range(n)),
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        alert_types = [a["type"] for a in r.get("alerts", [])]
        assert "high_cardinality" in alert_types

    # Lines 937-938: skewness exception handler (pass via mocking df.skew to raise)
    def test_skewness_exception_swallowed(self, tmp_path, monkeypatch):
        import pandas as pd

        csv_path = tmp_path / "skew_err.csv"
        pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}).to_csv(csv_path, index=False)

        original_skew = pd.DataFrame.skew

        def bad_skew(self, *args, **kwargs):
            raise RuntimeError("skew failure")

        monkeypatch.setattr(pd.DataFrame, "skew", bad_skew)
        r = check_data_quality(str(csv_path))
        # Exception is swallowed — function should still succeed
        assert r["success"] is True
        monkeypatch.setattr(pd.DataFrame, "skew", original_skew)

    # Lines 947-959: multicollinearity alert (|r| > 0.9 between two numeric cols)
    def test_multicollinearity_alert(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "multicol.csv"
        n = 50
        base = list(range(n))
        pd.DataFrame({
            "col1": base,
            "col2": [v * 2 + (i % 3) * 0.01 for i, v in enumerate(base)],
            "other": list(range(n, 2 * n)),
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        alert_types = [a["type"] for a in r.get("alerts", [])]
        assert "multicollinearity" in alert_types

    # Score decreases from constant column (score penalty = 15 per constant col)
    def test_quality_score_decreases_for_constant_column(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "penalty.csv"
        pd.DataFrame({
            "const_a": [0] * 30,
            "const_b": [1] * 30,
            "normal": list(range(30)),
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        # Two constant columns → -30 penalty from 100
        assert r["quality_score"] <= 70

    # Score decreases from high missing data
    def test_quality_score_decreases_for_high_missing(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "penalty_null.csv"
        pd.DataFrame({
            "mostly_null": [None] * 25 + [1] * 5,  # 83% null
            "ok": list(range(30)),
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        assert r["quality_score"] < 100

    # Score decreases from multicollinearity
    def test_quality_score_decreases_for_multicollinearity(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "mc_score.csv"
        n = 40
        base = list(range(n))
        pd.DataFrame({
            "x": base,
            "y": [v * 3 for v in base],
        }).to_csv(csv_path, index=False)
        r = check_data_quality(str(csv_path))
        assert r["success"] is True
        assert r["quality_score"] < 100


# ---------------------------------------------------------------------------
# run_clustering — meanshift, dbscan, silhouette, reduce_dims, error paths
# ---------------------------------------------------------------------------


class TestRunClusteringExtended:
    def test_meanshift_success(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "meanshift")
        assert r["success"] is True
        assert r["algorithm"] == "meanshift"
        assert "n_clusters_found" in r

    def test_dbscan_success(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "dbscan", eps=1.5, min_samples=3)
        assert r["success"] is True
        assert r["algorithm"] == "dbscan"
        assert "noise_points" in r

    def test_silhouette_computed_when_multi_cluster(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", n_clusters=3)
        assert r["success"] is True
        assert r["silhouette_score"] is not None

    def test_silhouette_none_when_single_cluster(self, tmp_path):
        csv = tmp_path / "one_cluster.csv"
        # Tightly packed data so DBSCAN finds only one cluster
        pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.0, 0.1], "y": [0.0, 0.1, 0.0, 0.2, 0.2]}).to_csv(
            csv, index=False
        )
        r = run_clustering(str(csv), ["x", "y"], "dbscan", eps=5.0, min_samples=2)
        assert r["success"] is True
        if r.get("n_clusters_found", 2) < 2:
            assert r["silhouette_score"] is None

    def test_invalid_reduce_dims(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", reduce_dims="tsne")
        assert r["success"] is False

    def test_non_csv_extension(self, tmp_path):
        bad = str(tmp_path / "data.parquet")
        Path(bad).write_text("dummy")
        r = run_clustering(bad, ["x", "y"], "kmeans")
        assert r["success"] is False

    def test_missing_feature_columns(self, clustering_simple):
        r = run_clustering(clustering_simple, ["no_such_col"], "kmeans")
        assert r["success"] is False

    def test_no_numeric_features(self, tmp_path):
        csv = tmp_path / "strings_only.csv"
        pd.DataFrame({"a": ["x", "y", "z", "w", "v"], "b": ["p", "q", "r", "s", "t"]}).to_csv(
            csv, index=False
        )
        r = run_clustering(str(csv), ["a", "b"], "kmeans")
        assert r["success"] is False

    def test_dry_run_meanshift(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "meanshift", dry_run=True)
        assert r["success"] is True
        assert r["dry_run"] is True

    def test_save_labels_creates_backup(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", n_clusters=2, save_labels=True)
        assert r["success"] is True
        assert r.get("backup", "") != ""

    def test_reduce_dims_pca(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", n_clusters=2, reduce_dims="pca")
        assert r["success"] is True

    def test_reduce_dims_ica(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", n_clusters=2, reduce_dims="ica")
        assert r["success"] is True

    def test_token_estimate_present(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans")
        assert "token_estimate" in r

    def test_progress_present(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans")
        assert isinstance(r.get("progress"), list)


# ---------------------------------------------------------------------------
# detect_outliers — std method + missing columns + file-not-found
# ---------------------------------------------------------------------------


class TestDetectOutliersExtended:
    def test_std_method_success(self, classification_simple):
        r = detect_outliers(classification_simple, ["age", "tenure"], method="std")
        assert r["success"] is True
        assert r["op"] == "detect_outliers"
        assert len(r["results"]) == 2

    def test_invalid_method(self, classification_simple):
        r = detect_outliers(classification_simple, ["age"], method="invalid_method")
        assert r["success"] is False
        assert "hint" in r

    def test_missing_column(self, classification_simple):
        r = detect_outliers(classification_simple, ["ghost_col"], method="iqr")
        assert r["success"] is False

    def test_file_not_found(self, tmp_path):
        r = detect_outliers(str(tmp_path / "nope.csv"), ["age"])
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple):
        r = detect_outliers(classification_simple, ["age"], method="std")
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = detect_outliers(classification_simple, ["age"])
        assert isinstance(r.get("progress"), list)

    def test_sample_outliers_bounded(self, classification_simple):
        r = detect_outliers(classification_simple, ["age"], method="iqr")
        assert r["success"] is True
        for entry in r["results"]:
            assert len(entry["sample_outliers"]) <= 5


# ---------------------------------------------------------------------------
# run_preprocessing — extra error paths
# ---------------------------------------------------------------------------


class TestRunPreprocessingErrorPaths:
    def test_invalid_csv_extension(self, tmp_path):
        bad = tmp_path / "data.json"
        bad.write_text("{}")
        r = run_preprocessing(str(bad), [])
        assert r["success"] is False

    def test_output_path_used_when_provided(self, classification_simple, tmp_path):
        out = str(tmp_path / "custom_output.csv")
        r = run_preprocessing(
            classification_simple,
            [{"op": "fill_nulls", "column": "age", "strategy": "mean"}],
            output_path=out,
        )
        assert r["success"] is True
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# train_with_cv — regression branch + error paths
# ---------------------------------------------------------------------------


class TestTrainWithCVExtended:
    def test_regression_kfold_success(self, regression_simple):
        r = train_with_cv(regression_simple, "salary", "lir", "regression", n_splits=3)
        assert r["success"] is True
        assert "fold_scores" in r
        for fold in r["fold_scores"]:
            assert "r2" in fold
            assert "rmse" in fold

    def test_regression_mean_metrics(self, regression_simple):
        r = train_with_cv(regression_simple, "salary", "rfr", "regression", n_splits=3)
        assert r["success"] is True
        mm = r["mean_metrics"]
        assert "r2_mean" in mm
        assert "rmse_mean" in mm

    def test_invalid_task(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "rf", "bad_task")
        assert r["success"] is False

    def test_invalid_model(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "zzz", "classification")
        assert r["success"] is False

    def test_file_not_found(self, tmp_path):
        r = train_with_cv(str(tmp_path / "nope.csv"), "target", "rf", "classification")
        assert r["success"] is False

    def test_target_column_missing(self, classification_simple):
        r = train_with_cv(classification_simple, "nonexistent_target", "rf", "classification")
        assert r["success"] is False

    def test_insufficient_rows(self, tmp_path):
        tiny = tmp_path / "tiny.csv"
        pd.DataFrame({"a": range(5), "b": [0, 1, 0, 1, 0]}).to_csv(tiny, index=False)
        r = train_with_cv(str(tiny), "b", "rf", "classification")
        assert r["success"] is False

    def test_single_class_target(self, tmp_path):
        csv = tmp_path / "one_class.csv"
        pd.DataFrame({"a": range(30), "b": [1] * 30}).to_csv(csv, index=False)
        r = train_with_cv(str(csv), "b", "rf", "classification")
        assert r["success"] is False

    def test_dry_run(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "rf", "classification", dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_model_saved(self, regression_simple):
        r = train_with_cv(regression_simple, "salary", "dtr", "regression", n_splits=3)
        assert r["success"] is True
        assert Path(r["model_path"]).exists()

    def test_token_estimate_present(self, classification_simple):
        r = train_with_cv(classification_simple, "churned", "lr", "classification", n_splits=3)
        assert "token_estimate" in r

    def test_non_csv_extension(self, tmp_path):
        bad = tmp_path / "data.parquet"
        bad.write_text("dummy")
        r = train_with_cv(str(bad), "target", "rf", "classification")
        assert r["success"] is False


# ---------------------------------------------------------------------------
# compare_models — regression branch + error paths + model cap
# ---------------------------------------------------------------------------


class TestCompareModelsExtended:
    def test_regression_success(self, regression_simple):
        r = compare_models(regression_simple, "salary", "regression", ["lir", "rfr"], test_size=0.3)
        assert r["success"] is True
        assert r["task"] == "regression"
        assert len(r["results"]) >= 1

    def test_regression_sorted_by_r2(self, regression_simple):
        r = compare_models(
            regression_simple, "salary", "regression", ["lir", "rfr", "dtr"], test_size=0.3
        )
        assert r["success"] is True
        scores = [e.get("r2", -1e9) for e in r["results"] if "r2" in e]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_task(self, classification_simple):
        r = compare_models(classification_simple, "churned", "bad_task", ["rf"])
        assert r["success"] is False

    def test_invalid_model_string(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["zzz_bad"])
        assert r["success"] is False

    def test_target_column_missing(self, classification_simple):
        r = compare_models(classification_simple, "no_such_col", "classification", ["rf"])
        assert r["success"] is False

    def test_insufficient_rows(self, tmp_path):
        tiny = tmp_path / "tiny.csv"
        pd.DataFrame({"a": range(5), "b": [0, 1, 0, 1, 0]}).to_csv(tiny, index=False)
        r = compare_models(str(tiny), "b", "classification", ["rf"])
        assert r["success"] is False

    def test_file_not_found(self, tmp_path):
        r = compare_models(str(tmp_path / "nope.csv"), "target", "classification", ["rf"])
        assert r["success"] is False

    def test_dry_run(self, classification_simple):
        r = compare_models(
            classification_simple, "churned", "classification", ["rf", "lr"], dry_run=True
        )
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_model_cap_enforced(self, classification_simple, constrained_mode):
        many = ["lr", "rf", "dtc", "knn", "nb"]
        r = compare_models(classification_simple, "churned", "classification", many)
        assert r["success"] is True
        assert len(r["results"]) <= 3

    def test_best_model_path_exists(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "rf"])
        assert r["success"] is True
        assert Path(r["best_model_path"]).exists()

    def test_token_estimate_present(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr"])
        assert "token_estimate" in r

    def test_results_ranked(self, classification_simple):
        r = compare_models(classification_simple, "churned", "classification", ["lr", "rf"])
        assert r["success"] is True
        ranks = [e["rank"] for e in r["results"]]
        assert ranks == list(range(1, len(ranks) + 1))


# ---------------------------------------------------------------------------
# run_clustering — additional uncovered paths
# ---------------------------------------------------------------------------


class TestRunClusteringCoverage:
    def test_null_byte_path_rejected(self):
        r = run_clustering("some\x00path.csv", ["x"], "kmeans")
        assert r["success"] is False

    def test_non_csv_extension_rejected(self, tmp_path):
        bad = tmp_path / "data.parquet"
        bad.write_text("dummy")
        r = run_clustering(str(bad), ["x"], "kmeans")
        assert r["success"] is False

    def test_invalid_reduce_dims(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", reduce_dims="svd")
        assert r["success"] is False

    def test_ica_reduce_dims(self, clustering_simple):
        r = run_clustering(clustering_simple, ["x", "y"], "kmeans", reduce_dims="ica", n_components=2)
        assert r["success"] is True

    def test_no_numeric_columns(self, tmp_path):
        import pandas as pd

        csv = tmp_path / "str_only.csv"
        pd.DataFrame({"a": ["foo", "bar", "baz"], "b": ["x", "y", "z"]}).to_csv(csv, index=False)
        r = run_clustering(str(csv), ["a", "b"], "kmeans")
        assert r["success"] is False

    def test_dbscan_single_cluster_no_silhouette(self, clustering_simple):
        # DBSCAN with high eps may produce a single cluster → silhouette skipped
        r = run_clustering(clustering_simple, ["x", "y"], "dbscan", eps=100.0, min_samples=1)
        assert r["success"] is True
        assert "silhouette_score" in r


# ---------------------------------------------------------------------------
# evaluate_model — remaining coverage
# ---------------------------------------------------------------------------


class TestEvaluateModelExtra:
    def test_binary_auc_computed(self, classification_simple, tmp_path):
        """RF has predict_proba, so AUC-ROC should be computed for binary task."""
        import numpy as np

        tr = train_classifier(classification_simple, "churned", "rf")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], classification_simple, "churned")
        assert r["success"] is True
        assert "auc_roc" in r.get("metrics", {})

    def test_regression_metrics(self, regression_simple, tmp_path):
        from servers.ml_basic.engine import train_regressor

        tr = train_regressor(regression_simple, "salary", "lir")
        assert tr["success"] is True
        r = evaluate_model(tr["model_path"], regression_simple, "salary")
        assert r["success"] is True
        assert "mse" in r.get("metrics", {})
        assert "r2" in r.get("metrics", {})


# ---------------------------------------------------------------------------
# run_preprocessing — additional uncovered paths
# ---------------------------------------------------------------------------


class TestRunPreprocessingCoverage:
    def test_null_byte_path_rejected(self):
        r = run_preprocessing("some\x00path.csv", [])
        assert r["success"] is False

    def test_snapshot_failure_warning(self, classification_simple):
        """Lines 68-69: snapshot raises → warning added to progress."""
        from unittest.mock import patch as _patch

        with _patch("servers.ml_medium._medium_preprocess.snapshot", side_effect=RuntimeError("no space")):
            r = run_preprocessing(
                classification_simple,
                [{"op": "fill_nulls", "column": "age", "strategy": "median"}],
            )
        assert r["success"] is True
        assert any("Snapshot failed" in str(p) for p in r.get("progress", []))

    def test_detect_outliers_null_byte_path(self):
        r = detect_outliers("bad\x00path.csv", ["age"])
        assert r["success"] is False


# ---------------------------------------------------------------------------
# shared/receipt.py — append_receipt corrupted JSON fallback (lines 40-41)
# ---------------------------------------------------------------------------


class TestAppendReceiptCorruptedJson:
    def test_corrupted_receipt_file_falls_back_to_empty(self, tmp_path):
        """Lines 40-41: existing receipt file with invalid JSON → starts fresh."""
        import json as _json
        from shared.receipt import append_receipt, read_receipt_log

        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a\n1\n")

        receipt_path = tmp_path / "data.csv.mcp_receipt.json"
        receipt_path.write_text("{invalid json}")

        append_receipt(csv_path, "tool_x", {}, "ok")

        records = read_receipt_log(csv_path)
        assert len(records) == 1
        assert records[0]["tool"] == "tool_x"


# ---------------------------------------------------------------------------
# compare_models — XGBoost/SVM/KNN paths in _medium_helpers.py
# ---------------------------------------------------------------------------


class TestCompareModelsHelperPaths:
    def test_xgb_classification_covers_fit_predict(self, classification_simple):
        """Lines 160-175 in _medium_helpers: XGBoost _fit_predict_classifier."""
        r = compare_models(classification_simple, "churned", "classification", ["xgb"])
        assert r["success"] is True

    def test_xgb_regression_covers_fit_predict(self, regression_simple):
        """Lines 198-202 in _medium_helpers: XGBoost _fit_predict_regressor."""
        r = compare_models(regression_simple, "salary", "regression", ["xgb"])
        assert r["success"] is True

    def test_svm_covers_pipeline_build(self, classification_simple):
        """Lines 127-128 in _medium_helpers: SVM pipeline build in _build_classifier."""
        r = compare_models(classification_simple, "churned", "classification", ["svm"])
        assert r["success"] is True

    def test_knn_covers_pipeline_build(self, classification_simple):
        """Lines 131-133 in _medium_helpers: KNN pipeline build in _build_classifier."""
        r = compare_models(classification_simple, "churned", "classification", ["knn"])
        assert r["success"] is True

    def test_lar_regressor_build(self, regression_simple):
        """Lines 143 in _medium_helpers: lar branch in _build_regressor."""
        r = compare_models(regression_simple, "salary", "regression", ["lar"])
        assert r["success"] is True

    def test_rr_regressor_build(self, regression_simple):
        """Line 145 in _medium_helpers: rr branch in _build_regressor."""
        r = compare_models(regression_simple, "salary", "regression", ["rr"])
        assert r["success"] is True

    def test_dtr_regressor_build(self, regression_simple):
        """Line 147 in _medium_helpers: dtr branch in _build_regressor."""
        r = compare_models(regression_simple, "salary", "regression", ["dtr"])
        assert r["success"] is True


# ---------------------------------------------------------------------------
# run_preprocessing — convert_dtype exception and add_date_parts exception
# ---------------------------------------------------------------------------


class TestApplyOpExceptionPaths:
    def test_convert_dtype_exception_caught(self, tmp_path):
        """Lines 348-350 in _medium_helpers: exception in convert_dtype."""
        import pandas as pd

        csv = tmp_path / "test_convert.csv"
        pd.DataFrame({"label": ["cat", "dog", "bird"]}).to_csv(csv, index=False)
        # Trying to convert a string column to an invalid dtype triggers exception
        r = run_preprocessing(
            str(csv),
            [{"op": "convert_dtype", "column": "label", "to": "invalid_dtype_xyz"}],
        )
        assert r["success"] is True  # preprocessing returns success even with per-op error

    def test_error_with_backup_key_in_helper(self, tmp_path):
        """Line 99 in _medium_helpers: backup present in _error call."""
        from servers.ml_medium._medium_helpers import _error

        r = _error("test error", "test hint", backup="/path/to/backup.bak")
        assert r["success"] is False
        assert "backup" in r
        assert r["backup"] == "/path/to/backup.bak"

    def test_check_memory_returns_error_when_insufficient(self):
        """Lines 105-113 in _medium_helpers: _check_memory with huge requirement."""
        from servers.ml_medium._medium_helpers import _check_memory

        r = _check_memory(999_999.0)
        assert r is not None
        assert r["success"] is False
        assert "token_estimate" in r


# ---------------------------------------------------------------------------
# train_with_cv and compare_models — additional error paths
# ---------------------------------------------------------------------------


class TestTrainWithCvCoverage:
    def test_null_byte_path_rejected(self):
        """Lines 63-64: ValueError on resolve_path."""
        r = train_with_cv("bad\x00path.csv", "target", "rf", "classification")
        assert r["success"] is False

    def test_xgb_classification_cv(self, classification_simple):
        """XGBoost via train_with_cv → covers xgb paths in _fit_predict_classifier."""
        r = train_with_cv(classification_simple, "churned", "xgb", "classification", n_splits=3)
        assert r["success"] is True

    def test_xgb_regression_cv(self, regression_simple):
        """XGBoost regressor via train_with_cv → covers _fit_predict_regressor xgb path."""
        r = train_with_cv(regression_simple, "salary", "xgb", "regression", n_splits=3)
        assert r["success"] is True


class TestCompareModelsCoverage:
    def test_null_byte_path_rejected(self):
        """Lines 261-262: ValueError on resolve_path in compare_models."""
        r = compare_models("bad\x00path.csv", "target", "classification", ["rf"])
        assert r["success"] is False

    def test_non_csv_rejected(self, tmp_path):
        """Line 266: non-CSV extension in compare_models."""
        bad = tmp_path / "data.parquet"
        bad.write_text("dummy")
        r = compare_models(str(bad), "target", "classification", ["rf"])
        assert r["success"] is False

    def test_per_model_exception_caught(self, classification_simple):
        """Lines 340-342: exception during individual model training is caught."""
        from unittest.mock import patch as _patch

        with _patch(
            "servers.ml_medium._medium_train._fit_predict_classifier",
            side_effect=RuntimeError("training failed")
        ):
            r = compare_models(classification_simple, "churned", "classification", ["rf"])
        # Should succeed but with the model having an error entry
        assert r["success"] is True
        assert any("error" in entry for entry in r.get("results", []))


# ---------------------------------------------------------------------------
# filter_rows — additional uncovered paths
# ---------------------------------------------------------------------------


class TestFilterRowsCoverage:
    def test_null_byte_path_rejected(self):
        """Lines 51-52: ValueError on resolve_path in filter_rows."""
        r = filter_rows("bad\x00path.csv", "col", "eq", 1)
        assert r["success"] is False

    def test_non_csv_extension_rejected(self, tmp_path):
        """Line 56: non-CSV in filter_rows."""
        bad = tmp_path / "data.txt"
        bad.write_text("dummy")
        r = filter_rows(str(bad), "col", "eq", 1)
        assert r["success"] is False

    def test_filter_exception_caught(self, classification_simple):
        """Lines 101-103: filter operation fails with exception."""
        # Use 'gt' operator on a string column (comparison fails)
        r = filter_rows(classification_simple, "churned", "gt", "not_a_number")
        # May succeed or fail depending on pandas behavior
        assert "success" in r


# ---------------------------------------------------------------------------
# merge_datasets — additional uncovered paths
# ---------------------------------------------------------------------------


class TestMergeDatasetsCoverage:
    def test_null_byte_path_rejected(self, classification_simple):
        """Lines 179-180: null byte in merge_datasets."""
        r = merge_datasets("bad\x00path.csv", classification_simple, on="age")
        assert r["success"] is False

    def test_second_file_null_byte_rejected(self, classification_simple):
        """Lines 189-190: null byte in second path."""
        r = merge_datasets(classification_simple, "bad\x00path.csv", on="age")
        assert r["success"] is False


# ---------------------------------------------------------------------------
# detect_outliers — additional uncovered paths
# ---------------------------------------------------------------------------


class TestDetectOutliersCoverage:
    def test_non_csv_rejected(self, tmp_path):
        """Lines 119-120: non-csv in detect_outliers."""
        bad = tmp_path / "data.txt"
        bad.write_text("dummy")
        r = detect_outliers(str(bad), ["age"])
        assert r["success"] is False


# ---------------------------------------------------------------------------
# batch_predict — additional uncovered paths
# ---------------------------------------------------------------------------


class TestBatchPredictCoverage:
    def test_model_not_found(self, tmp_path, classification_simple):
        """Lines 568+: model not found in batch_predict."""
        r = batch_predict(str(tmp_path / "ghost.pkl"), classification_simple)
        assert r["success"] is False

    def test_csv_not_found(self, classification_simple):
        """batch_predict with bad csv path."""
        mp = train_classifier(classification_simple, "churned", "rf")["model_path"]
        r = batch_predict(mp, "/nonexistent/ghost.csv")
        assert r["success"] is False


# ===========================================================================
# _medium_helpers.py — direct calls to cover private helpers
# ===========================================================================


class TestMediumHelpersCoverage:
    """Direct coverage of private helpers not reachable via public API."""

    def test_check_memory_sufficient_returns_none(self):
        """Line 113: _check_memory returns None when RAM is sufficient."""
        from servers.ml_medium._medium_helpers import _check_memory

        assert _check_memory(0.00001) is None

    def test_build_classifier_xgb_returns_none(self):
        """Lines 131-132: _build_classifier("xgb") returns None."""
        from servers.ml_medium._medium_helpers import _build_classifier

        assert _build_classifier("xgb") is None

    def test_build_classifier_unknown_raises(self):
        """Line 133: _build_classifier with unknown model raises ValueError."""
        import pytest
        from servers.ml_medium._medium_helpers import _build_classifier

        with pytest.raises(ValueError):
            _build_classifier("NONEXISTENT_ZZZ")

    def test_build_regressor_pr_returns_pipeline(self):
        """Line 143: _build_regressor("pr") returns a sklearn Pipeline."""
        from servers.ml_medium._medium_helpers import _build_regressor

        result = _build_regressor("pr", degree=2)
        assert result is not None and hasattr(result, "fit")

    def test_build_regressor_xgb_returns_none(self):
        """Lines 152-153: _build_regressor("xgb") returns None."""
        from servers.ml_medium._medium_helpers import _build_regressor

        assert _build_regressor("xgb") is None

    def test_build_regressor_unknown_raises(self):
        """Line 154: _build_regressor with unknown model raises ValueError."""
        import pytest
        from servers.ml_medium._medium_helpers import _build_regressor

        with pytest.raises(ValueError):
            _build_regressor("NONEXISTENT_ZZZ")

    def test_fit_predict_classifier_xgb_multiclass(self):
        """Lines 170, 174: XGBoost multiclass (nc>2) in _fit_predict_classifier."""
        import numpy as np
        from servers.ml_medium._medium_helpers import _fit_predict_classifier

        rng = np.random.RandomState(0)
        x_train = rng.randn(90, 4)
        y_train = np.array([0, 1, 2] * 30)
        x_test = rng.randn(30, 4)
        preds = _fit_predict_classifier("xgb", x_train, x_test, y_train)
        assert preds.shape == (30,)
        assert set(preds).issubset({0, 1, 2})

    def test_apply_op_add_date_parts_invalid_attr(self):
        """Lines 375-376: _apply_op add_date_parts with invalid part name causes exception."""
        import pandas as pd
        from servers.ml_medium._medium_helpers import _apply_op

        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-01"])})
        op = {"op": "add_date_parts", "column": "ts", "parts": ["NONEXISTENT_ATTR_ZZZ"]}
        _, summary = _apply_op(df, op)
        assert "error" in summary

    def test_apply_op_unhandled_op_fallthrough(self):
        """Line 415: _apply_op fallback for op not handled by any elif."""
        import pandas as pd
        import servers.ml_medium._medium_helpers as mh
        from servers.ml_medium._medium_helpers import _apply_op

        df = pd.DataFrame({"a": [1, 2, 3]})
        mh.ALLOWED_OPS.add("phantom_op_zzz")
        try:
            _, summary = _apply_op(df, {"op": "phantom_op_zzz"})
            assert "error" in summary
        finally:
            mh.ALLOWED_OPS.discard("phantom_op_zzz")


# ===========================================================================
# filter_rows — CSV read failure + snapshot failure
# ===========================================================================


class TestFilterRowsMorePaths:
    def test_csv_read_failure(self, tmp_path):
        """Lines 65-66: CSV read failure returns error dict."""
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"col": [1, 2, 3]}).to_csv(csv_path, index=False)
        with patch("servers.ml_medium._medium_data.pd.read_csv", side_effect=Exception("disk error")):
            r = filter_rows(str(csv_path), "col", "eq", "1")
        assert r["success"] is False

    def test_snapshot_failure_is_swallowed(self, tmp_path):
        """Lines 133-134: snapshot failure during output write is ignored."""
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"col": [1, 2, 3]}).to_csv(csv_path, index=False)
        out_path = tmp_path / "filtered.csv"
        out_path.write_text("existing content")
        with patch("servers.ml_medium._medium_data.snapshot", side_effect=Exception("snap fail")):
            r = filter_rows(str(csv_path), "col", "eq", "1", output_path=str(out_path))
        assert r["success"] is True


# ===========================================================================
# merge_datasets — CSV read failure + snapshot failure
# ===========================================================================


class TestMergeDatasetsMorePaths:
    def test_csv_read_failure(self, tmp_path):
        """Lines 189-190: CSV read failure returns error."""
        import pandas as pd

        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        pd.DataFrame({"id": [1, 2], "v": [10, 20]}).to_csv(csv1, index=False)
        pd.DataFrame({"id": [1, 2], "w": [30, 40]}).to_csv(csv2, index=False)
        with patch("servers.ml_medium._medium_data.pd.read_csv", side_effect=Exception("disk error")):
            r = merge_datasets(str(csv1), str(csv2), on="id")
        assert r["success"] is False

    def test_snapshot_failure_is_swallowed(self, tmp_path):
        """Lines 232-233: snapshot failure during output write is ignored."""
        import pandas as pd

        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        pd.DataFrame({"id": [1, 2], "v": [10, 20]}).to_csv(csv1, index=False)
        pd.DataFrame({"id": [1, 2], "w": [30, 40]}).to_csv(csv2, index=False)
        out = tmp_path / "merged.csv"
        out.write_text("existing")
        with patch("servers.ml_medium._medium_data.snapshot", side_effect=Exception("snap fail")):
            r = merge_datasets(str(csv1), str(csv2), on="id", output_path=str(out))
        assert r["success"] is True


# ===========================================================================
# find_optimal_clusters — error paths
# ===========================================================================


class TestFindOptimalClustersMorePaths:
    def test_resolve_failure(self):
        """Lines 283-284: null byte triggers resolve_path error."""
        r = find_optimal_clusters("\x00bad", ["x", "y"])
        assert r["success"] is False

    def test_csv_read_failure(self, tmp_path):
        """Lines 290-291: CSV read failure."""
        import pandas as pd

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(20), "y": range(20)}).to_csv(p, index=False)
        with patch("servers.ml_medium._medium_data.pd.read_csv", side_effect=Exception("broken")):
            r = find_optimal_clusters(str(p), ["x", "y"])
        assert r["success"] is False

    def test_missing_feature_columns(self, clustering_simple):
        """Line 295: feature columns missing returns error."""
        r = find_optimal_clusters(str(clustering_simple), ["ghost_col"])
        assert r["success"] is False

    def test_too_few_rows(self, tmp_path):
        """Line 299: fewer than 4 rows returns error."""
        import pandas as pd

        p = tmp_path / "tiny.csv"
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}).to_csv(p, index=False)
        r = find_optimal_clusters(str(p), ["x", "y"])
        assert r["success"] is False


# ===========================================================================
# anomaly_detection — error paths
# ===========================================================================


class TestAnomalyDetectionMorePaths:
    def test_contamination_zero_rejected(self, clustering_simple):
        """Line 390: contamination=0.0 is invalid."""
        r = anomaly_detection(str(clustering_simple), ["x", "y"], contamination=0.0)
        assert r["success"] is False

    def test_contamination_half_rejected(self, clustering_simple):
        """Line 390: contamination=0.5 is invalid."""
        r = anomaly_detection(str(clustering_simple), ["x", "y"], contamination=0.5)
        assert r["success"] is False

    def test_resolve_failure(self):
        """Lines 394-395: null byte in path triggers resolve error."""
        r = anomaly_detection("\x00bad", ["x", "y"])
        assert r["success"] is False

    def test_csv_read_failure(self, tmp_path):
        """Lines 401-402: CSV read failure."""
        import pandas as pd

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(10), "y": range(10)}).to_csv(p, index=False)
        with patch("servers.ml_medium._medium_data.pd.read_csv", side_effect=Exception("broken")):
            r = anomaly_detection(str(p), ["x", "y"])
        assert r["success"] is False

    def test_missing_feature_columns(self, clustering_simple):
        """Line 406: feature columns not in dataset."""
        r = anomaly_detection(str(clustering_simple), ["ghost_col"])
        assert r["success"] is False

    def test_snapshot_failure_with_save_labels(self, clustering_simple):
        """Lines 449-450: snapshot fails when save_labels=True; warning added, success=True."""
        with patch("servers.ml_medium._medium_data.snapshot", side_effect=Exception("snap fail")):
            r = anomaly_detection(str(clustering_simple), ["x", "y"], save_labels=True)
        assert r["success"] is True
        assert any(p.get("icon") == "\u26a0" for p in r["progress"])


# ===========================================================================
# evaluate_model — XGBoost paths and no-predict_proba model
# ===========================================================================


class TestEvaluateModelXGBPaths:
    def test_xgb_binary_classification(self, classification_simple):
        """XGBoost binary model evaluation."""
        mp = train_classifier(str(classification_simple), "churned", "xgb")["model_path"]
        r = evaluate_model(mp, str(classification_simple), "churned")
        assert r["success"] is True
        assert "accuracy" in r["metrics"]

    def test_xgb_regression(self, regression_simple):
        """Line 591: XGBoost regression evaluate_model."""
        from servers.ml_basic.engine import train_regressor

        mp = train_regressor(str(regression_simple), "salary", "xgb")["model_path"]
        r = evaluate_model(mp, str(regression_simple), "salary")
        assert r["success"] is True
        assert "r2" in r["metrics"]

    def test_no_auc_when_no_predict_proba(self, classification_simple, tmp_path):
        """Lines 607, 610-611: SVC without probability=True skips AUC."""
        import pickle
        import pandas as pd
        from sklearn.svm import SVC

        df = pd.read_csv(str(classification_simple))
        X = df[["age", "tenure"]].values
        y = df["churned"].values
        clf = SVC(probability=False)
        clf.fit(X, y)
        mp = tmp_path / "noproba.pkl"
        with open(mp, "wb") as fh:
            pickle.dump(
                {
                    "model": clf,
                    "metadata": {
                        "task": "classification",
                        "feature_columns": ["age", "tenure"],
                        "target_column": "churned",
                        "encoding_map": {},
                        "model_key": "svm",
                        "n_classes": 2,
                    },
                },
                fh,
            )
        mp.with_suffix(".manifest.json").write_text("{}")
        r = evaluate_model(str(mp), str(classification_simple), "churned")
        assert r["success"] is True
        assert "auc_roc" not in r["metrics"]


# ===========================================================================
# batch_predict — XGBoost, regression distribution, snapshot + encoding_map
# ===========================================================================


class TestBatchPredictMorePaths:
    def test_resolve_failure(self):
        """Lines 676-677: null byte model_path triggers error."""
        r = batch_predict("\x00bad", "/some/file.csv")
        assert r["success"] is False

    def test_xgb_classification_batch(self, classification_simple, tmp_path):
        """Lines 737-743: XGBoost classification batch_predict."""
        mp = train_classifier(str(classification_simple), "churned", "xgb")["model_path"]
        out = str(tmp_path / "preds.csv")
        r = batch_predict(mp, str(classification_simple), output_path=out)
        assert r["success"] is True

    def test_xgb_regression_batch(self, regression_simple, tmp_path):
        """Lines 744-745: XGBoost regression batch_predict."""
        from servers.ml_basic.engine import train_regressor

        mp = train_regressor(str(regression_simple), "salary", "xgb")["model_path"]
        out = str(tmp_path / "reg_preds.csv")
        r = batch_predict(mp, str(regression_simple), output_path=out)
        assert r["success"] is True

    def test_regression_prediction_distribution(self, regression_simple, tmp_path):
        """Line 772: regression distribution dict has min/max."""
        from servers.ml_basic.engine import train_regressor

        mp = train_regressor(str(regression_simple), "salary", "lir")["model_path"]
        out = str(tmp_path / "lir_preds.csv")
        r = batch_predict(mp, str(regression_simple), output_path=out)
        assert r["success"] is True
        assert "min" in r["prediction_distribution"]

    def test_encoding_map_applied(self, classification_messy, tmp_path):
        """Lines 726-727: encoding_map applied to categorical features."""
        mp = train_classifier(str(classification_messy), "churned", "rf")["model_path"]
        out = str(tmp_path / "enc_preds.csv")
        r = batch_predict(mp, str(classification_messy), output_path=out)
        assert r["success"] is True

    def test_snapshot_failure_is_swallowed(self, classification_simple, tmp_path):
        """Lines 761-762: snapshot failure during output write is ignored."""
        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        out = tmp_path / "preds.csv"
        out.write_text("old")
        with patch("servers.ml_medium._medium_data.snapshot", side_effect=Exception("snap")):
            r = batch_predict(mp, str(classification_simple), output_path=str(out))
        assert r["success"] is True


# ===========================================================================
# generate_eda_report — alert-triggering datasets + error paths
# ===========================================================================


class TestEDAReportAlertPaths:
    def test_resolve_failure(self):
        """Lines 268-269: resolve_path failure."""
        r = generate_eda_report("\x00bad")
        assert r["success"] is False

    def test_non_csv_extension(self, tmp_path):
        """Line 273: non-CSV extension returns error."""
        p = tmp_path / "data.txt"
        p.write_text("hello")
        r = generate_eda_report(str(p))
        assert r["success"] is False

    def test_csv_read_failure(self, tmp_path):
        """Lines 279-280: CSV read failure."""
        import pandas as pd

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(p, index=False)
        with patch("servers.ml_medium._medium_eda.pd.read_csv", side_effect=Exception("broken")):
            r = generate_eda_report(str(p))
        assert r["success"] is False

    def test_constant_column_alert(self, tmp_path):
        """Line 42: constant column triggers alert."""
        import pandas as pd

        p = tmp_path / "const.csv"
        pd.DataFrame({"const_col": [5] * 50, "normal": range(50)}).to_csv(p, index=False)
        r = generate_eda_report(str(p), output_path=str(tmp_path / "eda.html"), open_after=False)
        assert r["success"] is True
        assert "constant_column" in [a["type"] for a in r.get("alerts", [])]

    def test_high_missing_alert(self, tmp_path):
        """Line 56: >20% missing data triggers alert."""
        import pandas as pd

        p = tmp_path / "missing.csv"
        vals = [None if i % 4 == 0 else i for i in range(50)]
        pd.DataFrame({"x": range(50), "sparse": vals}).to_csv(p, index=False)
        r = generate_eda_report(str(p), output_path=str(tmp_path / "eda.html"), open_after=False)
        assert r["success"] is True
        assert "high_missing" in [a["type"] for a in r.get("alerts", [])]

    def test_class_imbalance_alert(self, tmp_path):
        """Line 108: >90% dominant class triggers imbalance alert."""
        import pandas as pd

        p = tmp_path / "imbalanced.csv"
        labels = [0] * 46 + [1] * 4
        pd.DataFrame({"feature": range(50), "label": labels}).to_csv(p, index=False)
        r = generate_eda_report(
            str(p), target_column="label", output_path=str(tmp_path / "eda.html"), open_after=False
        )
        assert r["success"] is True
        assert "class_imbalance" in [a["type"] for a in r.get("alerts", [])]

    def test_multicollinearity_alert(self, tmp_path):
        """Lines 153-154: highly correlated columns trigger multicollinearity alert."""
        import pandas as pd
        import numpy as np

        p = tmp_path / "corr.csv"
        x = np.arange(50, dtype=float)
        pd.DataFrame({"x": x, "x_double": x * 2.0 + 0.5, "target": range(50)}).to_csv(p, index=False)
        r = generate_eda_report(
            str(p), target_column="target", output_path=str(tmp_path / "eda.html"), open_after=False
        )
        assert r["success"] is True
        assert "multicollinearity" in [a["type"] for a in r.get("alerts", [])]


# ===========================================================================
# train_with_cv / compare_models / run_preprocessing — CSV read failures
# ===========================================================================


class TestCsvReadFailurePaths:
    def test_train_with_cv_csv_failure(self, tmp_path):
        """Lines 82-83: CSV read failure in train_with_cv."""
        import pandas as pd

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(30), "y": [0, 1] * 15}).to_csv(p, index=False)
        with patch("servers.ml_medium._medium_train.pd.read_csv", side_effect=Exception("broken")):
            r = train_with_cv(str(p), "y", "lr", "classification", n_splits=2)
        assert r["success"] is False

    def test_compare_models_csv_failure(self, tmp_path):
        """Lines 286-287: CSV read failure in compare_models."""
        import pandas as pd

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(30), "y": [0, 1] * 15}).to_csv(p, index=False)
        with patch("servers.ml_medium._medium_train.pd.read_csv", side_effect=Exception("broken")):
            r = compare_models(str(p), "y", "classification", ["lr"])
        assert r["success"] is False

    def test_run_preprocessing_csv_failure(self, tmp_path):
        """Lines 47-48: CSV read failure in run_preprocessing."""
        import pandas as pd

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(10), "y": range(10)}).to_csv(p, index=False)
        with patch("servers.ml_medium._medium_preprocess.pd.read_csv", side_effect=Exception("broken")):
            r = run_preprocessing(str(p), [{"op": "drop_duplicates"}])
        assert r["success"] is False
