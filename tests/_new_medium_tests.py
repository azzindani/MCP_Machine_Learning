"""New targeted tests to append to test_ml_medium.py"""
# This file is appended to test_ml_medium.py via a helper script.

MEDIUM_ADDITIONS = r'''

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
        r = generate_eda_report(str(p), output_path=str(tmp_path / "eda.html"), open_browser=False)
        assert r["success"] is True
        assert "constant_column" in [a["type"] for a in r.get("alerts", [])]

    def test_high_missing_alert(self, tmp_path):
        """Line 56: >20% missing data triggers alert."""
        import pandas as pd

        p = tmp_path / "missing.csv"
        vals = [None if i % 4 == 0 else i for i in range(50)]
        pd.DataFrame({"x": range(50), "sparse": vals}).to_csv(p, index=False)
        r = generate_eda_report(str(p), output_path=str(tmp_path / "eda.html"), open_browser=False)
        assert r["success"] is True
        assert "high_missing" in [a["type"] for a in r.get("alerts", [])]

    def test_class_imbalance_alert(self, tmp_path):
        """Line 108: >90% dominant class triggers imbalance alert."""
        import pandas as pd

        p = tmp_path / "imbalanced.csv"
        labels = [0] * 46 + [1] * 4
        pd.DataFrame({"feature": range(50), "label": labels}).to_csv(p, index=False)
        r = generate_eda_report(
            str(p), target_column="label", output_path=str(tmp_path / "eda.html"), open_browser=False
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
            str(p), target_column="target", output_path=str(tmp_path / "eda.html"), open_browser=False
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
'''
