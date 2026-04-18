"""New targeted tests to append to test_ml_advanced.py"""

ADVANCED_ADDITIONS = r'''

# ===========================================================================
# tune_hyperparameters — additional error paths
# ===========================================================================


class TestTuneMorePaths:
    def test_resolve_failure(self):
        """Lines 69-70: null byte in file_path triggers resolve error."""
        from servers.ml_advanced.engine import tune_hyperparameters

        r = tune_hyperparameters("\x00bad", "target", "lr", "classification")
        assert r["success"] is False

    def test_csv_read_failure(self, tmp_path):
        """Lines 111-112: CSV read failure in tune_hyperparameters."""
        from unittest.mock import patch
        import pandas as pd
        from servers.ml_advanced.engine import tune_hyperparameters

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(30), "y": [0, 1] * 15}).to_csv(p, index=False)
        with patch("servers.ml_advanced.engine.pd.read_csv", side_effect=Exception("broken")):
            r = tune_hyperparameters(str(p), "y", "lr", "classification")
        assert r["success"] is False


# ===========================================================================
# export_model — additional error paths
# ===========================================================================


class TestExportMorePaths:
    def test_resolve_failure_model_path(self):
        """Lines 247-248: null byte in model_path triggers resolve error."""
        from servers.ml_advanced.engine import export_model

        r = export_model("\x00bad.pkl")
        assert r["success"] is False

    def test_load_model_failure(self, classification_simple, tmp_path):
        """Lines 288-289: load_model failure returns error."""
        from unittest.mock import patch
        from servers.ml_advanced.engine import export_model
        from servers.ml_basic.engine import train_classifier

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        with patch("servers.ml_advanced.engine._load_model", side_effect=Exception("corrupt")):
            r = export_model(mp)
        assert r["success"] is False


# ===========================================================================
# read_model_report — additional error paths
# ===========================================================================


class TestReadModelReportMorePaths:
    def test_resolve_failure(self):
        """Lines 335-336: null byte in model_path."""
        from servers.ml_advanced.engine import read_model_report

        r = read_model_report("\x00bad")
        assert r["success"] is False

    def test_load_model_failure(self, classification_simple, tmp_path):
        """Lines 344-345: _load_model failure returns error."""
        from unittest.mock import patch
        from servers.ml_advanced.engine import read_model_report
        from servers.ml_basic.engine import train_classifier

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        with patch("servers.ml_advanced.engine._load_model", side_effect=Exception("corrupt")):
            r = read_model_report(mp)
        assert r["success"] is False

    def test_long_classification_report_truncated(self, classification_simple, tmp_path):
        """Line 364: classification_report > 500 chars is truncated."""
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        from servers.ml_advanced.engine import read_model_report

        mp = tmp_path / "long_report.pkl"
        clf = RandomForestClassifier(n_estimators=2, random_state=42)
        payload = {
            "model": clf,
            "metadata": {
                "task": "classification",
                "feature_columns": ["x"],
                "target_column": "y",
                "encoding_map": {},
                "metrics": {},
                "classification_report": "x" * 600,
            },
        }
        with open(mp, "wb") as fh:
            pickle.dump(payload, fh)
        mp.with_suffix(".manifest.json").write_text("{}")
        r = read_model_report(str(mp))
        assert r["success"] is True
        assert len(r["classification_report"]) <= 500

    def test_manifest_json_parse_failure(self, classification_simple, tmp_path):
        """Lines 371-372: corrupt manifest JSON is swallowed."""
        from servers.ml_basic.engine import train_classifier
        from servers.ml_advanced.engine import read_model_report
        from pathlib import Path

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        Path(mp).with_suffix(".manifest.json").write_text("NOT_VALID_JSON!!!")
        r = read_model_report(mp)
        assert r["success"] is True


# ===========================================================================
# run_profiling_report — additional error paths
# ===========================================================================


class TestProfilingMorePaths:
    def test_resolve_failure(self):
        """Lines 422-423: null byte in file_path."""
        from servers.ml_advanced.engine import run_profiling_report

        r = run_profiling_report("\x00bad")
        assert r["success"] is False

    def test_csv_read_failure(self, tmp_path):
        """Lines 431-432: CSV read failure."""
        from unittest.mock import patch
        import pandas as pd
        from servers.ml_advanced.engine import run_profiling_report

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(10)}).to_csv(p, index=False)
        with patch("servers.ml_advanced.engine.pd.read_csv", side_effect=Exception("broken")):
            r = run_profiling_report(str(p))
        assert r["success"] is False


# ===========================================================================
# apply_dimensionality_reduction — additional error paths
# ===========================================================================


class TestDRMorePaths:
    def test_resolve_failure(self):
        """Lines 535-536: null byte in file_path."""
        from servers.ml_advanced.engine import apply_dimensionality_reduction

        r = apply_dimensionality_reduction("\x00bad", ["x", "y"], "pca")
        assert r["success"] is False

    def test_csv_read_failure(self, tmp_path):
        """Lines 548-549: CSV read failure."""
        from unittest.mock import patch
        import pandas as pd
        from servers.ml_advanced.engine import apply_dimensionality_reduction

        p = tmp_path / "data.csv"
        pd.DataFrame({"x": range(10), "y": range(10)}).to_csv(p, index=False)
        with patch("servers.ml_advanced.engine.pd.read_csv", side_effect=Exception("broken")):
            r = apply_dimensionality_reduction(str(p), ["x", "y"], "pca")
        assert r["success"] is False

    def test_snapshot_failure_continues(self, clustering_simple, tmp_path):
        """Lines 582-583: snapshot failure is logged as warning, does not abort."""
        from unittest.mock import patch
        from servers.ml_advanced.engine import apply_dimensionality_reduction

        out = tmp_path / "reduced.csv"
        out.write_text("existing")
        with patch("servers.ml_advanced.engine.snapshot", side_effect=Exception("snap fail")):
            r = apply_dimensionality_reduction(
                str(clustering_simple), ["x", "y"], "pca", output_path=str(out)
            )
        assert r["success"] is True


# ===========================================================================
# generate_training_report — additional error paths
# ===========================================================================


class TestTrainingReportMorePaths:
    def test_resolve_failure(self):
        """Lines 645-646: null byte in model_path."""
        from servers.ml_advanced.engine import generate_training_report

        r = generate_training_report("\x00bad")
        assert r["success"] is False

    def test_load_model_failure(self, classification_simple):
        """Lines 655-656: _load_model failure returns error."""
        from unittest.mock import patch
        from servers.ml_advanced.engine import generate_training_report
        from servers.ml_basic.engine import train_classifier

        mp = train_classifier(str(classification_simple), "churned", "rf")["model_path"]
        with patch("servers.ml_advanced.engine._load_model", side_effect=Exception("corrupt")):
            r = generate_training_report(mp)
        assert r["success"] is False


# ===========================================================================
# plot_roc_curve — XGBoost Booster path + no-proba model
# ===========================================================================


class TestPlotRocXGBAndNoProba:
    def test_xgb_booster_roc_curve(self, classification_simple, tmp_path):
        """Lines 130-142: XGBoost Booster (no predict_proba) path in plot_roc_curve."""
        from servers.ml_basic.engine import train_classifier
        from servers.ml_advanced.engine import plot_roc_curve

        mp = train_classifier(str(classification_simple), "churned", "xgb")["model_path"]
        out = str(tmp_path / "roc_xgb.html")
        r = plot_roc_curve(mp, str(classification_simple), output_path=out, open_after=False)
        assert r["success"] is True

    def test_no_proba_model_returns_error(self, classification_simple, tmp_path):
        """Line 145: model without predict_proba returns error."""
        import pickle
        import pandas as pd
        from sklearn.svm import SVC
        from servers.ml_advanced.engine import plot_roc_curve

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
                    },
                },
                fh,
            )
        mp.with_suffix(".manifest.json").write_text("{}")
        out = str(tmp_path / "roc_noproba.html")
        r = plot_roc_curve(str(mp), str(classification_simple), output_path=out, open_after=False)
        assert r["success"] is False

    def test_encoding_map_applied_in_roc(self, classification_messy, tmp_path):
        """Line 94: encoding_map applied when model trained on categorical data."""
        from servers.ml_basic.engine import train_classifier
        from servers.ml_advanced.engine import plot_roc_curve

        mp = train_classifier(str(classification_messy), "churned", "rf")["model_path"]
        out = str(tmp_path / "roc_enc.html")
        r = plot_roc_curve(mp, str(classification_messy), output_path=out, open_after=False)
        assert r["success"] is True


# ===========================================================================
# plot_learning_curve — categorical column encoding path
# ===========================================================================


class TestPlotLearningCurveCategorical:
    def test_categorical_columns_encoded(self, classification_messy, tmp_path):
        """Lines 299-300: categorical columns encoded via LabelEncoder in learning curve."""
        from servers.ml_advanced.engine import plot_learning_curve

        out = str(tmp_path / "lc_cat.html")
        r = plot_learning_curve(
            str(classification_messy), "churned", "rf", "classification", cv=3,
            output_path=out, open_after=False
        )
        assert r["success"] is True


# ===========================================================================
# plot_predictions_vs_actual — encoding map + XGBoost Booster
# ===========================================================================


class TestPlotPredictionsMorePaths:
    def test_encoding_map_applied(self, regression_messy, tmp_path):
        """Lines 475-476: encoding_map applied when model has categorical features."""
        from servers.ml_basic.engine import train_regressor
        from servers.ml_advanced.engine import plot_predictions_vs_actual

        mp = train_regressor(str(regression_messy), "salary", "rfr")["model_path"]
        out = str(tmp_path / "pred_enc.html")
        r = plot_predictions_vs_actual(mp, str(regression_messy), output_path=out, open_after=False)
        assert r["success"] is True

    def test_xgb_regression_predictions(self, regression_simple, tmp_path):
        """Lines 494-495: XGBoost Booster in plot_predictions_vs_actual."""
        from servers.ml_basic.engine import train_regressor
        from servers.ml_advanced.engine import plot_predictions_vs_actual

        mp = train_regressor(str(regression_simple), "salary", "xgb")["model_path"]
        out = str(tmp_path / "pred_xgb.html")
        r = plot_predictions_vs_actual(mp, str(regression_simple), output_path=out, open_after=False)
        assert r["success"] is True


# ===========================================================================
# generate_cluster_report — exception path
# ===========================================================================


class TestClusterReportExceptionPath:
    def test_exception_returns_error_dict(self, clustering_simple, tmp_path):
        """Lines 811-812: exception in generate_cluster_report returns error dict."""
        from unittest.mock import patch
        from servers.ml_medium.engine import run_clustering
        from servers.ml_advanced.engine import generate_cluster_report

        r = run_clustering(
            str(clustering_simple), ["x", "y"], algorithm="kmeans", save_labels=True
        )
        assert r["success"] is True

        out = str(tmp_path / "cr.html")
        with patch("servers.ml_advanced._adv_viz.pd.read_csv", side_effect=Exception("broken")):
            r2 = generate_cluster_report(
                str(clustering_simple), ["x", "y"], label_column="cluster_label",
                output_path=out, open_after=False
            )
        assert r2["success"] is False
'''
