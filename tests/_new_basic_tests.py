"""New targeted tests to append to test_ml_basic.py"""

BASIC_ADDITIONS = r'''

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

        with patch("servers.ml_basic._basic_train.roc_auc_score", side_effect=Exception("auc fail")):
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
'''
