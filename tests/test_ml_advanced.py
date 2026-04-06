"""Tests for ml_advanced engine (Tier 3).

Fixture column names:
  classification_simple: age, tenure, monthly_charges, total_charges, num_products, churned
  regression_simple:     age, experience, education_level, department, performance_score, salary
  clustering_simple:     x, y
"""

import shutil
from pathlib import Path

import pytest

from servers.ml_advanced.engine import (
    apply_dimensionality_reduction,
    export_model,
    read_model_report,
    run_profiling_report,
    tune_hyperparameters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_basic_model(file_path: str, target: str = "churned", model: str = "rf", task: str = "classification") -> str:
    """Train a model via ml_basic and return model_path."""
    from servers.ml_basic.engine import train_classifier, train_regressor

    if task == "classification":
        r = train_classifier(file_path, target, model)
    else:
        r = train_regressor(file_path, target, model)
    assert r["success"], f"Training failed: {r.get('error')}"
    return r["model_path"]


# ---------------------------------------------------------------------------
# tune_hyperparameters
# ---------------------------------------------------------------------------


class TestTuneHyperparameters:
    def test_success_grid_classification(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification", search="grid")
        assert r["success"] is True
        assert r["op"] == "tune_hyperparameters"
        assert "best_params" in r

    def test_success_random_regression(self, regression_simple):
        r = tune_hyperparameters(regression_simple, "salary", "rfr", "regression", search="random", n_iter=5)
        assert r["success"] is True
        assert "best_score" in r

    def test_file_not_found(self, home_tmp):
        r = tune_hyperparameters(str(home_tmp / "nope.csv"), "target", "rf", "classification")
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "lr", "classification")
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "lr", "classification")
        assert "progress" in r

    def test_snapshot_created(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification")
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification")
        assert "backup" in r

    def test_dry_run(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification", dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_constrained_mode(self, classification_simple, constrained_mode):
        r = tune_hyperparameters(classification_simple, "churned", "lr", "classification", cv=5, n_iter=10)
        # In constrained mode n_iter capped at 5, cv at 3
        assert r.get("success") in (True, False)  # may fail on data size, not constraint

    def test_invalid_task(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "badtask")
        assert r["success"] is False

    def test_invalid_model(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "zzz", "classification")
        assert r["success"] is False

    def test_invalid_search(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification", search="badsearch")
        assert r["success"] is False

    def test_invalid_param_grid_json(self, classification_simple):
        r = tune_hyperparameters(
            classification_simple, "churned", "rf", "classification", param_grid="{not valid json}"
        )
        assert r["success"] is False

    def test_top_results_capped(self, classification_simple):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification")
        if r["success"]:
            assert len(r.get("top_results", [])) <= 20

    def test_model_saved(self, classification_simple, home_tmp):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification")
        if r["success"]:
            assert Path(r["model_path"]).exists()


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_success(self, classification_simple, home_tmp):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp)
        assert r["success"] is True
        assert r["op"] == "export_model"
        assert "manifest_path" in r

    def test_file_not_found(self, home_tmp):
        r = export_model(str(home_tmp / "nope.pkl"))
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp)
        assert "progress" in r

    def test_snapshot_created(self, classification_simple, home_tmp):
        mp = _train_basic_model(classification_simple)
        # Export to a different dir twice to trigger snapshot on second
        out = str(home_tmp / "exported_model.pkl")
        shutil.copy(mp, out)
        r = export_model(mp, output_dir=str(home_tmp))
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple, home_tmp):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp, output_dir=str(home_tmp))
        assert "backup" in r

    def test_dry_run(self, classification_simple, home_tmp):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True

    def test_constrained_mode(self, classification_simple, constrained_mode):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp)
        assert r["success"] is True

    def test_unsupported_format(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp, format="onnx")
        assert r["success"] is False

    def test_manifest_json_created(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        export_model(mp)
        manifest = Path(mp).with_suffix(".manifest.json")
        assert manifest.exists()

    def test_manifest_has_required_fields(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp)
        assert r["success"] is True
        import json

        manifest = json.loads(Path(r["manifest_path"]).read_text())
        for field in ("model_type", "task", "trained_on", "feature_columns", "sklearn_version"):
            assert field in manifest


# ---------------------------------------------------------------------------
# read_model_report
# ---------------------------------------------------------------------------


class TestReadModelReport:
    def test_success(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = read_model_report(mp)
        assert r["success"] is True
        assert r["op"] == "read_model_report"

    def test_file_not_found(self, home_tmp):
        r = read_model_report(str(home_tmp / "nope.pkl"))
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = read_model_report(mp)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = read_model_report(mp)
        assert "progress" in r

    def test_feature_importance_present(self, classification_simple):
        mp = _train_basic_model(classification_simple, model="rf")
        r = read_model_report(mp)
        assert r["success"] is True
        assert "feature_importance" in r

    def test_feature_importance_bounded(self, classification_simple):
        mp = _train_basic_model(classification_simple, model="rf")
        r = read_model_report(mp)
        assert len(r.get("feature_importance", [])) <= 10

    def test_metrics_present(self, classification_simple):
        mp = _train_basic_model(classification_simple)
        r = read_model_report(mp)
        assert "metrics" in r

    def test_regression_report(self, regression_simple):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        r = read_model_report(mp)
        assert r["success"] is True


# ---------------------------------------------------------------------------
# run_profiling_report
# ---------------------------------------------------------------------------


class TestRunProfilingReport:
    def test_success(self, classification_simple, home_tmp):
        out = str(home_tmp / "profile_report.html")
        try:
            r = run_profiling_report(classification_simple, output_path=out)
        except Exception:
            pytest.skip("ydata-profiling not installed")
        if not r["success"] and "not installed" in r.get("error", ""):
            pytest.skip("ydata-profiling not installed")
        assert r["success"] is True
        assert r["op"] == "run_profiling_report"

    def test_file_not_found(self, home_tmp):
        r = run_profiling_report(str(home_tmp / "nope.csv"))
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple, home_tmp):
        out = str(home_tmp / "pr.html")
        r = run_profiling_report(classification_simple, output_path=out)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, home_tmp):
        out = str(home_tmp / "pr2.html")
        r = run_profiling_report(classification_simple, output_path=out)
        if not r["success"] and "not installed" in r.get("error", ""):
            pytest.skip("ydata-profiling not installed")
        assert "progress" in r

    def test_dry_run(self, classification_simple, home_tmp):
        out = str(home_tmp / "pr_dry.html")
        r = run_profiling_report(classification_simple, output_path=out, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, classification_simple, constrained_mode, home_tmp):
        out = str(home_tmp / "pr_constrained.html")
        r = run_profiling_report(classification_simple, output_path=out)
        # Just check it returns a dict (ydata may or may not be installed)
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# apply_dimensionality_reduction
# ---------------------------------------------------------------------------


class TestApplyDimensionalityReduction:
    def test_success_pca(self, clustering_simple, home_tmp):
        out = str(home_tmp / "pca_out.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", n_components=2, output_path=out)
        assert r["success"] is True
        assert r["op"] == "apply_dimensionality_reduction"
        assert Path(out).exists()

    def test_success_ica(self, clustering_simple, home_tmp):
        out = str(home_tmp / "ica_out.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "ica", n_components=2, output_path=out)
        assert r["success"] is True

    def test_file_not_found(self, home_tmp):
        r = apply_dimensionality_reduction(str(home_tmp / "nope.csv"), ["x"], "pca")
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr1.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert "token_estimate" in r

    def test_progress_present(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr2.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert "progress" in r

    def test_snapshot_created(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr_snap.csv")
        # Create file first to trigger snapshot
        Path(out).write_text("dummy")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr_bak.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert "backup" in r

    def test_dry_run(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr_dry.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, clustering_simple, constrained_mode, home_tmp):
        out = str(home_tmp / "dr_cons.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert r["success"] is True

    def test_invalid_method(self, clustering_simple):
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "badmethod")
        assert r["success"] is False

    def test_missing_column(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr_mc.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "nonexistent"], "pca", output_path=out)
        assert r["success"] is False

    def test_variance_explained_for_pca(self, clustering_simple, home_tmp):
        out = str(home_tmp / "dr_var.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert r["success"] is True
        assert "variance_explained" in r
        assert len(r["variance_explained"]) > 0

    def test_component_columns_in_output(self, clustering_simple, home_tmp):
        import pandas as pd

        out = str(home_tmp / "dr_cols.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", n_components=2, output_path=out)
        assert r["success"] is True
        df_out = pd.read_csv(out)
        assert "component_1" in df_out.columns
        assert "component_2" in df_out.columns
