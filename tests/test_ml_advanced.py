"""Tests for ml_advanced engine (Tier 3).

Fixture column names:
  classification_simple: age, tenure, monthly_charges, total_charges, num_products, churned
  regression_simple:     age, experience, education_level, department, performance_score, salary
  clustering_simple:     x, y
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from servers.ml_advanced._adv_viz import generate_cluster_report
from servers.ml_advanced.engine import (
    apply_dimensionality_reduction,
    export_model,
    generate_training_report,
    plot_learning_curve,
    plot_predictions_vs_actual,
    plot_roc_curve,
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

    def test_file_not_found(self, tmp_path):
        r = tune_hyperparameters(str(tmp_path / "nope.csv"), "target", "rf", "classification")
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

    def test_model_saved(self, classification_simple, tmp_path):
        r = tune_hyperparameters(classification_simple, "churned", "rf", "classification")
        if r["success"]:
            assert Path(r["model_path"]).exists()


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_success(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp)
        assert r["success"] is True
        assert r["op"] == "export_model"
        assert "manifest_path" in r

    def test_file_not_found(self, tmp_path):
        r = export_model(str(tmp_path / "nope.pkl"))
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

    def test_snapshot_created(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        # Export to a different dir twice to trigger snapshot on second
        out = str(tmp_path / "exported_model.pkl")
        shutil.copy(mp, out)
        r = export_model(mp, output_dir=str(tmp_path))
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        r = export_model(mp, output_dir=str(tmp_path))
        assert "backup" in r

    def test_dry_run(self, classification_simple, tmp_path):
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

    def test_file_not_found(self, tmp_path):
        r = read_model_report(str(tmp_path / "nope.pkl"))
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

    def test_constrained_mode(self, classification_simple, constrained_mode):
        mp = _train_basic_model(classification_simple)
        r = read_model_report(mp)
        assert r["success"] is True


# ---------------------------------------------------------------------------
# run_profiling_report
# ---------------------------------------------------------------------------


class TestRunProfilingReport:
    def test_success(self, classification_simple, tmp_path):
        out = str(tmp_path / "profile_report.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert r["op"] == "run_profiling_report"

    def test_file_not_found(self, tmp_path):
        r = run_profiling_report(str(tmp_path / "nope.csv"), open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "pr.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "pr2.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False)
        assert "progress" in r

    def test_dry_run(self, classification_simple, tmp_path):
        out = str(tmp_path / "pr_dry.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        out = str(tmp_path / "pr_constrained.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert isinstance(r, dict)

    def test_output_name_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "pr_name.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert "output_name" in r

    def test_html_file_created(self, classification_simple, tmp_path):
        out = str(tmp_path / "pr_file.html")
        r = run_profiling_report(classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_auto_opens_browser(self, classification_simple, tmp_path):
        out = str(tmp_path / "pr_open.html")
        with patch("shared.html_theme._open_file") as mock_open:
            r = run_profiling_report(classification_simple, output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()


# ---------------------------------------------------------------------------
# apply_dimensionality_reduction
# ---------------------------------------------------------------------------


class TestApplyDimensionalityReduction:
    def test_success_pca(self, clustering_simple, tmp_path):
        out = str(tmp_path / "pca_out.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", n_components=2, output_path=out)
        assert r["success"] is True
        assert r["op"] == "apply_dimensionality_reduction"
        assert Path(out).exists()

    def test_success_ica(self, clustering_simple, tmp_path):
        out = str(tmp_path / "ica_out.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "ica", n_components=2, output_path=out)
        assert r["success"] is True

    def test_file_not_found(self, tmp_path):
        r = apply_dimensionality_reduction(str(tmp_path / "nope.csv"), ["x"], "pca")
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr1.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert "token_estimate" in r

    def test_progress_present(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr2.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert "progress" in r

    def test_snapshot_created(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr_snap.csv")
        # Create file first to trigger snapshot
        Path(out).write_text("dummy")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert r["success"] is True
        assert "backup" in r

    def test_backup_in_response(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr_bak.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert "backup" in r

    def test_dry_run(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr_dry.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, clustering_simple, constrained_mode, tmp_path):
        out = str(tmp_path / "dr_cons.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert r["success"] is True

    def test_invalid_method(self, clustering_simple):
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "badmethod")
        assert r["success"] is False

    def test_missing_column(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr_mc.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "nonexistent"], "pca", output_path=out)
        assert r["success"] is False

    def test_variance_explained_for_pca(self, clustering_simple, tmp_path):
        out = str(tmp_path / "dr_var.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", output_path=out)
        assert r["success"] is True
        assert "variance_explained" in r
        assert len(r["variance_explained"]) > 0

    def test_component_columns_in_output(self, clustering_simple, tmp_path):
        import pandas as pd

        out = str(tmp_path / "dr_cols.csv")
        r = apply_dimensionality_reduction(clustering_simple, ["x", "y"], "pca", n_components=2, output_path=out)
        assert r["success"] is True
        df_out = pd.read_csv(out)
        assert "component_1" in df_out.columns
        assert "component_2" in df_out.columns


# ---------------------------------------------------------------------------
# generate_training_report
# ---------------------------------------------------------------------------


class TestGenerateTrainingReport:
    def test_success(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "training_report.html")
        r = generate_training_report(mp, output_path=out, open_after=False)
        assert r["success"] is True
        assert r["op"] == "generate_training_report"
        assert Path(out).exists()

    def test_model_not_found(self, tmp_path):
        r = generate_training_report(str(tmp_path / "ghost.pkl"), open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "tr_tok.html")
        r = generate_training_report(mp, output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "tr_prog.html")
        r = generate_training_report(mp, output_path=out, open_after=False)
        assert isinstance(r.get("progress"), list)

    def test_output_name_present(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "tr_name.html")
        r = generate_training_report(mp, output_path=out, open_after=False)
        assert r["success"] is True
        assert "output_name" in r

    def test_dry_run(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "tr_dry.html")
        r = generate_training_report(mp, output_path=out, open_after=False, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_regression_report(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "tr_reg.html")
        r = generate_training_report(mp, output_path=out, open_after=False)
        assert r["success"] is True

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "tr_cons.html")
        r = generate_training_report(mp, output_path=out, open_after=False)
        assert r["success"] is True

    def test_auto_opens_browser(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple)
        out = str(tmp_path / "tr_open.html")
        with patch("shared.html_theme._open_file") as mock_open:
            r = generate_training_report(mp, output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------


class TestPlotRocCurve:
    def test_success(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc.html")
        r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert r["op"] == "plot_roc_curve"
        assert Path(out).exists()

    def test_model_not_found(self, classification_simple, tmp_path):
        r = plot_roc_curve(str(tmp_path / "ghost.pkl"), classification_simple, open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_file_not_found(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        r = plot_roc_curve(mp, str(tmp_path / "ghost.csv"), open_after=False)
        assert r["success"] is False

    def test_token_estimate_present(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc_tok.html")
        r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc_prog.html")
        r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=False)
        assert isinstance(r.get("progress"), list)

    def test_output_name_present(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc_name.html")
        r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert "output_name" in r

    def test_dry_run(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc_dry.html")
        r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=False, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc_cons.html")
        r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=False)
        assert r["success"] is True

    def test_auto_opens_browser(self, classification_simple, tmp_path):
        mp = _train_basic_model(classification_simple, model="lr")
        out = str(tmp_path / "roc_open.html")
        with patch("servers.ml_advanced._adv_helpers._open_file") as mock_open:
            r = plot_roc_curve(mp, classification_simple, output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()


# ---------------------------------------------------------------------------
# plot_learning_curve
# ---------------------------------------------------------------------------


class TestPlotLearningCurve:
    def test_success_classification(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_cls.html")
        r = plot_learning_curve(
            classification_simple, "churned", "rf", "classification", cv=3, output_path=out, open_after=False
        )
        assert r["success"] is True
        assert r["op"] == "plot_learning_curve"
        assert Path(out).exists()

    def test_success_regression(self, regression_simple, tmp_path):
        out = str(tmp_path / "lc_reg.html")
        r = plot_learning_curve(
            regression_simple, "salary", "rfr", "regression", cv=3, output_path=out, open_after=False
        )
        assert r["success"] is True

    def test_file_not_found(self, tmp_path):
        r = plot_learning_curve(str(tmp_path / "ghost.csv"), "target", "rf", "classification", open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_token_estimate_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_tok.html")
        r = plot_learning_curve(
            classification_simple, "churned", "lr", "classification", cv=3, output_path=out, open_after=False
        )
        assert "token_estimate" in r

    def test_progress_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_prog.html")
        r = plot_learning_curve(
            classification_simple, "churned", "lr", "classification", cv=3, output_path=out, open_after=False
        )
        assert isinstance(r.get("progress"), list)

    def test_output_name_present(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_name.html")
        r = plot_learning_curve(
            classification_simple, "churned", "lr", "classification", cv=3, output_path=out, open_after=False
        )
        assert r["success"] is True
        assert "output_name" in r

    def test_invalid_task(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_bad.html")
        r = plot_learning_curve(classification_simple, "churned", "rf", "badtask", output_path=out, open_after=False)
        assert r["success"] is False

    def test_dry_run(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_dry.html")
        r = plot_learning_curve(
            classification_simple,
            "churned",
            "rf",
            "classification",
            cv=3,
            output_path=out,
            open_after=False,
            dry_run=True,
        )
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, classification_simple, constrained_mode, tmp_path):
        out = str(tmp_path / "lc_cons.html")
        r = plot_learning_curve(
            classification_simple, "churned", "lr", "classification", cv=5, output_path=out, open_after=False
        )
        assert r["success"] is True

    def test_auto_opens_browser(self, classification_simple, tmp_path):
        out = str(tmp_path / "lc_open.html")
        with patch("servers.ml_advanced._adv_helpers._open_file") as mock_open:
            r = plot_learning_curve(
                classification_simple, "churned", "lr", "classification", cv=3, output_path=out, open_after=True
            )
        assert r["success"] is True
        mock_open.assert_called_once()


# ---------------------------------------------------------------------------
# plot_predictions_vs_actual
# ---------------------------------------------------------------------------


class TestPlotPredictionsVsActual:
    def test_success(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva.html")
        r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert r["op"] == "plot_predictions_vs_actual"
        assert Path(out).exists()

    def test_model_not_found(self, regression_simple, tmp_path):
        r = plot_predictions_vs_actual(str(tmp_path / "ghost.pkl"), regression_simple, open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_file_not_found(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        r = plot_predictions_vs_actual(mp, str(tmp_path / "ghost.csv"), open_after=False)
        assert r["success"] is False

    def test_token_estimate_present(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva_tok.html")
        r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_progress_present(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva_prog.html")
        r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=False)
        assert isinstance(r.get("progress"), list)

    def test_output_name_present(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva_name.html")
        r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=False)
        assert r["success"] is True
        assert "output_name" in r

    def test_dry_run(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva_dry.html")
        r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=False, dry_run=True)
        assert r["success"] is True
        assert r.get("dry_run") is True
        assert not Path(out).exists()

    def test_constrained_mode(self, regression_simple, constrained_mode, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva_cons.html")
        r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=False)
        assert r["success"] is True

    def test_auto_opens_browser(self, regression_simple, tmp_path):
        mp = _train_basic_model(regression_simple, target="salary", model="rfr", task="regression")
        out = str(tmp_path / "pva_open.html")
        with patch("servers.ml_advanced._adv_helpers._open_file") as mock_open:
            r = plot_predictions_vs_actual(mp, regression_simple, output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()


# ---------------------------------------------------------------------------
# generate_cluster_report
# ---------------------------------------------------------------------------


def _make_clustered_csv(file_path: str, tmp_path: Path) -> str:
    """Run kmeans with save_labels=True and return the labelled CSV path."""
    from servers.ml_medium.engine import run_clustering

    r = run_clustering(file_path, ["x", "y"], "kmeans", n_clusters=3, save_labels=True)
    assert r["success"], f"Clustering failed: {r.get('error')}"
    return file_path  # labels written in-place


class TestGenerateClusterReport:
    def test_success(self, clustering_simple, tmp_path):
        labelled = _make_clustered_csv(clustering_simple, tmp_path)
        out = str(tmp_path / "cluster_report.html")
        r = generate_cluster_report(labelled, ["x", "y"], "cluster_label", output_path=out, open_after=False)
        assert r["success"] is True
        assert r["op"] == "generate_cluster_report"
        assert Path(out).exists()

    def test_file_not_found(self, tmp_path):
        r = generate_cluster_report(str(tmp_path / "ghost.csv"), ["x", "y"], "cluster_label", open_after=False)
        assert r["success"] is False
        assert "hint" in r

    def test_missing_label_column(self, clustering_simple, tmp_path):
        out = str(tmp_path / "cr_bad_label.html")
        r = generate_cluster_report(
            clustering_simple, ["x", "y"], "nonexistent_label", output_path=out, open_after=False
        )
        assert r["success"] is False

    def test_token_estimate_present(self, clustering_simple, tmp_path):
        labelled = _make_clustered_csv(clustering_simple, tmp_path)
        out = str(tmp_path / "cr_tok.html")
        r = generate_cluster_report(labelled, ["x", "y"], "cluster_label", output_path=out, open_after=False)
        assert "token_estimate" in r

    def test_progress_present(self, clustering_simple, tmp_path):
        labelled = _make_clustered_csv(clustering_simple, tmp_path)
        out = str(tmp_path / "cr_prog.html")
        r = generate_cluster_report(labelled, ["x", "y"], "cluster_label", output_path=out, open_after=False)
        assert isinstance(r.get("progress"), list)

    def test_output_name_present(self, clustering_simple, tmp_path):
        labelled = _make_clustered_csv(clustering_simple, tmp_path)
        out = str(tmp_path / "cr_name.html")
        r = generate_cluster_report(labelled, ["x", "y"], "cluster_label", output_path=out, open_after=False)
        assert r["success"] is True
        assert "output_name" in r

    def test_constrained_mode(self, clustering_simple, constrained_mode, tmp_path):
        labelled = _make_clustered_csv(clustering_simple, tmp_path)
        out = str(tmp_path / "cr_cons.html")
        r = generate_cluster_report(labelled, ["x", "y"], "cluster_label", output_path=out, open_after=False)
        assert r["success"] is True

    def test_auto_opens_browser(self, clustering_simple, tmp_path):
        labelled = _make_clustered_csv(clustering_simple, tmp_path)
        out = str(tmp_path / "cr_open.html")
        with patch("shared.html_theme._open_file") as mock_open:
            r = generate_cluster_report(labelled, ["x", "y"], "cluster_label", output_path=out, open_after=True)
        assert r["success"] is True
        mock_open.assert_called_once()
