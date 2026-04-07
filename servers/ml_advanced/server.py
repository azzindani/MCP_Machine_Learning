"""ml_advanced server — Tier 3 MCP tool wrappers. Zero domain logic."""

from __future__ import annotations

import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from fastmcp import FastMCP

try:
    from . import engine
except ImportError:
    from servers.ml_advanced import engine

mcp = FastMCP("ml-advanced")


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def tune_hyperparameters(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    search: str = "grid",
    param_grid: str = "",
    cv: int = 5,
    n_iter: int = 10,
    dry_run: bool = False,
) -> dict:
    """Tune hyperparameters via grid or random search. search: grid random."""
    return engine.tune_hyperparameters(file_path, target_column, model, task, search, param_grid, cv, n_iter, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def export_model(
    model_path: str,
    output_dir: str = "",
    format: str = "pickle",
    dry_run: bool = False,
) -> dict:
    """Export trained model with metadata manifest. format: pickle."""
    return engine.export_model(model_path, output_dir, format, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def read_model_report(model_path: str) -> dict:
    """Read model metrics, feature importance, confusion matrix."""
    return engine.read_model_report(model_path)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def run_profiling_report(
    file_path: str,
    output_path: str = "",
    sample_rows: int = 0,
    dry_run: bool = False,
) -> dict:
    """Generate ydata-profiling HTML report for dataset."""
    return engine.run_profiling_report(file_path, output_path, sample_rows, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def apply_dimensionality_reduction(
    file_path: str,
    feature_columns: list[str],
    method: str,
    n_components: int = 2,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Reduce dimensions with PCA or ICA. Saves reduced dataset."""
    return engine.apply_dimensionality_reduction(file_path, feature_columns, method, n_components, output_path, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def generate_training_report(
    model_path: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate HTML report: metrics, confusion matrix, feature importance."""
    return engine.generate_training_report(model_path, theme, output_path, open_browser, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def plot_roc_curve(
    model_path: str,
    file_path: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Plot ROC curve for classifier. Saves interactive HTML."""
    return engine.plot_roc_curve(model_path, file_path, theme, output_path, open_browser, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def plot_learning_curve(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    cv: int = 5,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Plot train vs val score by training size. HTML output."""
    return engine.plot_learning_curve(
        file_path, target_column, model, task, cv, theme, output_path, open_browser, dry_run
    )


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def plot_predictions_vs_actual(
    model_path: str,
    file_path: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Scatter predicted vs actual for regression. HTML output."""
    return engine.plot_predictions_vs_actual(model_path, file_path, theme, output_path, open_browser, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def generate_cluster_report(
    file_path: str,
    feature_columns: list[str],
    label_column: str,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate HTML cluster visualization with PCA scatter and profile."""
    return engine.generate_cluster_report(
        file_path, feature_columns, label_column, theme, output_path, open_browser, dry_run
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
