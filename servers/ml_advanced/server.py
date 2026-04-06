"""ml_advanced server — Tier 3 MCP tool wrappers. Zero domain logic."""

import argparse
import logging
import sys

from fastmcp import FastMCP

from . import engine

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

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
    return engine.tune_hyperparameters(
        file_path, target_column, model, task, search, param_grid, cv, n_iter, dry_run
    )


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
    return engine.apply_dimensionality_reduction(
        file_path, feature_columns, method, n_components, output_path, dry_run
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ml-advanced MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port, path="/mcp")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
