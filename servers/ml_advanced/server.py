"""ml_advanced server — Tier 3 MCP tool wrappers. Zero domain logic."""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

try:
    from shared.arg_errors import contract_errors
    from shared.deploy_auth import build_auth, build_oauth_bridge
    from shared.token_estimate import measure_responses

    from . import engine
except ImportError:
    from servers.ml_advanced import engine
    from shared.arg_errors import contract_errors
    from shared.deploy_auth import build_auth, build_oauth_bridge
    from shared.token_estimate import measure_responses

_VERSION = "0.1.2"  # keep in sync with pyproject.toml [project].version

_oauth_bridge = build_oauth_bridge(
    "ML", state_dir=os.environ.get("ML_ADVANCED_OAUTH_STATE_DIR", "/tmp/ml-advanced-oauth-state")
)
_public_origin = os.environ.get("ML_PUBLIC_URL", "").rstrip("/")
_base_url = f"{_public_origin}/advanced" if _public_origin else None
_HOST = os.environ.get("ML_ADVANCED_HOST", "127.0.0.1")
_PORT = int(os.environ.get("ML_ADVANCED_PORT", "8822"))
_token_verifier, _auth_settings = build_auth("ML", _base_url, _oauth_bridge)

mcp = FastMCP(
    "ml-advanced",
    host=_HOST,
    port=_PORT,
    token_verifier=_token_verifier,
    auth=_auth_settings,
)
if _oauth_bridge is not None:
    _oauth_bridge.register_routes(mcp)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness check. Unauthenticated."""
    return JSONResponse({"status": "ok", "version": _VERSION})


@mcp.custom_route("/version", methods=["GET"])
async def version(request: Request) -> JSONResponse:
    """Report running version. Unauthenticated."""
    return JSONResponse({"current": _VERSION})


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
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
    output_path: str = "",
) -> dict:
    """Tune hyperparameters via grid or random search. search: grid random."""
    return engine.tune_hyperparameters(
        file_path, target_column, model, task, search, param_grid, cv, n_iter, dry_run, output_path
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def export_model(
    model_path: str,
    output_dir: str = "",
    format: str = "pickle",
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Export model + manifest. Pickle carries a 32-byte signature prefix."""
    return engine.export_model(model_path, output_dir, format, dry_run, return_content)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def read_model_report(model_path: str) -> dict:
    """Read model metrics, feature importance, confusion matrix."""
    return engine.read_model_report(model_path)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def run_profiling_report(
    file_path: str,
    output_path: str = "",
    sample_rows: int = 0,
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
    theme: str = "device",
) -> dict:
    """Generate Plotly HTML profile report for a dataset."""
    return engine.run_profiling_report(file_path, output_path, sample_rows, open_after, dry_run, return_content, theme)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def apply_dimensionality_reduction(
    file_path: str,
    feature_columns: list[str],
    method: str,
    n_components: int = 2,
    output_path: str = "",
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Reduce dimensions with PCA or ICA. Saves reduced dataset."""
    return engine.apply_dimensionality_reduction(
        file_path, feature_columns, method, n_components, output_path, dry_run, return_content
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def generate_training_report(
    model_path: str,
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Generate HTML report: metrics, confusion matrix, feature importance."""
    return engine.generate_training_report(model_path, theme, output_path, open_after, dry_run, return_content)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def plot_roc_curve(
    model_path: str,
    file_path: str,
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Plot ROC curve for classifier. Saves interactive HTML."""
    return engine.plot_roc_curve(model_path, file_path, theme, output_path, open_after, dry_run, return_content)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def plot_learning_curve(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    cv: int = 5,
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Plot train vs val score by training size. HTML output."""
    return engine.plot_learning_curve(
        file_path, target_column, model, task, cv, theme, output_path, open_after, dry_run, return_content
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def plot_predictions_vs_actual(
    model_path: str,
    file_path: str,
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Scatter predicted vs actual for regression. HTML output."""
    return engine.plot_predictions_vs_actual(
        model_path, file_path, theme, output_path, open_after, dry_run, return_content
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def generate_cluster_report(
    file_path: str,
    feature_columns: list[str],
    label_column: str,
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Generate HTML cluster visualization with PCA scatter and profile."""
    return engine.generate_cluster_report(
        file_path, feature_columns, label_column, theme, output_path, open_after, dry_run, return_content
    )


# Every tool above reports what its response actually costs; see
# shared/token_estimate.py for why this is a choke point and not 101 edits.
measure_responses(mcp)
# A known argument with the WRONG TYPE is rejected by pydantic before any of
# this runs, and used to escape as a raw dump with no success/hint/token_estimate
# and a pydantic.dev URL. Give it the fleet's failure shape instead.
contract_errors(mcp)


def main() -> None:
    parser = argparse.ArgumentParser(description="ml_advanced MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default=os.environ.get("ML_ADVANCED_TRANSPORT", "stdio")
    )
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
