"""ml_medium server — Tier 2 MCP tool wrappers. Zero domain logic."""

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
    from shared.arg_alias import missing, missing_list, pick, pick_list
    from shared.arg_errors import contract_errors
    from shared.deploy_auth import build_auth, build_oauth_bridge
    from shared.progress import info
    from shared.token_estimate import measure_responses

    from . import engine
except ImportError:
    from servers.ml_medium import engine
    from shared.arg_alias import missing, missing_list, pick, pick_list
    from shared.arg_errors import contract_errors
    from shared.deploy_auth import build_auth, build_oauth_bridge
    from shared.progress import info
    from shared.token_estimate import measure_responses

_VERSION = "0.1.2"  # keep in sync with pyproject.toml [project].version

_oauth_bridge = build_oauth_bridge(
    "ML", state_dir=os.environ.get("ML_MEDIUM_OAUTH_STATE_DIR", "/tmp/ml-medium-oauth-state")
)
_public_origin = os.environ.get("ML_PUBLIC_URL", "").rstrip("/")
_base_url = f"{_public_origin}/medium" if _public_origin else None
_HOST = os.environ.get("ML_MEDIUM_HOST", "127.0.0.1")
_PORT = int(os.environ.get("ML_MEDIUM_PORT", "8821"))
_token_verifier, _auth_settings = build_auth("ML", _base_url, _oauth_bridge)

mcp = FastMCP(
    "ml-medium",
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
def run_preprocessing(
    file_path: str,
    ops: list[dict],
    output_path: str = "",
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Apply preprocessing ops to dataset. Snapshot before write."""
    return engine.run_preprocessing(file_path, ops, output_path, dry_run, return_content)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def detect_outliers(
    file_path: str,
    columns: list[str] = [],
    method: str = "iqr",
    th1: float = 0.25,
    th3: float = 0.75,
    feature_columns: list[str] = [],
) -> dict:
    """Detect outliers in columns. feature_columns= also accepted. iqr or std."""
    chosen, note = pick_list("detect_outliers", "columns", columns, feature_columns)
    if not chosen:
        return missing_list("detect_outliers", "columns", "feature_columns")
    result = engine.detect_outliers(file_path, chosen, method, th1, th3)
    if note:
        result.setdefault("progress", []).append(info(note))
    return result


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def train_with_cv(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
    dry_run: bool = False,
    output_path: str = "",
) -> dict:
    """Train with K-fold CV. Returns per-fold and mean scores."""
    return engine.train_with_cv(file_path, target_column, model, task, n_splits, random_state, dry_run, output_path)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def compare_models(
    file_path: str,
    target_column: str,
    task: str,
    models: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
    output_path: str = "",
) -> dict:
    """Train multiple models, return sorted comparison table."""
    return engine.compare_models(file_path, target_column, task, models, test_size, random_state, dry_run, output_path)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def run_clustering(
    file_path: str,
    feature_columns: list[str],
    algorithm: str,
    n_clusters: int = 3,
    eps: float = 3.0,
    min_samples: int = 5,
    reduce_dims: str = "",
    n_components: int = 2,
    save_labels: bool = False,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Cluster dataset. algorithm: kmeans meanshift dbscan."""
    return engine.run_clustering(
        file_path,
        feature_columns,
        algorithm,
        n_clusters,
        eps,
        min_samples,
        reduce_dims,
        n_components,
        save_labels,
        output_path,
        dry_run,
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def read_receipt(file_path: str) -> dict:
    """Read operation history for a file. Returns log entries."""
    return engine.read_receipt(file_path)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def generate_eda_report(
    file_path: str,
    target_column: str = "",
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Generate interactive HTML EDA report. theme: light dark."""
    return engine.generate_eda_report(file_path, target_column, theme, output_path, open_after, dry_run, return_content)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def find_optimal_clusters(
    file_path: str,
    feature_columns: list[str],
    max_k: int = 10,
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    return_content: bool = False,
) -> dict:
    """Find optimal K via elbow + silhouette. Saves HTML chart."""
    return engine.find_optimal_clusters(
        file_path, feature_columns, max_k, theme, output_path, open_after, return_content
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def anomaly_detection(
    file_path: str,
    feature_columns: list[str],
    method: str = "isolation_forest",
    contamination: float = 0.05,
    save_labels: bool = False,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Detect anomalies. method: isolation_forest lof."""
    return engine.anomaly_detection(
        file_path, feature_columns, method, contamination, save_labels, output_path, dry_run
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def check_data_quality(file_path: str) -> dict:
    """Return JSON quality score 0-100 with typed alerts per column."""
    return engine.check_data_quality(file_path)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def evaluate_model(
    model_path: str,
    test_file_path: str = "",
    target_column: str = "",
    file_path: str = "",
) -> dict:
    """Score model on a labeled CSV. test_file_path= or file_path=."""
    # get_predictions and batch_predict both name this `file_path`, so a caller
    # chaining train -> evaluate writes that and pydantic refuses the call
    # before this server can say which name it wanted. test_file_path keeps its
    # original position so a positional caller still binds it correctly.
    chosen, note = pick("evaluate_model", "test_file_path", test_file_path, file_path)
    if not chosen:
        return missing("evaluate_model", "test_file_path", "file_path")
    if not target_column.strip():
        return missing("evaluate_model", "target_column", "target_column")
    result = engine.evaluate_model(model_path, chosen, target_column)
    if note:
        result.setdefault("progress", []).append(info(note))
    return result


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def batch_predict(
    model_path: str,
    file_path: str,
    output_path: str = "",
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """Predict all rows, save to CSV. No row limit. Returns output path."""
    return engine.batch_predict(model_path, file_path, output_path, dry_run, return_content)


# Every tool above reports what its response actually costs; see
# shared/token_estimate.py for why this is a choke point and not 101 edits.
measure_responses(mcp)
# A known argument with the WRONG TYPE is rejected by pydantic before any of
# this runs, and used to escape as a raw dump with no success/hint/token_estimate
# and a pydantic.dev URL. Give it the fleet's failure shape instead.
contract_errors(mcp)


def main() -> None:
    parser = argparse.ArgumentParser(description="ml_medium MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default=os.environ.get("ML_MEDIUM_TRANSPORT", "stdio")
    )
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
