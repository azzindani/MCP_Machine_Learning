"""ml_basic server — Tier 1 MCP tool wrappers. Zero domain logic."""

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
    from servers.ml_basic import engine
    from shared.arg_errors import contract_errors
    from shared.deploy_auth import build_auth, build_oauth_bridge
    from shared.token_estimate import measure_responses

_VERSION = "0.1.1"  # keep in sync with pyproject.toml [project].version

_oauth_bridge = build_oauth_bridge(
    "ML", state_dir=os.environ.get("ML_BASIC_OAUTH_STATE_DIR", "/tmp/ml-basic-oauth-state")
)
_public_origin = os.environ.get("ML_PUBLIC_URL", "").rstrip("/")
_base_url = f"{_public_origin}/basic" if _public_origin else None
_HOST = os.environ.get("ML_BASIC_HOST", "127.0.0.1")
_PORT = int(os.environ.get("ML_BASIC_PORT", "8820"))
_token_verifier, _auth_settings = build_auth("ML", _base_url, _oauth_bridge)

mcp = FastMCP(
    "ml-basic",
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
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def inspect_dataset(file_path: str) -> dict:
    """Inspect dataset schema, row count, dtypes, null summary."""
    return engine.inspect_dataset(file_path)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def read_column_profile(file_path: str, column_name: str) -> dict:
    """Profile one column. Returns stats, null count, top values."""
    return engine.read_column_profile(file_path, column_name)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def search_columns(
    file_path: str,
    has_nulls: bool = False,
    dtype: str = "",
    name_contains: str = "",
    max_results: int = 20,
) -> dict:
    """Search columns by condition. Returns names only, no data."""
    return engine.search_columns(file_path, has_nulls, dtype, name_contains, max_results)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def read_rows(file_path: str, start: int, end: int) -> dict:
    """Read bounded row slice. Max rows enforced by hardware mode."""
    return engine.read_rows(file_path, start, end)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def train_classifier(
    file_path: str,
    target_column: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    class_weight: str = "",
    return_train_score: bool = False,
    dry_run: bool = False,
    output_path: str = "",
) -> dict:
    """Train classifier on CSV. model: lr svm rf dtc knn nb xgb."""
    return engine.train_classifier(
        file_path,
        target_column,
        model,
        test_size,
        random_state,
        class_weight,
        return_train_score,
        dry_run,
        output_path,
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def train_regressor(
    file_path: str,
    target_column: str,
    model: str,
    degree: int = 5,
    alpha: float = 0.01,
    n_estimators: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
    output_path: str = "",
) -> dict:
    """Train regressor on CSV. model: lir pr lar rr dtr rfr xgb."""
    return engine.train_regressor(
        file_path,
        target_column,
        model,
        degree,
        alpha,
        n_estimators,
        test_size,
        random_state,
        dry_run,
        output_path,
    )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def get_predictions(model_path: str, file_path: str, max_rows: int = 20, return_proba: bool = False) -> dict:
    """Run predictions with saved model. Returns bounded prediction list."""
    return engine.get_predictions(model_path, file_path, max_rows, return_proba)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file/model to previous snapshot. Empty timestamp = list."""
    return engine.restore_version(file_path, timestamp)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def predict_single(model_path: str, input_data: str | dict) -> dict:
    """Predict on one record: a JSON string or an object. No CSV needed."""
    return engine.predict_single(model_path, input_data)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
)
def list_models(directory: str = "") -> dict:
    """List saved .pkl models. Empty scans the server's model output dir."""
    return engine.list_models(directory)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
)
def split_dataset(
    file_path: str,
    test_size: float = 0.2,
    stratify_column: str = "",
    output_dir: str = "",
    random_state: int = 42,
) -> dict:
    """Split CSV into train/test CSV files and save both."""
    return engine.split_dataset(file_path, test_size, stratify_column, output_dir, random_state)


# Every tool above reports what its response actually costs; see
# shared/token_estimate.py for why this is a choke point and not 101 edits.
measure_responses(mcp)
# A known argument with the WRONG TYPE is rejected by pydantic before any of
# this runs, and used to escape as a raw dump with no success/hint/token_estimate
# and a pydantic.dev URL. Give it the fleet's failure shape instead.
contract_errors(mcp)


def main() -> None:
    parser = argparse.ArgumentParser(description="ml_basic MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default=os.environ.get("ML_BASIC_TRANSPORT", "stdio"))
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
