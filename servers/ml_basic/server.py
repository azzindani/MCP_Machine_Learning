"""ml_basic server — Tier 1 MCP tool wrappers. Zero domain logic."""

import argparse
import logging
import sys

from fastmcp import FastMCP

from . import engine

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP("ml-basic")


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def inspect_dataset(file_path: str) -> dict:
    """Inspect dataset schema, row count, dtypes, null summary."""
    return engine.inspect_dataset(file_path)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def read_column_profile(file_path: str, column_name: str) -> dict:
    """Profile one column. Returns stats, null count, top values."""
    return engine.read_column_profile(file_path, column_name)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
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
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def read_rows(file_path: str, start: int, end: int) -> dict:
    """Read bounded row slice. Max rows enforced by hardware mode."""
    return engine.read_rows(file_path, start, end)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def train_classifier(
    file_path: str,
    target_column: str,
    model: str,
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train classifier on CSV. model: lr svm rf dtc knn nb xgb."""
    return engine.train_classifier(file_path, target_column, model, test_size, random_state, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
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
) -> dict:
    """Train regressor on CSV. model: lir pr lar rr dtr rfr xgb."""
    return engine.train_regressor(
        file_path, target_column, model, degree, alpha, n_estimators, test_size, random_state, dry_run
    )


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def get_predictions(model_path: str, file_path: str, max_rows: int = 20) -> dict:
    """Run predictions with saved model. Returns bounded prediction list."""
    return engine.get_predictions(model_path, file_path, max_rows)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file/model to previous snapshot. Empty timestamp = list."""
    return engine.restore_version(file_path, timestamp)


@mcp.tool(
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False}
)
def predict_single(model_path: str, input_data: str) -> dict:
    """Predict on one JSON record. No CSV file needed."""
    return engine.predict_single(model_path, input_data)


@mcp.tool(
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False}
)
def list_models(directory: str = "") -> dict:
    """List all saved .pkl models with metadata. Empty = home dir."""
    return engine.list_models(directory)


@mcp.tool(
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ml-basic MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port, path="/mcp")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
