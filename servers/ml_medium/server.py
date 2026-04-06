"""ml_medium server — Tier 2 MCP tool wrappers. Zero domain logic."""

import argparse
import logging
import sys

from fastmcp import FastMCP

from . import engine

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP("ml-medium")


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def run_preprocessing(
    file_path: str,
    ops: list[dict],
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Apply preprocessing ops to dataset. Snapshot before write."""
    return engine.run_preprocessing(file_path, ops, output_path, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def detect_outliers(
    file_path: str,
    columns: list[str],
    method: str = "iqr",
    th1: float = 0.25,
    th3: float = 0.75,
) -> dict:
    """Detect outliers in numeric columns. method: iqr std."""
    return engine.detect_outliers(file_path, columns, method, th1, th3)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def train_with_cv(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train with K-fold CV. Returns per-fold and mean scores."""
    return engine.train_with_cv(file_path, target_column, model, task, n_splits, random_state, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def compare_models(
    file_path: str,
    target_column: str,
    task: str,
    models: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    dry_run: bool = False,
) -> dict:
    """Train multiple models, return sorted comparison table."""
    return engine.compare_models(file_path, target_column, task, models, test_size, random_state, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
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
        dry_run,
    )


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def read_receipt(file_path: str) -> dict:
    """Read operation history for a file. Returns log entries."""
    return engine.read_receipt(file_path)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def generate_eda_report(
    file_path: str,
    target_column: str = "",
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate interactive HTML EDA report. theme: light dark."""
    return engine.generate_eda_report(file_path, target_column, theme, output_path, open_browser, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def filter_rows(
    file_path: str,
    column: str,
    operator: str,
    value: str = "",
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Filter rows by column condition. Saves filtered CSV."""
    return engine.filter_rows(file_path, column, operator, value, output_path, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def merge_datasets(
    file_path_1: str,
    file_path_2: str,
    on: str,
    how: str = "left",
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Merge two CSVs on a key column. how: left right inner outer."""
    return engine.merge_datasets(file_path_1, file_path_2, on, how, output_path, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def find_optimal_clusters(
    file_path: str,
    feature_columns: list[str],
    max_k: int = 10,
    theme: str = "light",
    output_path: str = "",
    open_browser: bool = True,
) -> dict:
    """Find optimal K via elbow + silhouette. Saves HTML chart."""
    return engine.find_optimal_clusters(file_path, feature_columns, max_k, theme, output_path, open_browser)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def anomaly_detection(
    file_path: str,
    feature_columns: list[str],
    method: str = "isolation_forest",
    contamination: float = 0.05,
    save_labels: bool = False,
    dry_run: bool = False,
) -> dict:
    """Detect anomalies. method: isolation_forest lof."""
    return engine.anomaly_detection(file_path, feature_columns, method, contamination, save_labels, dry_run)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def check_data_quality(file_path: str) -> dict:
    """Return JSON quality score 0-100 with typed alerts per column."""
    return engine.check_data_quality(file_path)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def evaluate_model(
    model_path: str,
    test_file_path: str,
    target_column: str,
) -> dict:
    """Score saved model on labeled test CSV. Returns metrics dict."""
    return engine.evaluate_model(model_path, test_file_path, target_column)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def batch_predict(
    model_path: str,
    file_path: str,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Predict all rows, save to CSV. No row limit. Returns output path."""
    return engine.batch_predict(model_path, file_path, output_path, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="ml-medium MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port, path="/mcp")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
