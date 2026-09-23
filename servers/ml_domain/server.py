"""The whole machine-learning surface as four domain tools -- one endpoint, `action` plus `args`.

The three tier servers list 33 tools between them; a model connected to all of
them reads 33 names on every turn. This endpoint lists four -- data, train,
predict, report -- and each tool's `action` is one of those 33 tools by its own
name. Schemas,
validation, wrappers and answers are the tiers' own: see shared/domain_tools.py.
The tier endpoints keep serving unchanged, for small local models and for
every client already connected to one.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

from servers.ml_advanced.server import mcp as advanced  # noqa: E402
from servers.ml_basic.server import mcp as basic  # noqa: E402
from servers.ml_medium.server import mcp as medium  # noqa: E402
from shared.arg_errors import contract_errors  # noqa: E402
from shared.deploy_auth import build_auth, build_oauth_bridge  # noqa: E402
from shared.domain_tools import register_domains  # noqa: E402
from shared.strict_args import enforce_known_arguments  # noqa: E402

_VERSION = "0.2.0"  # keep in sync with pyproject.toml [project].version

_oauth_bridge = build_oauth_bridge(
    "ML", state_dir=os.environ.get("ML_DOMAIN_OAUTH_STATE_DIR", "/tmp/ml-domain-oauth-state")
)
_public_origin = os.environ.get("ML_PUBLIC_URL", "").rstrip("/")
_HOST = os.environ.get("ML_DOMAIN_HOST", "127.0.0.1")
_PORT = int(os.environ.get("ML_DOMAIN_PORT", "8824"))
_token_verifier, _auth_settings = build_auth("ML", _public_origin or None, _oauth_bridge)

mcp = FastMCP("ml", host=_HOST, port=_PORT, token_verifier=_token_verifier, auth=_auth_settings)
if _oauth_bridge is not None:
    _oauth_bridge.register_routes(mcp)

# Each domain: what it is for, then its actions -- each an existing tier tool.
DOMAINS = {
    "ml_data": (
        "Look at and prepare a dataset: schema, profile, search, rows, quality, outliers, split, preprocess, reduce, undo.",
        [
            (basic, "inspect_dataset"),
            (basic, "read_column_profile"),
            (basic, "search_columns"),
            (basic, "read_rows"),
            (medium, "check_data_quality"),
            (medium, "detect_outliers"),
            (basic, "split_dataset"),
            (medium, "run_preprocessing"),
            (advanced, "apply_dimensionality_reduction"),
            (basic, "restore_version"),
            (medium, "read_receipt"),
        ],
    ),
    "ml_train": (
        "Fit a model: classifier, regressor, cross-validation, model comparison, tuning, clustering, anomalies.",
        [
            (basic, "train_classifier"),
            (basic, "train_regressor"),
            (medium, "train_with_cv"),
            (medium, "compare_models"),
            (advanced, "tune_hyperparameters"),
            (medium, "run_clustering"),
            (medium, "find_optimal_clusters"),
            (medium, "anomaly_detection"),
        ],
    ),
    "ml_predict": (
        "Use a trained model: predict one row or a file, evaluate it, list, export or describe models.",
        [
            (basic, "get_predictions"),
            (basic, "predict_single"),
            (medium, "batch_predict"),
            (medium, "evaluate_model"),
            (basic, "list_models"),
            (advanced, "export_model"),
            (advanced, "read_model_report"),
        ],
    ),
    "ml_report": (
        "Reports and charts as HTML: EDA, profiling, training and cluster reports, ROC, learning curve, fit.",
        [
            (medium, "generate_eda_report"),
            (advanced, "run_profiling_report"),
            (advanced, "generate_training_report"),
            (advanced, "generate_cluster_report"),
            (advanced, "plot_roc_curve"),
            (advanced, "plot_learning_curve"),
            (advanced, "plot_predictions_vs_actual"),
        ],
    ),
}
register_domains(mcp, DOMAINS)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness check. Unauthenticated."""
    return JSONResponse({"status": "ok", "version": _VERSION, "tools": len(DOMAINS)})


# A wrong-typed `args` or an unknown top-level key gets the fleet's failure
# shape, as on every tier; per-action arguments are checked by the dispatcher.
contract_errors(mcp)
enforce_known_arguments(mcp)


def main() -> None:
    parser = argparse.ArgumentParser(description="ml domain MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default=os.environ.get("ML_DOMAIN_TRANSPORT", "stdio")
    )
    args = parser.parse_args()
    mcp.run(transport="streamable-http" if args.transport == "http" else "stdio")


if __name__ == "__main__":
    main()
