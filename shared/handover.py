"""Handover protocol for multi-tool call loops (Ring 0).

Every tool response includes a 'handover' field that tells the calling
LLM what workflow step was just completed and what to call next.
"""

from __future__ import annotations

NEXT_STEP: dict[str, str] = {
    "LOCATE": "INSPECT",
    "INSPECT": "PATCH",
    "PATCH": "VERIFY",
    "VERIFY": "LOCATE",
}

STEP_TOOLS: dict[str, list[str]] = {
    "LOCATE": ["inspect_dataset", "search_columns", "list_models"],
    "INSPECT": [
        "read_column_profile",
        "read_rows",
        "read_model_report",
        "detect_outliers",
        "check_data_quality",
    ],
    "PATCH": [
        "train_classifier",
        "train_regressor",
        "run_preprocessing",
        "run_clustering",
        "tune_hyperparameters",
        "batch_predict",
        "apply_dimensionality_reduction",
    ],
    "VERIFY": [
        "get_predictions",
        "evaluate_model",
        "read_model_report",
        "generate_training_report",
        "plot_roc_curve",
        "plot_learning_curve",
        "plot_predictions_vs_actual",
        "generate_cluster_report",
    ],
}


def make_handover(
    workflow_step: str,
    suggested_tools: list[str],
    carry_forward: dict[str, object] | None = None,
) -> dict:
    """Return a handover dict for inclusion in every tool response.

    workflow_step   : current step (LOCATE | INSPECT | PATCH | VERIFY)
    suggested_tools : ordered list of recommended next tools
    carry_forward   : exact key-value pairs caller should pass to next tool
    """
    return {
        "workflow_step": workflow_step,
        "workflow_next": NEXT_STEP.get(workflow_step, ""),
        "suggested_tools": suggested_tools,
        "carry_forward": carry_forward or {},
    }
