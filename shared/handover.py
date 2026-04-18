"""Universal handover protocol for multi-MCP-server tool call loops.

Workflow steps (server-agnostic, in order):
  COLLECT -> INSPECT -> CLEAN -> PREPARE -> TRAIN -> EVALUATE -> REPORT

Compatible with MCP_Data_Analyst handover.py — both servers emit the same
handover schema so the calling LLM can chain tools across servers.

Legacy step names (LOCATE/PATCH/VERIFY) are accepted and mapped to canonical
names for backward compatibility with existing callers.
"""

from __future__ import annotations

from datetime import UTC, datetime

WORKFLOW_STEPS: list[str] = ["COLLECT", "INSPECT", "CLEAN", "PREPARE", "TRAIN", "EVALUATE", "REPORT"]

# Legacy step names -> canonical mapping
_LEGACY_STEP_MAP: dict[str, str] = {
    "LOCATE": "COLLECT",
    "PATCH": "TRAIN",
    "VERIFY": "EVALUATE",
}

DOMAIN_SERVERS: dict[str, str] = {
    "data": "MCP_Data_Analyst",
    "ml": "MCP_Machine_Learning",
    "office": "MCP_Office",
    "fs": "MCP_FileSystem",
    "search": "MCP_Search",
}

STEP_TOOLS: dict[str, list[str]] = {
    "COLLECT": ["inspect_dataset", "search_columns", "list_models"],
    "INSPECT": [
        "read_column_profile",
        "read_rows",
        "read_model_report",
        "detect_outliers",
        "check_data_quality",
    ],
    "CLEAN": [
        "run_preprocessing",
        "apply_patch",
        "smart_impute",
    ],
    "PREPARE": [
        "run_preprocessing",
        "feature_engineering",
    ],
    "TRAIN": [
        "train_classifier",
        "train_regressor",
        "run_preprocessing",
        "run_clustering",
        "train_with_cv",
        "compare_models",
        "tune_hyperparameters",
        "apply_dimensionality_reduction",
    ],
    "EVALUATE": [
        "get_predictions",
        "evaluate_model",
        "read_model_report",
        "plot_roc_curve",
        "plot_learning_curve",
        "plot_predictions_vs_actual",
        "generate_cluster_report",
    ],
    "REPORT": [
        "generate_eda_report",
        "generate_cluster_report",
        "plot_predictions_vs_actual",
    ],
}

# Legacy NEXT_STEP kept for any code that imported it directly
NEXT_STEP: dict[str, str] = {
    "LOCATE": "INSPECT",
    "INSPECT": "TRAIN",
    "TRAIN": "EVALUATE",
    "EVALUATE": "REPORT",
    "REPORT": "",
    "COLLECT": "INSPECT",
    "CLEAN": "PREPARE",
    "PREPARE": "TRAIN",
}


def _normalize_step(step: str) -> str:
    normalized = step.upper()
    if normalized in _LEGACY_STEP_MAP:
        return _LEGACY_STEP_MAP[normalized]
    if normalized in WORKFLOW_STEPS:
        return normalized
    return normalized


def _next_step(step: str) -> str:
    try:
        idx = WORKFLOW_STEPS.index(step)
        return WORKFLOW_STEPS[idx + 1] if idx + 1 < len(WORKFLOW_STEPS) else ""
    except ValueError:
        return NEXT_STEP.get(step, "")


def make_context(
    op: str,
    summary: str,
    artifacts: list[dict] | None = None,
) -> dict:
    """Return a context dict capturing what this tool just did.

    op        : tool/operation name (e.g. "train_classifier")
    summary   : plain-English description of what happened and the result
    artifacts : list of {"type": str, "path": str, "role": str, ...} dicts
    """
    return {
        "op": op,
        "summary": summary,
        "artifacts": artifacts or [],
        "timestamp": datetime.now(UTC).isoformat(),
    }


def make_handover(
    workflow_step: str,
    suggested_next: list[str] | list[dict],  # type: ignore[type-arg]
    carry_forward: dict | None = None,
) -> dict:
    """Return a handover dict for inclusion in every tool response.

    Accepts both old-style list[str] and new-style list[dict] suggested_next
    for backward compatibility with existing train_classifier/train_regressor code.

    workflow_step   : current step (canonical or legacy name)
    suggested_next  : list of tool names (legacy) OR
                      list of {"tool", "server", "domain", "reason"} dicts (new)
    carry_forward   : exact params the LLM should pass to the next tool call
    """
    step = _normalize_step(workflow_step)

    normalized: list[dict] = []
    for s in suggested_next:
        if isinstance(s, str):
            normalized.append({"tool": s, "server": "", "domain": "ml", "reason": ""})
        else:
            normalized.append(
                {
                    "tool": s.get("tool", ""),
                    "server": s.get("server", ""),
                    "domain": s.get("domain", "ml"),
                    "reason": s.get("reason", ""),
                }
            )

    return {
        "workflow_step": step,
        "workflow_next": _next_step(step),
        "suggested_next": normalized,
        "suggested_tools": [s["tool"] for s in normalized],  # legacy compat
        "carry_forward": carry_forward or {},
    }


__all__ = [
    "WORKFLOW_STEPS",
    "DOMAIN_SERVERS",
    "STEP_TOOLS",
    "NEXT_STEP",
    "make_context",
    "make_handover",
]
