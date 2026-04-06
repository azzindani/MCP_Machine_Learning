"""Hardware mode helpers — read MCP_CONSTRAINED_MODE env var.

Never hardcode row/result limits. Always call these helpers.
"""

import os


def is_constrained_mode() -> bool:
    return os.environ.get("MCP_CONSTRAINED_MODE", "0") == "1"


def get_max_rows() -> int:
    return 20 if is_constrained_mode() else 100


def get_max_results() -> int:
    return 10 if is_constrained_mode() else 50


def get_max_depth() -> int:
    return 3 if is_constrained_mode() else 5


def get_max_columns() -> int:
    return 20 if is_constrained_mode() else 50


def get_cv_folds() -> int:
    return 3 if is_constrained_mode() else 5


def get_max_models() -> int:
    return 3 if is_constrained_mode() else 7
