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


def get_n_iter() -> int:
    return 5 if is_constrained_mode() else 10


def get_sklearn_working_memory_mb() -> int:
    """Chunk budget for sklearn's pairwise routines, in MB.

    sklearn defaults this to 1024, which is the whole memory limit of the
    container these servers run in. silhouette_score on 10,000 samples peaked
    at 921 MB of a 1 GiB cap under that default and got the process OOM-killed
    mid-sweep, taking all three ML sub-servers down with it. The same call at
    64 MB peaks at 205 MB and returns the identical score.
    """
    return 32 if is_constrained_mode() else 64


def get_silhouette_sample_cap() -> int:
    """Rows to sample before scoring — silhouette is O(n^2) in the sample."""
    return 2_000 if is_constrained_mode() else 10_000
