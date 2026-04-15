"""Algorithm registry — Ring 2 pure constants, no I/O.

Canonical sets of supported algorithm keys. Add new algorithms here;
engine validation code calls allowed_classifiers() / allowed_regressors()
instead of referencing hardcoded sets scattered in helpers.

Extension: call register_classifier("myalgo") at server startup and
implement the corresponding branch in the engine without touching
any other module.
"""

from __future__ import annotations

# Core built-in algorithm keys (immutable)
CLASSIFIERS: frozenset[str] = frozenset({"lr", "svm", "rf", "dtc", "knn", "nb", "xgb"})
REGRESSORS: frozenset[str] = frozenset({"lir", "pr", "lar", "rr", "dtr", "rfr", "xgb"})
CLUSTERERS: frozenset[str] = frozenset({"kmeans", "meanshift", "dbscan"})
REDUCERS: frozenset[str] = frozenset({"pca", "ica"})

# Extension sets — populated at server startup via register_*()
_EXTRA_CLASSIFIERS: set[str] = set()
_EXTRA_REGRESSORS: set[str] = set()


def register_classifier(key: str) -> None:
    """Register an additional classifier key. Call before server starts."""
    _EXTRA_CLASSIFIERS.add(key)


def register_regressor(key: str) -> None:
    """Register an additional regressor key. Call before server starts."""
    _EXTRA_REGRESSORS.add(key)


def allowed_classifiers() -> frozenset[str]:
    """Return all valid classifier keys including extensions."""
    return CLASSIFIERS | frozenset(_EXTRA_CLASSIFIERS)


def allowed_regressors() -> frozenset[str]:
    """Return all valid regressor keys including extensions."""
    return REGRESSORS | frozenset(_EXTRA_REGRESSORS)
