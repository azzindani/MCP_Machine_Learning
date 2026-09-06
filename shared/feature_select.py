"""Choose what a model trains on.

Every trainer here used every column but the target, and offered no way to say
otherwise. That is fine until the caller finds a leak, which is the moment the
tools are *for*: the credit-risk review's instruction was

    agent must not ship without time-split + drop of id/member_id/total_payment

and "drop" was not expressible. Detection shipped; the action did not.

Found live in round 27. A clicks model scored r2 0.983 on a feature set holding
`link_clicks`, a strict subset of the target. The obvious next call --

    train_regressor(target_column="clicks", feature_columns=[...nine columns...])

-- returned `success: true` with metrics identical to four decimal places,
because `feature_columns` was not a parameter and the bundled FastMCP dropped
it in silence. The caller is told the leak is gone and it is not. Two defects
compounding: no way to act, and no way to know the action was ignored.

`enforce_known_arguments` closes the second. This closes the first.

Either name may be given, never both: `feature_columns` states the whole set,
`exclude_columns` removes from the default. A column named in neither call
still cannot be silently wrong -- an unknown name is refused with the list of
what the file actually holds.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def select_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> tuple[list[str], str, dict[str, Any] | None]:
    """(features, note, error). `error` is a ready return value when not None.

    The note is for `progress`: a caller who narrows a feature set should see
    the narrowing confirmed, because the whole reason this exists is that a
    silently ignored narrowing looks exactly like a successful one.
    """
    available = [c for c in df.columns if c != target_column]

    def _refuse(error: str, hint: str) -> dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "hint": hint,
            "progress": [],
            "token_estimate": (len(error) + len(hint)) // 4 + 10,
        }

    if feature_columns and exclude_columns:
        return (
            [],
            "",
            _refuse(
                "Pass feature_columns or exclude_columns, not both.",
                "feature_columns states the whole set; exclude_columns removes from the default. "
                "Two answers to the same question cannot both be honoured.",
            ),
        )

    if feature_columns:
        unknown = [c for c in feature_columns if c not in df.columns]
        if unknown:
            return ([], "", _refuse(f"feature_columns not in the file: {unknown}", f"Available: {available}"))
        if target_column in feature_columns:
            return (
                [],
                "",
                _refuse(
                    f"feature_columns contains the target '{target_column}'.",
                    "A model cannot use the answer as a feature. Drop it from feature_columns.",
                ),
            )
        return (list(feature_columns), f"{len(feature_columns)} of {len(available)} columns, as given", None)

    if exclude_columns:
        unknown = [c for c in exclude_columns if c not in df.columns]
        if unknown:
            return ([], "", _refuse(f"exclude_columns not in the file: {unknown}", f"Available: {available}"))
        kept = [c for c in available if c not in exclude_columns]
        if not kept:
            return (
                [],
                "",
                _refuse(
                    "exclude_columns removes every feature.",
                    f"Nothing would be left to train on. Available: {available}",
                ),
            )
        dropped = [c for c in exclude_columns if c != target_column]
        return (kept, f"{len(kept)} columns, excluding {dropped}", None)

    return (available, "", None)
