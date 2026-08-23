"""Shared ML utility functions — canonical implementations used by all tier helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# A model that scores perfectly has almost always been handed the answer. A real
# run trained on campaign_type while campaign_platform stayed in the feature set;
# the two are the same fact under different names, so accuracy, f1 and AUC all
# came back at 1.000 with a clean confusion matrix and no caveat anywhere in the
# result or the generated training report. Anyone reading that — a person or a
# dispatching model — takes it as a great model.
_NEAR_PERFECT_SCORE = 0.999
_MAX_FEATURES_SCANNED = 60


def find_determinant_features(df: pd.DataFrame, target_column: str, feature_cols: list[str]) -> list[str]:
    """Return features whose value alone fixes the target — i.e. leaks it.

    Only low-cardinality columns are considered: a column holding one distinct
    value per row (an id, a timestamp) partitions the target perfectly by
    construction, which says nothing about leakage.
    """
    if target_column not in df.columns or df.empty:
        return []
    row_limit = max(2, len(df) // 2)
    culprits: list[str] = []
    for column in feature_cols[:_MAX_FEATURES_SCANNED]:
        if column not in df.columns:
            continue
        distinct = df[column].nunique(dropna=False)
        if distinct < 2 or distinct > row_limit:
            continue
        try:
            worst = df.groupby(column, observed=True)[target_column].nunique(dropna=False).max()
        except (TypeError, ValueError):
            continue
        if worst is not None and worst <= 1:
            culprits.append(column)
    return culprits


def leakage_warning(df: pd.DataFrame, target_column: str, feature_cols: list[str], score: float) -> str:
    """Explain a near-perfect score, or '' when the score is ordinary.

    `score` is accuracy for classifiers and R² for regressors — both are 1.0 at
    perfection, so the same threshold reads correctly for either.
    """
    if score < _NEAR_PERFECT_SCORE:
        return ""
    culprits = find_determinant_features(df, target_column, feature_cols)
    if culprits:
        named = ", ".join(f"'{c}'" for c in culprits[:5])
        more = f" (and {len(culprits) - 5} more)" if len(culprits) > 5 else ""
        return (
            f"Score of {score:.4f} is explained by leakage: {named}{more} "
            f"determine '{target_column}' exactly, so the model is reading the answer "
            "rather than predicting it. Drop those columns and retrain to get a real score."
        )
    return (
        f"Score of {score:.4f} is near-perfect, which usually means the target leaked into "
        "the features. No single column determines the target, so check for combinations, "
        "duplicate rows shared across the train/test split, or a feature derived from the target."
    )


def typical_row(df_raw: pd.DataFrame, feature_cols: list[str]) -> dict:
    """A representative value per feature, in the dataset's own vocabulary.

    Recorded at training time so an embedded prediction panel opens on a
    plausible example instead of a row of zeros. Categorical columns keep their
    original labels — the panel offers those, not the encoded integers.
    """
    defaults: dict = {}
    for column in feature_cols:
        if column not in df_raw.columns:
            continue
        series = df_raw[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = round(float(series.median()), 6)
        else:
            modes = series.mode()
            if len(modes):
                defaults[column] = str(modes.iloc[0])
    return defaults


def _auto_preprocess(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict, list[str]]:
    """Drop null targets, label-encode categoricals, fill numeric nulls.

    Returns: (processed_df, encoding_map, encoded_columns)
    """
    df = df.dropna(subset=[target_column]).copy()
    encoding_map: dict = {}
    encoded_cols: list[str] = []

    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("nan").astype(str))
            encoding_map[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            encoded_cols.append(col)

    # Encode target column if it is categorical (handles string labels like "yes"/"no")
    if (
        pd.api.types.is_string_dtype(df[target_column])
        or df[target_column].dtype == object
        or str(df[target_column].dtype) == "category"
    ):
        le_tgt = LabelEncoder()
        df[target_column] = le_tgt.fit_transform(df[target_column].astype(str))
        encoding_map[f"__target__{target_column}"] = {str(cls): int(idx) for idx, cls in enumerate(le_tgt.classes_)}

    # +/-inf (e.g. a ratio column divided by zero) is not a valid model input either —
    # treat it the same as a missing value so it gets median-filled below, instead of
    # reaching sklearn raw and crashing with "Input X contains infinity".
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # fill numeric nulls with median (vectorized — single pass)
    if len(num_cols) > 0:
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)

    return df, encoding_map, encoded_cols


def bounded_silhouette(x, labels, cap: int | None = None, random_state: int = 42) -> float | None:
    """silhouette_score with both of its costs bounded, or None if not scoreable.

    Two separate budgets, and only one of them was set. The sample cap keeps the
    O(n^2) comparison count down; sklearn's `working_memory` caps the chunk it
    allocates while doing it, and that one defaults to 1024 MB -- the entire
    memory limit of the container. A single run_clustering call on the 16,834-row
    fixture peaked at 962 MB of a 1 GiB cap and the kernel killed the process,
    which took ml_basic, ml_medium and ml_advanced down together. Same score at
    64 MB, a quarter of the memory.
    """
    import numpy as np
    import sklearn
    from sklearn.metrics import silhouette_score

    from shared.platform_utils import get_silhouette_sample_cap, get_sklearn_working_memory_mb

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None

    sample_cap = cap or get_silhouette_sample_cap()
    if len(x) > sample_cap:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(x), sample_cap, replace=False)
        x, labels = x[idx], labels[idx]
        if len(set(labels.tolist())) < 2:
            return None

    with sklearn.config_context(working_memory=get_sklearn_working_memory_mb()):
        return round(float(silhouette_score(x, labels)), 4)
