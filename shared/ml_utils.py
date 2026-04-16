"""Shared ML utility functions — canonical implementations used by all tier helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

    # fill numeric nulls with median (vectorized — single pass)
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)

    return df, encoding_map, encoded_cols
