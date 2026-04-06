"""ml_medium engine — Tier 2 ML logic. Zero MCP imports."""

from ._medium_cluster import read_receipt, run_clustering
from ._medium_data import (
    anomaly_detection,
    batch_predict,
    check_data_quality,
    evaluate_model,
    filter_rows,
    find_optimal_clusters,
    merge_datasets,
)
from ._medium_preprocess import detect_outliers, run_preprocessing
from ._medium_reports import generate_eda_report
from ._medium_train import compare_models, train_with_cv

__all__ = [
    "run_preprocessing",
    "detect_outliers",
    "train_with_cv",
    "compare_models",
    "run_clustering",
    "read_receipt",
    "generate_eda_report",
    "check_data_quality",
    "filter_rows",
    "merge_datasets",
    "find_optimal_clusters",
    "anomaly_detection",
    "evaluate_model",
    "batch_predict",
]
