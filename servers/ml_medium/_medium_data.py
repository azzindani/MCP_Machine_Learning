"""ml_medium data tools — filter_rows, merge_datasets, find_optimal_clusters,
anomaly_detection, evaluate_model, batch_predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.handover import make_context, make_handover

from ._medium_helpers import (
    _error,
    _read_csv,
    _save_chart,
    append_receipt,
    get_output_dir,
    info,
    ok,
    resolve_path,
    snapshot,
    warn,
)

FILTER_OPS = {
    "eq",
    "ne",
    "gt",
    "lt",
    "gte",
    "lte",
    "contains",
    "not_contains",
    "is_null",
    "not_null",
    "starts_with",
    "ends_with",
}


def filter_rows(
    file_path: str,
    column: str,
    operator: str,
    value: str = "",
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Filter dataset rows by condition. operator: eq ne gt lt gte lte contains is_null not_null."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv, got {path.suffix!r}", "Provide a CSV file.")
    if operator not in FILTER_OPS:
        return _error(
            f"Unknown operator: '{operator}'. Allowed: {', '.join(sorted(FILTER_OPS))}",
            "Use one of: eq ne gt lt gte lte contains is_null not_null starts_with ends_with",
        )

    if path.stat().st_size == 0:
        return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")
    try:
        df = _read_csv(str(path))
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")

    if column not in df.columns:
        return _error(f"Column '{column}' not found.", "Use inspect_dataset() to list column names.")

    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows"))
    original_count = len(df)

    col = df[column]
    try:
        if operator == "eq":
            mask = col.astype(str) == str(value)
        elif operator == "ne":
            mask = col.astype(str) != str(value)
        elif operator == "gt":
            mask = pd.to_numeric(col, errors="coerce") > float(value)
        elif operator == "lt":
            mask = pd.to_numeric(col, errors="coerce") < float(value)
        elif operator == "gte":
            mask = pd.to_numeric(col, errors="coerce") >= float(value)
        elif operator == "lte":
            mask = pd.to_numeric(col, errors="coerce") <= float(value)
        elif operator == "contains":
            mask = col.astype(str).str.contains(str(value), na=False)
        elif operator == "not_contains":
            mask = ~col.astype(str).str.contains(str(value), na=False)
        elif operator == "is_null":
            mask = col.isnull()
        elif operator == "not_null":
            mask = col.notnull()
        elif operator == "starts_with":
            mask = col.astype(str).str.startswith(str(value), na=False)
        elif operator == "ends_with":
            mask = col.astype(str).str.endswith(str(value), na=False)
        else:
            mask = pd.Series([True] * len(df))
    except Exception as exc:
        return _error(f"Filter failed: {exc}", "Check value type matches column dtype.")

    df_filtered = df[mask].copy()
    kept = len(df_filtered)
    removed = original_count - kept
    progress.append(ok("Filter applied", f"{kept:,} rows kept, {removed:,} removed"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "filter_rows",
            "dry_run": True,
            "rows_kept": kept,
            "rows_removed": removed,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    out_path = Path(output_path) if output_path else get_output_dir() / f"{path.stem}_filtered.csv"
    try:
        out_resolved = resolve_path(str(out_path))
    except ValueError:
        out_resolved = out_path

    backup = ""
    if out_resolved.exists():
        try:
            backup = snapshot(str(out_resolved))
        except Exception:
            pass

    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(out_resolved, index=False)
    progress.append(ok("Saved filtered dataset", out_resolved.name))
    append_receipt(
        str(path), "filter_rows", {"column": column, "operator": operator, "value": value}, "success", backup
    )  # noqa: E501

    resp = {
        "success": True,
        "op": "filter_rows",
        "output_path": str(out_resolved),
        "rows_original": original_count,
        "rows_kept": kept,
        "rows_removed": removed,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "filter_rows",
        f"Filtered {path.name}: kept {kept:,} of {original_count:,} rows ({column} {operator} {value!r})",
        [{"type": "csv", "path": str(out_resolved), "role": "filtered_dataset"}],
    )
    resp["handover"] = make_handover(
        "CLEAN",
        ["inspect_dataset", "train_classifier", "train_regressor"],
        {"file_path": str(out_resolved)},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: merge_datasets
# ---------------------------------------------------------------------------


def merge_datasets(
    file_path_1: str,
    file_path_2: str,
    on: str,
    how: str = "left",
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Merge two CSVs on a key column. how: left right inner outer."""
    progress: list[dict] = []
    if how not in ("left", "right", "inner", "outer"):
        return _error(f"Unknown how: '{how}'.", "Use: left right inner outer")

    try:
        p1 = resolve_path(file_path_1)
        p2 = resolve_path(file_path_2)
    except ValueError as exc:
        return _error(str(exc), "Check both paths are inside your home directory.")

    for p in (p1, p2):
        if not p.exists():
            return _error(f"File not found: {p}", "Check the file path.")

    for p_check in (p1, p2):
        if p_check.stat().st_size == 0:
            return _error(f"File is empty: {p_check.name}", "Verify the file has header + data rows.")
    try:
        df1 = _read_csv(str(p1))
        df2 = _read_csv(str(p2))
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check files are valid CSVs.")

    progress.append(ok(f"Loaded {p1.name}", f"{len(df1):,} rows × {len(df1.columns)} cols"))
    progress.append(ok(f"Loaded {p2.name}", f"{len(df2):,} rows × {len(df2.columns)} cols"))

    # Validate key column exists in both
    keys = [k.strip() for k in on.split(",")]
    for k in keys:
        if k not in df1.columns:
            return _error(f"Key '{k}' not in {p1.name}.", "Use inspect_dataset() to list column names.")
        if k not in df2.columns:
            return _error(f"Key '{k}' not in {p2.name}.", "Use inspect_dataset() to list column names.")

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "merge_datasets",
            "dry_run": True,
            "left_rows": len(df1),
            "right_rows": len(df2),
            "join_keys": keys,
            "how": how,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    on_param = keys[0] if len(keys) == 1 else keys
    df_merged = pd.merge(df1, df2, on=on_param, how=how, suffixes=("", "_right"))
    progress.append(ok("Merged", f"{len(df_merged):,} rows × {len(df_merged.columns)} cols"))

    out_path = Path(output_path) if output_path else get_output_dir() / f"{p1.stem}_merged.csv"
    try:
        out_resolved = resolve_path(str(out_path))
    except ValueError:
        out_resolved = out_path

    backup = ""
    if out_resolved.exists():
        try:
            backup = snapshot(str(out_resolved))
        except Exception:
            pass

    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_resolved, index=False)
    progress.append(ok("Saved merged dataset", out_resolved.name))
    append_receipt(str(p1), "merge_datasets", {"file_2": p2.name, "on": on, "how": how}, "success", backup)

    resp = {
        "success": True,
        "op": "merge_datasets",
        "output_path": str(out_resolved),
        "left_rows": len(df1),
        "right_rows": len(df2),
        "merged_rows": len(df_merged),
        "merged_columns": len(df_merged.columns),
        "join_keys": keys,
        "how": how,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "merge_datasets",
        f"Merged {p1.name} + {p2.name} on {on} ({how}): {len(df_merged):,} rows → {out_resolved.name}",
        [{"type": "csv", "path": str(out_resolved), "role": "merged_dataset"}],
    )
    resp["handover"] = make_handover(
        "PREPARE",
        ["inspect_dataset", "train_classifier", "train_regressor", "run_preprocessing"],
        {"file_path": str(out_resolved)},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: find_optimal_clusters
# ---------------------------------------------------------------------------


def find_optimal_clusters(
    file_path: str,
    feature_columns: list[str],
    max_k: int = 10,
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
) -> dict:
    """Find optimal K for K-Means via elbow + silhouette. Saves HTML chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.cluster import MiniBatchKMeans as _MBKMeans
    from sklearn.metrics import silhouette_score

    from shared.html_theme import calc_chart_height, get_theme, plotly_template

    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")
    if path.stat().st_size == 0:
        return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

    try:
        df = _read_csv(str(path))
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return _error(f"Columns not found: {', '.join(missing)}", "Use inspect_dataset() for column names.")

    x = df[feature_columns].select_dtypes(include="number").dropna().values
    if x.shape[0] < 4:
        return _error("Need at least 4 rows for cluster analysis.", "Provide a larger dataset.")

    from sklearn.preprocessing import StandardScaler as _SS

    x_scaled = _SS().fit_transform(x)

    max_k = min(max_k, x_scaled.shape[0] - 1, 15)
    k_range = list(range(2, max_k + 1))

    # Use MiniBatchKMeans for large datasets (much faster)
    use_mini = len(x_scaled) > 50_000
    # Subsample for silhouette (O(n²) — impractical on large data)
    sil_cap = min(len(x_scaled), 10_000)
    if len(x_scaled) > sil_cap:
        rng = np.random.RandomState(42)
        sil_idx = rng.choice(len(x_scaled), sil_cap, replace=False)
    else:
        sil_idx = np.arange(len(x_scaled))

    inertias, silhouettes = [], []
    for k in k_range:
        if use_mini:
            km = _MBKMeans(n_clusters=k, random_state=42, max_iter=100, batch_size=1024)
        else:
            km = _KMeans(n_clusters=k, random_state=42, max_iter=100)
        labels = km.fit_predict(x_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(x_scaled[sil_idx], labels[sil_idx])))
        progress.append(info(f"k={k}", f"inertia={km.inertia_:.1f} sil={silhouettes[-1]:.3f}"))

    best_k = k_range[int(np.argmax(silhouettes))]
    t = get_theme(theme)
    tmpl = plotly_template(theme)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Elbow Curve (Inertia)", "Silhouette Score"])
    fig.add_trace(
        go.Scatter(x=k_range, y=inertias, mode="lines+markers", name="Inertia", marker_color=t["accent"]), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=k_range, y=silhouettes, mode="lines+markers", name="Silhouette", marker_color=t["success"]),
        row=1,
        col=2,
    )
    fig.add_vline(x=best_k, line_dash="dash", line_color=t["danger"], annotation_text=f"Best K={best_k}", row=1, col=2)
    fig.update_layout(
        title=f"Optimal Clusters for {path.name}",
        template=tmpl,
        height=calc_chart_height(420, mode="fixed"),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    out_abs, out_name = _save_chart(fig, output_path, "optimal_k", path, open_after, theme)
    progress.append(ok("Saved elbow chart", out_name))

    resp: dict = {
        "success": True,
        "op": "find_optimal_clusters",
        "best_k": best_k,
        "k_range": k_range,
        "inertias": [round(v, 2) for v in inertias],
        "silhouette_scores": [round(v, 4) for v in silhouettes],
        "output_path": out_abs,
        "output_name": out_name,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "find_optimal_clusters",
        f"Elbow analysis on {path.name}: best K={best_k} by silhouette score",
        [{"type": "html", "path": out_abs, "role": "elbow_chart"}],
    )
    resp["handover"] = make_handover(
        "INSPECT",
        ["run_clustering", "generate_cluster_report"],
        {"file_path": file_path, "feature_columns": feature_columns},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: anomaly_detection
# ---------------------------------------------------------------------------


def anomaly_detection(
    file_path: str,
    feature_columns: list[str],
    method: str = "isolation_forest",
    contamination: float = 0.05,
    save_labels: bool = False,
    dry_run: bool = False,
) -> dict:
    """Detect anomalies. method: isolation_forest lof. Adds anomaly_score column if save_labels=True."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    progress: list[dict] = []
    if method not in ("isolation_forest", "lof"):
        return _error(f"Unknown method: '{method}'.", "Use 'isolation_forest' or 'lof'.")
    if not 0 < contamination < 0.5:
        return _error(f"contamination={contamination} must be in (0, 0.5).", "Use e.g. contamination=0.05")

    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check the file path.")
    if path.stat().st_size == 0:
        return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")

    try:
        df = _read_csv(str(path))
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is valid.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return _error(f"Columns not found: {', '.join(missing)}", "Use inspect_dataset() for column names.")

    x = df[feature_columns].select_dtypes(include="number").fillna(0).values
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows, {len(feature_columns)} features"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "anomaly_detection",
            "dry_run": True,
            "method": method,
            "contamination": contamination,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    from sklearn.preprocessing import StandardScaler as _SS

    x_scaled = _SS().fit_transform(x)

    if method == "isolation_forest":
        det = IsolationForest(contamination=contamination, random_state=42)
        labels = det.fit_predict(x_scaled)  # -1 = anomaly, 1 = normal
        scores = det.score_samples(x_scaled)  # lower = more anomalous
    else:
        det = LocalOutlierFactor(contamination=contamination)
        labels = det.fit_predict(x_scaled)
        scores = det.negative_outlier_factor_

    anomaly_mask = labels == -1
    n_anomalies = int(anomaly_mask.sum())
    anomaly_pct = round(n_anomalies / len(df) * 100, 2)
    progress.append(ok("Detected anomalies", f"{n_anomalies} ({anomaly_pct}%) anomalous rows"))

    # Top anomaly indices
    top_anomaly_idx = np.argsort(scores)[: min(10, n_anomalies)].tolist()

    backup = ""
    if save_labels:
        try:
            backup = snapshot(str(path))
        except Exception as exc:
            progress.append(warn("Snapshot failed", str(exc)))
        df["anomaly_score"] = scores
        df["is_anomaly"] = anomaly_mask.astype(int)
        df.to_csv(path, index=False)
        progress.append(ok("Saved anomaly labels", path.name))

    append_receipt(
        str(path), "anomaly_detection", {"method": method, "contamination": contamination}, "success", backup
    )

    resp = {
        "success": True,
        "op": "anomaly_detection",
        "method": method,
        "contamination": contamination,
        "n_anomalies": n_anomalies,
        "anomaly_pct": anomaly_pct,
        "top_anomaly_indices": top_anomaly_idx,
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "anomaly_detection",
        f"Detected {n_anomalies} anomaly rows ({anomaly_pct}%) in {path.name} using {method}",
    )
    resp["handover"] = make_handover(
        "INSPECT",
        ["run_preprocessing", "filter_rows", "inspect_dataset"],
        {"file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


# ---------------------------------------------------------------------------
# Tool: check_data_quality (JSON summary — model-readable)
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    test_file_path: str,
    target_column: str,
) -> dict:
    """Score a saved model on a labeled test CSV. Returns metrics dict."""
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    from shared.file_utils import resolve_path
    from shared.progress import ok

    progress = []
    try:
        mp = resolve_path(model_path)
        dp = resolve_path(test_file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file paths.", "token_estimate": 30}

    if not mp.exists():
        return {
            "success": False,
            "error": f"Model not found: {model_path}",
            "hint": "Train a model first.",
            "token_estimate": 30,
        }
    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {test_file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    try:
        import xgboost as xgb

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model_obj = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "classification")
        feature_columns = metadata.get("feature_columns", [])
        encoding_map = metadata.get("encoding_map", {})
        scaler = metadata.get("scaler")
        poly = metadata.get("poly")

        if dp.stat().st_size == 0:
            return {
                "success": False,
                "error": f"File is empty: {dp.name}",
                "hint": "Verify the file has header + data rows.",
                "token_estimate": 30,
            }
        df = _read_csv(str(dp))
        progress.append(ok("Loaded test data", f"{len(df)} rows"))

        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not in test file.",
                "hint": "Use inspect_dataset() to check column names.",
                "token_estimate": 30,
            }

        # Encode categoricals using stored map
        for col, mapping in encoding_map.items():
            if col in df.columns and col != target_column:
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        available = [c for c in feature_columns if c in df.columns]
        if not available:
            return {
                "success": False,
                "error": "No feature columns found in test file.",
                "hint": "Use the same dataset schema as training.",
                "token_estimate": 30,
            }

        X = df[available].fillna(0).values.astype(float)
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)

        y_true = df[target_column].values

        # Encode target if categorical
        from sklearn.preprocessing import LabelEncoder

        le = None
        if not pd.api.types.is_numeric_dtype(y_true):
            le = LabelEncoder()
            y_true = le.fit_transform(y_true)

        model_key = metadata.get("model_key", "")
        if model_key == "xgb" or isinstance(model_obj, xgb.Booster):
            dmat = xgb.DMatrix(X)
            raw = model_obj.predict(dmat)
            n_classes = metadata.get("n_classes", 2)
            if task == "classification":
                if n_classes > 2:
                    y_pred = np.argmax(raw, axis=1)
                else:
                    y_pred = (raw > 0.5).astype(int)
            else:
                y_pred = raw
        else:
            y_pred = model_obj.predict(X)

        metrics: dict = {}
        if task == "classification":
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}
            if len(set(y_true)) == 2:
                try:
                    if hasattr(model_obj, "predict_proba"):
                        y_prob = model_obj.predict_proba(X)[:, 1]
                    elif model_key == "xgb":
                        y_prob = raw if raw.ndim == 1 else raw[:, 1]
                    else:
                        y_prob = None
                    if y_prob is not None:
                        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
                except Exception:
                    pass
            from servers.ml_basic.engine import _confusion_dict

            metrics["confusion_matrix"] = _confusion_dict(y_true, y_pred)
        else:
            mse = float(mean_squared_error(y_true, y_pred))
            metrics = {
                "mse": round(mse, 4),
                "rmse": round(float(np.sqrt(mse)), 4),
                "r2": round(float(r2_score(y_true, y_pred)), 4),
            }

        progress.append(
            ok("Evaluated model", ", ".join(f"{k}={v}" for k, v in metrics.items() if not isinstance(v, dict))[:80])
        )

        resp = {
            "success": True,
            "op": "evaluate_model",
            "model_path": str(mp),
            "test_file": dp.name,
            "task": task,
            "target_column": target_column,
            "test_rows": len(df),
            "metrics": metrics,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["context"] = make_context(
            "evaluate_model",
            f"Evaluated {mp.name} on {dp.name}: "
            + ", ".join(f"{k}={v}" for k, v in metrics.items() if not isinstance(v, dict))[:80],
            [{"type": "model", "path": str(mp), "role": "evaluated_model"}],
        )
        resp["handover"] = make_handover(
            "EVALUATE",
            ["read_model_report", "plot_roc_curve", "generate_training_report"],
            {"model_path": model_path, "file_path": test_file_path},
        )
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check model and test file compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# Tool: batch_predict (all rows → CSV, no row limit)
# ---------------------------------------------------------------------------


def batch_predict(
    model_path: str,
    file_path: str,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Run predictions on all rows and save to CSV. No row limit."""
    import pickle

    import numpy as np
    import pandas as pd

    from shared.file_utils import resolve_path
    from shared.progress import info, ok
    from shared.version_control import snapshot

    progress = []
    try:
        mp = resolve_path(model_path)
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file paths.", "token_estimate": 30}

    if not mp.exists():
        return {
            "success": False,
            "error": f"Model not found: {model_path}",
            "hint": "Train a model first.",
            "token_estimate": 30,
        }
    if not dp.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check file path.",
            "token_estimate": 30,
        }

    if dry_run:
        return {
            "success": True,
            "op": "batch_predict",
            "dry_run": True,
            "model_path": str(mp),
            "file_path": str(dp),
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import xgboost as xgb

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model_obj = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "classification")
        feature_columns = metadata.get("feature_columns", [])
        encoding_map = metadata.get("encoding_map", {})
        scaler = metadata.get("scaler")
        poly = metadata.get("poly")
        model_key = metadata.get("model_key", "")
        n_classes = metadata.get("n_classes", 2)

        if dp.stat().st_size == 0:
            return {
                "success": False,
                "error": f"File is empty: {dp.name}",
                "hint": "Verify the file has header + data rows.",
                "token_estimate": 30,
            }
        df = _read_csv(str(dp))
        progress.append(ok("Loaded data", f"{len(df):,} rows"))

        for col, mapping in encoding_map.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        available = [c for c in feature_columns if c in df.columns]
        X = df[available].fillna(0).values.astype(float)
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)

        if model_key == "xgb" or isinstance(model_obj, xgb.Booster):
            dmat = xgb.DMatrix(X)
            raw = model_obj.predict(dmat)
            if task == "classification":
                if n_classes > 2:
                    preds = np.argmax(raw, axis=1)
                else:
                    preds = (raw > 0.5).astype(int)
            else:
                preds = raw
        else:
            preds = model_obj.predict(X)

        df["prediction"] = preds
        progress.append(ok("Generated predictions", f"{len(preds):,} rows"))

        out_path_str = output_path or str(get_output_dir() / f"{dp.stem}_predictions.csv")
        from pathlib import Path

        out = Path(out_path_str).resolve()

        backup = ""
        if out.exists():
            try:
                backup = snapshot(str(out))
            except Exception:
                pass

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        progress.append(ok("Saved predictions", out.name))

        # Distribution summary
        if task == "classification":
            dist = {str(k): int(v) for k, v in pd.Series(preds).value_counts().sort_index().items()}
        else:
            dist = {
                "min": round(float(preds.min()), 4),
                "max": round(float(preds.max()), 4),
                "mean": round(float(preds.mean()), 4),
            }

        resp = {
            "success": True,
            "op": "batch_predict",
            "output_path": str(out),
            "row_count": len(preds),
            "task": task,
            "prediction_distribution": dist,
            "backup": backup,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["context"] = make_context(
            "batch_predict",
            f"Generated {len(preds):,} predictions from {mp.name} on {dp.name} → {out.name}",
            [{"type": "csv", "path": str(out), "role": "predictions_csv"}],
        )
        resp["handover"] = make_handover(
            "EVALUATE",
            ["evaluate_model", "read_model_report", "inspect_dataset"],
            {"model_path": model_path, "file_path": str(out)},
        )
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check model and data compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# Tool: check_data_quality (JSON summary — model-readable)
# ---------------------------------------------------------------------------


def check_data_quality(file_path: str) -> dict:
    """Return JSON data quality summary with score 0-100. No HTML."""
    from shared.file_utils import resolve_path
    from shared.progress import ok

    progress = []
    try:
        path = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file path.", "token_estimate": 30}
    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "hint": "Check that file_path is absolute.",
            "token_estimate": 30,
        }
    if path.stat().st_size == 0:
        return {
            "success": False,
            "error": f"File is empty: {path.name}",
            "hint": "Verify the file has header + data rows.",
            "token_estimate": 30,
        }

    try:
        df = _read_csv(str(path))
    except Exception as exc:
        return {"success": False, "error": str(exc), "hint": "Check the file is a valid CSV.", "token_estimate": 30}
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    n_rows, n_cols = len(df), len(df.columns)
    alerts = []
    score = 100.0

    # Compute stats in bulk (vectorized — single pass each)
    nunique_all = df.nunique(dropna=True)
    null_counts = df.isnull().sum()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # 1. Constant columns
    for col in nunique_all.index:
        if nunique_all[col] <= 1:
            alerts.append(
                {
                    "type": "constant_column",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' has only 1 unique value.",
                    "recommendation": f"Drop column '{col}' — it contains no information.",
                }
            )
            score -= 15

    # 2. High missing
    null_summary = []
    for col in null_counts.index:
        nc = int(null_counts[col])
        null_pct = nc / n_rows * 100 if n_rows > 0 else 0
        if null_pct > 0:
            null_summary.append({"column": col, "null_count": nc, "null_pct": round(null_pct, 2)})
        if null_pct > 20:
            alerts.append(
                {
                    "type": "high_missing",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' is {null_pct:.1f}% null.",
                    "recommendation": f"Use run_preprocessing fill_nulls or drop column '{col}'.",
                }
            )
            score -= 15

    # 3. Duplicate rows
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / n_rows * 100 if n_rows > 0 else 0
    if dup_count > 0:
        alerts.append(
            {
                "type": "duplicate_rows",
                "severity": "medium",
                "message": f"{dup_count} duplicate rows ({dup_pct:.1f}%).",
                "recommendation": "Use run_preprocessing op 'drop_duplicates' to remove them.",
            }
        )
        score -= min(10, dup_pct * 0.3)

    # 4. Zero-inflated numeric (vectorized)
    if num_cols and n_rows > 0:
        zero_pcts = (df[num_cols] == 0).sum() / n_rows * 100
        for col in num_cols:
            zp = float(zero_pcts[col])
            if zp > 50:
                alerts.append(
                    {
                        "type": "zero_inflated",
                        "severity": "medium",
                        "column": col,
                        "message": f"Column '{col}' is {zp:.1f}% zeros.",
                        "recommendation": f"Consider log_transform or separate zero/nonzero modeling for '{col}'.",
                    }
                )
                score -= 8

    # 5. High cardinality (use pre-computed nunique)
    for col in cat_cols:
        n_unique = int(nunique_all[col])
        ratio = n_unique / n_rows if n_rows > 0 else 0
        if ratio > 0.5 and n_unique > 20:
            alerts.append(
                {
                    "type": "high_cardinality",
                    "severity": "medium",
                    "column": col,
                    "message": f"Column '{col}' has {n_unique} unique values ({ratio * 100:.1f}% of rows).",
                    "recommendation": f"Consider drop_column or target-encoding for '{col}'.",
                }
            )
            score -= 8

    # 6. Extreme skewness (vectorized)
    if num_cols:
        try:
            skews = df[num_cols].skew()
            for col in num_cols:
                skew = float(skews[col])
                if abs(skew) > 2:
                    alerts.append(
                        {
                            "type": "extreme_skewness",
                            "severity": "medium",
                            "column": col,
                            "message": f"Column '{col}' skewness = {skew:.2f}.",
                            "recommendation": f"Apply log_transform to column '{col}' before training.",
                        }
                    )
                    score -= 8
        except Exception:
            pass

    # 7. Multicollinearity
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols].corr(method="pearson").abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        c1, c2 = num_cols[i], num_cols[j]
                        alerts.append(
                            {
                                "type": "multicollinearity",
                                "severity": "high",
                                "column_pair": [c1, c2],
                                "message": f"Columns '{c1}' and '{c2}' have |r|={corr_matrix.iloc[i, j]:.2f}.",
                                "recommendation": f"Drop one of '{c1}' or '{c2}' to reduce multicollinearity.",
                            }
                        )
                        score -= 15
        except Exception:
            pass

    # Cap score
    score = max(0.0, min(100.0, score))

    # Summarize null cols with high missing
    high_missing_cols = [c["column"] for c in alerts if c.get("type") == "high_missing"]
    constant_cols = [c["column"] for c in alerts if c.get("type") == "constant_column"]

    resp = {
        "success": True,
        "op": "check_data_quality",
        "file": path.name,
        "row_count": n_rows,
        "column_count": n_cols,
        "quality_score": round(score, 1),
        "alerts_count": len(alerts),
        "alerts_high": sum(1 for a in alerts if a.get("severity") == "high"),
        "alerts_medium": sum(1 for a in alerts if a.get("severity") == "medium"),
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_pct, 2),
        "null_summary": null_summary[:20],
        "high_missing_columns": high_missing_cols,
        "constant_columns": constant_cols,
        "alerts": alerts[:30],
        "progress": progress,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "check_data_quality",
        f"Quality score {round(score, 1)}/100 for {path.name}: {len(alerts)} alert(s) ({sum(1 for a in alerts if a.get('severity') == 'high')} high)",
    )
    resp["handover"] = make_handover(
        "INSPECT",
        ["run_preprocessing", "generate_eda_report", "detect_outliers"],
        {"file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
