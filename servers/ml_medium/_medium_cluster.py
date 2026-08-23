"""ml_medium clustering tools — run_clustering, read_receipt."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.handover import make_context, make_handover

from ._medium_helpers import (
    ALLOWED_CLUSTER_ALGOS,
    DBSCAN,
    PCA,
    FastICA,
    KMeans,
    MeanShift,
    StandardScaler,
    _error,
    _read_csv,
    append_receipt,
    bounded_silhouette,
    ok,
    read_receipt_log,
    resolve_path,
    snapshot,
    warn,
)


def run_clustering(
    file_path: str,
    feature_columns: list[str],
    algorithm: str,
    n_clusters: int = 3,
    eps: float = 3.0,
    min_samples: int = 5,
    reduce_dims: str = "",
    n_components: int = 2,
    save_labels: bool = False,
    output_path: str = "",
    dry_run: bool = False,
) -> dict:
    """Cluster dataset. algorithm: kmeans meanshift dbscan."""
    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    if algorithm not in ALLOWED_CLUSTER_ALGOS:
        return _error(
            f"Unknown algorithm: '{algorithm}'. Allowed: {', '.join(sorted(ALLOWED_CLUSTER_ALGOS))}",
            "Use 'kmeans', 'meanshift', or 'dbscan'.",
        )

    if reduce_dims and reduce_dims not in ("pca", "ica"):
        return _error(f"Unknown reduce_dims: '{reduce_dims}'.", "Use 'pca', 'ica', or '' (none).")

    if path.stat().st_size == 0:
        return _error(f"File is empty: {path.name}", "Verify the file has header + data rows.")
    try:
        df = _read_csv(str(path))
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return _error(
            f"Columns not found: {', '.join(missing[:5])}",
            "Use inspect_dataset() to list valid column names.",
        )

    numeric_df = df[feature_columns].select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        return _error("No numeric feature columns found.", "Select numeric columns for clustering.")

    # +/-inf (e.g. a ratio column divided by zero) and nulls are not valid
    # clustering input — median-fill both instead of letting sklearn crash
    # with a raw "Input X contains NaN". Median-fill (not row-drop) keeps
    # every row aligned with the original df for save_labels.
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    null_counts = numeric_df.isna().sum()
    filled_cols = [c for c in numeric_df.columns if null_counts[c] > 0]
    if filled_cols:
        numeric_df = numeric_df.fillna(numeric_df.median())
        progress.append(warn(f"Median-filled {len(filled_cols)} column(s) with nulls/inf", ", ".join(filled_cols)))

    x = numeric_df.values

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "run_clustering",
            "dry_run": True,
            "algorithm": algorithm,
            "feature_columns": feature_columns,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    # Scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    progress.append(ok("Scaled features", "StandardScaler"))

    # Optional dimensionality reduction
    if reduce_dims:
        nc = min(n_components, x_scaled.shape[1])
        if reduce_dims == "pca":
            reducer = PCA(n_components=nc)
        else:
            reducer = FastICA(n_components=nc)
        x_scaled = reducer.fit_transform(x_scaled)
        progress.append(ok(f"Reduced dims with {reduce_dims.upper()}", f"{nc} components"))

    # Cluster (use MiniBatchKMeans for large datasets — much faster)
    if algorithm == "kmeans":
        if len(x_scaled) > 50_000:
            from sklearn.cluster import MiniBatchKMeans

            clf = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, random_state=42, batch_size=1024, n_init=10)
        else:
            # Same ten restarts find_optimal_clusters uses, so the k it
            # recommended and the clustering produced here agree.
            clf = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42, n_init=10)
        labels = clf.fit_predict(x_scaled)
        inertia = float(clf.inertia_)
        n_found = n_clusters
        extra = {"inertia": round(inertia, 4)}
    elif algorithm == "meanshift":
        clf = MeanShift()
        labels = clf.fit_predict(x_scaled)
        n_found = len(np.unique(labels))
        extra = {"n_clusters_found": n_found}
    else:  # dbscan
        clf = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clf.fit_predict(x_scaled)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int((labels == -1).sum())
        extra = {"n_clusters_found": n_found, "noise_points": noise}

    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {str(int(u)): int(c) for u, c in zip(unique, counts)}
    progress.append(ok(f"Clustered with {algorithm}", f"{n_found} clusters"))

    # Silhouette score (needs at least 2 clusters and not all noise).
    # The sample cap was here already; the other half of the cost was not.
    # sklearn's `working_memory` defaults to 1024 MB -- the whole container --
    # and this call peaked at 962 MB of a 1 GiB cap, got OOM-killed, and took
    # all three ML sub-servers with it. bounded_silhouette caps both.
    silhouette = None
    if n_found >= 2 and len(set(labels)) >= 2:
        try:
            non_noise = labels != -1
            x_sil = x_scaled[non_noise]
            l_sil = labels[non_noise]
            if len(x_sil) > n_found:
                silhouette = bounded_silhouette(x_sil, l_sil)
        except Exception:
            pass

    # save_labels rewrites the caller's dataset in place, which was the only way
    # to keep the labels. output_path gives them somewhere else to land: the
    # source is then never touched, so there is nothing to snapshot. Naming a
    # destination is itself a request to write, so it works without save_labels.
    backup = ""
    out = resolve_path(output_path) if output_path else path
    if save_labels or output_path:
        if out == path:
            try:
                backup = snapshot(str(path))
            except Exception as exc:
                progress.append(warn("Snapshot failed", str(exc)))
        df["cluster_label"] = labels
        df.to_csv(out, index=False)
        progress.append(ok("Saved labels", f"cluster_label column added to {out.name}"))

    append_receipt(
        str(path), "run_clustering", {"algorithm": algorithm, "feature_columns": feature_columns}, "success", backup
    )

    resp = {
        "success": True,
        "op": "run_clustering",
        "algorithm": algorithm,
        "feature_columns": feature_columns,
        "label_counts": label_counts,
        "silhouette_score": silhouette,
        "output_path": str(out) if (save_labels or output_path) else "",
        "backup": backup,
        "progress": progress,
        "token_estimate": 0,
        **extra,
    }
    resp["context"] = make_context(
        "run_clustering",
        f"Clustered {path.name} with {algorithm}: {n_found} cluster(s), silhouette={silhouette}",
    )
    resp["handover"] = make_handover(
        "TRAIN",
        ["find_optimal_clusters", "generate_cluster_report", "generate_eda_report"],
        {"file_path": file_path, "feature_columns": feature_columns},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


def read_receipt(file_path: str) -> dict:
    """Read operation history for a file. Returns log entries."""
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")

    log = read_receipt_log(str(path))
    resp: dict = {
        "success": True,
        "op": "read_receipt",
        "file": Path(file_path).name,
        "entry_count": len(log),
        "entries": log,
        "token_estimate": 0,
    }
    resp["context"] = make_context(
        "read_receipt",
        f"Read {len(log)} operation log entries for {Path(file_path).name}",
    )
    resp["handover"] = make_handover(
        "INSPECT",
        ["inspect_dataset", "restore_version", "run_preprocessing"],
        {"file_path": file_path},
    )
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
