"""ml_advanced visualization functions — plot_roc_curve, plot_learning_curve,
plot_predictions_vs_actual, generate_cluster_report."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from shared.file_utils import atomic_write_text, read_csv as _read_csv
from shared.handover import make_context, make_handover
from shared.html_theme import apply_fig_theme, calc_chart_height, get_theme, plotly_template
from shared.progress import info, ok

from ._adv_helpers import _save_chart, get_output_path, resolve_path

# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    model_path: str,
    file_path: str,
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Plot ROC curve for a classifier model. Saves interactive HTML."""
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
            "hint": "Train a classifier first.",
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
            "op": "plot_roc_curve",
            "dry_run": True,
            "model_path": str(mp),
            "file_path": str(dp),
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.graph_objects as go
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "classification")
        if task != "classification":
            return {
                "success": False,
                "error": "ROC curve is only for classifiers.",
                "hint": "Use plot_predictions_vs_actual for regression.",
                "token_estimate": 30,
            }

        feature_columns = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", "")
        encoding_map = metadata.get("encoding_map", {})

        if dp.stat().st_size == 0:
            return {"success": False, "error": f"File is empty: {dp.name}", "hint": "Verify the file has header + data rows.", "token_estimate": 30}
        df = _read_csv(str(dp))
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        # Encode
        for col, mapping in encoding_map.items():
            if col in df.columns and col != target_column:
                df[col] = df[col].map(mapping).fillna(df[col])

        available = [c for c in feature_columns if c in df.columns]
        if not available:
            return {
                "success": False,
                "error": "No feature columns found in dataset.",
                "hint": "Use the same dataset used for training.",
                "token_estimate": 30,
            }

        X = df[available].select_dtypes(include="number").fillna(0)
        y_true = df[target_column] if target_column in df.columns else None

        if y_true is None:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not found.",
                "hint": "Provide the same dataset used for training.",
                "token_estimate": 30,
            }

        # Encode target if needed
        if not pd.api.types.is_numeric_dtype(y_true):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_true = pd.Series(le.fit_transform(y_true))

        classes = sorted(y_true.unique())
        n_classes = len(classes)

        # Get probabilities
        has_proba = hasattr(model, "predict_proba")
        if not has_proba:
            # Try XGBoost Booster
            try:
                import xgboost as xgb

                if isinstance(model, xgb.Booster):
                    dmat = xgb.DMatrix(X)
                    raw_preds = model.predict(dmat)
                    if raw_preds.ndim == 1:
                        y_prob = np.column_stack([1 - raw_preds, raw_preds])
                    else:
                        y_prob = raw_preds
                    has_proba = True
            except Exception:
                pass

        if not has_proba:
            return {
                "success": False,
                "error": "Model does not support probability estimates for ROC curve.",
                "hint": "Use a classifier that supports predict_proba (lr, rf, xgb, nb, dtc).",
                "token_estimate": 40,
            }

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)

        t = get_theme(theme)
        tmpl = plotly_template(theme)
        fig = go.Figure()

        auc_scores = {}
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            auc_scores["binary"] = round(roc_auc, 4)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC (AUC = {roc_auc:.3f})",
                    line=dict(color=t["accent"], width=2),
                )
            )
        else:
            y_bin = label_binarize(y_true, classes=classes)
            colors = [t["accent"], t["success"], t["warning"], t["danger"]]
            for i, cls in enumerate(classes[:10]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores[str(cls)] = round(roc_auc, 4)
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"Class {cls} (AUC={roc_auc:.3f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )

        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random (AUC=0.5)",
                line=dict(color=t["grid_color"], dash="dash"),
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1),
            template=tmpl,
            height=calc_chart_height(450, mode="fixed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )

        out_abs, out_name = _save_chart(fig, output_path, "roc_curve", dp, open_after, theme)
        progress.append(ok("Saved ROC curve", out_name))

        resp = {
            "success": True,
            "op": "plot_roc_curve",
            "output_path": out_abs,
            "output_name": out_name,
            "auc_scores": auc_scores,
            "n_classes": n_classes,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["context"] = make_context(
            "plot_roc_curve",
            f"Plotted ROC curve for {mp.name} — {n_classes} class(es), AUC: {list(auc_scores.values())[0] if auc_scores else 'n/a'}",
            [{"type": "report", "path": out_abs, "role": "roc_chart"}],
        )
        resp["handover"] = make_handover(
            step="REPORT",
            suggested_tools=["plot_learning_curve", "generate_training_report", "read_model_report"],
            carry_forward={"model_path": str(mp)},
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
# plot_learning_curve
# ---------------------------------------------------------------------------


def plot_learning_curve(
    file_path: str,
    target_column: str,
    model: str,
    task: str,
    cv: int = 5,
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Plot learning curve (train vs val score vs training size). HTML output."""
    progress = []
    try:
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file path.", "token_estimate": 30}

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
            "op": "plot_learning_curve",
            "dry_run": True,
            "file_path": str(dp),
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import importlib

        import plotly.graph_objects as go
        from sklearn.model_selection import learning_curve
        from sklearn.preprocessing import LabelEncoder

        if dp.stat().st_size == 0:
            return {"success": False, "error": f"File is empty: {dp.name}", "hint": "Verify the file has header + data rows.", "token_estimate": 30}
        df = _read_csv(str(dp))
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not found.",
                "hint": "Use inspect_dataset() to list column names.",
                "token_estimate": 30,
            }

        df = df.dropna(subset=[target_column])
        y = df[target_column]
        # Encode categoricals
        for col in df.select_dtypes(include=["object", "string"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        X = df.drop(columns=[target_column]).select_dtypes(include="number").fillna(0)

        # Build estimator
        CLASSIFIERS = {
            "lr": ("sklearn.linear_model", "LogisticRegression", {"random_state": 42, "max_iter": 200}),
            "rf": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50, "random_state": 42}),
            "dtc": ("sklearn.tree", "DecisionTreeClassifier", {"random_state": 42}),
            "knn": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
            "svm": ("sklearn.svm", "SVC", {"kernel": "rbf", "gamma": "auto", "random_state": 42}),
        }
        REGRESSORS = {
            "lir": ("sklearn.linear_model", "LinearRegression", {}),
            "rfr": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 50, "random_state": 42}),
            "dtr": ("sklearn.tree", "DecisionTreeRegressor", {"random_state": 42}),
        }
        model_map = CLASSIFIERS if task == "classification" else REGRESSORS
        if model not in model_map:
            allowed = ", ".join(model_map.keys())
            return {
                "success": False,
                "error": f"Unknown model '{model}'. Allowed: {allowed}",
                "hint": "Check model string.",
                "token_estimate": 30,
            }

        mod_name, cls_name, kwargs = model_map[model]
        cls = getattr(importlib.import_module(mod_name), cls_name)
        estimator = cls(**kwargs)

        scoring = "accuracy" if task == "classification" else "r2"
        train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(  # type: ignore[misc]
            estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
        )
        progress.append(ok("Computed learning curves", f"{len(train_sizes_abs)} points"))

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        t = get_theme(theme)
        tmpl = plotly_template(theme)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=train_sizes_abs.tolist(),
                y=train_mean.tolist(),
                mode="lines+markers",
                name="Train score",
                line=dict(color=t["accent"], width=2),
                error_y=dict(type="data", array=train_std.tolist(), visible=True, color=t["accent"]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=train_sizes_abs.tolist(),
                y=val_mean.tolist(),
                mode="lines+markers",
                name="Validation score",
                line=dict(color=t["success"], width=2),
                error_y=dict(type="data", array=val_std.tolist(), visible=True, color=t["success"]),
            )
        )

        fig.update_layout(
            title=f"Learning Curve — {model} ({task})",
            xaxis_title="Training Examples",
            yaxis_title=scoring.upper(),
            template=tmpl,
            height=calc_chart_height(450, mode="fixed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )

        out_abs, out_name = _save_chart(fig, output_path, f"{model}_learning_curve", dp, open_after, theme)
        progress.append(ok("Saved learning curve", out_name))

        resp = {
            "success": True,
            "op": "plot_learning_curve",
            "output_path": out_abs,
            "output_name": out_name,
            "final_train_score": round(float(train_mean[-1]), 4),
            "final_val_score": round(float(val_mean[-1]), 4),
            "scoring": scoring,
            "progress": progress,
            "token_estimate": 0,
        }
        resp["context"] = make_context(
            "plot_learning_curve",
            f"Plotted learning curve for {model} ({task}) on {dp.name}: val={round(float(val_mean[-1]), 3)}",
            [{"type": "report", "path": out_abs, "role": "learning_curve_chart"}],
        )
        resp["handover"] = make_handover(
            step="REPORT",
            suggested_tools=["tune_hyperparameters", "generate_training_report", "compare_models"],
            carry_forward={"file_path": str(dp), "model": model, "task": task},
        )
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "hint": "Check data and model compatibility.",
            "token_estimate": 30,
        }


# ---------------------------------------------------------------------------
# plot_predictions_vs_actual
# ---------------------------------------------------------------------------


def plot_predictions_vs_actual(
    model_path: str,
    file_path: str,
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Scatter plot of predicted vs actual values for regression. HTML."""
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
            "hint": "Train a regressor first.",
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
            "op": "plot_predictions_vs_actual",
            "dry_run": True,
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.graph_objects as go
        from sklearn.metrics import mean_squared_error, r2_score

        with open(mp, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        metadata = payload.get("metadata", {})
        progress.append(ok("Loaded model", mp.name))

        task = metadata.get("task", "regression")
        if task != "regression":
            return {
                "success": False,
                "error": "This chart is only for regression models.",
                "hint": "Use plot_roc_curve for classifiers.",
                "token_estimate": 30,
            }

        feature_columns = metadata.get("feature_columns", [])
        target_column = metadata.get("target_column", "")
        encoding_map = metadata.get("encoding_map", {})

        if dp.stat().st_size == 0:
            return {"success": False, "error": f"File is empty: {dp.name}", "hint": "Verify the file has header + data rows.", "token_estimate": 30}
        df = _read_csv(str(dp))
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        for col, mapping in encoding_map.items():
            if col in df.columns and col != target_column:
                df[col] = df[col].map(mapping).fillna(df[col])

        available = [c for c in feature_columns if c in df.columns]
        X = df[available].select_dtypes(include="number").fillna(0)
        y_true = df[target_column].values if target_column in df.columns else None

        if y_true is None:
            return {
                "success": False,
                "error": f"Target '{target_column}' not found.",
                "hint": "Provide the same dataset used for training.",
                "token_estimate": 30,
            }

        try:
            import xgboost as xgb

            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(X)
                y_pred = model.predict(dmat)
            else:
                y_pred = model.predict(X)
        except Exception:
            y_pred = model.predict(X)

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        # Subsample for scatter readability (>10K markers are unreadable)
        scat_cap = min(len(y_true), 10_000)
        if len(y_true) > scat_cap:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_true), scat_cap, replace=False)
            plot_true = y_true[idx]
            plot_pred = y_pred[idx]
        else:
            plot_true = y_true
            plot_pred = y_pred

        t = get_theme(theme)
        tmpl = plotly_template(theme)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_true.tolist(),
                y=plot_pred.tolist(),
                mode="markers",
                name="Predictions",
                marker=dict(color=t["accent"], opacity=0.6, size=6),
            )
        )
        # Perfect prediction line
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        fig.add_trace(
            go.Scatter(
                x=[mn, mx],
                y=[mn, mx],
                mode="lines",
                name="Perfect fit",
                line=dict(color=t["danger"], dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Predictions vs Actual — R²={r2:.3f}",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            template=tmpl,
            height=calc_chart_height(450, mode="fixed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )

        out_abs, out_name = _save_chart(fig, output_path, "pred_vs_actual", dp, open_after, theme)
        progress.append(ok("Saved chart", out_name))

        resp = {
            "success": True,
            "op": "plot_predictions_vs_actual",
            "output_path": out_abs,
            "output_name": out_name,
            "metrics": {"mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
            "n_points": len(y_true),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["context"] = make_context(
            "plot_predictions_vs_actual",
            f"Plotted predictions vs actual for {mp.name}: R²={round(r2, 3)}, RMSE={round(rmse, 3)}",
            [{"type": "report", "path": out_abs, "role": "pred_vs_actual_chart"}],
        )
        resp["handover"] = make_handover(
            step="REPORT",
            suggested_tools=["tune_hyperparameters", "generate_training_report", "read_model_report"],
            carry_forward={"model_path": str(mp)},
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
# generate_cluster_report
# ---------------------------------------------------------------------------


def generate_cluster_report(
    file_path: str,
    feature_columns: list[str],
    label_column: str,
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate HTML cluster visualization report with scatter and profile."""
    from shared.html_theme import (
        _open_file,
        build_html_report,
        data_table_html,
        metrics_cards_html,
        plotly_div,
        plotly_template,
    )

    progress = []
    try:
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Check file path.", "token_estimate": 30}

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
            "op": "generate_cluster_report",
            "dry_run": True,
            "progress": [info("Dry run — no files written")],
            "token_estimate": 40,
        }

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if dp.stat().st_size == 0:
            return {"success": False, "error": f"File is empty: {dp.name}", "hint": "Verify the file has header + data rows.", "token_estimate": 30}
        df = _read_csv(str(dp))
        progress.append(ok("Loaded data", f"{len(df)} rows"))

        if label_column not in df.columns:
            return {
                "success": False,
                "error": f"Label column '{label_column}' not found.",
                "hint": "Use run_clustering(save_labels=True) first.",
                "token_estimate": 30,
            }

        missing_feats = [c for c in feature_columns if c not in df.columns]
        if missing_feats:
            return {
                "success": False,
                "error": f"Feature columns not found: {', '.join(missing_feats[:5])}",
                "hint": "Use inspect_dataset() to list column names.",
                "token_estimate": 30,
            }

        X = df[feature_columns].select_dtypes(include="number").fillna(0)
        labels = df[label_column].astype(str)
        n_clusters = labels.nunique()

        tmpl = plotly_template(theme)
        sections = []

        # --- Summary cards ---
        label_counts = labels.value_counts().sort_values(ascending=False).to_dict()
        summary = {
            "n_clusters": n_clusters,
            "n_samples": len(df),
            "n_features": len(X.columns),
        }
        sections.append(
            {
                "id": "summary",
                "heading": "Summary",
                "html": metrics_cards_html(summary),
            }
        )

        # --- Cluster size table ---
        size_rows = [
            {"cluster": str(k), "count": int(v), "pct": f"{v / len(df) * 100:.1f}%"} for k, v in label_counts.items()
        ]
        sections.append(
            {
                "id": "cluster_sizes",
                "heading": "Cluster Sizes",
                "html": data_table_html(size_rows),
            }
        )

        # --- PCA scatter (2D) — subsample for readability ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(2, X_scaled.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)
            explained = [round(float(v), 3) for v in pca.explained_variance_ratio_]

            # Subsample for scatter readability (>10K points are unreadable)
            scat_cap = min(len(coords), 10_000)
            if len(coords) > scat_cap:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(coords), scat_cap, replace=False)
                scat_coords = coords[idx]
                scat_labels = labels.values[idx]
                scat_note = f" (sampled {scat_cap:,}/{len(coords):,})"
            else:
                scat_coords = coords
                scat_labels = labels.values
                scat_note = ""

            scatter_df = pd.DataFrame(
                {
                    "PC1": scat_coords[:, 0],
                    "PC2": scat_coords[:, 1],
                    "cluster": scat_labels,
                }
            )
            fig_scatter = px.scatter(
                scatter_df,
                x="PC1",
                y="PC2",
                color="cluster",
                render_mode="svg",
                title=f"PCA Scatter — {explained[0] * 100:.1f}%/{explained[1] * 100:.1f}% variance{scat_note}",
                template=tmpl,
            )
            apply_fig_theme(fig_scatter, theme)
            sections.append(
                {
                    "id": "scatter",
                    "heading": "PCA Cluster Scatter",
                    "html": plotly_div(fig_scatter, height=480, theme=theme),
                }
            )
            progress.append(ok("PCA scatter", f"var explained: {sum(explained) * 100:.1f}%"))

        # --- Feature means per cluster ---
        df_feat = X.copy()
        df_feat["_cluster"] = labels.values
        cluster_profile = df_feat.groupby("_cluster").mean().round(3)
        profile_rows = []
        for idx, row in cluster_profile.iterrows():
            r = {"cluster": str(idx)}
            r.update({col: round(float(val), 3) for col, val in row.items()})
            profile_rows.append(r)

        sections.append(
            {
                "id": "feature_profile",
                "heading": "Feature Means by Cluster",
                "html": data_table_html(profile_rows),
            }
        )

        # --- Bar chart: cluster sizes (sorted highest first) ---
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        bar_x = [str(k) for k, _ in sorted_labels]
        bar_y = [v for _, v in sorted_labels]
        fig_bar = go.Figure(
            go.Bar(
                x=bar_x,
                y=bar_y,
                marker=dict(color=bar_y, colorscale="Blues", reversescale=False),
            )
        )
        bar_h = calc_chart_height(len(bar_x), mode="bar", extra_base=40)
        fig_bar.update_layout(
            title="Cluster Sizes",
            xaxis_title="Cluster",
            yaxis_title="Count",
            xaxis=dict(categoryorder="array", categoryarray=bar_x),
            template=tmpl,
            height=bar_h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig_bar, theme)
        sections.append(
            {
                "id": "size_chart",
                "heading": "Cluster Size Chart",
                "html": plotly_div(fig_bar, height=bar_h, theme=theme),
            }
        )

        html = build_html_report(
            title=f"Cluster Report — {dp.name}",
            subtitle="",
            sections=sections,
            theme=theme,
            open_after=False,
            output_path="",
            sidebar_title="Cluster Report",
            sidebar_meta=f"{dp.name}<br>Clusters: {n_clusters} &middot; Samples: {len(df):,}",
        )

        out = get_output_path(output_path, dp, "cluster_report", "html")
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, html)
        progress.append(ok("Saved cluster report", out.name))

        if open_after:
            _open_file(out)

        resp = {
            "success": True,
            "op": "generate_cluster_report",
            "output_path": str(out),
            "output_name": out.name,
            "n_clusters": n_clusters,
            "n_samples": len(df),
            "sections_generated": len(sections),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["context"] = make_context(
            "generate_cluster_report",
            f"Generated cluster report for {dp.name}: {n_clusters} clusters, {len(df):,} samples → {out.name}",
            [{"type": "report", "path": str(out), "role": "cluster_report"}],
        )
        resp["handover"] = make_handover(
            step="REPORT",
            suggested_tools=["find_optimal_clusters", "run_clustering", "check_data_quality"],
            carry_forward={"file_path": str(dp), "label_column": label_column},
        )
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    except Exception as exc:
        return {"success": False, "error": str(exc), "hint": "Check data and label column.", "token_estimate": 30}


__all__ = [
    "generate_cluster_report",
    "plot_learning_curve",
    "plot_predictions_vs_actual",
    "plot_roc_curve",
]
