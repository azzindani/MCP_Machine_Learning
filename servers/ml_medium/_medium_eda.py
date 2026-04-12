"""ml_medium EDA tools — generate_eda_report and quality helpers."""

from __future__ import annotations

import pandas as pd

from shared.file_utils import atomic_write_text

from ._medium_helpers import (
    _error,
    append_receipt,
    get_output_path,
    ok,
    resolve_path,
)


def _compute_quality_score(df: pd.DataFrame, alerts: list[dict]) -> float:
    """Score 0–100. Start at 100, deduct for each alert by severity."""
    severity_weights = {"high": 15, "medium": 8, "low": 3}
    deductions = sum(severity_weights.get(a.get("severity", "low"), 3) for a in alerts)
    # Additional structural deductions
    miss_pct = df.isnull().sum().sum() / max(len(df) * len(df.columns), 1) * 100
    dup_pct = df.duplicated().sum() / max(len(df), 1) * 100
    deductions += min(miss_pct * 0.5, 20)  # up to 20 pts for missingness
    deductions += min(dup_pct * 0.3, 10)  # up to 10 pts for duplicates
    return round(max(0.0, min(100.0, 100.0 - deductions)), 1)


def _run_quality_alerts(df: pd.DataFrame, target_column: str = "") -> list[dict]:
    """Run 8 quality checks. Returns list of alert dicts."""
    alerts: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Pre-compute stats in bulk (vectorized — single pass each)
    nunique_all = df.nunique(dropna=False)
    null_pcts = df.isnull().mean() * 100

    # 1. Constant columns (single unique value)
    for col in nunique_all.index:
        if nunique_all[col] <= 1:
            alerts.append(
                {
                    "type": "constant_column",
                    "severity": "high",
                    "column": col,
                    "message": f"Column '{col}' has only 1 unique value — provides no information.",
                    "recommendation": f"Drop '{col}' with run_preprocessing op 'drop_column'.",
                }
            )

    # 2. High missing data (>20%)
    for col in null_pcts.index:
        miss_pct = float(null_pcts[col])
        if miss_pct > 20:
            alerts.append(
                {
                    "type": "high_missing",
                    "severity": "high",
                    "column": col,
                    "missing_pct": round(miss_pct, 1),
                    "message": f"Column '{col}' is {miss_pct:.1f}% missing.",
                    "recommendation": "Fill with run_preprocessing fill_nulls or drop if >50%.",
                }
            )

    # 3. Zero-inflated distributions (vectorized)
    feat_num = [c for c in numeric_cols if c != target_column]
    if feat_num:
        zero_pcts = (df[feat_num] == 0).mean() * 100
        for col in feat_num:
            zp = float(zero_pcts[col])
            if zp > 50:
                alerts.append(
                    {
                        "type": "zero_inflated",
                        "severity": "medium",
                        "column": col,
                        "zero_pct": round(zp, 1),
                        "message": f"Column '{col}' is {zp:.1f}% zeros — may need log transform.",
                        "recommendation": "Consider log1p transform or treat zeros as a separate indicator.",
                    }
                )

    # 4. High cardinality in categoricals (use pre-computed nunique)
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != target_column]
    for col in cat_cols:
        n_unique = int(nunique_all[col])
        ratio = n_unique / max(len(df), 1)
        if ratio > 0.5 and n_unique > 20:
            alerts.append(
                {
                    "type": "high_cardinality",
                    "severity": "medium",
                    "column": col,
                    "unique_count": n_unique,
                    "message": f"Column '{col}' has {n_unique} unique values ({ratio * 100:.1f}% of rows) — likely an ID or free-text field.",  # noqa: E501
                    "recommendation": f"Drop '{col}' or encode only top-N categories before training.",
                }
            )

    # 5. Class imbalance (>90% dominance) — only for target or low-cardinality cols
    check_imbal = [target_column] if target_column and target_column in df.columns else []
    check_imbal += [c for c in cat_cols if df[c].nunique() <= 10]
    for col in check_imbal[:5]:  # cap to 5
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.90:
            alerts.append(
                {
                    "type": "class_imbalance",
                    "severity": "high" if col == target_column else "medium",
                    "column": col,
                    "dominant_class": str(vc.index[0]),
                    "dominant_pct": round(float(vc.iloc[0]) * 100, 1),
                    "message": f"'{col}' is {vc.iloc[0] * 100:.1f}% '{vc.index[0]}' — severe class imbalance.",
                    "recommendation": "Use stratify=y in train split. Consider SMOTE or class_weight='balanced'.",
                }
            )

    # 6. Extreme skewness (|skew| > 2) — vectorized
    if feat_num:
        try:
            skews = df[feat_num].skew()
            for col in feat_num:
                skew = float(skews[col])
                if abs(skew) > 2:
                    direction = "right" if skew > 0 else "left"
                    alerts.append(
                        {
                            "type": "extreme_skewness",
                            "severity": "medium",
                            "column": col,
                            "skewness": round(skew, 2),
                            "message": f"Column '{col}' is {direction}-skewed "
                            f"(skew={skew:.2f}) — may hurt linear models.",
                            "recommendation": "Apply log transform or use "
                            "run_preprocessing 'cap_outliers' to reduce skew.",
                        }
                    )
        except Exception:
            pass

    # 7. Multicollinearity (|r| > 0.9 between feature pairs)
    if len(numeric_cols) >= 2:
        feat_cols = [c for c in numeric_cols if c != target_column]
        if len(feat_cols) >= 2:
            corr = df[feat_cols].corr().abs()
            seen: set[tuple] = set()
            for i, c1 in enumerate(feat_cols):
                for c2 in feat_cols[i + 1 :]:
                    r = corr.loc[c1, c2]
                    if r > 0.9 and (c1, c2) not in seen:
                        seen.add((c1, c2))
                        alerts.append(
                            {
                                "type": "multicollinearity",
                                "severity": "medium",
                                "columns": [c1, c2],
                                "correlation": round(float(r), 3),
                                "message": f"'{c1}' and '{c2}' are highly correlated (r={r:.3f}).",
                                "recommendation": "Consider dropping one of them, or use PCA via apply_dimensionality_reduction.",  # noqa: E501
                            }
                        )

    # 8. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / len(df) * 100, 1)
        alerts.append(
            {
                "type": "duplicate_rows",
                "severity": "medium" if dup_pct > 5 else "low",
                "count": dup_count,
                "pct": dup_pct,
                "message": f"{dup_count} duplicate rows found ({dup_pct}% of data).",
                "recommendation": "Remove with run_preprocessing op 'drop_duplicates'.",
            }
        )

    return alerts


def _alerts_html(alerts: list[dict], t: dict) -> str:
    """Render alerts as styled HTML cards — matches Data Analyst pattern."""
    if not alerts:
        return (
            '<div class="alert-panel">'
            '<div class="alert-item info">'
            '<span class="alert-badge info">OK</span> No data quality issues detected.'
            "</div></div>"
        )
    sev_badge = {"high": "error", "medium": "warning", "low": "info"}
    sev_label = {
        "high": lambda a: a["type"].replace("_", " ").upper(),
        "medium": lambda a: a["type"].replace("_", " ").upper(),
        "low": lambda a: a["type"].replace("_", " ").upper(),
    }
    parts = []
    for a in alerts:
        sev = a.get("severity", "low")
        badge_cls = sev_badge.get(sev, "info")
        label = sev_label.get(sev, lambda x: "INFO")(a)
        msg = a.get("message", "")
        rec = a.get("recommendation", "")
        full_msg = f"{msg}<br><small style='color:var(--text-muted)'>{rec}</small>" if rec else msg
        parts.append(
            f'<div class="alert-item {badge_cls}">'
            f'<span class="alert-badge {badge_cls}">{label}</span>'
            f"<span>{full_msg}</span></div>"
        )
    return f'<div class="alert-panel">{"".join(parts)}</div>'


def _quality_score_html(score: float, alerts: list[dict], t: dict) -> str:
    """Render quality score gauge + alert summary cards."""
    high = sum(1 for a in alerts if a.get("severity") == "high")
    med = sum(1 for a in alerts if a.get("severity") == "medium")
    low = sum(1 for a in alerts if a.get("severity") == "low")

    score_cls = "good" if score >= 80 else ("warn" if score >= 60 else "bad")
    score_card = (
        f'<div class="cards">'
        f'<div class="card {score_cls}">'
        f'  <div class="num">{score}</div><div class="lbl">Quality Score</div>'
        f"</div>"
        f'<div class="card bad">'
        f'  <div class="num">{high}</div><div class="lbl">High Severity</div>'
        f"</div>"
        f'<div class="card warn">'
        f'  <div class="num">{med}</div><div class="lbl">Medium Severity</div>'
        f"</div>"
        f'<div class="card good">'
        f'  <div class="num">{low}</div><div class="lbl">Low Severity</div>'
        f"</div>"
        f"</div>"
    )
    return score_card + _alerts_html(alerts, t)


def generate_eda_report(
    file_path: str,
    target_column: str = "",
    theme: str = "dark",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
) -> dict:
    """Generate interactive HTML EDA report with Plotly charts."""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from shared.html_theme import (
        _open_file,
        apply_fig_theme,
        build_html_report,
        calc_chart_height,
        data_table_html,
        get_theme,
        metrics_cards_html,
        plotly_div,
        plotly_template,
    )

    progress: list[dict] = []
    try:
        path = resolve_path(file_path)
    except ValueError as exc:
        return _error(str(exc), "Check that file_path is inside your home directory.")
    if not path.exists():
        return _error(f"File not found: {file_path}", "Check that file_path is absolute and the CSV file exists.")
    if path.suffix.lower() != ".csv":
        return _error(f"Expected .csv file, got {path.suffix!r}", "Provide a CSV file path.")

    out_path = get_output_path(output_path, path, "eda_report", "html")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return _error(f"Failed to read CSV: {exc}", "Check the file is a valid CSV.")
    progress.append(ok(f"Loaded {path.name}", f"{len(df):,} rows × {len(df.columns)} cols"))

    if dry_run:
        resp: dict = {
            "success": True,
            "op": "generate_eda_report",
            "dry_run": True,
            "output_path": str(out_path),
            "progress": progress,
            "token_estimate": 0,
        }
        resp["token_estimate"] = len(str(resp)) // 4
        return resp

    t = get_theme(theme)
    tmpl = plotly_template(theme)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    sections: list[dict] = []

    # ── 1. Quality Alerts + Score ──────────────────────────────────────────
    alerts = _run_quality_alerts(df, target_column)
    quality_score = _compute_quality_score(df, alerts)
    progress.append(ok("Quality analysis", f"score={quality_score}/100, {len(alerts)} alerts"))

    missing_total = int(df.isnull().sum().sum())
    missing_pct = round(missing_total / max(len(df) * len(df.columns), 1) * 100, 2)
    dup_rows = int(df.duplicated().sum())

    score_cls = "good" if quality_score >= 80 else ("warn" if quality_score >= 60 else "bad")
    dup_cls = "warn" if dup_rows > 0 else "good"
    overview_html = metrics_cards_html(
        {
            "rows": f"{len(df):,}",
            "columns": len(df.columns),
            "numeric": len(numeric_cols),
            "categorical": len(cat_cols),
            "quality_score": quality_score,
            "missing_cells": f"{missing_pct}%",
            "duplicates": f"{dup_rows:,}",
        },
        styles={
            "quality_score": score_cls,
            "duplicates": dup_cls,
        },
    )
    sections.append(
        {
            "id": "quality",
            "heading": "Data Quality",
            "html": overview_html + _quality_score_html(quality_score, alerts, t),
        }
    )

    # ── 2. Missing values pattern ──────────────────────────────────────────
    if missing_total > 0:
        null_series = df.isnull().sum().sort_values(ascending=False)
        null_series = null_series[null_series > 0]

        fig_miss = px.bar(
            x=null_series.values,
            y=null_series.index,
            orientation="h",
            title="Missing Values per Column",
            labels={"x": "Missing Count", "y": "Column"},
            template=tmpl,
            color=null_series.values / len(df) * 100,
            color_continuous_scale="Reds",
        )
        fig_miss.update_coloraxes(colorbar_title="% Missing")
        miss_h = calc_chart_height(len(null_series), mode="bar", extra_base=30)
        fig_miss.update_layout(
            yaxis=dict(autorange="reversed"),
            height=miss_h,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        apply_fig_theme(fig_miss, theme)
        sections.append(
            {
                "id": "missing",
                "heading": "Missing Values",
                "html": plotly_div(fig_miss, height=miss_h, theme=theme),
            }
        )
        progress.append(ok("Missing values chart", f"{len(null_series)} cols affected"))

    # ── 3. Distributions — histogram + box plot side by side ──────────────
    if numeric_cols:
        show_nums = numeric_cols[:12]
        n = len(show_nums)
        # 2 charts per col: histogram left, box right
        rows_n = n

        fig_dist = make_subplots(
            rows=rows_n,
            cols=2,
            subplot_titles=[f"{c} — histogram" if i % 2 == 0 else f"{c} — box" for c in show_nums for i in range(2)],
            horizontal_spacing=0.08,
            vertical_spacing=0.05,
        )
        for i, col in enumerate(show_nums):
            r = i + 1
            clean = df[col].dropna()
            fig_dist.add_trace(
                go.Histogram(x=clean, name=col, showlegend=False, marker_color=t["accent"]),
                row=r,
                col=1,
            )
            fig_dist.add_trace(
                go.Box(x=clean, name=col, showlegend=False, marker_color=t["accent"], boxpoints="outliers"),
                row=r,
                col=2,
            )
        dist_h = calc_chart_height(rows_n, mode="subplot")
        fig_dist.update_layout(
            title="Numeric Distributions (Histogram + Box Plot)",
            template=tmpl,
            height=dist_h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig_dist, theme)
        sections.append(
            {
                "id": "distributions",
                "heading": "Numeric Distributions",
                "html": plotly_div(fig_dist, height=dist_h, theme=theme),
            }
        )
        progress.append(ok("Distribution charts", f"{len(show_nums)} cols (histogram + box)"))

    # ── 4. Correlation — Pearson AND Spearman ─────────────────────────────
    if len(numeric_cols) >= 2:
        feat_cols = [c for c in numeric_cols if c != target_column] or numeric_cols

        fig_corr = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Pearson Correlation", "Spearman Correlation"],
            horizontal_spacing=0.12,
        )
        for col_idx, method in enumerate(["pearson", "spearman"], start=1):
            corr = df[feat_cols].corr(method=method)
            fig_corr.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                    showscale=(col_idx == 1),
                    name=method,
                ),
                row=1,
                col=col_idx,
            )
        corr_h = calc_chart_height(len(feat_cols), mode="heatmap", extra_base=60)
        fig_corr.update_layout(
            template=tmpl,
            height=corr_h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig_corr, theme)
        sections.append(
            {
                "id": "correlation",
                "heading": "Correlation (Pearson + Spearman)",
                "html": plotly_div(fig_corr, height=corr_h, theme=theme),
            }
        )
        progress.append(ok("Correlation heatmaps", f"Pearson + Spearman, {len(feat_cols)} features"))

    # ── 5. Categorical columns ─────────────────────────────────────────────
    show_cats = [c for c in cat_cols if c != target_column][:6]
    if show_cats:
        cat_html = ""
        for col in show_cats:
            vc = df[col].value_counts().head(15)
            fig_cat = px.bar(
                x=vc.values,
                y=vc.index.astype(str),
                orientation="h",
                title=f"{col} — Top Values",
                labels={"x": "Count", "y": col},
                template=tmpl,
                color=vc.values,
                color_continuous_scale="Blues",
            )
            cat_h = calc_chart_height(len(vc), mode="bar")
            fig_cat.update_layout(
                yaxis=dict(autorange="reversed"),
                height=cat_h,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            apply_fig_theme(fig_cat, theme)
            cat_html += plotly_div(fig_cat, height=cat_h, theme=theme)
        sections.append({"id": "categorical", "heading": "Categorical Columns", "html": cat_html})
        progress.append(ok("Categorical charts", f"{len(show_cats)} cols"))

    # ── 6. Target column analysis ──────────────────────────────────────────
    if target_column and target_column in df.columns:
        tgt = df[target_column]
        if tgt.nunique() <= 20:
            vc = tgt.value_counts()
            fig_tgt = px.pie(
                names=vc.index.astype(str),
                values=vc.values,
                title=f"Target Distribution: {target_column}",
                template=tmpl,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            fig_tgt = px.histogram(
                df,
                x=target_column,
                nbins=40,
                title=f"Target Distribution: {target_column}",
                template=tmpl,
            )
        fig_tgt.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        apply_fig_theme(fig_tgt, theme)
        sections.append(
            {
                "id": "target",
                "heading": f"Target Column: {target_column}",
                "html": plotly_div(fig_tgt, height=400, theme=theme),
            }
        )
        progress.append(ok("Target distribution", f"{tgt.nunique()} unique values"))

    # ── 7. Summary statistics table ────────────────────────────────────────
    if numeric_cols:
        desc = df[numeric_cols[:12]].describe().T.round(3)
        desc["skewness"] = df[numeric_cols[:12]].skew().round(3)
        desc.index.name = "column"
        rows_data = [{"column": idx, **row.to_dict()} for idx, row in desc.iterrows()]
        sections.append(
            {
                "id": "stats",
                "heading": "Summary Statistics",
                "html": data_table_html(rows_data),
            }
        )

    # ── Build and write report ─────────────────────────────────────────────
    html = build_html_report(
        title=f"EDA Report — {path.name}",
        subtitle="",
        sections=sections,
        theme=theme,
        open_after=False,
        output_path="",
        sidebar_title="EDA Report",
        sidebar_meta=f"{path.name}<br>{len(df):,} rows &times; {len(df.columns)} cols",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, html)
    progress.append(ok("Saved HTML report", out_path.name))

    if open_after:
        _open_file(out_path)

    file_size_kb = out_path.stat().st_size // 1024
    append_receipt(str(path), "generate_eda_report", {"theme": theme, "target_column": target_column}, "success", "")

    resp = {
        "success": True,
        "op": "generate_eda_report",
        "output_path": str(out_path),
        "output_name": out_path.name,
        "file_size_kb": file_size_kb,
        "quality_score": quality_score,
        "alerts_count": len(alerts),
        "alerts_high": sum(1 for a in alerts if a.get("severity") == "high"),
        "alerts_medium": sum(1 for a in alerts if a.get("severity") == "medium"),
        "alerts_low": sum(1 for a in alerts if a.get("severity") == "low"),
        "alerts": alerts,
        "charts_generated": len(sections),
        "progress": progress,
        "token_estimate": 0,
    }
    resp["token_estimate"] = len(str(resp)) // 4
    return resp


__all__ = [
    "generate_eda_report",
    "_compute_quality_score",
    "_run_quality_alerts",
    "_alerts_html",
    "_quality_score_html",
]
