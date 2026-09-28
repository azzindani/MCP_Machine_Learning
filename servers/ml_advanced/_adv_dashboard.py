"""generate_model_dashboard: a model read against labelled rows, the way a data scientist reads one.

The training report says what a model scored when it was trained. It cannot
say how sure the model is when it is right, where the threshold should sit
when a missed churner costs five times a wasted call, which segments it fails,
what moves its predictions, or whether the rows it now meets look like the
ones it learned from -- each of those needs the model's predictions on
labelled rows. This scores a file with the model and draws them:

- performance against a baseline, per class, a normalised confusion heatmap;
- ROC and precision-recall curves, calibration and its Brier score;
- a threshold slider that recounts the confusion matrix and the cost of the
  errors (`cost_fp`, `cost_fn`) in the page, with the cheapest threshold named;
- lift and cumulative gain: how much of the positive class the top-scored
  tenth, fifth and half of the rows hold;
- permutation importance, partial dependence of the top features, and a linear
  model's coefficients or odds ratios;
- errors by segment, the most confident mistakes, leakage suspects;
- drift against the training file (`train_file_path`), per feature, as PSI;
- a leaderboard when other models are named (`compare_model_paths`).

Every number is computed here from the scored rows; the page recomputes only
the threshold, from the same scores.
"""

from __future__ import annotations

import json
import logging
from html import escape as html_escape
from typing import Any

import numpy as np
import pandas as pd

from shared.file_utils import embed_content, resolve_path
from shared.file_utils import read_csv as _read_csv
from shared.handover import make_context, make_handover
from shared.leakage import leakage_note, leakage_suspects
from shared.ml_utils import target_labels
from shared.progress import ok, warn

from ._adv_helpers import _error, _load_model, get_output_path

logger = logging.getLogger(__name__)

MAX_EXPLAIN_ROWS = 4000  # permutation importance and partial dependence score a sample this large
MAX_THRESHOLD_ROWS = 20000  # scores carried into the page for the slider
PD_FEATURES = 3
PD_POINTS = 15
MAX_SEGMENT_LEVELS = 20
MIN_SEGMENT_ROWS = 30
PSI_MODERATE, PSI_MAJOR = 0.1, 0.25


def _plain(value: Any) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), 4) if isinstance(value, float) else str(value)


class _Scorer:
    """A saved model and its manifest, scoring raw rows exactly as evaluate_model does."""

    def __init__(self, model_obj: Any, metadata: dict) -> None:
        self.model = model_obj
        self.meta = metadata
        self.task = metadata.get("task", "classification")
        self.features = list(metadata.get("feature_columns", []))
        self.encoding = metadata.get("encoding_map", {}) or {}
        self.labels = target_labels(metadata) if self.task == "classification" else None

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        df = frame.copy()
        for col, mapping in self.encoding.items():
            if col in df.columns and not str(col).startswith("__target__"):
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        X = df[self.features].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
        if self.meta.get("scaler") is not None:
            X = self.meta["scaler"].transform(X)
        if self.meta.get("poly") is not None:
            X = self.meta["poly"].transform(X)
        return X

    def truth(self, series: pd.Series) -> np.ndarray:
        if self.task != "classification":
            return np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
        if self.labels:
            code_of = {label: i for i, label in enumerate(self.labels)}
            return np.array([code_of.get(str(v), -1) for v in series])
        return np.asarray(pd.to_numeric(series, errors="coerce").fillna(-1), dtype=int)

    def scores(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Predictions, and for a classifier each row's class probabilities."""
        import xgboost as xgb

        X = self.matrix(frame)
        if self.meta.get("model_key") == "xgb" or isinstance(self.model, xgb.Booster):
            raw = self.model.predict(xgb.DMatrix(X))
            if self.task != "classification":
                return raw, None
            proba = raw if raw.ndim == 2 else np.column_stack([1 - raw, raw])
            return proba.argmax(axis=1), proba
        pred = self.model.predict(X)
        proba = (
            self.model.predict_proba(X)
            if self.task == "classification" and hasattr(self.model, "predict_proba")
            else None
        )
        return pred, proba

    def name(self, code: int) -> str:
        return self.labels[code] if self.labels and 0 <= code < len(self.labels) else str(code)


# ---------------------------------------------------------------------------
# What is measured
# ---------------------------------------------------------------------------


def _headline_metric(task: str, y: np.ndarray, pred: np.ndarray, pos: np.ndarray | None) -> tuple[str, float]:
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    if task != "classification":
        return "r2", float(r2_score(y, pred))
    if pos is not None and len(set(y)) == 2:
        return "auc", float(roc_auc_score(y, pos))
    return "accuracy", float(accuracy_score(y, pred))


def _classification(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, scorer: _Scorer) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    classes = sorted(set(int(v) for v in y) | set(int(v) for v in pred))
    majority = int(pd.Series(y).mode().iloc[0])
    out: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1_weighted": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "baseline": {
            "rule": f"always predicting {scorer.name(majority)!r}, the most common class",
            "accuracy": round(float(np.mean(y == majority)), 4),
        },
    }
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    out["per_class"] = [
        {"class": scorer.name(c), "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
         "f1": round(float(f[i]), 4), "support": int(s[i])}
        for i, c in enumerate(classes)
    ]  # fmt: skip
    out["confusion"] = {
        "labels": [scorer.name(c) for c in classes],
        "counts": confusion_matrix(y, pred, labels=classes).tolist(),
        "normalised": np.nan_to_num(confusion_matrix(y, pred, labels=classes, normalize="true")).round(4).tolist(),
    }
    if proba is not None and len(classes) == 2 and proba.shape[1] == 2:
        pos = proba[:, 1]
        out["auc"] = round(float(roc_auc_score(y, pos)), 4)
        out["brier"] = round(float(brier_score_loss(y, pos)), 4)
        out["baseline"]["brier"] = round(float(brier_score_loss(y, np.full(len(y), np.mean(y)))), 4)
    return out


def _curves(y: np.ndarray, pos: np.ndarray) -> dict:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, pos)
    prec, rec, _ = precision_recall_curve(y, pos)
    frac, mean = calibration_curve(y, pos, n_bins=10, strategy="quantile")
    step = max(1, len(fpr) // 400)
    return {
        "roc": {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()},
        "pr": {"recall": rec[::step].round(4).tolist(), "precision": prec[::step].round(4).tolist()},
        "average_precision": round(float(average_precision_score(y, pos)), 4),
        "calibration": {"predicted": mean.round(4).tolist(), "observed": frac.round(4).tolist()},
    }


def _threshold(y: np.ndarray, pos: np.ndarray, cost_fp: float, cost_fn: float) -> dict:
    """The confusion counts and error cost at each threshold, the default and the cheapest."""

    def at(t: float) -> dict:
        hit = pos >= t
        tp, fp = int(np.sum(hit & (y == 1))), int(np.sum(hit & (y == 0)))
        fn, tn = int(np.sum(~hit & (y == 1))), int(np.sum(~hit & (y == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {"threshold": round(t, 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 4), "recall": round(recall, 4),
                "cost": round(fp * cost_fp + fn * cost_fn, 4)}  # fmt: skip

    grid = [at(t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    best = min(grid, key=lambda g: (g["cost"], abs(g["threshold"] - 0.5)))
    return {"default": at(0.5), "cheapest": best, "cost_fp": cost_fp, "cost_fn": cost_fn}


def _lift(y: np.ndarray, pos: np.ndarray) -> dict:
    """Cumulative gain and lift by tenth of the rows, highest scores first."""
    order = np.argsort(-pos, kind="stable")
    hits = np.cumsum(y[order] == 1)
    total = max(int(hits[-1]), 1)
    rows = []
    for tenth in range(1, 11):
        n = max(1, int(round(len(y) * tenth / 10)))
        gain = hits[n - 1] / total
        rows.append({"share_of_rows": tenth / 10, "share_of_positives": round(float(gain), 4),
                     "lift": round(float(gain / (tenth / 10)), 3)})  # fmt: skip
    return {"deciles": rows, "positives": total}


def _regression(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    base = np.full(len(y), float(np.mean(y)))
    return {
        "r2": round(float(r2_score(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "baseline": {
            "rule": "always predicting the mean of these rows",
            "rmse": round(float(np.sqrt(mean_squared_error(y, base))), 4),
            "mae": round(float(mean_absolute_error(y, base)), 4),
        },
    }


def _metric_on(scorer: _Scorer, frame: pd.DataFrame, y: np.ndarray, name: str) -> float:
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    pred, proba = scorer.scores(frame)
    if name == "auc" and proba is not None:
        return float(roc_auc_score(y, proba[:, 1]))
    if name == "r2":
        return float(r2_score(y, pred))
    return float(accuracy_score(y, pred))


def _importance(
    scorer: _Scorer, frame: pd.DataFrame, y: np.ndarray, metric: str, rng: np.random.Generator
) -> list[dict]:
    """How much the headline metric drops when one feature's values are shuffled; three shuffles each."""
    base = _metric_on(scorer, frame, y, metric)
    out = []
    for feature in scorer.features:
        drops = []
        for _ in range(3):
            shuffled = frame.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].values)
            drops.append(base - _metric_on(scorer, shuffled, y, metric))
        out.append(
            {"feature": feature, "drop": round(float(np.mean(drops)), 4), "spread": round(float(np.std(drops)), 4)}
        )
    return sorted(out, key=lambda r: -r["drop"])


def _partial_dependence(scorer: _Scorer, frame: pd.DataFrame, feature: str) -> dict | None:
    """The average prediction with every row's `feature` set to each value in turn."""
    values = frame[feature].dropna()
    if values.empty:
        return None
    numeric = pd.api.types.is_numeric_dtype(values) and values.nunique() > 12
    grid = (
        sorted(set(np.quantile(values.astype(float), np.linspace(0.02, 0.98, PD_POINTS)).round(6)))
        if numeric
        else list(values.astype(str).value_counts().index[:12])
    )
    ys = []
    for v in grid:
        probe = frame.copy()
        probe[feature] = v
        pred, proba = scorer.scores(probe)
        ys.append(float(np.mean(proba[:, 1] if proba is not None and proba.shape[1] == 2 else pred)))
    return {"feature": feature, "numeric": bool(numeric), "x": [float(g) if numeric else str(g) for g in grid],
            "y": [round(v, 4) for v in ys]}  # fmt: skip


def _segments(frame: pd.DataFrame, correct: np.ndarray, error: np.ndarray, columns: list[str], task: str) -> list[dict]:
    """How the model does in each value of each segment column, weakest first, with how many rows back it."""
    out = []
    overall = float(np.mean(correct)) if task == "classification" else float(np.mean(error))
    for col in columns:
        groups = frame[col].astype(str).fillna("(missing)")
        for value, idx in groups.groupby(groups).groups.items():
            rows = len(idx)
            if rows < MIN_SEGMENT_ROWS:
                continue
            pos = frame.index.get_indexer(idx)
            score = float(np.mean(correct[pos])) if task == "classification" else float(np.mean(error[pos]))
            gap = (overall - score) if task == "classification" else (score - overall)
            out.append(
                {"column": col, "value": str(value), "rows": rows, "score": round(score, 4), "gap": round(gap, 4)}
            )
    return sorted(out, key=lambda s: -s["gap"])


def _psi(train: pd.Series, now: pd.Series) -> float | None:
    """Population stability: how far `now`'s distribution has moved from `train`'s."""
    a, b = train.dropna(), now.dropna()
    if a.empty or b.empty:
        return None
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and a.nunique() > 10:
        edges = np.unique(np.quantile(a.astype(float), np.linspace(0, 1, 11)))
        if len(edges) < 3:
            return None
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a.astype(float), edges)[0] / len(a)
        pb = np.histogram(b.astype(float), edges)[0] / len(b)
    else:
        levels = sorted(set(a.astype(str)) | set(b.astype(str)))
        pa = a.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).values
        pb = b.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).values
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

_SLIDER_JS = r"""
<script>
(function(){
  var d=JSON.parse(document.getElementById('th-data').textContent),el=document.getElementById('th-range');
  if(!el)return;
  function draw(){
    var t=+el.value,tp=0,fp=0,fn=0,tn=0;
    for(var i=0;i<d.p.length;i++){var hit=d.p[i]>=t;if(d.y[i]===1){if(hit)tp++;else fn++;}else{if(hit)fp++;else tn++;}}
    var pr=tp+fp?tp/(tp+fp):0,rc=tp+fn?tp/(tp+fn):0,cost=fp*d.cfp+fn*d.cfn;
    var set=function(id,v){var e=document.getElementById(id);if(e)e.textContent=v;};
    set('th-t',t.toFixed(2));set('th-tp',tp);set('th-fp',fp);set('th-fn',fn);set('th-tn',tn);
    set('th-pr',(pr*100).toFixed(1)+'%');set('th-rc',(rc*100).toFixed(1)+'%');set('th-cost',cost.toLocaleString('en-US'));
  }
  el.addEventListener('input',draw);draw();
})();
</script>
"""


def _slider_html(th: dict, scores: np.ndarray, y: np.ndarray, positive: str) -> str:
    data = json.dumps({"p": [round(float(v), 4) for v in scores], "y": [int(v) for v in y],
                       "cfp": th["cost_fp"], "cfn": th["cost_fn"]}).replace("</", "<\\/")  # fmt: skip
    best = th["cheapest"]
    return (
        f'<script type="application/json" id="th-data">{data}</script>'
        f"<p>Rows scored at or above the threshold are predicted <b>{html_escape(positive)}</b>. "
        f"A false positive costs {th['cost_fp']:g} and a false negative {th['cost_fn']:g}; the cheapest threshold "
        f"is <b>{best['threshold']:.2f}</b> (cost {best['cost']:,g}, against {th['default']['cost']:,g} at 0.50).</p>"
        '<label style="display:block;margin:.5rem 0">Threshold <b id="th-t"></b> '
        f'<input id="th-range" type="range" min="0.01" max="0.99" step="0.01" value="{best["threshold"]:.2f}" '
        'style="width:100%"></label>'
        "<table><thead><tr><th></th><th>Predicted positive</th><th>Predicted negative</th></tr></thead><tbody>"
        '<tr><td>Actually positive</td><td id="th-tp"></td><td id="th-fn"></td></tr>'
        '<tr><td>Actually negative</td><td id="th-fp"></td><td id="th-tn"></td></tr></tbody></table>'
        '<p>Precision <b id="th-pr"></b> · recall <b id="th-rc"></b> · cost of the errors <b id="th-cost"></b></p>'
    )


def generate_model_dashboard(
    model_path: str,
    file_path: str,
    target_column: str = "",
    compare_model_paths: list[str] | None = None,
    segment_columns: list[str] | None = None,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    train_file_path: str = "",
    theme: str = "device",
    output_path: str = "",
    open_after: bool = True,
    dry_run: bool = False,
    return_content: bool = False,
) -> dict:
    """A model read against labelled rows: performance, threshold, explanation, errors, drift."""
    import plotly.graph_objects as go

    from shared.html_theme import apply_fig_theme, build_html_report, data_table_html, metrics_cards_html, plotly_div
    from shared.html_theme import plotly_template as template_for

    progress: list[dict] = []
    try:
        mp = resolve_path(model_path)
        dp = resolve_path(file_path, (".csv",))
    except ValueError as exc:
        return _error(str(exc), "Check that model_path and file_path are inside your home directory.")
    if not mp.exists():
        return _error(f"Model file not found: {model_path}", "Train one with ml_train first.")
    if not dp.exists():
        return _error(f"File not found: {file_path}", "Pass a labelled CSV the model can score, e.g. the test split.")
    if cost_fp < 0 or cost_fn < 0:
        return _error(
            "cost_fp and cost_fn are costs: zero or more",
            "e.g. cost_fp=1, cost_fn=5 when a miss costs five times a false alarm.",
        )
    try:
        model_obj, metadata = _load_model(str(mp))
    except Exception as exc:
        return _error(f"Failed to load model: {exc}", "Check model_path points to a model ml_train saved.")
    scorer = _Scorer(model_obj, metadata)
    target = target_column or metadata.get("target_column", "")
    df = _read_csv(str(dp))
    if target not in df.columns:
        return _error(
            f"Target column {target!r} is not in {dp.name}. Its columns: {', '.join(map(str, df.columns[:30]))}.",
            "Pass target_column, or a file with the column the model was trained to predict.",
        )
    lacking = [c for c in scorer.features if c not in df.columns]
    if lacking:
        return _error(
            f"{dp.name} lacks the model's feature column(s): {', '.join(lacking[:8])}",
            "Score a file with the columns the model was trained on.",
        )
    df = df[df[target].notna()].reset_index(drop=True)
    out = get_output_path(output_path, mp, "model_dashboard", "html")
    if dry_run:
        return {"success": True, "op": "generate_model_dashboard", "dry_run": True, "output_path": str(out),
                "rows": len(df), "task": scorer.task, "progress": progress, "token_estimate": 40}  # fmt: skip

    y = scorer.truth(df[target])
    if scorer.task == "classification" and (y < 0).any():
        unseen = sorted(set(df[target].astype(str)[y < 0]))[:5]
        return _error(
            f"{target} holds value(s) the model never saw in training: {', '.join(unseen)}",
            "Score rows whose target is one of the model's classes.",
        )
    pred, proba = scorer.scores(df)
    progress.append(ok("Scored rows", f"{len(df):,} rows of {dp.name}"))
    binary = scorer.task == "classification" and proba is not None and proba.shape[1] == 2
    pos = proba[:, 1] if binary else None
    positive = scorer.name(1) if binary else ""
    rng = np.random.default_rng(42)
    tmpl = template_for(theme)

    def fig_html(fig: Any, height: int = 380) -> str:
        fig.update_layout(template=tmpl, height=height, margin={"l": 40, "r": 20, "t": 30, "b": 40})
        apply_fig_theme(fig, theme)
        return plotly_div(fig, height=height, theme=theme)

    sections: list[dict] = []
    resp: dict[str, Any] = {"success": True, "op": "generate_model_dashboard", "task": scorer.task, "rows": len(df)}

    # --- Performance, against a baseline -------------------------------------
    if scorer.task == "classification":
        perf = _classification(y, pred, proba, scorer)
        base = perf["baseline"]
        lift = perf["accuracy"] - base["accuracy"]
        headline = f"{perf['accuracy']:.1%} accurate against {base['accuracy']:.1%} for {base['rule']}" + (
            f"; AUC {perf['auc']:.3f}" if "auc" in perf else ""
        )
        cards = {"Accuracy": f"{perf['accuracy']:.1%}", "Baseline": f"{base['accuracy']:.1%}",
                 "F1 (weighted)": f"{perf['f1_weighted']:.3f}"}  # fmt: skip
        if "auc" in perf:
            cards["AUC"] = f"{perf['auc']:.3f}"
        verdict = "no better than" if lift <= 0 else f"{lift * 100:.1f} points above"
        body = metrics_cards_html(cards) + (
            f"<p>{html_escape(base['rule'][0].upper() + base['rule'][1:])} is right {base['accuracy']:.1%} of the "
            f"time; this model is {html_escape(verdict)} that.</p>"
        )
        cv = metadata.get("cv_mean_metrics")
        if isinstance(cv, dict) and cv:
            body += (
                "<p>Cross-validated when trained: "
                + html_escape(", ".join(f"{k} {v}" for k, v in cv.items() if not isinstance(v, dict)))
                + "</p>"
            )
        body += data_table_html(perf["per_class"])
        sections.append({"id": "performance", "heading": "Performance", "html": body})
        conf = perf["confusion"]
        heat = go.Figure(
            go.Heatmap(
                z=conf["normalised"], x=[f"predicted {n}" for n in conf["labels"]], y=[f"actual {n}" for n in conf["labels"]],
                text=[[f"{v:.0%}<br>{c:,}" for v, c in zip(row, counts, strict=True)] for row, counts in zip(conf["normalised"], conf["counts"], strict=True)],
                texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1, showscale=False,
                hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>",
            )
        )  # fmt: skip
        heat.update_layout(yaxis={"autorange": "reversed"})
        sections.append(
            {"id": "confusion", "heading": "Confusion matrix, share of each actual class", "html": fig_html(heat, 360)}
        )
        resp["metrics"] = {k: perf[k] for k in ("accuracy", "f1_weighted", "auc", "brier") if k in perf}
        resp["baseline"] = base
        resp["per_class"] = perf["per_class"]
        metric = "auc" if binary else "accuracy"
    else:
        perf = _regression(y, pred)
        base = perf["baseline"]
        headline = f"R² {perf['r2']:.3f}; RMSE {perf['rmse']:,.4g} against {base['rmse']:,.4g} for {base['rule']}"
        cards = {"R²": f"{perf['r2']:.3f}", "RMSE": f"{perf['rmse']:,.4g}", "Baseline RMSE": f"{base['rmse']:,.4g}",
                 "MAE": f"{perf['mae']:,.4g}"}  # fmt: skip
        sections.append({"id": "performance", "heading": "Performance", "html": metrics_cards_html(cards)})
        resid = y - pred
        scatter = go.Figure(go.Scattergl(x=y, y=pred, mode="markers", marker={"size": 4, "opacity": 0.5}, name="rows"))
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash"}, name="perfect"))
        scatter.update_layout(xaxis_title="actual", yaxis_title="predicted")
        sections.append({"id": "fit", "heading": "Predicted against actual", "html": fig_html(scatter)})
        res = go.Figure(go.Scattergl(x=pred, y=resid, mode="markers", marker={"size": 4, "opacity": 0.5}))
        res.add_hline(y=0, line_dash="dash")
        res.update_layout(xaxis_title="predicted", yaxis_title="actual − predicted")
        sections.append({"id": "residuals", "heading": "Residuals", "html": fig_html(res)})
        resp["metrics"] = {k: perf[k] for k in ("r2", "rmse", "mae")}
        resp["baseline"] = base
        metric = "r2"

    # --- Curves, calibration, threshold, lift --------------------------------
    if binary and pos is not None:
        curves = _curves(y, pos)
        roc = go.Figure(
            go.Scatter(x=curves["roc"]["fpr"], y=curves["roc"]["tpr"], mode="lines", name=f"AUC {perf['auc']:.3f}")
        )
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="chance"))
        roc.update_layout(xaxis_title="false positive rate", yaxis_title="true positive rate")
        pr = go.Figure(go.Scatter(x=curves["pr"]["recall"], y=curves["pr"]["precision"], mode="lines",
                                  name=f"AP {curves['average_precision']:.3f}"))  # fmt: skip
        pr.add_hline(y=float(np.mean(y == 1)), line_dash="dash", annotation_text="share of positives")
        pr.update_layout(xaxis_title="recall", yaxis_title="precision")
        sections.append({"id": "roc", "heading": "ROC curve", "html": fig_html(roc)})
        sections.append({"id": "pr", "heading": "Precision and recall", "html": fig_html(pr)})
        cal = go.Figure(go.Scatter(x=curves["calibration"]["predicted"], y=curves["calibration"]["observed"],
                                   mode="lines+markers", name="model"))  # fmt: skip
        cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="calibrated"))
        cal.update_layout(xaxis_title="predicted probability", yaxis_title=f"share actually {positive}")
        sections.append({
            "id": "calibration", "heading": "Calibration",
            "html": f"<p>Brier score {perf['brier']:.4f}, against {perf['baseline']['brier']:.4f} for predicting the "
                    "share of positives for every row: lower is better. On the dashed line a score of 0.3 means 30% of "
                    "such rows are positive.</p>" + fig_html(cal),
        })  # fmt: skip
        th = _threshold(y, pos, float(cost_fp), float(cost_fn))
        keep = (
            np.arange(len(y)) if len(y) <= MAX_THRESHOLD_ROWS else rng.choice(len(y), MAX_THRESHOLD_ROWS, replace=False)
        )
        sections.append({"id": "threshold", "heading": "Threshold and the cost of errors",
                         "html": _slider_html(th, pos[keep], y[keep], positive)})  # fmt: skip
        gains = _lift(y, pos)
        gain_fig = go.Figure(go.Scatter(x=[0] + [d["share_of_rows"] for d in gains["deciles"]],
                                        y=[0] + [d["share_of_positives"] for d in gains["deciles"]],
                                        mode="lines+markers", name="model"))  # fmt: skip
        gain_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="random"))
        gain_fig.update_layout(xaxis_title="share of rows, highest scores first", yaxis_title=f"share of the {positive}",
                               xaxis_tickformat=".0%", yaxis_tickformat=".0%")  # fmt: skip
        top = gains["deciles"][1]
        sections.append({
            "id": "lift", "heading": "Lift and gain",
            "html": f"<p>Targeting the top {top['share_of_rows']:.0%} of rows by score reaches "
                    f"<b>{top['share_of_positives']:.0%}</b> of the {html_escape(positive)} rows, {top['lift']:.1f} times what "
                    "a random pick of the same size would.</p>" + fig_html(gain_fig),
        })  # fmt: skip
        resp["curves"] = {"average_precision": curves["average_precision"], "calibration": curves["calibration"]}
        resp["threshold"] = th
        resp["lift"] = gains["deciles"]

    # --- What moves the predictions -------------------------------------------
    sample = df if len(df) <= MAX_EXPLAIN_ROWS else df.sample(MAX_EXPLAIN_ROWS, random_state=42).reset_index(drop=True)
    y_sample = scorer.truth(sample[target])
    importance = _importance(scorer, sample, y_sample, metric, rng)
    imp = [r for r in importance if r["drop"] != 0][:15] or importance[:15]
    bars = go.Figure(go.Bar(x=[r["drop"] for r in imp][::-1], y=[r["feature"] for r in imp][::-1], orientation="h",
                            error_x={"type": "data", "array": [r["spread"] for r in imp][::-1]}))  # fmt: skip
    bars.update_layout(xaxis_title=f"drop in {metric} when the feature is shuffled")
    sections.append({
        "id": "importance", "heading": "Permutation importance",
        "html": f"<p>Each feature's values shuffled across {len(sample):,} rows, three times: how much {metric} falls "
                "is how much the model leans on it. Near zero means the model does without it.</p>" + fig_html(bars, 60 + 26 * len(imp)),
    })  # fmt: skip
    resp["importance"] = importance
    pd_rows = []
    for r in importance[:PD_FEATURES]:
        part = _partial_dependence(scorer, sample.head(1000), r["feature"])
        if part is None:
            continue
        pd_rows.append(part)
        fig = go.Figure(
            go.Scatter(x=part["x"], y=part["y"], mode="lines+markers")
            if part["numeric"]
            else go.Bar(x=part["x"], y=part["y"])
        )
        fig.update_layout(xaxis_title=part["feature"],
                          yaxis_title=f"average predicted {'probability of ' + positive if binary else target}")  # fmt: skip
        sections.append(
            {
                "id": f"pd_{len(pd_rows)}",
                "heading": f"Partial dependence: {part['feature']}",
                "html": fig_html(fig, 320),
            }
        )
    resp["partial_dependence"] = pd_rows
    from .engine import _coefficient_rows

    coefficients = _coefficient_rows(model_obj, metadata)
    if coefficients:
        sections.append(
            {"id": "coefficients", "heading": "Coefficients", "html": data_table_html(coefficients, max_rows=60)}
        )
        resp["coefficients"] = coefficients

    # --- Where it fails -------------------------------------------------------
    correct = (pred == y).astype(float) if scorer.task == "classification" else np.zeros(len(y))
    error = np.abs(y - pred) if scorer.task != "classification" else 1 - correct
    chosen = [c for c in (segment_columns or []) if c in df.columns]
    if segment_columns and len(chosen) < len(segment_columns):
        missing = [c for c in segment_columns if c not in df.columns]
        return _error(
            f"segment_columns names column(s) not in {dp.name}: {', '.join(missing)}",
            "Name columns of the scored file.",
        )
    if not chosen:
        chosen = [
            c for c in df.columns
            if c != target and not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= MAX_SEGMENT_LEVELS
        ][:6]  # fmt: skip
    segments = _segments(df, correct, error, chosen, scorer.task)
    weak = [s for s in segments if s["gap"] > (0.05 if scorer.task == "classification" else 0)][:10]
    if segments:
        label = "accuracy" if scorer.task == "classification" else "mean absolute error"
        rows = [{"segment": f"{s['column']} = {s['value']}", "rows": s["rows"], label: s["score"],
                 "worse than overall by": s["gap"]} for s in segments[:20]]  # fmt: skip
        sections.append({"id": "segments", "heading": "Errors by segment",
                         "html": f"<p>Segments with at least {MIN_SEGMENT_ROWS} rows, weakest first.</p>" + data_table_html(rows)})  # fmt: skip
    resp["segments"] = segments[:20]
    resp["weak_segments"] = weak
    if scorer.task == "classification" and proba is not None:
        confidence = proba.max(axis=1)
        wrong = np.where(correct == 0)[0]
        worst = wrong[np.argsort(-confidence[wrong])][:10]
    else:
        worst = np.argsort(-error)[:10]
    shown_cols = [target, *scorer.features[:8]]
    worst_rows = []
    for i in worst:
        # A missing cell is null in the response, never a NaN a JSON reader rejects.
        row = {c: _plain(df.at[i, c]) for c in shown_cols}
        row["predicted"] = scorer.name(int(pred[i])) if scorer.task == "classification" else round(float(pred[i]), 4)
        if scorer.task == "classification" and proba is not None:
            row["confidence"] = round(float(proba[i].max()), 3)
        worst_rows.append(row)
    if worst_rows:
        heading = "The most confident mistakes" if scorer.task == "classification" else "The largest misses"
        sections.append({"id": "worst", "heading": heading, "html": data_table_html(worst_rows)})
    resp["worst_predictions"] = worst_rows
    suspects = leakage_suspects(df, target, [c for c in scorer.features if c in df.columns])
    note = leakage_note(suspects, None)
    if suspects:
        sections.insert(
            0, {"id": "leakage", "heading": "Scores may not be real", "html": f"<p>{html_escape(note)}</p>"}
        )
    resp["leakage_suspects"] = suspects[:10]

    # --- Drift against the rows it learned from --------------------------------
    if train_file_path:
        try:
            tp_ = resolve_path(train_file_path, (".csv",))
        except ValueError as exc:
            return _error(str(exc), "Pass the file the model was trained on.")
        if not tp_.exists():
            return _error(f"File not found: {train_file_path}", "Pass the file the model was trained on.")
        train = _read_csv(str(tp_))
        drift = []
        for feature in scorer.features:
            if feature in train.columns:
                value = _psi(train[feature], df[feature])
                if value is not None:
                    band = "major" if value >= PSI_MAJOR else "moderate" if value >= PSI_MODERATE else "stable"
                    drift.append({"feature": feature, "psi": round(value, 4), "shift": band})
        drift.sort(key=lambda d: -d["psi"])
        moved = [d for d in drift if d["shift"] != "stable"]
        sections.append({
            "id": "drift", "heading": "Drift from the training rows",
            "html": f"<p>Population stability index per feature, {tp_.name} against {dp.name}: under {PSI_MODERATE} is "
                    f"stable, {PSI_MODERATE}-{PSI_MAJOR} a moderate shift, above {PSI_MAJOR} a major one. "
                    f"{len(moved)} of {len(drift)} features have moved.</p>" + data_table_html(drift),
        })  # fmt: skip
        resp["drift"] = drift

    # --- Other models on the same rows ---------------------------------------
    if compare_model_paths:
        board = []
        for raw in [model_path, *compare_model_paths]:
            try:
                other_obj, other_meta = _load_model(str(resolve_path(raw)))
                other = _Scorer(other_obj, other_meta)
                o_pred, o_proba = other.scores(df)
                o_y = other.truth(df[target])
                name, value = _headline_metric(
                    other.task, o_y, o_pred, o_proba[:, 1] if o_proba is not None and o_proba.shape[1] == 2 else None
                )
                cv = other_meta.get("cv_mean_metrics") or {}
                board.append({"model": resolve_path(raw).name, "type": str(other_meta.get("model_type", "")),
                              name: round(value, 4), "features": len(other.features),
                              "cross_validated": ", ".join(f"{k} {v}" for k, v in cv.items() if not isinstance(v, dict)) or "-"})  # fmt: skip
            except Exception as exc:
                return _error(
                    f"compare_model_paths: {raw} could not be scored on {dp.name}: {exc}",
                    "Compare models trained for the same target.",
                )
        key = next(k for k in ("auc", "accuracy", "r2") if k in board[0])
        board.sort(key=lambda r: -float(r.get(key, float("-inf"))))
        sections.append({"id": "leaderboard", "heading": "Leaderboard, on these rows", "html": data_table_html(board)})
        resp["leaderboard"] = board

    sections.insert(0, {"id": "answer", "heading": "The answer first", "html": f"<p><b>{html_escape(headline)}</b></p>"
                        + (f"<p>Weakest segment: {html_escape(weak[0]['column'])} = {html_escape(weak[0]['value'])} "
                           f"({weak[0]['rows']:,} rows).</p>" if weak else "")})  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    build_html_report(
        title=f"Model dashboard: {mp.stem}",
        subtitle=f"{metadata.get('model_type', '')} — {scorer.task} — scored on {dp.name}",
        sections=sections,
        theme=theme,
        open_after=open_after,
        output_path=str(out),
        extra_body=_SLIDER_JS if binary else "",
    )
    progress.append(ok("Model dashboard saved", out.name))
    if suspects:
        progress.append(
            warn(f"{len(suspects)} possible leakage suspect(s)", ", ".join(str(s["feature"]) for s in suspects[:3]))
        )
    resp.update(
        {
            "headline": headline,
            "output_path": str(out),
            "output_name": out.name,
            "sections_generated": [s["id"] for s in sections],
            "progress": progress,
        }
    )
    resp["context"] = make_context(
        "generate_model_dashboard",
        f"Model dashboard for {mp.name} on {dp.name}: {headline}",
        [{"type": "report", "path": str(out), "role": "model_dashboard"}],
    )
    resp["handover"] = make_handover(
        workflow_step="REPORT",
        suggested_next=["tune_hyperparameters", "compare_models", "generate_training_report"],
        carry_forward={"model_path": str(mp), "file_path": str(dp)},
    )
    embed_content(resp, out, return_content)
    resp["token_estimate"] = len(str(resp)) // 4
    return resp
