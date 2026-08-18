"""Export a fitted model to plain JavaScript so a report can predict in-page.

A training report that only shows metrics tells you how the model scored. It
cannot tell you what the model *does* — what happens when spend doubles, which
way a feature pushes the answer. That normally needs a served endpoint, which
means state, a network and something to keep running.

Everything needed is already in the model metadata: the fitted estimator, the
LabelEncoder maps built during preprocessing, and the scaler when one was used.
Emitting those as a JS scoring function makes the report answer questions on its
own — offline, from a file:// URL, with nothing running behind it.

Only estimator families whose scoring is a short, exact expression are exported:
linear models, single trees, and tree ensembles. Anything else (SVM with an RBF
kernel, KNN, which need the training set itself) reports that it cannot be
embedded rather than shipping an approximation that quietly disagrees with the
real model.
"""

from __future__ import annotations

import json
from typing import Any

# A forest of 100 deep trees serialises into megabytes of JS. Past this many
# nodes the panel is dropped rather than doubling the size of the report.
_MAX_TREE_NODES = 60_000


class ModelNotEmbeddable(Exception):
    """Raised when an estimator has no exact, compact JS equivalent."""


def _tree_arrays(tree: Any) -> dict:
    """Flatten one sklearn tree into the arrays its scoring loop needs."""
    t = tree.tree_
    # value has shape (n_nodes, n_outputs, n_classes); reports are single-output.
    values = [[float(v) for v in node[0]] for node in t.value]
    return {
        "left": [int(v) for v in t.children_left],
        "right": [int(v) for v in t.children_right],
        "feature": [int(v) for v in t.feature],
        "threshold": [float(v) for v in t.threshold],
        "value": values,
    }


def _node_count(trees: list[dict]) -> int:
    return sum(len(t["left"]) for t in trees)


def extract_model(model: Any, metadata: dict) -> dict:
    """Return a JSON-serialisable description of `model`'s scoring rule.

    Raises ModelNotEmbeddable when the estimator has no compact exact form.
    """
    name = type(model).__name__
    task = metadata.get("task", "classification")

    if name in ("LinearRegression", "Ridge", "Lasso", "LogisticRegression"):
        coef = model.coef_
        intercept = model.intercept_
        multi = getattr(coef, "ndim", 1) > 1
        return {
            "kind": "linear",
            "task": task,
            "coef": [[float(c) for c in row] for row in coef] if multi else [[float(c) for c in coef]],
            "intercept": [float(v) for v in intercept] if hasattr(intercept, "__len__") else [float(intercept)],
            "logistic": name == "LogisticRegression",
        }

    if name in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        trees = [_tree_arrays(model)]
        if _node_count(trees) > _MAX_TREE_NODES:
            raise ModelNotEmbeddable(f"tree has {_node_count(trees):,} nodes")
        return {"kind": "forest", "task": task, "trees": trees, "normalise": task == "classification"}

    if name in ("RandomForestClassifier", "RandomForestRegressor"):
        trees = [_tree_arrays(est) for est in model.estimators_]
        if _node_count(trees) > _MAX_TREE_NODES:
            raise ModelNotEmbeddable(f"forest has {_node_count(trees):,} nodes across {len(trees)} trees")
        return {"kind": "forest", "task": task, "trees": trees, "normalise": task == "classification"}

    raise ModelNotEmbeddable(f"{name} has no compact exact JavaScript form")


def build_payload(model: Any, metadata: dict) -> dict:
    """Assemble everything the in-page scorer needs, including preprocessing.

    The encoders and scaler matter as much as the estimator: a value typed into
    the page has to travel the same path the training rows did, or the prediction
    is quietly wrong.
    """
    spec = extract_model(model, metadata)
    features = list(metadata.get("feature_columns", []))
    target = metadata.get("target_column", "")
    encoding_map = metadata.get("encoding_map", {}) or {}

    scaler = metadata.get("scaler")
    scaling = None
    if scaler is not None and hasattr(scaler, "mean_"):
        scaling = {
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        }

    # Categorical features become a dropdown of their real training values.
    choices = {name: list(encoding_map[name].keys()) for name in features if name in encoding_map}
    codes = {name: {k: int(v) for k, v in encoding_map[name].items()} for name in features if name in encoding_map}

    target_labels: list[str] = []
    target_key = f"__target__{target}"
    if target_key in encoding_map:
        pairs = sorted(encoding_map[target_key].items(), key=lambda kv: kv[1])
        target_labels = [str(k) for k, _ in pairs]

    return {
        "model": spec,
        "features": features,
        "target": target,
        "choices": choices,
        "codes": codes,
        "scaling": scaling,
        "targetLabels": target_labels,
    }


_SCORER_JS = """
(function(){
  const P = window.__MODEL__;
  if (!P) return;

  function encode(values){
    return P.features.map(function(name, i){
      const raw = values[i];
      const table = P.codes[name];
      if (table) return table[raw] !== undefined ? table[raw] : 0;
      const n = parseFloat(raw);
      return isFinite(n) ? n : 0;
    });
  }

  function scale(row){
    if (!P.scaling) return row;
    return row.map(function(v, i){ return (v - P.scaling.mean[i]) / (P.scaling.scale[i] || 1); });
  }

  function treeValue(tree, row){
    let node = 0;
    while (tree.left[node] !== -1) {
      node = row[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
  }

  function forest(spec, row){
    const total = [];
    spec.trees.forEach(function(tree){
      const leaf = treeValue(tree, row);
      const sum = spec.normalise ? leaf.reduce(function(a, b){ return a + b; }, 0) || 1 : 1;
      leaf.forEach(function(v, i){ total[i] = (total[i] || 0) + v / sum; });
    });
    return total.map(function(v){ return v / spec.trees.length; });
  }

  function linear(spec, row){
    return spec.coef.map(function(coefs, k){
      let acc = spec.intercept[k] || 0;
      for (let i = 0; i < coefs.length; i++) acc += coefs[i] * row[i];
      return spec.logistic ? 1 / (1 + Math.exp(-acc)) : acc;
    });
  }

  function predict(values){
    const row = scale(encode(values));
    const spec = P.model;
    const out = spec.kind === 'linear' ? linear(spec, row) : forest(spec, row);
    // `value` is exact; `label` is rounded for display only. Anything scripting
    // against this should read `value`.
    if (spec.task !== 'classification') return { label: out[0].toPrecision(6), value: out[0], scores: null };

    let scores = out;
    if (spec.kind === 'linear' && out.length === 1) scores = [1 - out[0], out[0]];
    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    const label = P.targetLabels[best] !== undefined ? P.targetLabels[best] : String(best);
    return { label: label, scores: scores };
  }

  // Exposed deliberately: the panel calls it, and so can anything else on the
  // page. window.__mdlPredict(valuesInFeatureOrder) -> {label, scores}.
  window.__mdlPredict = predict;

  if (typeof document === 'undefined') return;
  const form = document.getElementById('mdl-form');
  if (!form) return;
  const outEl = document.getElementById('mdl-out');
  const barsEl = document.getElementById('mdl-bars');

  function run(){
    const values = P.features.map(function(name){
      const el = form.querySelector('[data-feature="' + CSS.escape(name) + '"]');
      return el ? el.value : '';
    });
    let result;
    try { result = predict(values); }
    catch (err) { outEl.textContent = 'error: ' + err.message; return; }

    outEl.textContent = result.label;
    if (!barsEl) return;
    if (!result.scores) { barsEl.innerHTML = ''; return; }
    barsEl.innerHTML = result.scores.map(function(p, i){
      const name = P.targetLabels[i] !== undefined ? P.targetLabels[i] : ('class ' + i);
      const pct = Math.max(0, Math.min(1, p)) * 100;
      return '<div class="mdl-row"><span class="mdl-name">' + name + '</span>' +
             '<span class="mdl-track"><span class="mdl-fill" style="width:' + pct.toFixed(1) + '%"></span></span>' +
             '<span class="mdl-pct">' + pct.toFixed(1) + '%</span></div>';
    }).join('');
  }

  form.addEventListener('input', run);
  form.addEventListener('change', run);
  run();
})();
"""

_PANEL_CSS = """
.mdl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(13rem,1fr));gap:.75rem;margin-bottom:1rem}
.mdl-field{display:flex;flex-direction:column;gap:.25rem}
.mdl-field label{font-size:.75rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
.mdl-field input,.mdl-field select{background:var(--card);color:var(--text);border:1px solid var(--border);
  border-radius:.375rem;padding:.4rem .5rem;font-size:.875rem;font-family:inherit}
.mdl-result{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem .75rem;padding:.9rem 1rem;
  background:var(--card);border:1px solid var(--border);border-radius:.5rem;margin-bottom:.75rem}
.mdl-result .lbl{font-size:.75rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
.mdl-result .val{font-size:1.5rem;font-weight:600;color:var(--accent);
  min-width:0;overflow-wrap:anywhere}
.mdl-row{display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-size:.8125rem}
.mdl-name{flex:0 1 9rem;min-width:0;color:var(--muted);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mdl-track{flex:1 1 auto;min-width:4rem;height:.5rem;background:var(--border);
  border-radius:.25rem;overflow:hidden}
.mdl-fill{display:block;height:100%;background:var(--accent)}
.mdl-pct{flex:0 0 3.5rem;text-align:right;font-variant-numeric:tabular-nums}
/* On a phone the class name must not take half the row from the bar it labels. */
@media (max-width:30rem){
  .mdl-name{flex:0 1 5.5rem}
  .mdl-pct{flex:0 0 3rem;font-size:.75rem}
  .mdl-result .val{font-size:1.25rem}
}
"""


def _field_html(name: str, payload: dict, defaults: dict) -> str:
    import html as _html

    safe = _html.escape(name)
    value = defaults.get(name, "")
    options = payload["choices"].get(name)
    if options:
        opts = "".join(
            f'<option value="{_html.escape(str(o))}"{" selected" if str(o) == str(value) else ""}>'
            f"{_html.escape(str(o))}</option>"
            for o in options
        )
        control = f'<select data-feature="{safe}">{opts}</select>'
    else:
        control = f'<input type="number" step="any" data-feature="{safe}" value="{_html.escape(str(value))}">'
    return f'<div class="mdl-field"><label>{safe}</label>{control}</div>'


def prediction_panel(payload: dict, defaults: dict | None = None) -> tuple[str, str]:
    """Return (section_html, script_html) for a working prediction panel."""
    defaults = defaults or {}
    fields = "".join(_field_html(name, payload, defaults) for name in payload["features"])
    target = payload["target"] or "prediction"
    body = (
        f"<style>{_PANEL_CSS}</style>"
        "<p>Change any value to see the model's answer update. Everything runs in this "
        "page — no server, no network.</p>"
        f'<form id="mdl-form" class="mdl-grid" onsubmit="return false">{fields}</form>'
        f'<div class="mdl-result"><span class="lbl">{target}</span>'
        '<span class="val" id="mdl-out">—</span></div>'
        '<div id="mdl-bars"></div>'
    )
    script = f"<script>window.__MODEL__={json.dumps(payload, separators=(',', ':'))};{_SCORER_JS}</script>"
    return body, script
