"""The embedded scorer must agree with sklearn, not merely run.

A prediction panel that renders but disagrees with the model it claims to be is
worse than no panel: it looks authoritative and is wrong. These tests execute the
generated JavaScript in node where it is available and compare its answers
against the fitted estimator's own, row for row.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shared.model_js import (
    ModelNotEmbeddable,
    build_payload,
    extract_model,
    prediction_panel,
)

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


@pytest.fixture()
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "spend": rng.uniform(0, 100, n).round(3),
            "clicks": rng.integers(0, 50, n),
            "channel": rng.choice([0, 1, 2], n),
            "label": rng.integers(0, 2, n),
        }
    )


def _js_predict(payload: dict, rows: list[list]) -> list:
    """Run the emitted scorer in node and return its prediction per row."""
    _, script = prediction_panel(payload)
    body = script.replace("<script>", "").replace("</script>", "")
    # The emitted scorer publishes window.__mdlPredict and skips its DOM wiring
    # when there is no document, so it runs unmodified under node.
    program = (
        "global.window = global;\n"
        f"{body}\n"
        f"const rows = {json.dumps(rows)};\n"
        "console.log(JSON.stringify(rows.map(r => window.__mdlPredict(r))));\n"
    )
    return _run_node(program)


def _run_node(program: str) -> list:
    """Run `program` under node and parse its JSON output.

    Written to a file rather than passed with -e: an embedded forest is far past
    the Windows command-line length limit. Decoded as UTF-8 explicitly, since the
    default codec there is cp1252 and would mangle any non-ASCII output.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "run.js"
        script.write_text(program, encoding="utf-8")
        out = subprocess.run([NODE, str(script)], capture_output=True, timeout=120)
    if out.returncode != 0:
        raise AssertionError(f"node failed: {out.stderr.decode('utf-8', 'replace')[:500]}")
    return json.loads(out.stdout.decode("utf-8"))


class TestWhatCanBeEmbedded:
    def test_linear_regression(self, frame):
        model = LinearRegression().fit(frame[["spend", "clicks"]], frame["label"])
        assert extract_model(model, {"task": "regression"})["kind"] == "linear"

    def test_random_forest(self, frame):
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(frame[["spend", "clicks"]], frame["label"])
        assert extract_model(model, {"task": "classification"})["kind"] == "forest"

    def test_svm_is_refused_rather_than_approximated(self, frame):
        """An RBF SVM needs its support vectors; there is no short exact form.
        Shipping an approximation would silently disagree with the real model."""
        model = SVC().fit(frame[["spend", "clicks"]], frame["label"])
        with pytest.raises(ModelNotEmbeddable):
            extract_model(model, {"task": "classification"})

    def test_refusal_names_the_estimator(self, frame):
        model = SVC().fit(frame[["spend", "clicks"]], frame["label"])
        with pytest.raises(ModelNotEmbeddable, match="SVC"):
            extract_model(model, {"task": "classification"})

    def test_an_oversized_forest_is_refused(self, frame):
        """A deep 100-tree forest serialises to megabytes of JS; the panel is
        dropped rather than doubling the size of the report."""
        import shared.model_js as mj

        big = RandomForestClassifier(n_estimators=40, random_state=0).fit(frame[["spend", "clicks"]], frame["label"])
        original = mj._MAX_TREE_NODES
        mj._MAX_TREE_NODES = 10
        try:
            with pytest.raises(ModelNotEmbeddable, match="nodes"):
                extract_model(big, {"task": "classification"})
        finally:
            mj._MAX_TREE_NODES = original


class TestPayloadCarriesPreprocessing:
    def test_categorical_choices_come_from_training_values(self, frame):
        model = RandomForestClassifier(n_estimators=3, random_state=0).fit(frame[["spend", "channel"]], frame["label"])
        payload = build_payload(
            model,
            {
                "task": "classification",
                "feature_columns": ["spend", "channel"],
                "target_column": "label",
                "encoding_map": {"channel": {"email": 0, "search": 1, "social": 2}},
            },
        )
        assert payload["choices"]["channel"] == ["email", "search", "social"]
        assert payload["codes"]["channel"]["social"] == 2

    def test_target_labels_are_ordered_by_code(self, frame):
        model = RandomForestClassifier(n_estimators=3, random_state=0).fit(frame[["spend", "clicks"]], frame["label"])
        payload = build_payload(
            model,
            {
                "task": "classification",
                "feature_columns": ["spend", "clicks"],
                "target_column": "label",
                "encoding_map": {"__target__label": {"yes": 1, "no": 0}},
            },
        )
        assert payload["targetLabels"] == ["no", "yes"]

    def test_scaler_is_carried_so_inputs_take_the_training_path(self, frame):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(frame[["spend", "clicks"]])
        model = LinearRegression().fit(scaler.transform(frame[["spend", "clicks"]]), frame["label"])
        payload = build_payload(
            model,
            {
                "task": "regression",
                "feature_columns": ["spend", "clicks"],
                "target_column": "label",
                "scaler": scaler,
            },
        )
        assert payload["scaling"]["mean"] == pytest.approx(list(scaler.mean_))


@needs_node
class TestJsAgreesWithSklearn:
    """The whole point: the page's answer must be the model's answer."""

    def _payload(self, model, features, task, target_labels=None):
        meta = {
            "task": task,
            "feature_columns": features,
            "target_column": "label",
            "encoding_map": {"__target__label": target_labels} if target_labels else {},
        }
        return build_payload(model, meta)

    def test_random_forest_classifier_matches(self, frame):
        features = ["spend", "clicks"]
        model = RandomForestClassifier(n_estimators=8, random_state=0).fit(frame[features], frame["label"])
        payload = self._payload(model, features, "classification", {"no": 0, "yes": 1})
        rows = frame[features].head(25).values.tolist()
        js = _js_predict(payload, [[str(v) for v in r] for r in rows])
        expected = model.predict(np.array(rows))
        labels = ["no", "yes"]
        assert [r["label"] for r in js] == [labels[int(e)] for e in expected]

    def test_forest_probabilities_match(self, frame):
        features = ["spend", "clicks"]
        model = RandomForestClassifier(n_estimators=8, random_state=0).fit(frame[features], frame["label"])
        payload = self._payload(model, features, "classification", {"no": 0, "yes": 1})
        rows = frame[features].head(10).values.tolist()
        js = _js_predict(payload, [[str(v) for v in r] for r in rows])
        expected = model.predict_proba(np.array(rows))
        for got, want in zip(js, expected):
            assert got["scores"] == pytest.approx(list(want), abs=1e-9)

    def test_decision_tree_matches(self, frame):
        features = ["spend", "clicks"]
        model = DecisionTreeClassifier(random_state=0, max_depth=6).fit(frame[features], frame["label"])
        payload = self._payload(model, features, "classification", {"no": 0, "yes": 1})
        rows = frame[features].head(25).values.tolist()
        js = _js_predict(payload, [[str(v) for v in r] for r in rows])
        labels = ["no", "yes"]
        assert [r["label"] for r in js] == [labels[int(e)] for e in model.predict(np.array(rows))]

    def test_linear_regression_matches(self, frame):
        features = ["spend", "clicks"]
        model = LinearRegression().fit(frame[features], frame["spend"] * 2 + frame["clicks"])
        payload = self._payload(model, features, "regression")
        rows = frame[features].head(15).values.tolist()
        js = _js_predict(payload, [[str(v) for v in r] for r in rows])
        expected = model.predict(np.array(rows))
        for got, want in zip(js, expected):
            assert got["value"] == pytest.approx(float(want), rel=1e-12)

    def test_logistic_regression_matches(self, frame):
        features = ["spend", "clicks"]
        model = LogisticRegression(max_iter=500).fit(frame[features], frame["label"])
        payload = self._payload(model, features, "classification", {"no": 0, "yes": 1})
        rows = frame[features].head(15).values.tolist()
        js = _js_predict(payload, [[str(v) for v in r] for r in rows])
        labels = ["no", "yes"]
        assert [r["label"] for r in js] == [labels[int(e)] for e in model.predict(np.array(rows))]

    def test_random_forest_regressor_matches(self, frame):
        features = ["spend", "clicks"]
        model = RandomForestRegressor(n_estimators=8, random_state=0).fit(frame[features], frame["spend"])
        payload = self._payload(model, features, "regression")
        rows = frame[features].head(15).values.tolist()
        js = _js_predict(payload, [[str(v) for v in r] for r in rows])
        expected = model.predict(np.array(rows))
        for got, want in zip(js, expected):
            assert got["value"] == pytest.approx(float(want), rel=1e-12)
            # label is display-rounded; value is what anything scripting reads
            assert float(got["label"]) == pytest.approx(got["value"], rel=1e-5)

    def test_categorical_input_takes_the_encoding_path(self, frame):
        """A label typed into the page must become the same integer the training
        row did, or the prediction is quietly wrong."""
        features = ["spend", "channel"]
        model = RandomForestClassifier(n_estimators=8, random_state=0).fit(frame[features], frame["label"])
        meta = {
            "task": "classification",
            "feature_columns": features,
            "target_column": "label",
            "encoding_map": {
                "channel": {"email": 0, "search": 1, "social": 2},
                "__target__label": {"no": 0, "yes": 1},
            },
        }
        payload = build_payload(model, meta)
        names = ["email", "search", "social"]
        rows = frame[features].head(20).values.tolist()
        js = _js_predict(payload, [[str(r[0]), names[int(r[1])]] for r in rows])
        labels = ["no", "yes"]
        assert [r["label"] for r in js] == [labels[int(e)] for e in model.predict(np.array(rows))]
