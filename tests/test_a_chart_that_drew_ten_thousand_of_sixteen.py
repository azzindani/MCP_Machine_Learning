"""The reply said 16,834 points. The file held exactly 10,000.

    plot_predictions_vs_actual(...)
      -> n_points: 16834
         the written HTML: one "Predictions" trace, 10,000 x/y values

    generate_cluster_report(...)
      -> n_samples: 16834
         the PCA scatter: 1024 + 3709 + 4371 + 896 = 10,000 points

Both thin the scatter to 10,000 markers, which is right -- more than that is an
unreadable smear, and the thinning is seeded so it is reproducible. What was
wrong is that neither said so where a caller reads numbers. The cluster report
put "(sampled 10,000/16,834)" in the chart title, so someone looking at the
picture could tell; its response said `n_samples: 16834` and nothing else, so
someone reading the reply could not. plot_predictions_vs_actual said nothing
anywhere.

The third sibling had it right the whole time. plot_learning_curve reports
`rows_used`, `rows_in_file` and `sampled`, above a comment reading "the caller
cannot tell from the chart. Say it in the payload, not only in the progress
log." Two of the three tools that sample did not follow it.

The distinction that decides the field names: plot_learning_curve samples
before it *computes*, so its scores are a sample's scores. These two compute
over every row and thin only the drawing. Calling that `sampled` would tell a
caller the R-squared came from 10,000 rows, which is false. Hence
`plot_sampled` / `scatter_sampled`, with the metrics untouched.

Found by round 15's axis: decode the trace and count what is actually in it.
The reply is self-consistent either way -- only the file disagrees.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from servers.ml_advanced import engine as ma
from servers.ml_basic import engine as mb

CAP = 10_000


def readable(html_path: Path) -> str:
    """The page with plotly's JSON escapes turned back into characters.

    A chart title is written inside plotly's JSON payload, where '/' becomes
    \\u002f and an em dash becomes \\u2014 -- so grepping the file for the title
    a human would read finds nothing, and a test written that way fails against
    correct code.
    """
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), raw)


def points_in(html_path: Path, trace_name: str = "") -> int:
    """Count the x-values plotly actually wrote, across every scatter trace.

    Plotly encodes long numeric arrays as base64 `bdata`, so the values are not
    greppable as text; the length is read from the JSON instead.
    """
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    total = 0
    for blob in re.findall(r"Plotly\.newPlot\(\s*[^,]+,\s*(\[.*?\])\s*,\s*\{", html, re.S):
        try:
            traces = json.loads(blob)
        except json.JSONDecodeError:
            continue
        for tr in traces:
            if tr.get("mode") not in ("markers", "markers+lines"):
                continue
            if trace_name and tr.get("name") != trace_name:
                continue
            x = tr.get("x")
            if isinstance(x, list):
                total += len(x)
            elif isinstance(x, dict) and "bdata" in x:
                import base64

                width = {"f8": 8, "f4": 4, "i8": 8, "i4": 4}.get(x.get("dtype", "f8"), 8)
                total += len(base64.b64decode(x["bdata"])) // width
    return total


@pytest.fixture(scope="module")
def big_regression(tmp_path_factory) -> Path:
    """More rows than the 10,000 cap, so the thinning actually fires."""
    rng = np.random.RandomState(0)
    n = 16_834
    x1 = rng.normal(50, 12, n)
    x2 = rng.normal(5, 2, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": 2.5 * x1 + 4 * x2 + rng.normal(0, 3, n)})
    path = tmp_path_factory.mktemp("big") / "big.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def trained(big_regression: Path, tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("m") / "m.pkl"
    r = mb.train_regressor(str(big_regression), target_column="target", model="lir", output_path=str(out))
    assert r["success"] is True, r.get("error")
    return r["model_path"]


class TestPredictionsVsActual:
    def test_the_reply_says_how_many_it_drew(self, big_regression: Path, trained: str, tmp_path: Path) -> None:
        out = tmp_path / "pva.html"
        r = ma.plot_predictions_vs_actual(trained, str(big_regression), output_path=str(out), open_after=False)
        assert r["success"] is True, r.get("error")
        assert r["plot_sampled"] is True, r
        assert r["points_plotted"] == CAP, r["points_plotted"]

    def test_that_number_is_what_the_file_holds(self, big_regression: Path, trained: str, tmp_path: Path) -> None:
        """The check the sweep ran by hand."""
        out = tmp_path / "pva.html"
        r = ma.plot_predictions_vs_actual(trained, str(big_regression), output_path=str(out), open_after=False)
        assert points_in(out, "Predictions") == r["points_plotted"]

    def test_the_metrics_are_still_over_every_row(self, big_regression: Path, trained: str, tmp_path: Path) -> None:
        """The reason this is not called `sampled`."""
        out = tmp_path / "pva.html"
        r = ma.plot_predictions_vs_actual(trained, str(big_regression), output_path=str(out), open_after=False)
        assert r["n_points"] == 16_834, r["n_points"]
        assert r["n_points"] > r["points_plotted"]

    def test_the_hint_separates_the_two(self, big_regression: Path, trained: str, tmp_path: Path) -> None:
        out = tmp_path / "pva.html"
        r = ma.plot_predictions_vs_actual(trained, str(big_regression), output_path=str(out), open_after=False)
        hint = r.get("hint", "")
        assert "10,000" in hint and "16,834" in hint, hint
        assert "over all 16,834 rows" in hint, hint

    def test_the_chart_title_says_it_too(self, big_regression: Path, trained: str, tmp_path: Path) -> None:
        out = tmp_path / "pva.html"
        ma.plot_predictions_vs_actual(trained, str(big_regression), output_path=str(out), open_after=False)
        assert "drew 10,000/16,834" in readable(out)


class TestASmallDatasetIsNotMarkedSampled:
    """The flag must fire on thinning, not on every call."""

    def test_flag_is_false(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(1)
        n = 400
        df = pd.DataFrame({"x1": rng.normal(0, 1, n)})
        df["target"] = 3 * df["x1"] + rng.normal(0, 0.1, n)
        src = tmp_path / "small.csv"
        df.to_csv(src, index=False)
        m = mb.train_regressor(str(src), target_column="target", model="lir", output_path=str(tmp_path / "s.pkl"))
        out = tmp_path / "small.html"
        r = ma.plot_predictions_vs_actual(m["model_path"], str(src), output_path=str(out), open_after=False)
        assert r["success"] is True, r.get("error")
        assert r["plot_sampled"] is False, r
        assert r["points_plotted"] == r["n_points"] == n
        assert "drew" not in readable(out)[:4000]

    def test_no_sampling_hint(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(2)
        n = 300
        df = pd.DataFrame({"x1": rng.normal(0, 1, n)})
        df["target"] = df["x1"] * 2
        src = tmp_path / "s2.csv"
        df.to_csv(src, index=False)
        m = mb.train_regressor(str(src), target_column="target", model="lir", output_path=str(tmp_path / "s2.pkl"))
        r = ma.plot_predictions_vs_actual(
            m["model_path"], str(src), output_path=str(tmp_path / "s2.html"), open_after=False
        )
        assert "draws" not in r.get("hint", "")


class TestClusterReport:
    @pytest.fixture()
    def labelled(self, big_regression: Path, tmp_path: Path) -> Path:
        df = pd.read_csv(big_regression)
        df["cluster"] = (df.index % 4).astype(int)
        path = tmp_path / "labelled.csv"
        df.to_csv(path, index=False)
        return path

    def test_the_reply_says_how_many_the_scatter_drew(self, labelled: Path, tmp_path: Path) -> None:
        out = tmp_path / "cl.html"
        r = ma.generate_cluster_report(
            str(labelled), feature_columns=["x1", "x2"], label_column="cluster", output_path=str(out), open_after=False
        )
        assert r["success"] is True, r.get("error")
        assert r["scatter_sampled"] is True, r
        assert r["scatter_points_plotted"] == CAP, r["scatter_points_plotted"]

    def test_that_number_is_what_the_file_holds(self, labelled: Path, tmp_path: Path) -> None:
        out = tmp_path / "cl.html"
        r = ma.generate_cluster_report(
            str(labelled), feature_columns=["x1", "x2"], label_column="cluster", output_path=str(out), open_after=False
        )
        assert points_in(out) == r["scatter_points_plotted"], points_in(out)

    def test_the_clustered_total_is_unchanged(self, labelled: Path, tmp_path: Path) -> None:
        r = ma.generate_cluster_report(
            str(labelled),
            feature_columns=["x1", "x2"],
            label_column="cluster",
            output_path=str(tmp_path / "cl.html"),
            open_after=False,
        )
        assert r["n_samples"] == 16_834
        assert r["n_samples"] > r["scatter_points_plotted"]

    def test_the_title_note_it_always_had_is_still_there(self, labelled: Path, tmp_path: Path) -> None:
        out = tmp_path / "cl.html"
        ma.generate_cluster_report(
            str(labelled), feature_columns=["x1", "x2"], label_column="cluster", output_path=str(out), open_after=False
        )
        assert "sampled 10,000/16,834" in readable(out)


class TestTheCounterWouldNoticeAnEmptyChart:
    """A reader that always returned 0 would make every check above vacuous."""

    def test_it_counts_a_known_number_of_points(self, tmp_path: Path) -> None:
        import plotly.graph_objects as go

        out = tmp_path / "probe.html"
        fig = go.Figure(go.Scatter(x=list(range(37)), y=list(range(37)), mode="markers", name="Predictions"))
        fig.write_html(str(out))
        assert points_in(out, "Predictions") == 37
