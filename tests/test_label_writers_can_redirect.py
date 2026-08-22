"""run_clustering and anomaly_detection could only label the file they read.

Both take save_labels, which snapshots the caller's dataset and then rewrites it
in place with a new column. That is opt-in and recoverable, so it is not a bug --
but it was the only way to keep the labels at all. Every dataset-writing tool on
the Data_Analyst transform server takes output_path; these two did not, so
"score this file but leave it alone" was impossible.

The rule now:

    neither             analysis only, nothing written  (unchanged default)
    save_labels=True    written in place, snapshot first (unchanged)
    output_path=X       written to X, source untouched, no snapshot
    both                written to X -- the named destination wins

Naming a destination is itself a request to write, so output_path works without
save_labels; requiring both would just be a second switch for one decision.

The wrappers passed save_labels positionally, so inserting output_path before
dry_run would have bound the dry_run bool to output_path -- the same trap that
came with the same fix on run_cleaning_pipeline.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from servers.ml_medium.engine import anomaly_detection, run_clustering


@pytest.fixture()
def csv(tmp_path: Path) -> Path:
    p = tmp_path / "points.csv"
    rng = np.random.default_rng(0)
    n = 90
    pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(8, 1, n // 2)]),
            "y": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(8, 1, n // 2)]),
        }
    ).to_csv(p, index=False)
    return p


FEATURES = ["x", "y"]


class TestClusteringCanRedirect:
    def test_it_writes_the_labels_elsewhere(self, csv: Path, tmp_path: Path):
        out = tmp_path / "labelled.csv"
        r = run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, output_path=str(out))
        assert r["success"] is True, r.get("error")
        assert "cluster_label" in pd.read_csv(out).columns

    def test_the_source_is_untouched(self, csv: Path, tmp_path: Path):
        before = csv.read_bytes()
        run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, output_path=str(tmp_path / "l.csv"))
        assert csv.read_bytes() == before

    def test_no_snapshot_when_the_source_is_not_the_target(self, csv: Path, tmp_path: Path):
        r = run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, output_path=str(tmp_path / "l.csv"))
        assert r["backup"] == ""

    def test_it_reports_where_the_labels_went(self, csv: Path, tmp_path: Path):
        out = tmp_path / "l.csv"
        r = run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, output_path=str(out))
        assert r["output_path"] == str(out)

    def test_output_path_alone_is_enough(self, csv: Path, tmp_path: Path):
        """No save_labels: naming a destination is the request to write."""
        out = tmp_path / "l.csv"
        run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, output_path=str(out))
        assert out.is_file()


class TestClusteringDefaultsAreUnchanged:
    def test_analysis_only_writes_nothing(self, csv: Path):
        before = csv.read_bytes()
        r = run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2)
        assert r["success"] is True, r.get("error")
        assert csv.read_bytes() == before
        assert r["output_path"] == ""

    def test_save_labels_still_writes_in_place(self, csv: Path):
        r = run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, save_labels=True)
        assert r["success"] is True, r.get("error")
        assert "cluster_label" in pd.read_csv(csv).columns

    def test_save_labels_still_snapshots(self, csv: Path):
        r = run_clustering(str(csv), FEATURES, "kmeans", n_clusters=2, save_labels=True)
        assert r["backup"], "an in-place rewrite must stay recoverable"


class TestAnomalyDetectionCanRedirect:
    def test_it_writes_the_scores_elsewhere(self, csv: Path, tmp_path: Path):
        out = tmp_path / "scored.csv"
        r = anomaly_detection(str(csv), FEATURES, output_path=str(out))
        assert r["success"] is True, r.get("error")
        cols = pd.read_csv(out).columns
        assert "anomaly_score" in cols
        assert "is_anomaly" in cols

    def test_the_source_is_untouched(self, csv: Path, tmp_path: Path):
        before = csv.read_bytes()
        anomaly_detection(str(csv), FEATURES, output_path=str(tmp_path / "s.csv"))
        assert csv.read_bytes() == before

    def test_no_snapshot_when_the_source_is_not_the_target(self, csv: Path, tmp_path: Path):
        r = anomaly_detection(str(csv), FEATURES, output_path=str(tmp_path / "s.csv"))
        assert r["backup"] == ""

    def test_analysis_only_writes_nothing(self, csv: Path):
        before = csv.read_bytes()
        r = anomaly_detection(str(csv), FEATURES)
        assert r["success"] is True, r.get("error")
        assert csv.read_bytes() == before
        assert r["output_path"] == ""

    def test_save_labels_still_writes_in_place(self, csv: Path):
        r = anomaly_detection(str(csv), FEATURES, save_labels=True)
        assert r["success"] is True, r.get("error")
        assert "anomaly_score" in pd.read_csv(csv).columns
        assert r["backup"]


class TestTheWrappersDoNotMisbindArguments:
    """Both called the engine positionally with save_labels in the slot before
    dry_run. Inserting output_path there without updating them would have made
    dry_run=True write a file named "True"."""

    @pytest.mark.parametrize("name", ["run_clustering", "anomaly_detection"])
    def test_output_path_precedes_dry_run(self, name: str):
        from servers.ml_medium import server

        fn = getattr(getattr(server, name), "fn", getattr(server, name))
        params = list(inspect.signature(fn).parameters)
        assert params.index("output_path") < params.index("dry_run")

    @pytest.mark.parametrize("name", ["run_clustering", "anomaly_detection"])
    def test_the_wrapper_and_engine_agree(self, name: str):
        from servers.ml_medium import engine, server

        fn = getattr(getattr(server, name), "fn", getattr(server, name))
        assert list(inspect.signature(fn).parameters) == list(inspect.signature(getattr(engine, name)).parameters)

    def test_a_wrapper_dry_run_is_still_a_dry_run(self, csv: Path):
        from servers.ml_medium import server

        fn = getattr(server.run_clustering, "fn", server.run_clustering)
        before = csv.read_bytes()
        r = fn(str(csv), FEATURES, "kmeans", n_clusters=2, dry_run=True)
        assert r.get("dry_run") is True
        assert csv.read_bytes() == before


class TestEveryLabelWriterOffersTheChoice:
    """These two were the gap. Fail if a third appears without it."""

    def test_no_save_labels_tool_lacks_output_path(self):
        from servers.ml_medium import server

        missing = []
        for name in dir(server):
            tool = getattr(server, name)
            fn = getattr(tool, "fn", None)
            if fn is None or not callable(fn) or name.startswith("_"):
                continue
            params = inspect.signature(fn).parameters
            if "save_labels" in params and "output_path" not in params:
                missing.append(name)
        assert not missing, f"these rewrite the caller's dataset with no way to redirect: {missing}"
