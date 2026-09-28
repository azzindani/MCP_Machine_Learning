"""run_clustering names the column its labels went into; the cluster report says what columns there are.

The sweep clustered, then asked generate_cluster_report for label_column
"cluster". run_clustering had written `cluster_label` and never said so, and
the refusal was "Use run_clustering(save_labels=True) first" -- which had just
been done. The answer now names the column and hands it forward, and the
refusal lists the file's columns.
"""

from __future__ import annotations

import pytest

from servers.ml_advanced._adv_viz import generate_cluster_report
from servers.ml_medium._medium_cluster import run_clustering


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_DATA_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def clustered(home, clustering_simple):
    r = run_clustering(
        str(clustering_simple), ["x", "y"], algorithm="kmeans", n_clusters=3, output_path=str(home / "k.csv")
    )
    assert r["success"] is True, r
    return r


def test_the_label_column_is_named_and_handed_forward(home, clustered):
    assert clustered["label_column"] == "cluster_label"
    carried = clustered["handover"]["carry_forward"]
    assert carried == {"file_path": str(home / "k.csv"), "feature_columns": ["x", "y"], "label_column": "cluster_label"}
    report = generate_cluster_report(**carried, output_path=str(home / "c.html"), open_after=False)
    assert report["success"] is True, report


def test_a_wrong_label_column_is_answered_with_the_columns(home, clustered):
    r = generate_cluster_report(str(home / "k.csv"), ["x", "y"], "cluster", open_after=False)
    assert r["success"] is False
    assert "Its columns: x, y, cluster_label." in r["error"]
    assert "label_column='cluster_label'" in r["hint"]


def test_a_file_never_clustered_is_sent_to_run_clustering(home, clustering_simple):
    r = generate_cluster_report(str(clustering_simple), ["x", "y"], "cluster", open_after=False)
    assert r["success"] is False
    assert "Its columns: x, y." in r["error"]
    assert r["hint"].startswith("Run run_clustering")


def test_nothing_written_names_no_column(home, clustering_simple):
    r = run_clustering(str(clustering_simple), ["x", "y"], algorithm="kmeans", n_clusters=3)
    assert r["label_column"] == "" and "label_column" not in r["handover"]["carry_forward"]
