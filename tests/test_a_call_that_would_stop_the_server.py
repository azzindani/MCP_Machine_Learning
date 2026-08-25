"""Clustering refuses what it cannot afford, and says which columns it used.

Two findings from the round-14 sweep, both in the clustering pair.

**run_clustering(algorithm="dbscan")** on the 16,834-row reference dataset
peaked at ~4.1 GB against a 1 GiB container. The process was OOM-killed and
every tool on the server went down with it; the caller saw only a closed
socket -- the sweep recorded "The socket connection was closed unexpectedly"
and, on the retry, "Streamable HTTP error: Error POSTing to endpoint:". It ran
fine locally, which is why nothing had caught it: the defect only exists
against the deployed memory limit.

DBSCAN materialises every point's eps-neighbourhood at once, so its cost is
O(n x neighbourhood size) -- and on data with many near-identical rows the
neighbourhood grows with n too. Measured: 39 MB at n=2,000, 325 MB at n=8,000,
~4,100 MB at n=16,834. The cap refuses rather than sampling, because sampling
changes which points come back as noise.

**find_optimal_clusters** dropped non-numeric feature columns with
select_dtypes and then checked the ROW count, not the column count -- so a call
naming only categorical columns went past the guard into StandardScaler and
raised "Found array with 0 feature(s)" out of the tool, past the return-value
contract, naming neither the columns nor what to send. A mixed call was quieter
and worse: it clustered on whatever was numeric and the response never said so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from servers.ml_medium.engine import find_optimal_clusters, run_clustering
from shared.platform_utils import get_dbscan_row_cap


@pytest.fixture
def small(tmp_path):
    f = tmp_path / "small.csv"
    rng = range(60)
    pd.DataFrame(
        {
            "device": ["mobile", "desktop", "tablet"] * 20,
            "clicks": [i % 7 for i in rng],
            "spends": [float(i % 11) for i in rng],
        }
    ).to_csv(f, index=False)
    return f


# --- the memory refusal -----------------------------------------------------


def test_dbscan_refuses_a_dataset_it_cannot_hold(tmp_path):
    n = get_dbscan_row_cap() + 1
    f = tmp_path / "big.csv"
    pd.DataFrame({"a": [i % 50 for i in range(n)], "b": [float(i % 30) for i in range(n)]}).to_csv(f, index=False)
    r = run_clustering(str(f), feature_columns=["a", "b"], algorithm="dbscan")
    assert r["success"] is False
    assert f"{n:,}" in r["error"], "the refusal must name the row count it was given"
    assert f"{get_dbscan_row_cap():,}" in r["error"], "and the limit"
    # A refusal that does not say what to do instead just moves the problem.
    assert "kmeans" in r["hint"]
    assert "sample" in r["hint"]


def test_kmeans_has_no_such_limit(tmp_path):
    """The cap is DBSCAN's, not clustering's -- k-means is linear in n."""
    n = get_dbscan_row_cap() + 1
    f = tmp_path / "big.csv"
    pd.DataFrame({"a": [i % 50 for i in range(n)], "b": [float(i % 30) for i in range(n)]}).to_csv(f, index=False)
    r = run_clustering(str(f), feature_columns=["a", "b"], algorithm="kmeans", n_clusters=3)
    assert r["success"] is True, r.get("error")


def test_dbscan_still_runs_under_the_cap(small):
    r = run_clustering(str(small), feature_columns=["clicks", "spends"], algorithm="dbscan", eps=1.0, min_samples=3)
    assert r["success"] is True, r.get("error")
    assert "n_clusters_found" in r


def test_the_cap_tightens_in_constrained_mode(monkeypatch):
    monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
    tight = get_dbscan_row_cap()
    monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)
    assert tight < get_dbscan_row_cap()


# --- naming the features actually used --------------------------------------


def test_all_categorical_features_are_refused_by_name(tmp_path, small):
    r = find_optimal_clusters(str(small), feature_columns=["device"], max_k=3, open_after=False)
    assert r["success"] is False
    assert "device" in r["error"]
    assert "numeric" in r["error"]
    assert "label_encode" in r["hint"], "the hint should name the way out"


def test_a_mixed_call_says_which_columns_it_clustered_on(tmp_path, small):
    r = find_optimal_clusters(str(small), feature_columns=["clicks", "device", "spends"], max_k=3, open_after=False)
    assert r["success"] is True, r.get("error")
    assert r["features_used"] == ["clicks", "spends"]
    assert r["features_skipped"] == ["device"]
    warnings = [p for p in r["progress"] if p.get("icon") == "⚠"]
    assert any("device" in str(p) for p in warnings), r["progress"]


def test_an_all_numeric_call_skips_nothing(tmp_path, small):
    r = find_optimal_clusters(str(small), feature_columns=["clicks", "spends"], max_k=3, open_after=False)
    assert r["success"] is True, r.get("error")
    assert r["features_used"] == ["clicks", "spends"]
    assert r["features_skipped"] == []
    assert not [p for p in r["progress"] if p.get("icon") == "⚠" and "skipped" in str(p)]
