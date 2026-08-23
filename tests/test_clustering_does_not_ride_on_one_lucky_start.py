"""K-Means here must not depend on a single initialisation.

sklearn changed the `n_init` default to "auto" in 1.4, which for k-means++ means
**one** start. A coverage sweep recomputing find_optimal_clusters' numbers on the
16,834-row reference dataset found k=6 disagreeing, and the cause was exactly
that: one start against the ten a person naturally recomputes with.

    k   inertia n_init=auto   inertia n_init=10
    2            18975.35            18975.35
    3            10402.21            10401.14
    4             7418.75             7418.68
    5             4951.32             4950.72
    6             4413.60             4149.00      <-- 6% worse

The inertia is what the elbow curve plots, so a bad start bends the curve this
tool exists to draw. The sharper problem is the recommendation: the *worse*
clustering at k=6 scores the *higher* silhouette (0.7845 vs 0.7356), and best_k
is the argmax of the silhouettes -- so an unlucky start can choose the answer.

run_clustering is pinned to the same ten restarts, so the k this tool recommends
and the clustering the next tool produces agree.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "ad_data_full.csv"


class TestTheRestartCountIsPinned:
    @pytest.mark.parametrize(
        "module,func",
        [
            ("servers.ml_medium._medium_data", "find_optimal_clusters"),
            ("servers.ml_medium._medium_cluster", "run_clustering"),
        ],
    )
    def test_n_init_is_never_left_to_the_library(self, module: str, func: str):
        import importlib

        src = inspect.getsource(getattr(importlib.import_module(module), func))
        constructions = [ln for ln in src.splitlines() if "KMeans(" in ln]
        assert constructions, f"no KMeans construction found in {func}"
        # the call may wrap across lines; check the whole source instead
        for ln in constructions:
            block = src[src.index(ln) :]
            end = block.index(")") if ")" in block else len(block)
            assert "n_init" in block[: end + 200], f"{func}: {ln.strip()} leaves n_init to sklearn"


class TestTheAnswerIsStableAcrossRuns:
    @pytest.fixture(scope="class")
    def csv(self, tmp_path_factory) -> str:
        import shutil

        dst = tmp_path_factory.mktemp("cluster") / "data.csv"
        shutil.copy2(FIXTURE, dst)
        return str(dst)

    def test_two_identical_calls_agree(self, csv, tmp_path):
        from servers.ml_medium.engine import find_optimal_clusters

        a = find_optimal_clusters(
            csv, ["spends", "impressions", "clicks"], max_k=6, output_path=str(tmp_path / "a.html"), open_after=False
        )
        b = find_optimal_clusters(
            csv, ["spends", "impressions", "clicks"], max_k=6, output_path=str(tmp_path / "b.html"), open_after=False
        )
        assert a["success"] is True, a.get("error")
        assert a["best_k"] == b["best_k"]
        assert a["inertias"] == b["inertias"]
        assert a["silhouette_scores"] == b["silhouette_scores"]

    def test_the_inertia_matches_ten_restarts(self, csv, tmp_path):
        """The number the elbow plots is the good optimum, not a lucky start."""
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        from servers.ml_medium.engine import find_optimal_clusters

        r = find_optimal_clusters(
            csv, ["spends", "impressions", "clicks"], max_k=6, output_path=str(tmp_path / "c.html"), open_after=False
        )
        assert r["success"] is True, r.get("error")

        df = pd.read_csv(csv)
        xs = StandardScaler().fit_transform(df[["spends", "impressions", "clicks"]].dropna())
        for k, reported in zip(r["k_range"], r["inertias"], strict=True):
            expected = KMeans(n_clusters=k, random_state=42, max_iter=100, n_init=10).fit(xs).inertia_
            assert reported == pytest.approx(expected, rel=0.01), (
                f"k={k}: reported {reported}, ten restarts give {expected:.2f}"
            )
        assert np.isfinite(r["inertias"]).all()
