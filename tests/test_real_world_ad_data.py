"""Integration tests against the full real (not synthetic) Ad_Data.csv dataset
with a derived ctr_pct column — the exact scenario that surfaced two real bugs
during the 2026-08-12 comprehensive real-world tool sweep (read_column_profile
silently nulling stats on +inf, and every training tool crashing on +inf).
See ad_data_full_with_ctr fixture in conftest.py.
"""

from __future__ import annotations

from servers.ml_basic.engine import read_column_profile, train_classifier, train_regressor
from servers.ml_medium.engine import run_clustering


class TestRealWorldInfHandling:
    def test_read_column_profile_reports_inf_not_null(self, ad_data_full_with_ctr):
        r = read_column_profile(str(ad_data_full_with_ctr), "ctr_pct")
        assert r["success"] is True
        profile = r["profile"]
        assert profile["inf_count"] == 4
        assert profile["mean"] is not None
        assert profile["std"] is not None
        assert profile["max"] is not None

    def test_train_classifier_does_not_crash_on_inf_column(self, ad_data_full_with_ctr):
        r = train_classifier(
            str(ad_data_full_with_ctr),
            target_column="campaign_platform",
            model="rf",
        )
        assert r["success"] is True

    def test_train_regressor_does_not_crash_on_inf_column(self, ad_data_full_with_ctr):
        r = train_regressor(
            str(ad_data_full_with_ctr),
            target_column="clicks",
            model="dtr",
        )
        assert r["success"] is True

    def test_run_clustering_does_not_crash_on_null_column(self, ad_data_full_with_ctr):
        """Regression test: run_clustering crashed with a raw sklearn
        "Input X contains NaN" on link_clicks' real null values (546 of
        them in the full dataset) — found training real clusters on the
        full Ad_Data.csv. Median-fill keeps rows aligned for save_labels,
        unlike find_optimal_clusters' dropna() (which has no such
        requirement)."""
        r = run_clustering(
            str(ad_data_full_with_ctr),
            feature_columns=["spends", "impressions", "clicks", "link_clicks"],
            algorithm="kmeans",
            n_clusters=4,
        )
        assert r["success"] is True
        assert "link_clicks" in " ".join(p.get("detail", "") for p in r["progress"])
