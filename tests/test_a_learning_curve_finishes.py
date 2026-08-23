"""A learning curve on the reference dataset must return, and say what it used.

learning_curve() fits the estimator train_sizes x cv times -- ten sizes against
five folds is fifty fits. On the full 16,834-row reference dataset with a 50-tree
forest a coverage sweep got `MCP error -32001`: the request timed out and the
tool returned nothing at all. It was also the only call in this repo asking for
`n_jobs=-1`, in a 1 GiB container with four CPUs where every worker holds its own
copy of the data -- the same shape as the sklearn working_memory default that
OOM-killed all three ML sub-servers during an earlier sweep.

Every sibling here already bounds its work: silhouette scoring samples to 10,000
rows, clustering switches to MiniBatchKMeans above 50,000, profiling takes
sample_rows. This one did not.

The sampling is the fix and the disclosure is half of it -- a score computed on
a sample is not the score on the file, and nothing in the chart says which.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.platform_utils import get_fit_n_jobs, get_learning_curve_row_cap  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ad_data_full.csv"


@pytest.fixture(scope="module")
def csv(tmp_path_factory) -> str:
    import shutil

    dst = tmp_path_factory.mktemp("lc") / "data.csv"
    shutil.copy2(FIXTURE, dst)
    return str(dst)


class TestTheWorkIsBounded:
    def test_no_call_asks_for_every_core(self):
        import inspect

        from servers.ml_advanced import _adv_viz

        src = inspect.getsource(_adv_viz)
        assert "n_jobs=-1" not in src, "n_jobs=-1 takes every core in a 1 GiB container"

    def test_the_worker_count_is_small(self):
        assert 1 <= get_fit_n_jobs() <= 2, get_fit_n_jobs()

    def test_the_row_cap_is_below_the_reference_dataset(self):
        assert get_learning_curve_row_cap() < 16_834


class TestItReturnsOnTheFullFixture:
    def test_it_finishes_and_reports_the_sample(self, csv, tmp_path):
        from servers.ml_advanced.engine import plot_learning_curve

        started = time.time()
        r = plot_learning_curve(
            csv,
            target_column="device",
            model="rf",
            task="classification",
            output_path=str(tmp_path / "lc.html"),
            open_after=False,
        )
        elapsed = time.time() - started
        assert r["success"] is True, r.get("error")
        assert r["rows_in_file"] == 16834, r
        assert r["sampled"] is True
        assert r["rows_used"] == get_learning_curve_row_cap()
        assert elapsed < 120, f"took {elapsed:.0f}s -- still too slow to return over MCP"

    def test_the_hint_says_the_scores_are_from_a_sample(self, csv, tmp_path):
        from servers.ml_advanced.engine import plot_learning_curve

        r = plot_learning_curve(
            csv,
            target_column="device",
            model="rf",
            task="classification",
            output_path=str(tmp_path / "lc2.html"),
            open_after=False,
        )
        assert "sample" in r["hint"].lower(), r.get("hint")
        assert str(r["rows_used"]) in r["hint"]

    def test_the_chart_is_written(self, csv, tmp_path):
        from servers.ml_advanced.engine import plot_learning_curve

        out = tmp_path / "lc3.html"
        r = plot_learning_curve(
            csv,
            target_column="device",
            model="rf",
            task="classification",
            output_path=str(out),
            open_after=False,
        )
        assert r["success"] is True, r.get("error")
        assert Path(r["output_path"]).exists()
        assert Path(r["output_path"]).stat().st_size > 1000

    def test_a_small_file_is_not_sampled(self, tmp_path):
        import pandas as pd

        from servers.ml_advanced.engine import plot_learning_curve

        small = tmp_path / "small.csv"
        pd.read_csv(FIXTURE, nrows=400).to_csv(small, index=False)
        r = plot_learning_curve(
            str(small),
            target_column="device",
            model="rf",
            task="classification",
            output_path=str(tmp_path / "lc4.html"),
            open_after=False,
        )
        assert r["success"] is True, r.get("error")
        assert r["sampled"] is False
        assert r["rows_used"] == r["rows_in_file"]
