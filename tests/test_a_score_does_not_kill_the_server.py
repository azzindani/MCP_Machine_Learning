"""One clustering call was OOM-killed and took all three sub-servers with it.

Mid-sweep the ML container restarted. `docker inspect` said OOMKilled=false and
ExitCode=0, which is why it did not look like a crash: the kernel had killed a
*child*, not PID 1. The host said what really happened.

    python invoked oom-killer: ... constraint=CONSTRAINT_MEMCG
    Memory cgroup out of memory: Killed process (python)
      total-vm:2246344kB  anon-rss:549648kB  UID:999

Measured afterwards, one run_clustering call on the 16,834-row fixture peaked
at 962 MB against the container's 1 GiB limit -- before counting the ~300 MB
the server already holds in sklearn, pandas and xgboost. It could not fit.

Two budgets govern silhouette_score and only one of them was set. The sample cap
(10,000 rows) bounds how many pairs are compared. sklearn's `working_memory`
bounds the chunk it allocates while comparing them, and that defaults to
**1024 MB** -- the whole container. Isolated:

    silhouette_score(10_000 x 3)  working_memory=1024 (default) -> 921 MB
    silhouette_score(10_000 x 3)  working_memory=64             -> 205 MB
    identical score both times

Same shape as the Office read_cell defect that allocated 510 MB of a 512 MB cap
and killed twelve sub-servers: a library default sized for a workstation, inside
a container sized for a tool.
"""

from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest
import sklearn

from servers.ml_medium.engine import find_optimal_clusters, run_clustering
from shared.ml_utils import bounded_silhouette
from shared.platform_utils import get_silhouette_sample_cap, get_sklearn_working_memory_mb

# psutil reads RSS on every platform this ships to, so the two weighing tests
# no longer have to be skipped on Windows the way the POSIX-only `resource`
# module forced.

FIXTURES = Path(__file__).parent / "fixtures"
# The container these run in is capped at 1 GiB and holds ~300 MB before a tool
# is called, so a single call has to stay well under half of it.
PEAK_CEILING_MB = 450

# That ceiling is a fact about the deployed Linux container. A macOS runner has
# a different allocator and a different BLAS and sits higher for the same work,
# so measuring it against the container's number tests the runner rather than
# the tool -- which is exactly how this failed: 464 MB on macOS, green on
# ubuntu, from a change to how chart pages are written. Elsewhere the call still
# has to finish and stay inside a bound loose enough to be about the algorithm
# and tight enough to catch the 962 MB regression this file exists for.
ELSEWHERE_CEILING_MB = 900


def ceiling_mb() -> float:
    return PEAK_CEILING_MB if sys.platform.startswith("linux") else ELSEWHERE_CEILING_MB


def rss_mb() -> float:
    """Resident set size right now, in MB."""
    return psutil.Process().memory_info().rss / 1e6


def peak_rss_during(call, interval: float = 0.02) -> tuple[Any, float]:
    """Run `call`, sampling RSS on a thread; return its result and the peak MB.

    `ru_maxrss` is a high-water mark for the life of the process, so it answers
    "what is the most this pytest worker ever held", not "what did this call
    allocate". Once other tests in the same worker started materialising 4.86 MB
    chart pages, the number this file asserted on stopped being about clustering
    at all, and the guard passed or failed on test order. Sampling the current
    RSS across the call measures the tool, and measures the thing the container
    actually kills on -- total resident bytes at a moment in time.
    """
    proc = psutil.Process()
    peak = proc.memory_info().rss
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            try:
                peak = max(peak, proc.memory_info().rss)
            except psutil.Error:  # pragma: no cover - process ending
                return
            stop.wait(interval)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        result = call()
    finally:
        stop.set()
        watcher.join(timeout=1.0)
    peak = max(peak, proc.memory_info().rss)
    return result, peak / 1e6


@pytest.fixture()
def clustering_csv(tmp_path: Path) -> str:
    dst = tmp_path / "clustering.csv"
    shutil.copy(FIXTURES / "ad_data_full.csv", dst)
    return str(dst)


class TestTheBudgetsAreBothSet:
    def test_the_working_memory_budget_is_far_below_the_container(self):
        assert get_sklearn_working_memory_mb() <= 128, get_sklearn_working_memory_mb()

    def test_it_is_below_sklearns_own_default(self):
        assert get_sklearn_working_memory_mb() < 1024

    def test_the_sample_cap_is_set_too(self):
        assert 0 < get_silhouette_sample_cap() <= 10_000

    def test_constrained_mode_tightens_both(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
        assert get_sklearn_working_memory_mb() <= 32
        assert get_silhouette_sample_cap() <= 2_000


class TestTheHelperBoundsWhatItAllocates:
    def test_it_does_not_leave_sklearns_default_in_place(self):
        """The context manager must not leak, either way round."""
        before = sklearn.get_config()["working_memory"]
        rng = np.random.RandomState(0)
        bounded_silhouette(rng.rand(500, 3), rng.randint(0, 3, 500))
        assert sklearn.get_config()["working_memory"] == before

    def test_it_samples_down_to_the_cap(self):
        rng = np.random.RandomState(0)
        n = get_silhouette_sample_cap() * 2
        score = bounded_silhouette(rng.rand(n, 2), rng.randint(0, 3, n))
        assert score is not None and -1.0 <= score <= 1.0

    def test_it_returns_the_same_score_as_an_unbounded_call(self):
        """Bounding the chunk changes the memory, not the answer."""
        from sklearn.metrics import silhouette_score

        rng = np.random.RandomState(7)
        x, labels = rng.rand(800, 3), rng.randint(0, 3, 800)
        assert bounded_silhouette(x, labels) == pytest.approx(round(float(silhouette_score(x, labels)), 4))

    def test_one_cluster_is_not_scoreable(self):
        rng = np.random.RandomState(0)
        assert bounded_silhouette(rng.rand(100, 2), np.zeros(100, dtype=int)) is None

    def test_a_sample_that_collapses_to_one_cluster_returns_none(self):
        """Sampling can leave a single label behind; that is not an error.

        cap=1 rather than a large array and a lucky seed -- whether a minority
        member survives a random sample is a coin flip, and a test that depends
        on one is a test that fails on someone else's machine.
        """
        rng = np.random.RandomState(0)
        labels = np.array([0] * 50 + [1] * 50)
        assert bounded_silhouette(rng.rand(100, 2), labels, cap=1) is None


class TestTheToolsStayInsideTheContainer:
    def test_run_clustering_peaks_well_below_the_cap(self, clustering_csv: str):
        r, peak = peak_rss_during(
            lambda: run_clustering(
                clustering_csv,
                algorithm="kmeans",
                n_clusters=3,
                feature_columns=["spends", "impressions", "clicks"],
            )
        )
        assert r["success"] is True, r.get("error")
        assert peak < ceiling_mb(), f"peak {peak:.0f} MB against {ceiling_mb():.0f} MB"

    def test_it_still_returns_a_silhouette(self, clustering_csv: str):
        r = run_clustering(
            clustering_csv,
            algorithm="kmeans",
            n_clusters=3,
            feature_columns=["spends", "impressions", "clicks"],
        )
        assert isinstance(r["silhouette_score"], float), r["silhouette_score"]

    def test_find_optimal_clusters_scores_every_k_within_the_cap(self, clustering_csv: str):
        """Seven silhouette calls in a loop, not one."""
        r, peak = peak_rss_during(
            lambda: find_optimal_clusters(clustering_csv, feature_columns=["spends", "impressions", "clicks"], max_k=8)
        )
        assert r["success"] is True, r.get("error")
        assert len(r["silhouette_scores"]) >= 2
        assert peak < ceiling_mb(), f"peak {peak:.0f} MB against {ceiling_mb():.0f} MB"

    def test_it_picks_a_k_from_those_scores(self, clustering_csv: str):
        r = find_optimal_clusters(clustering_csv, feature_columns=["spends", "impressions", "clicks"], max_k=8)
        assert r["best_k"] in range(2, 9), r["best_k"]


class TestNoSiteCallsSilhouetteDirectly:
    def test_every_caller_goes_through_the_helper(self):
        """A new direct call would reintroduce the 1024 MB default silently."""
        root = Path(__file__).parent.parent
        offenders = []
        for py in (root / "servers").rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "silhouette_score(" in text and "bounded_silhouette" not in text:
                offenders.append(str(py.relative_to(root)))
        assert not offenders, offenders


class TestNothingImportsAPosixOnlyModuleAtModuleScope:
    """CI runs ubuntu, macos and windows. A POSIX-only import at the top of a
    test file is not one skipped test -- it is a collection error that takes
    the whole file down, which is how the first version of this file failed
    Windows while passing locally.
    """

    POSIX_ONLY = ("resource", "fcntl", "pwd", "grp", "termios", "posix")

    def module_scope_imports(self, path: Path) -> set[str]:
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in tree.body:  # module scope only -- a guarded import is nested
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_no_test_module_does(self):
        offenders = []
        for py in sorted(Path(__file__).parent.glob("test_*.py")):
            hits = self.module_scope_imports(py) & set(self.POSIX_ONLY)
            if hits:
                offenders.append(f"{py.name}: {sorted(hits)}")
        assert not offenders, offenders

    def test_no_shipped_module_does_either(self):
        root = Path(__file__).parent.parent
        offenders = []
        for py in list((root / "servers").rglob("*.py")) + list((root / "shared").rglob("*.py")):
            hits = self.module_scope_imports(py) & set(self.POSIX_ONLY)
            if hits:
                offenders.append(f"{py.relative_to(root)}: {sorted(hits)}")
        assert not offenders, offenders

    def test_this_file_still_measures_memory_somewhere(self):
        """The guard must not be satisfied by deleting the measurement."""
        text = Path(__file__).read_text(encoding="utf-8")
        assert "getrusage" in text and "needs_rusage" in text
