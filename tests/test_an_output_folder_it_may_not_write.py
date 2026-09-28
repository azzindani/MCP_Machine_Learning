"""An output folder the server may not write in is an answer, not a crash.

The sweep ran ML against an output folder owned by another user. Eight actions
let the PermissionError escape: the caller read "Error executing tool ...:
[Errno 13] Permission denied" with no success, error or hint. train_regressor
caught it and blamed the data -- "Use inspect_dataset() and read_column_profile()
to verify your data first." Both now answer in the fleet's failure shape, with
a hint naming the folder.

The folder is refused by patching, not chmod: the suite also runs as root,
which chmod does not stop, and on Windows.
"""

from __future__ import annotations

import asyncio
import builtins
import os
import shutil
from pathlib import Path

import pytest

import unified_server  # type: ignore[reportMissingImports]

FIXTURES = Path(__file__).parent / "fixtures"
CLS, REG, CLU = "classification_simple.csv", "regression_simple.csv", "clustering_simple.csv"
CLU_FEATURES = ["x", "y"]


@pytest.fixture
def locked(tmp_path, monkeypatch):
    """A folder whose every write -- mkdir, open for writing, a rename into it -- is refused."""
    # The inputs are copies: a tool files its receipt beside the file it read.
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in (CLS, REG, CLU):
        shutil.copy(FIXTURES / name, inputs / name)
    folder = tmp_path / "locked"
    folder.mkdir()
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(folder))

    def inside(path) -> bool:
        return str(Path(os.fspath(path)).resolve()).startswith(str(folder.resolve()))

    def denied(path):
        return PermissionError(13, "Permission denied", os.fspath(path))

    real_mkdir, real_open, real_os_open = Path.mkdir, builtins.open, os.open
    real_replace, real_mkstemp = os.replace, __import__("tempfile").mkstemp

    def mkdir(self, *a, **kw):
        if inside(self) and not self.exists():
            raise denied(self)
        return real_mkdir(self, *a, **kw)

    def open_(file, mode="r", *a, **kw):
        if isinstance(file, (str, os.PathLike)) and any(c in mode for c in "wax+") and inside(file):
            raise denied(file)
        return real_open(file, mode, *a, **kw)

    def os_open(path, flags, *a, **kw):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT) and inside(path):
            raise denied(path)
        return real_os_open(path, flags, *a, **kw)

    def replace(src, dst, *a, **kw):
        if inside(dst):
            raise denied(dst)
        return real_replace(src, dst, *a, **kw)

    def mkstemp(*a, dir=None, **kw):  # noqa: A002 -- tempfile's own name
        if dir is not None and inside(dir):
            raise denied(dir)
        return real_mkstemp(*a, dir=dir, **kw)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(builtins, "open", open_)
    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr("tempfile.mkstemp", mkstemp)
    return folder


def _call(tier: str, name: str, args: dict) -> dict:
    manager = unified_server._TIERS[tier]._tool_manager
    return asyncio.run(manager.call_tool(name, args, convert_result=False))


CASES = [
    ("basic", "split_dataset", lambda o, i: {"file_path": f"{i}/{CLS}", "test_size": 0.2, "output_dir": f"{o}/split"}),
    (
        "medium",
        "run_preprocessing",
        lambda o, i: {
            "file_path": f"{i}/{CLS}",
            "output_path": f"{o}/clean.csv",
            "ops": [{"op": "drop_column", "column": "tenure"}],
        },
    ),
    (
        "advanced",
        "apply_dimensionality_reduction",
        lambda o, i: {
            "file_path": f"{i}/{REG}",
            "feature_columns": ["age", "experience"],
            "method": "pca",
            "n_components": 2,
            "output_path": f"{o}/pca.csv",
        },
    ),
    (
        "medium",
        "run_clustering",
        lambda o, i: {
            "file_path": f"{i}/{CLU}",
            "feature_columns": CLU_FEATURES,
            "algorithm": "kmeans",
            "n_clusters": 3,
            "save_labels": True,
            "output_path": f"{o}/k.csv",
        },
    ),
    (
        "medium",
        "find_optimal_clusters",
        lambda o, i: {
            "file_path": f"{i}/{CLU}",
            "feature_columns": CLU_FEATURES,
            "max_k": 4,
            "output_path": f"{o}/elbow.html",
        },
    ),
    (
        "medium",
        "anomaly_detection",
        lambda o, i: {
            "file_path": f"{i}/{CLU}",
            "feature_columns": CLU_FEATURES,
            "contamination": 0.05,
            "save_labels": True,
            "output_path": f"{o}/a.csv",
        },
    ),
    (
        "medium",
        "generate_eda_report",
        lambda o, i: {"file_path": f"{i}/{CLS}", "target_column": "churned", "output_path": f"{o}/eda.html"},
    ),
    ("advanced", "run_profiling_report", lambda o, i: {"file_path": f"{i}/{CLS}", "output_path": f"{o}/profile.html"}),
    (
        "basic",
        "train_regressor",
        lambda o, i: {
            "file_path": f"{i}/{REG}",
            "target_column": "salary",
            "model": "lir",
            "output_path": f"{o}/m/reg.pkl",
        },
    ),
]


@pytest.mark.parametrize(("tier", "name", "args"), CASES, ids=[c[1] for c in CASES])
def test_answered_in_the_failure_shape_naming_the_folder(locked, tier, name, args):
    r = _call(tier, name, args(locked, locked.parent / "in"))
    assert isinstance(r, dict), f"{name} escaped the contract: {r!r}"
    assert r["success"] is False, r
    assert r["op"] == name
    assert "may not write in" in r["hint"] and str(locked) in r["hint"], r["hint"]
    assert "verify your data" not in r["hint"]


def test_a_failure_that_was_not_a_permission_keeps_its_own_hint(tmp_path, monkeypatch, regression_simple):
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    r = _call(
        "basic",
        "train_regressor",
        {"file_path": str(regression_simple), "target_column": "no_such_column", "model": "lir"},
    )
    assert r["success"] is False
    assert "may not write in" not in r["hint"]
