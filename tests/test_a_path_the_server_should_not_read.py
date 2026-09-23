"""A remote server reads and writes only inside the folders it serves.

Every tool resolved its path with `Path(raw).resolve()`, so any authenticated
caller of the deployed server could name any file the container could read --
a dataset anywhere, a model file (which is a pickle), or /proc/self/environ
with the API keys. Model paths and several output paths skipped the resolver
entirely. A workspace `base_dir` and a workspace *name* ("../../etc") reached
anywhere too.

With MCP_CONFINE_PATHS on (the default for every HTTP deployment) a path must
lie inside MCP_OUTPUT_DIR, the workspace root, or MCP_ALLOWED_ROOTS, judged
after symlinks resolve; a relative path is read from the data folder. A local
stdio install is unchanged, except that `~` expands and a workspace name is
always a name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "servers" / "ml_medium"), str(ROOT / "servers" / "ml_basic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from servers.ml_basic import engine as basic  # noqa: E402
from shared.file_utils import PathOutsideRootError, resolve_path  # noqa: E402
from shared.workspace_utils import get_workspace_dir, get_workspace_root  # noqa: E402

OUTSIDE = "/etc/hostname" if Path("/etc/hostname").exists() else str(Path(__file__).resolve())


@pytest.fixture
def served(tmp_path, monkeypatch):
    data = tmp_path / "data"
    ws = tmp_path / "ws"
    data.mkdir()
    ws.mkdir()
    monkeypatch.setenv("MCP_CONFINE_PATHS", "1")
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(data))
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(ws))
    monkeypatch.delenv("MCP_DATA_ROOT", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)
    pd.DataFrame({"x": range(20), "y": [0, 1] * 10}).to_csv(data / "train.csv", index=False)
    return data


class TestConfined:
    def test_a_file_outside_is_refused(self, served):
        with pytest.raises(PathOutsideRootError, match="outside the folders"):
            resolve_path(OUTSIDE)

    def test_a_relative_path_is_read_from_the_data_folder(self, served):
        assert resolve_path("train.csv") == (served / "train.csv").resolve()

    def test_climbing_out_is_refused(self, served):
        with pytest.raises(PathOutsideRootError):
            resolve_path("../escape.csv")

    def test_a_symlink_is_judged_by_where_it_leads(self, served, tmp_path):
        outside = tmp_path / "outside.csv"
        outside.write_text("a\n1\n")
        link = served / "innocent.csv"
        try:
            link.symlink_to(outside)
        except OSError, NotImplementedError:
            pytest.skip("cannot create a symlink here")
        with pytest.raises(PathOutsideRootError):
            resolve_path(str(link))

    def test_a_workspace_base_dir_outside_is_refused(self, served, tmp_path):
        with pytest.raises(PathOutsideRootError, match="base_dir"):
            get_workspace_root(str(tmp_path / "elsewhere"))

    def test_the_tool_refuses_and_says_why(self, served):
        r = basic.inspect_dataset(OUTSIDE)
        assert r["success"] is False
        assert "outside the folders" in r["error"]

    def test_the_tool_reads_a_relative_path(self, served):
        r = basic.inspect_dataset("train.csv")
        assert r["success"] is True, r

    def test_a_model_outside_is_never_loaded(self, served, tmp_path):
        model = tmp_path / "elsewhere.pkl"
        model.write_bytes(b"not a model")
        r = basic.predict_single(str(model), '{"x": 1}')
        assert r["success"] is False
        assert "outside the folders" in str(r)


class TestLocal:
    def test_a_local_install_is_not_confined(self, monkeypatch):
        monkeypatch.delenv("MCP_CONFINE_PATHS", raising=False)
        assert resolve_path(OUTSIDE) == Path(OUTSIDE).resolve()

    def test_home_is_expanded(self, monkeypatch):
        monkeypatch.delenv("MCP_CONFINE_PATHS", raising=False)
        assert resolve_path("~/x.csv") == (Path.home() / "x.csv").resolve()

    @pytest.mark.parametrize("name", ["../../etc", "..", "a/../../b"])
    def test_a_workspace_name_is_always_a_name(self, tmp_path, monkeypatch, name):
        monkeypatch.delenv("MCP_CONFINE_PATHS", raising=False)
        monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="not a plain name"):
            get_workspace_dir(name)
