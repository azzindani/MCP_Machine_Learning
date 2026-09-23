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


class TestOutputsConfined:
    """Where a tool WRITES is held to the served folders too.

    A5-sec confined the paths tools read and missed five they write: batch_predict
    resolved its output with a bare `Path(...).resolve()` (a relative path landed
    beside the process, an absolute one anywhere), and four sites wrapped
    resolve_path in `except ValueError: use the raw path` -- which predates
    confinement and, because PathOutsideRootError is a ValueError here, turned
    every refused output into a write wherever it pointed. Found by driving the
    deployed server directly: batch_predict(output_path="sweep/preds.csv") tried
    to create /app/sweep.
    """

    @pytest.fixture
    def model(self, served):
        from servers.ml_basic import engine as b

        r = b.train_regressor("train.csv", "y", "lir", feature_columns=["x"], output_path="m.pkl")
        assert r["success"] is True, r
        return "m.pkl"

    def _refused(self, r: dict, target: Path) -> None:
        assert r["success"] is False, r
        assert "outside the folders" in r["error"]
        assert not target.exists(), "a refused output must not be written"

    def test_batch_predict_refuses_an_outside_output(self, served, model, tmp_path):
        from servers.ml_medium import engine as m

        target = tmp_path / "elsewhere" / "preds.csv"
        self._refused(m.batch_predict(model, "train.csv", output_path=str(target)), target)

    def test_batch_predict_writes_a_relative_output_into_the_data_folder(self, served, model):
        from servers.ml_medium import engine as m

        r = m.batch_predict(model, "train.csv", output_path="out/preds.csv")
        assert r["success"] is True, r
        assert (served / "out" / "preds.csv").exists()

    def test_run_preprocessing_refuses_an_outside_output(self, served, tmp_path):
        from servers.ml_medium import engine as m

        target = tmp_path / "elsewhere.csv"
        ops = [{"op": "fill_nulls", "column": "x", "strategy": "mean"}]
        self._refused(m.run_preprocessing("train.csv", ops, output_path=str(target)), target)

    def test_filter_rows_refuses_an_outside_output(self, served, tmp_path):
        from servers.ml_medium import engine as m

        target = tmp_path / "filtered.csv"
        self._refused(m.filter_rows("train.csv", "x", "gt", "5", output_path=str(target)), target)

    def test_merge_datasets_refuses_an_outside_output(self, served, tmp_path):
        from servers.ml_medium import engine as m

        target = tmp_path / "merged.csv"
        self._refused(m.merge_datasets("train.csv", "train.csv", "x", output_path=str(target)), target)

    def test_export_model_refuses_an_outside_output_dir(self, served, model, tmp_path):
        from servers.ml_advanced import engine as a

        target = tmp_path / "exported"
        r = a.export_model(model, output_dir=str(target))
        assert r["success"] is False, r
        assert "outside the folders" in r["error"]
        assert not target.exists()


class TestReceiptsLandBesideTheFile:
    """Three tools wrote their receipt beside the caller's raw string.

    train_classifier, train_regressor and split_dataset passed `file_path` --
    what the caller typed -- to append_receipt instead of the resolved path, so
    a relative name put `<name>.mcp_receipt.json` in the process's working
    directory (unwritable /app on a deployed server, where the receipt was
    silently lost) instead of beside the data. Found as a stray receipt in the
    repo root after a test run.
    """

    def test_a_training_receipt_is_written_beside_the_data(self, served, monkeypatch, tmp_path):
        from servers.ml_basic import engine as b

        elsewhere = tmp_path / "cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert b.train_regressor("train.csv", "y", "lir", feature_columns=["x"], output_path="m.pkl")["success"]
        assert (served / "train.csv.mcp_receipt.json").exists()
        assert not list(elsewhere.iterdir()), "nothing belongs in the process's working directory"


class TestARefusedPathIsAnAnswer:
    """A refusal comes back in the fleet's failure shape, whichever resolver raised it.

    Eight output resolvers let PathOutsideRootError propagate. Nothing was
    written -- the confinement held -- but the caller got "Error executing tool
    anomaly_detection: Path ... is outside" with no success, op or hint. Found
    live on the deployed server; the per-tool wrapper now answers it.
    """

    CASES = [
        ("ml_medium", "anomaly_detection", {"feature_columns": ["x", "y"]}, "output_path", "labels.csv"),
        (
            "ml_medium",
            "run_clustering",
            {"feature_columns": ["x", "y"], "algorithm": "kmeans", "n_clusters": 2},
            "output_path",
            "labels.csv",
        ),
        ("ml_basic", "split_dataset", {}, "output_dir", "splits"),
        ("ml_medium", "generate_eda_report", {}, "output_path", "eda.html"),
    ]

    @pytest.mark.parametrize(("tier", "tool", "args", "key", "leaf"), CASES, ids=[c[1] for c in CASES])
    def test_the_refusal_is_a_response(self, served, tmp_path, tier, tool, args, key, leaf):
        import importlib

        fn = importlib.import_module(f"servers.{tier}.server").mcp._tool_manager._tools[tool].fn
        target = tmp_path / "elsewhere" / leaf
        r = fn(file_path="train.csv", **args, **{key: str(target)})
        assert isinstance(r, dict), r
        assert r["success"] is False, r
        assert r["op"] == tool
        assert "outside the folders" in r["error"]
        assert "data folder" in r["hint"]
        assert not target.exists(), "a refused output must not be written"


class TestAPathOnTheCallersSideIsNamedAsOne:
    """A claude.ai upload path is refused for what it is, with the way in.

    `/mnt/user-data/uploads/Ad_Data.csv` is the only path a chat's model holds
    for an attached file. "Outside the folders this server can use" named the
    rule and sent it guessing folders on a server that cannot see the file.
    """

    def test_the_refusal_says_the_file_is_on_the_callers_side(self, served, monkeypatch):
        monkeypatch.setenv("MCP_FETCH_URLS", "1")
        with pytest.raises(PathOutsideRootError) as caught:
            resolve_path("/mnt/user-data/uploads/Ad_Data.csv")
        message = str(caught.value)
        assert "caller's side" in message
        assert "cannot see it" in message
        assert "link" in message

    def test_any_other_outside_path_keeps_the_plain_refusal(self, served):
        with pytest.raises(PathOutsideRootError, match="outside the folders"):
            resolve_path("/etc/hostname")


class TestAFileSentInlineIsRead:
    """A CSV sent as data:...;base64 is read by a registered tool like any file."""

    def test_inspect_reads_an_inline_csv(self, served):
        import base64
        import importlib

        uri = "data:text/csv;name=sent.csv;base64," + base64.b64encode(b"x,y\n1,2\n3,4\n5,6\n").decode()
        tools = importlib.import_module("servers.ml_basic.server").mcp._tool_manager._tools
        r = tools["inspect_dataset"].fn(file_path=uri)
        assert r["success"] is True, r
        assert "base64" not in str(r)
        assert (served / "inbox" / "sent.csv").exists()
        assert "3" in str(r.get("rows", r.get("row_count", "")))
