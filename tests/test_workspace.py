"""Tests for shared/workspace_utils.py and shared/handover.py (ML version)."""

import json
from pathlib import Path

import pytest

from shared.handover import DOMAIN_SERVERS, NEXT_STEP, STEP_TOOLS, WORKFLOW_STEPS, make_context, make_handover
from shared.workspace_utils import (
    _ALIAS_PREFIX,
    _LEGACY_ALIAS_PREFIX,
    create_project_dirs,
    create_workspace_dirs,
    get_project_dir,
    get_projects_root,
    get_workspace_dir,
    get_workspace_root,
    is_alias,
    load_manifest,
    register_file,
    resolve_alias,
    save_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(base: Path, name: str, alias: str, csv: Path) -> None:
    ws_dir = base / name
    ws_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "files": {
            alias: {
                "path": str(csv),
                "stage": "working",
                "rows": 2,
                "size_bytes": csv.stat().st_size if csv.exists() else 0,
                "registered": "2026-01-01T00:00:00+00:00",
            }
        },
        "pipelines": {},
        "pipeline_history": [],
        "updated": "2026-01-01T00:00:00+00:00",
    }
    (ws_dir / "workspace.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_legacy_workspace(base: Path, name: str, alias: str, csv: Path) -> None:
    ws_dir = base / name
    ws_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "files": {alias: {"path": str(csv), "stage": "raw", "rows": 2, "size_bytes": 0}},
        "pipelines": {},
        "pipeline_history": [],
    }
    (ws_dir / "project.json").write_text(json.dumps(manifest), encoding="utf-8")


def _empty_workspace(base: Path, name: str) -> None:
    ws_dir = base / name
    ws_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "files": {}, "pipelines": {}, "pipeline_history": [], "updated": ""}
    (ws_dir / "workspace.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# Alias constants
# ---------------------------------------------------------------------------


def test_ml_alias_prefix_value():
    assert _ALIAS_PREFIX == "workspace:"


def test_ml_legacy_alias_prefix_value():
    assert _LEGACY_ALIAS_PREFIX == "project:"


# ---------------------------------------------------------------------------
# get_workspace_root
# ---------------------------------------------------------------------------


def test_ml_ws_root_default():
    assert get_workspace_root() == Path.home() / "mcp_workspace"


def test_ml_ws_root_workspace_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    assert get_workspace_root() == tmp_path


def test_ml_ws_root_legacy_projects_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MCP_WORKSPACE_DIR", raising=False)
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    assert get_workspace_root() == tmp_path


def test_ml_ws_root_base_dir_arg(tmp_path):
    assert get_workspace_root(str(tmp_path)) == tmp_path


def test_ml_ws_root_workspace_env_priority(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    proj = tmp_path / "proj"
    ws.mkdir()
    proj.mkdir()
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(proj))
    assert get_workspace_root() == ws


def test_ml_backward_compat_get_root(tmp_path):
    assert get_projects_root(str(tmp_path)) == get_workspace_root(str(tmp_path))


def test_ml_backward_compat_get_dir(tmp_path):
    assert get_project_dir("foo", str(tmp_path)) == get_workspace_dir("foo", str(tmp_path))


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_ml_load_manifest_workspace_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _make_workspace(tmp_path, "ws1", "mydata", csv)
    m = load_manifest("ws1")
    assert m["name"] == "ws1"
    assert "mydata" in m["files"]


def test_ml_load_manifest_legacy_project_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _make_legacy_workspace(tmp_path, "legacyws", "raw_data", csv)
    m = load_manifest("legacyws")
    assert "raw_data" in m["files"]


def test_ml_load_manifest_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        load_manifest("nonexistent")


# ---------------------------------------------------------------------------
# save_manifest
# ---------------------------------------------------------------------------


def test_ml_save_manifest_creates_workspace_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "newws").mkdir()
    save_manifest({"name": "newws", "files": {}}, "newws")
    assert (tmp_path / "newws" / "workspace.json").exists()


# ---------------------------------------------------------------------------
# register_file
# ---------------------------------------------------------------------------


def test_ml_register_file_success(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "out.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    _empty_workspace(tmp_path, "reg_ws")
    m = register_file("reg_ws", str(csv), "clean", stage="working")
    assert "clean" in m["files"]
    assert m["files"]["clean"]["stage"] == "working"


def test_ml_register_file_default_stage_is_working(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "f.csv"
    csv.write_text("x\n1\n")
    _empty_workspace(tmp_path, "def_ws")
    m = register_file("def_ws", str(csv), "myfile")
    assert m["files"]["myfile"]["stage"] == "working"


def test_ml_register_file_invalid_stage(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    _empty_workspace(tmp_path, "ws_bad")
    with pytest.raises(ValueError, match="Invalid stage"):
        register_file("ws_bad", str(tmp_path / "f.csv"), "a", stage="bad")


def test_ml_register_file_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    _empty_workspace(tmp_path, "ws_miss")
    with pytest.raises(FileNotFoundError):
        register_file("ws_miss", str(tmp_path / "missing.csv"), "a")


# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------


def test_ml_resolve_alias_workspace_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _make_workspace(tmp_path, "resws", "mycsv", csv)
    assert resolve_alias("workspace:resws/mycsv") == csv.resolve()


def test_ml_resolve_alias_project_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _make_workspace(tmp_path, "legws", "mycsv", csv)
    assert resolve_alias("project:legws/mycsv") == csv.resolve()


def test_ml_resolve_alias_no_prefix_passthrough(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    assert resolve_alias(str(csv)) == csv.resolve()


def test_ml_resolve_alias_bad_format(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Invalid alias format"):
        resolve_alias("workspace:noslash")


def test_ml_resolve_alias_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_alias("workspace:ghost/alias")


def test_ml_resolve_alias_alias_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _make_workspace(tmp_path, "ws_noa", "real_alias", csv)
    with pytest.raises(ValueError, match="not found"):
        resolve_alias("workspace:ws_noa/wrong_alias")


# ---------------------------------------------------------------------------
# is_alias
# ---------------------------------------------------------------------------


def test_ml_is_alias_workspace_prefix():
    assert is_alias("workspace:proj/file") is True


def test_ml_is_alias_project_prefix():
    assert is_alias("project:proj/file") is True


def test_ml_is_alias_plain_path():
    assert is_alias("/home/user/data.csv") is False


# ---------------------------------------------------------------------------
# create_workspace_dirs (ML has models/ dir)
# ---------------------------------------------------------------------------


def test_ml_create_workspace_dirs_all_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    dirs = create_workspace_dirs("mlws")
    for key in ("root", "data_raw", "data_working", "data_trials", "reports", "models", "pipelines", "versions"):
        assert Path(dirs[key]).exists(), f"Missing dir: {key}"


def test_ml_create_project_dirs_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    dirs = create_project_dirs("alias_ws")
    assert Path(dirs["models"]).exists()


# ---------------------------------------------------------------------------
# handover — make_context
# ---------------------------------------------------------------------------


def test_ml_make_context_fields():
    ctx = make_context("train_classifier", "Trained RF on 1000 rows, accuracy=0.92")
    assert ctx["op"] == "train_classifier"
    assert "0.92" in ctx["summary"]
    assert ctx["artifacts"] == []
    assert "timestamp" in ctx


def test_ml_make_context_with_artifacts():
    arts = [{"type": "model", "path": "/out/model.pkl", "role": "output"}]
    ctx = make_context("train_classifier", "ok", artifacts=arts)
    assert ctx["artifacts"] == arts


# ---------------------------------------------------------------------------
# handover — make_handover (legacy step name mapping)
# ---------------------------------------------------------------------------


def test_ml_make_handover_canonical_step():
    h = make_handover("TRAIN", [])
    assert h["workflow_step"] == "TRAIN"
    assert h["workflow_next"] == "EVALUATE"


def test_ml_make_handover_legacy_locate_maps_to_collect():
    h = make_handover("LOCATE", [])
    assert h["workflow_step"] == "COLLECT"


def test_ml_make_handover_legacy_patch_maps_to_train():
    h = make_handover("PATCH", [])
    assert h["workflow_step"] == "TRAIN"


def test_ml_make_handover_legacy_verify_maps_to_evaluate():
    h = make_handover("VERIFY", [])
    assert h["workflow_step"] == "EVALUATE"


def test_ml_make_handover_str_list_backward_compat():
    h = make_handover("TRAIN", ["evaluate_model", "read_model_report"])
    assert h["suggested_next"][0]["tool"] == "evaluate_model"
    assert h["suggested_next"][1]["tool"] == "read_model_report"
    # Legacy suggested_tools key still present
    assert "suggested_tools" in h
    assert h["suggested_tools"] == ["evaluate_model", "read_model_report"]


def test_ml_make_handover_dict_list_new_style():
    suggestions = [{"tool": "check_data_quality", "server": "ml_medium", "domain": "ml", "reason": "verify"}]
    h = make_handover("PREPARE", suggestions, carry_forward={"file_path": "/tmp/data.csv"})
    assert h["suggested_next"][0]["server"] == "ml_medium"
    assert h["carry_forward"]["file_path"] == "/tmp/data.csv"


def test_ml_make_handover_full_chain():
    for i, step in enumerate(WORKFLOW_STEPS[:-1]):
        h = make_handover(step, [])
        assert h["workflow_next"] == WORKFLOW_STEPS[i + 1]


def test_ml_make_handover_report_no_next():
    h = make_handover("REPORT", [])
    assert h["workflow_next"] == ""


def test_ml_workflow_steps_order():
    assert WORKFLOW_STEPS == ["COLLECT", "INSPECT", "CLEAN", "PREPARE", "TRAIN", "EVALUATE", "REPORT"]


def test_ml_domain_servers_content():
    assert DOMAIN_SERVERS["data"] == "MCP_Data_Analyst"
    assert DOMAIN_SERVERS["ml"] == "MCP_Machine_Learning"


def test_ml_step_tools_all_steps_covered():
    for step in WORKFLOW_STEPS:
        assert step in STEP_TOOLS
        assert len(STEP_TOOLS[step]) > 0


def test_ml_next_step_legacy_dict():
    # NEXT_STEP kept for backward compat
    assert "LOCATE" in NEXT_STEP
    assert "TRAIN" in NEXT_STEP
