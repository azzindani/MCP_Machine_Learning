"""Tests for shared utilities: workspace_utils (via project_utils shim) and file_utils."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from shared.file_utils import read_csv, resolve_path
from shared.project_utils import (
    create_project_dirs,
    get_project_dir,
    get_projects_root,
    is_alias,
    load_manifest,
    register_file,
    resolve_alias,
    save_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(base_dir: Path, project_name: str, alias: str, file_path: Path) -> None:
    """Create a minimal workspace-compatible manifest for testing."""
    proj_dir = base_dir / project_name
    proj_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": project_name,
        "files": {
            alias: {
                "path": str(file_path),
                "stage": "working",
                "rows": 10,
                "size_bytes": 100,
                "registered": "2026-01-01T00:00:00+00:00",
            }
        },
        "pipelines": {},
        "pipeline_history": [],
        "updated": "2026-01-01T00:00:00+00:00",
    }
    # Write as project.json for backward-compat fallback testing
    (proj_dir / "project.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_empty_project(base_dir: Path, project_name: str) -> None:
    """Create a manifest with no registered files."""
    proj_dir = base_dir / project_name
    proj_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": project_name,
        "files": {},
        "pipelines": {},
        "pipeline_history": [],
        "updated": "",
    }
    (proj_dir / "project.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# get_projects_root
# ---------------------------------------------------------------------------


def test_get_projects_root_default():
    # Default changed from mcp_projects -> mcp_workspace with the workspace rename
    root = get_projects_root()
    assert root == Path.home() / "mcp_workspace"


def test_get_projects_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    root = get_projects_root()
    assert root == tmp_path


def test_get_projects_root_workspace_env_takes_priority(monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    proj = tmp_path / "proj"
    ws.mkdir()
    proj.mkdir()
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(proj))
    assert get_projects_root() == ws


def test_get_projects_root_base_dir_arg(tmp_path):
    root = get_projects_root(str(tmp_path))
    assert root == tmp_path


# ---------------------------------------------------------------------------
# is_alias
# ---------------------------------------------------------------------------


def test_is_alias_true():
    assert is_alias("project:myproj/clean_data") is True


def test_is_alias_workspace_prefix():
    assert is_alias("workspace:myproj/clean_data") is True


def test_is_alias_false_absolute():
    assert is_alias("/home/user/data.csv") is False


def test_is_alias_false_relative():
    assert is_alias("data/file.csv") is False


# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------


def test_resolve_alias_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    _make_project(tmp_path, "testproj", "mydata", csv_file)
    result = resolve_alias("project:testproj/mydata")
    assert result == csv_file.resolve()


def test_resolve_alias_workspace_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    _make_project(tmp_path, "testproj", "mydata", csv_file)
    result = resolve_alias("workspace:testproj/mydata")
    assert result == csv_file.resolve()


def test_resolve_alias_relative_path_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    proj_dir = tmp_path / "proj2"
    proj_dir.mkdir()
    (proj_dir / "data" / "working").mkdir(parents=True)
    csv_file = proj_dir / "data" / "working" / "clean.csv"
    csv_file.write_text("a,b\n1,2\n")
    manifest = {
        "name": "proj2",
        "files": {"clean": {"path": "data/working/clean.csv", "stage": "working"}},
        "pipelines": {},
        "pipeline_history": [],
    }
    (proj_dir / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = resolve_alias("project:proj2/clean")
    assert result == csv_file.resolve()


def test_resolve_alias_project_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_alias("project:nonexistent/alias")


def test_resolve_alias_alias_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    _make_project(tmp_path, "testproj", "mydata", csv_file)
    with pytest.raises(ValueError, match="not found"):
        resolve_alias("project:testproj/wrongalias")


def test_resolve_alias_bad_format(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Invalid alias format"):
        resolve_alias("project:noslash")


def test_resolve_alias_non_alias_passthrough(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    result = resolve_alias(str(csv_file))
    assert result == csv_file.resolve()


# ---------------------------------------------------------------------------
# register_file
# ---------------------------------------------------------------------------


def test_register_file_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    csv_file = tmp_path / "output.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")
    _make_empty_project(tmp_path, "myproj")
    manifest = register_file("myproj", str(csv_file), "clean_output", stage="working")
    assert "clean_output" in manifest["files"]
    assert manifest["files"]["clean_output"]["stage"] == "working"


def test_register_file_updates_manifest_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    csv_file = tmp_path / "out.csv"
    csv_file.write_text("x\n1\n")
    _make_empty_project(tmp_path, "myproj")
    register_file("myproj", str(csv_file), "out_alias")
    manifest = load_manifest("myproj")
    assert "out_alias" in manifest["files"]


def test_register_file_invalid_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    _make_empty_project(tmp_path, "myproj")
    with pytest.raises(ValueError, match="Invalid stage"):
        register_file("myproj", str(tmp_path / "f.csv"), "alias", stage="invalid")


def test_register_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    _make_empty_project(tmp_path, "myproj")
    with pytest.raises(FileNotFoundError):
        register_file("myproj", str(tmp_path / "missing.csv"), "alias")


# ---------------------------------------------------------------------------
# create_project_dirs
# ---------------------------------------------------------------------------


def test_create_project_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    dirs = create_project_dirs("newproj")
    assert Path(dirs["root"]).exists()
    assert Path(dirs["data_raw"]).exists()
    assert Path(dirs["models"]).exists()


# ---------------------------------------------------------------------------
# file_utils.resolve_path — alias support
# ---------------------------------------------------------------------------


def test_resolve_path_project_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    _make_project(tmp_path, "proj1", "mycsv", csv_file)
    result = resolve_path("project:proj1/mycsv")
    assert result == csv_file.resolve()


def test_resolve_path_workspace_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_WORKSPACE_DIR", str(tmp_path))
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    _make_project(tmp_path, "proj1", "mycsv", csv_file)
    result = resolve_path("workspace:proj1/mycsv")
    assert result == csv_file.resolve()


def test_resolve_path_project_alias_bad_project(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_PROJECTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Cannot resolve project alias"):
        resolve_path("project:nope/alias")


def test_resolve_path_normal_path(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    result = resolve_path(str(csv_file))
    assert result == csv_file.resolve()


def test_resolve_path_null_byte_rejected():
    with pytest.raises(ValueError, match="null byte"):
        resolve_path("some\x00path.csv")


def test_resolve_path_filesystem_root_rejected():
    root = "C:\\\\" if sys.platform == "win32" else "/"
    with pytest.raises(ValueError, match="filesystem root"):
        resolve_path(root)


def test_resolve_path_extension_check(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not allowed"):
        resolve_path(str(csv_file), allowed_extensions=(".pkl",))


# ---------------------------------------------------------------------------
# read_csv — encoding fallbacks
# ---------------------------------------------------------------------------


def test_read_csv_utf8(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,value\nalice,1\nbob,2\n", encoding="utf-8")
    df = read_csv(str(f))
    assert list(df.columns) == ["name", "value"]
    assert len(df) == 2


def test_read_csv_bom_utf8(tmp_path):
    f = tmp_path / "bom.csv"
    f.write_bytes(b"\xef\xbb\xbfname,value\nalice,1\n")
    df = read_csv(str(f))
    assert "name" in df.columns


def test_read_csv_cp1252(tmp_path):
    f = tmp_path / "win.csv"
    f.write_bytes("name,value\ncaf\xe9,1\n".encode("cp1252"))
    df = read_csv(str(f))
    assert len(df) == 1


def test_read_csv_strips_column_whitespace(tmp_path):
    f = tmp_path / "spaces.csv"
    f.write_text(" name , value \nalice,1\n", encoding="utf-8")
    df = read_csv(str(f))
    assert "name" in df.columns
    assert "value" in df.columns


def test_read_csv_max_rows(tmp_path):
    f = tmp_path / "big.csv"
    rows = ["a,b"] + [f"{i},{i * 2}" for i in range(100)]
    f.write_text("\n".join(rows), encoding="utf-8")
    df = read_csv(str(f), max_rows=10)
    assert len(df) == 10


def test_read_csv_semicolon_separator(tmp_path):
    f = tmp_path / "semi.csv"
    f.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    df = read_csv(str(f), separator=";")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
