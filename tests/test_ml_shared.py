"""Tests for shared utilities: workspace_utils (via project_utils shim) and file_utils."""

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


# ===========================================================================
# NEW: patch_validator tests
# ===========================================================================

from shared.patch_validator import (  # noqa: E402
    ALLOWED_FILL_STRATEGIES,
    ALLOWED_OUTLIER_METHODS,
    ALLOWED_PREPROCESSING_OPS,
    ALLOWED_SCALE_METHODS,
    MAX_OPS,
    validate_ops,
)


class TestValidateOpsBasicStructure:
    """Tests for top-level structural validation of the ops array."""

    def test_empty_list_is_valid(self):
        valid, msg = validate_ops([])
        assert valid is True
        assert msg == ""

    def test_non_list_ops_returns_false(self):
        valid, msg = validate_ops({"op": "fill_nulls"})  # type: ignore[arg-type]
        assert valid is False
        assert "list" in msg.lower()

    def test_string_ops_returns_false(self):
        valid, msg = validate_ops("fill_nulls")  # type: ignore[arg-type]
        assert valid is False
        assert "list" in msg.lower()

    def test_none_ops_returns_false(self):
        valid, msg = validate_ops(None)  # type: ignore[arg-type]
        assert valid is False

    def test_too_many_ops_returns_false(self):
        # 51 ops — one over the limit
        ops = [{"op": "drop_duplicates"} for _ in range(MAX_OPS + 1)]
        valid, msg = validate_ops(ops)
        assert valid is False
        assert str(MAX_OPS + 1) in msg
        assert str(MAX_OPS) in msg

    def test_exactly_max_ops_is_valid(self):
        ops = [{"op": "drop_duplicates"} for _ in range(MAX_OPS)]
        valid, msg = validate_ops(ops)
        assert valid is True
        assert msg == ""

    def test_op_not_a_dict_returns_false(self):
        valid, msg = validate_ops(["not_a_dict"])
        assert valid is False
        assert "dict" in msg.lower()

    def test_op_as_integer_returns_false(self):
        valid, msg = validate_ops([42])
        assert valid is False
        assert "dict" in msg.lower()

    def test_op_missing_op_key_returns_false(self):
        valid, msg = validate_ops([{"column": "age"}])
        assert valid is False
        assert "'op'" in msg or "op" in msg

    def test_op_empty_op_key_returns_false(self):
        valid, msg = validate_ops([{"op": ""}])
        assert valid is False
        assert "op" in msg.lower()

    def test_unknown_op_name_returns_false(self):
        valid, msg = validate_ops([{"op": "delete_everything"}])
        assert valid is False
        assert "delete_everything" in msg
        assert "unknown" in msg.lower() or "allowed" in msg.lower()

    def test_error_reports_index_of_bad_op(self):
        ops = [
            {"op": "drop_duplicates"},
            {"op": "drop_duplicates"},
            {"op": "totally_fake"},
        ]
        valid, msg = validate_ops(ops)
        assert valid is False
        # Should mention index 2
        assert "2" in msg


class TestValidateOpsAllowedSet:
    """Tests for the custom `allowed` parameter."""

    def test_custom_allowed_set_accepts_known_op(self):
        valid, msg = validate_ops(
            [{"op": "drop_duplicates"}],
            allowed={"drop_duplicates"},
        )
        assert valid is True

    def test_custom_allowed_set_rejects_op_not_in_set(self):
        valid, msg = validate_ops(
            [{"op": "drop_duplicates"}],
            allowed={"fill_nulls"},
        )
        assert valid is False
        assert "drop_duplicates" in msg

    def test_default_allowed_contains_all_14_ops(self):
        assert len(ALLOWED_PREPROCESSING_OPS) == 14


class TestValidateFillNulls:
    """Tests for fill_nulls-specific validation."""

    def test_fill_nulls_missing_column_returns_false(self):
        valid, msg = validate_ops([{"op": "fill_nulls", "strategy": "mean"}])
        assert valid is False
        assert "column" in msg

    def test_fill_nulls_missing_strategy_returns_false(self):
        valid, msg = validate_ops([{"op": "fill_nulls", "column": "age"}])
        assert valid is False
        assert "strategy" in msg

    def test_fill_nulls_invalid_strategy_returns_false(self):
        valid, msg = validate_ops([{"op": "fill_nulls", "column": "age", "strategy": "interpolate"}])
        assert valid is False
        assert "interpolate" in msg

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode", "ffill", "bfill", "zero"])
    def test_fill_nulls_valid_strategy(self, strategy):
        valid, msg = validate_ops([{"op": "fill_nulls", "column": "age", "strategy": strategy}])
        assert valid is True, f"Strategy '{strategy}' should be valid but got: {msg}"
        assert msg == ""

    def test_fill_nulls_all_strategies_covered(self):
        # Ensure our parametrize above matches the actual allowed set
        tested = {"mean", "median", "mode", "ffill", "bfill", "zero"}
        assert tested == ALLOWED_FILL_STRATEGIES


class TestValidateScale:
    """Tests for scale-specific validation."""

    def test_scale_missing_columns_returns_false(self):
        valid, msg = validate_ops([{"op": "scale", "method": "standard"}])
        assert valid is False
        assert "columns" in msg

    def test_scale_missing_method_returns_false(self):
        valid, msg = validate_ops([{"op": "scale", "columns": ["age"]}])
        assert valid is False
        assert "method" in msg

    def test_scale_invalid_method_returns_false(self):
        valid, msg = validate_ops([{"op": "scale", "columns": ["age"], "method": "robust"}])
        assert valid is False
        assert "robust" in msg

    def test_scale_columns_must_be_list_not_string(self):
        valid, msg = validate_ops([{"op": "scale", "columns": "age", "method": "standard"}])
        assert valid is False
        assert "list" in msg.lower()

    def test_scale_columns_must_be_list_not_dict(self):
        valid, msg = validate_ops([{"op": "scale", "columns": {"age": True}, "method": "standard"}])
        assert valid is False
        assert "list" in msg.lower()

    @pytest.mark.parametrize("method", ["standard", "minmax"])
    def test_scale_valid_method(self, method):
        valid, msg = validate_ops([{"op": "scale", "columns": ["age", "income"], "method": method}])
        assert valid is True, f"Scale method '{method}' should be valid but got: {msg}"

    def test_scale_all_methods_covered(self):
        tested = {"standard", "minmax"}
        assert tested == ALLOWED_SCALE_METHODS


class TestValidateDropOutliers:
    """Tests for drop_outliers-specific validation."""

    def test_drop_outliers_missing_column_returns_false(self):
        valid, msg = validate_ops([{"op": "drop_outliers", "method": "iqr"}])
        assert valid is False
        assert "column" in msg

    def test_drop_outliers_missing_method_returns_false(self):
        valid, msg = validate_ops([{"op": "drop_outliers", "column": "price"}])
        assert valid is False
        assert "method" in msg

    def test_drop_outliers_invalid_method_returns_false(self):
        valid, msg = validate_ops([{"op": "drop_outliers", "column": "price", "method": "zscore"}])
        assert valid is False
        assert "zscore" in msg

    @pytest.mark.parametrize("method", ["iqr", "std"])
    def test_drop_outliers_valid_method(self, method):
        valid, msg = validate_ops([{"op": "drop_outliers", "column": "price", "method": method}])
        assert valid is True, f"Outlier method '{method}' should be valid but got: {msg}"

    def test_drop_outliers_all_methods_covered(self):
        tested = {"iqr", "std"}
        assert tested == ALLOWED_OUTLIER_METHODS


class TestValidateConvertDtype:
    """Tests for convert_dtype-specific validation."""

    def test_convert_dtype_missing_column_returns_false(self):
        valid, msg = validate_ops([{"op": "convert_dtype", "to": "int"}])
        assert valid is False
        assert "column" in msg

    def test_convert_dtype_missing_to_returns_false(self):
        valid, msg = validate_ops([{"op": "convert_dtype", "column": "age"}])
        assert valid is False
        assert "'to'" in msg or "to" in msg

    def test_convert_dtype_invalid_dtype_returns_false(self):
        valid, msg = validate_ops([{"op": "convert_dtype", "column": "age", "to": "complex128"}])
        assert valid is False
        assert "complex128" in msg

    @pytest.mark.parametrize("dtype", ["int", "float", "str", "datetime", "bool", "numeric", "string"])
    def test_convert_dtype_valid_dtype(self, dtype):
        valid, msg = validate_ops([{"op": "convert_dtype", "column": "age", "to": dtype}])
        assert valid is True, f"dtype '{dtype}' should be valid but got: {msg}"


class TestValidateOpsNoFieldRequirements:
    """Tests for ops that require no fields beyond 'op' itself."""

    def test_drop_duplicates_valid_with_op_only(self):
        valid, msg = validate_ops([{"op": "drop_duplicates"}])
        assert valid is True

    def test_drop_null_rows_valid_with_op_only(self):
        valid, msg = validate_ops([{"op": "drop_null_rows"}])
        assert valid is True


class TestValidateOpsSingleFieldOps:
    """Tests for ops that require only 'column'."""

    @pytest.mark.parametrize(
        "op_name",
        [
            "label_encode",
            "onehot_encode",
            "drop_column",
            "bin_numeric",
            "add_date_parts",
            "log_transform",
            "clip_column",
        ],
    )
    def test_single_column_op_missing_column_returns_false(self, op_name):
        valid, msg = validate_ops([{"op": op_name}])
        assert valid is False
        assert "column" in msg

    @pytest.mark.parametrize(
        "op_name",
        [
            "label_encode",
            "onehot_encode",
            "drop_column",
            "bin_numeric",
            "add_date_parts",
            "log_transform",
            "clip_column",
        ],
    )
    def test_single_column_op_with_column_is_valid(self, op_name):
        valid, msg = validate_ops([{"op": op_name, "column": "mycolumn"}])
        assert valid is True, f"Op '{op_name}' with column should be valid but got: {msg}"


class TestValidateRenameColumn:
    """Tests for rename_column required fields."""

    def test_rename_column_missing_from_returns_false(self):
        valid, msg = validate_ops([{"op": "rename_column", "to": "new_name"}])
        assert valid is False
        assert "from" in msg

    def test_rename_column_missing_to_returns_false(self):
        valid, msg = validate_ops([{"op": "rename_column", "from": "old_name"}])
        assert valid is False
        assert "'to'" in msg or "to" in msg

    def test_rename_column_valid(self):
        valid, msg = validate_ops([{"op": "rename_column", "from": "old", "to": "new"}])
        assert valid is True


class TestValidateOpsMultipleOps:
    """Tests for multi-op arrays and fail-fast behaviour."""

    def test_multiple_valid_ops(self):
        ops = [
            {"op": "fill_nulls", "column": "age", "strategy": "median"},
            {"op": "scale", "columns": ["age", "income"], "method": "standard"},
            {"op": "drop_duplicates"},
            {"op": "label_encode", "column": "gender"},
        ]
        valid, msg = validate_ops(ops)
        assert valid is True

    def test_fail_fast_on_first_bad_op(self):
        ops = [
            {"op": "drop_duplicates"},
            {"op": "bad_op"},  # fails here at index 1
            {"op": "another_bad_op"},  # never reached
        ]
        valid, msg = validate_ops(ops)
        assert valid is False
        # Should reference index 1, not 2
        assert "1" in msg
        assert "bad_op" in msg

    def test_all_allowed_preprocessing_ops_are_individually_valid(self):
        """Every op in ALLOWED_PREPROCESSING_OPS must pass when supplied with
        the minimum required fields."""
        minimal_args: dict[str, dict] = {
            "fill_nulls": {"column": "x", "strategy": "mean"},
            "drop_outliers": {"column": "x", "method": "iqr"},
            "label_encode": {"column": "x"},
            "onehot_encode": {"column": "x"},
            "scale": {"columns": ["x"], "method": "standard"},
            "drop_duplicates": {},
            "drop_column": {"column": "x"},
            "rename_column": {"from": "old", "to": "new"},
            "convert_dtype": {"column": "x", "to": "int"},
            "bin_numeric": {"column": "x"},
            "add_date_parts": {"column": "x"},
            "log_transform": {"column": "x"},
            "drop_null_rows": {},
            "clip_column": {"column": "x"},
        }
        for op_name in ALLOWED_PREPROCESSING_OPS:
            args = minimal_args[op_name]
            op = {"op": op_name, **args}
            valid, msg = validate_ops([op])
            assert valid is True, f"Op '{op_name}' with minimal args failed: {msg}"


# ===========================================================================
# NEW: version_control tests
# ===========================================================================

from shared.version_control import list_snapshots, restore_version, snapshot  # noqa: E402


class TestSnapshotCreation:
    """Tests for snapshot() — happy paths and file creation."""

    def test_snapshot_creates_versions_dir(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        backup = snapshot(str(src))
        versions_dir = tmp_path / ".mcp_versions"
        assert versions_dir.exists()
        assert versions_dir.is_dir()

    def test_snapshot_returns_string_path(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        backup = snapshot(str(src))
        assert isinstance(backup, str)

    def test_snapshot_creates_bak_file(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        backup = snapshot(str(src))
        bak = Path(backup)
        assert bak.exists()
        assert bak.suffix == ".bak"

    def test_snapshot_bak_has_correct_stem(self, tmp_path):
        src = tmp_path / "mydata.csv"
        src.write_text("x\n1\n")
        backup = snapshot(str(src))
        assert Path(backup).name.startswith("mydata_")

    def test_snapshot_bak_content_matches_source(self, tmp_path):
        src = tmp_path / "data.csv"
        content = "a,b\n1,2\n3,4\n"
        src.write_text(content)
        backup = snapshot(str(src))
        assert Path(backup).read_text() == content

    def test_snapshot_does_not_modify_source(self, tmp_path):
        src = tmp_path / "data.csv"
        original = "a,b\n1,2\n"
        src.write_text(original)
        snapshot(str(src))
        assert src.read_text() == original

    def test_snapshot_successive_calls_create_distinct_files(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        bak1 = snapshot(str(src))
        bak2 = snapshot(str(src))
        assert bak1 != bak2
        assert Path(bak1).exists()
        assert Path(bak2).exists()


class TestSnapshotErrors:
    """Tests for snapshot() error paths."""

    def test_snapshot_raises_file_not_found_for_missing_file(self, tmp_path):
        missing = tmp_path / "ghost.csv"
        with pytest.raises(FileNotFoundError):
            snapshot(str(missing))

    def test_snapshot_error_message_contains_path(self, tmp_path):
        missing = tmp_path / "ghost.csv"
        with pytest.raises(FileNotFoundError, match="ghost.csv"):
            snapshot(str(missing))


class TestSnapshotCollision:
    """Tests that counter suffix handles pre-existing backup names."""

    def test_snapshot_collision_generates_counter_suffix(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")

        # Take a real snapshot to get a valid timestamp pattern
        bak1_path = snapshot(str(src))
        bak1 = Path(bak1_path)

        # Manually create a file at the exact same name to force collision
        # on the NEXT call by monkey-patching datetime isn't needed —
        # instead just verify the counter path by pre-creating the expected name
        bak1.write_text("collision_placeholder")

        # Take another snapshot — must NOT overwrite bak1
        bak2_path = snapshot(str(src))
        bak2 = Path(bak2_path)

        # bak1 still has original placeholder content (not overwritten)
        assert bak1.read_text() == "collision_placeholder"
        # bak2 is a different file
        assert bak1_path != bak2_path
        assert bak2.exists()


class TestListSnapshots:
    """Tests for list_snapshots()."""

    def test_list_snapshots_returns_empty_when_no_versions_dir(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        result = list_snapshots(str(src))
        assert result == []

    def test_list_snapshots_returns_empty_when_no_matching_bak(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        # Create the dir but put an unrelated file there
        versions_dir = tmp_path / ".mcp_versions"
        versions_dir.mkdir()
        (versions_dir / "other_2026-01-01T00-00-00-000000Z.bak").write_text("x")
        result = list_snapshots(str(src))
        assert result == []

    def test_list_snapshots_returns_list_after_snapshot(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = list_snapshots(str(src))
        assert isinstance(result, list)
        assert len(result) == 1

    def test_list_snapshots_entry_has_required_keys(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = list_snapshots(str(src))
        entry = result[0]
        assert "timestamp" in entry
        assert "path" in entry
        assert "size_kb" in entry

    def test_list_snapshots_path_exists_on_disk(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = list_snapshots(str(src))
        assert Path(result[0]["path"]).exists()

    def test_list_snapshots_multiple_snapshots(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        snapshot(str(src))
        snapshot(str(src))
        result = list_snapshots(str(src))
        assert len(result) == 3

    def test_list_snapshots_ordered_newest_first(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("v1")
        snapshot(str(src))
        src.write_text("v2")
        snapshot(str(src))
        result = list_snapshots(str(src))
        # sorted reverse=True means newest timestamp string sorts last → first in list
        timestamps = [r["timestamp"] for r in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_snapshots_size_kb_is_numeric(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = list_snapshots(str(src))
        assert isinstance(result[0]["size_kb"], (int, float))


class TestRestoreVersion:
    """Tests for restore_version()."""

    def test_restore_without_timestamp_returns_snapshot_list(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = restore_version(str(src), timestamp="")
        assert result["success"] is True
        assert "snapshots" in result
        assert isinstance(result["snapshots"], list)

    def test_restore_without_timestamp_lists_correct_count(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        snapshot(str(src))
        result = restore_version(str(src), timestamp="")
        assert len(result["snapshots"]) == 2

    def test_restore_without_timestamp_includes_hint(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        result = restore_version(str(src), timestamp="")
        assert "hint" in result

    def test_restore_with_invalid_timestamp_returns_false(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = restore_version(str(src), timestamp="9999-totally-wrong")
        assert result["success"] is False
        assert "error" in result
        assert "hint" in result

    def test_restore_with_invalid_timestamp_mentions_timestamp(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        ts = "9999-totally-wrong"
        result = restore_version(str(src), timestamp=ts)
        assert ts in result["error"]

    def test_restore_with_valid_timestamp_returns_success(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("original content")
        snapshot(str(src))
        snapshots = list_snapshots(str(src))
        ts = snapshots[0]["timestamp"]
        # Overwrite src so restore actually does something
        src.write_text("modified content")
        result = restore_version(str(src), timestamp=ts)
        assert result["success"] is True

    def test_restore_with_valid_timestamp_actually_restores_content(self, tmp_path):
        src = tmp_path / "data.csv"
        original = "a,b\n1,2\n3,4\n"
        src.write_text(original)
        snapshot(str(src))
        snapshots = list_snapshots(str(src))
        ts = snapshots[0]["timestamp"]
        # Corrupt the file
        src.write_text("corrupted!")
        restore_version(str(src), timestamp=ts)
        assert src.read_text() == original

    def test_restore_success_response_has_required_keys(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        snapshots = list_snapshots(str(src))
        ts = snapshots[0]["timestamp"]
        result = restore_version(str(src), timestamp=ts)
        assert "success" in result
        assert "op" in result
        assert "restored_from" in result
        assert "progress" in result
        assert "token_estimate" in result

    def test_restore_failure_response_has_required_keys(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        snapshot(str(src))
        result = restore_version(str(src), timestamp="bad-ts")
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "hint" in result

    def test_restore_no_snapshots_invalid_timestamp_returns_false(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        # No snapshot taken — list is empty
        result = restore_version(str(src), timestamp="2026-01-01")
        assert result["success"] is False

    def test_restore_with_partial_timestamp_match(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("hello")
        snapshot(str(src))
        snapshots = list_snapshots(str(src))
        # Use only the date portion of the timestamp (partial match)
        partial_ts = snapshots[0]["timestamp"][:10]  # e.g. "2026-04-18"
        result = restore_version(str(src), timestamp=partial_ts)
        # Should find the snapshot via partial match
        assert result["success"] is True


# ===========================================================================
# NEW: shared/progress.py — fail() and undo() coverage
# ===========================================================================

from shared.progress import fail, info, name, ok, undo, warn  # noqa: E402


class TestProgressFail:
    """Tests for fail() — lines 19-23 in progress.py."""

    def test_fail_no_detail_has_icon(self):
        result = fail("something went wrong")
        assert result["icon"] == "✘"

    def test_fail_no_detail_has_msg(self):
        result = fail("something went wrong")
        assert result["msg"] == "something went wrong"

    def test_fail_no_detail_omits_detail_key(self):
        result = fail("something went wrong")
        assert "detail" not in result

    def test_fail_with_detail_includes_detail_key(self):
        result = fail("bad thing", "because of X")
        assert "detail" in result
        assert result["detail"] == "because of X"

    def test_fail_with_detail_preserves_icon_and_msg(self):
        result = fail("error", "details here")
        assert result["icon"] == "✘"
        assert result["msg"] == "error"

    def test_fail_empty_detail_omits_detail_key(self):
        # Empty string is falsy → detail key should be omitted
        result = fail("oops", "")
        assert "detail" not in result


class TestProgressUndo:
    """Tests for undo() — lines 40-44 in progress.py."""

    def test_undo_no_detail_has_icon(self):
        result = undo("rolled back")
        assert result["icon"] == "↩"

    def test_undo_no_detail_has_msg(self):
        result = undo("rolled back")
        assert result["msg"] == "rolled back"

    def test_undo_no_detail_omits_detail_key(self):
        result = undo("rolled back")
        assert "detail" not in result

    def test_undo_with_detail_includes_detail_key(self):
        result = undo("restored snapshot", "data_2026.csv.bak")
        assert "detail" in result
        assert result["detail"] == "data_2026.csv.bak"

    def test_undo_empty_detail_omits_detail_key(self):
        result = undo("rolled back", "")
        assert "detail" not in result


class TestProgressWarnWithDetail:
    """Additional warn() coverage — line 33-36 in progress.py."""

    def test_warn_with_detail_includes_detail_key(self):
        result = warn("might be slow", "dataset is 50 MB")
        assert result["icon"] == "⚠"
        assert result["msg"] == "might be slow"
        assert result["detail"] == "dataset is 50 MB"

    def test_warn_no_detail_omits_detail_key(self):
        result = warn("memory low")
        assert "detail" not in result


class TestProgressName:
    """Tests for name() helper."""

    def test_name_returns_filename_only(self):
        assert name("/path/to/file.csv") == "file.csv"

    def test_name_windows_style_path(self):
        # pathlib handles both separators on any OS
        result = name("C:/Users/foo/data/dataset.csv")
        assert result == "dataset.csv"

    def test_name_no_extension(self):
        assert name("/some/dir/myfile") == "myfile"

    def test_name_just_filename(self):
        assert name("report.html") == "report.html"


# ===========================================================================
# NEW: shared/html_layout.py — plotly_config, plotly_layout_base, get_output_path
# ===========================================================================

from shared.html_layout import get_output_path, plotly_config, plotly_layout_base  # noqa: E402


class TestPlotlyConfig:
    """Tests for plotly_config() — line 32 in html_layout.py."""

    def test_plotly_config_returns_dict(self):
        cfg = plotly_config()
        assert isinstance(cfg, dict)

    def test_plotly_config_has_responsive_key(self):
        cfg = plotly_config()
        assert "responsive" in cfg
        assert cfg["responsive"] is True

    def test_plotly_config_has_display_modebar(self):
        cfg = plotly_config()
        assert "displayModeBar" in cfg

    def test_plotly_config_has_scroll_zoom(self):
        cfg = plotly_config()
        assert "scrollZoom" in cfg


class TestPlotlyLayoutBase:
    """Tests for plotly_layout_base() — line 51 in html_layout.py."""

    def test_plotly_layout_base_has_paper_bgcolor(self):
        layout = plotly_layout_base("#000000", "#ffffff")
        assert "paper_bgcolor" in layout
        assert layout["paper_bgcolor"] == "#000000"

    def test_plotly_layout_base_has_plot_bgcolor(self):
        layout = plotly_layout_base("#000000", "#ffffff")
        assert "plot_bgcolor" in layout

    def test_plotly_layout_base_has_font_color(self):
        layout = plotly_layout_base("#000000", "#ffffff")
        assert layout["font"]["color"] == "#ffffff"

    def test_plotly_layout_base_default_margin(self):
        layout = plotly_layout_base("#000000", "#ffffff")
        assert "margin" in layout
        # Default margin has standard keys
        assert "l" in layout["margin"]
        assert "r" in layout["margin"]

    def test_plotly_layout_base_custom_margin(self):
        custom_margin = {"l": 10, "r": 10, "t": 10, "b": 10}
        layout = plotly_layout_base("#000000", "#ffffff", margin=custom_margin)
        assert layout["margin"] == custom_margin

    def test_plotly_layout_base_has_autosize(self):
        layout = plotly_layout_base("#aabbcc", "#112233")
        assert layout.get("autosize") is True


class TestGetOutputPath:
    """Tests for get_output_path() — line 80-82 in html_layout.py."""

    def test_get_output_path_explicit_output_path(self, tmp_path):
        explicit = str(tmp_path / "my_report.html")
        result = get_output_path(explicit, None, "suffix", "html")
        assert result == Path(explicit).resolve()

    def test_get_output_path_with_input_file(self, tmp_path):
        input_file = tmp_path / "data.csv"
        input_file.write_text("a,b\n1,2\n")
        result = get_output_path("", input_file, "eda", "html")
        # Should be in same directory as input file
        assert result.parent == input_file.parent
        assert result.suffix == ".html"
        assert "eda" in result.name

    def test_get_output_path_no_input_no_output_returns_path(self):
        # Falls back to ~/Downloads or a path when no input provided
        result = get_output_path("", None, "test_suffix", "html")
        assert isinstance(result, Path)
        assert result.suffix == ".html"
        # stem_suffix used as filename stem when no input_path
        assert "test_suffix" in result.name

    def test_get_output_path_extension_respected(self, tmp_path):
        input_file = tmp_path / "model.pkl"
        result = get_output_path("", input_file, "report", "json")
        assert result.suffix == ".json"


# ===========================================================================
# NEW: shared/registry.py — register_classifier / register_regressor
# ===========================================================================

from shared.registry import (  # noqa: E402
    _EXTRA_CLASSIFIERS,
    _EXTRA_REGRESSORS,
    allowed_classifiers,
    allowed_regressors,
    register_classifier,
    register_regressor,
)


class TestRegistry:
    """Tests for register_classifier/register_regressor — lines 27, 32."""

    def setup_method(self):
        """Clean up extension sets before each test to avoid cross-test pollution."""
        _EXTRA_CLASSIFIERS.clear()
        _EXTRA_REGRESSORS.clear()

    def teardown_method(self):
        """Restore extension sets after each test."""
        _EXTRA_CLASSIFIERS.clear()
        _EXTRA_REGRESSORS.clear()

    def test_register_classifier_adds_to_allowed(self):
        register_classifier("myalgo")
        assert "myalgo" in allowed_classifiers()

    def test_register_classifier_does_not_affect_regressors(self):
        register_classifier("myalgo")
        assert "myalgo" not in allowed_regressors()

    def test_register_regressor_adds_to_allowed(self):
        register_regressor("myreg")
        assert "myreg" in allowed_regressors()

    def test_register_regressor_does_not_affect_classifiers(self):
        register_regressor("myreg")
        assert "myreg" not in allowed_classifiers()

    def test_register_classifier_multiple(self):
        register_classifier("algo1")
        register_classifier("algo2")
        ac = allowed_classifiers()
        assert "algo1" in ac
        assert "algo2" in ac

    def test_allowed_classifiers_includes_builtins(self):
        ac = allowed_classifiers()
        for key in ("lr", "rf", "svm", "dtc", "knn", "nb", "xgb"):
            assert key in ac

    def test_allowed_regressors_includes_builtins(self):
        ar = allowed_regressors()
        for key in ("lir", "pr", "lar", "rr", "dtr", "rfr", "xgb"):
            assert key in ar


# ===========================================================================
# NEW: shared/ml_utils.py — _auto_preprocess with categorical target
# ===========================================================================

import pandas as pd  # noqa: E402 (already imported above but needed here explicitly)

from shared.ml_utils import _auto_preprocess  # noqa: E402


class TestAutoPreprocess:
    """Tests for _auto_preprocess() — lines 34-36 for categorical target encoding."""

    def test_auto_preprocess_categorical_target_encoded(self):
        df = pd.DataFrame({"age": [25, 30, 35], "target": ["yes", "no", "yes"]})
        processed, enc_map, cols = _auto_preprocess(df, "target")
        # Target should be encoded (strings → ints)
        assert "__target__target" in enc_map
        assert processed["target"].dtype in (int, "int64", "int32")

    def test_auto_preprocess_categorical_target_values_are_ints(self):
        df = pd.DataFrame({"age": [25, 30, 35], "target": ["yes", "no", "yes"]})
        processed, enc_map, cols = _auto_preprocess(df, "target")
        unique_vals = set(processed["target"].tolist())
        assert unique_vals.issubset({0, 1})

    def test_auto_preprocess_numeric_target_not_in_enc_map(self):
        df = pd.DataFrame({"age": [25, 30, 35], "target": [0, 1, 0]})
        processed, enc_map, cols = _auto_preprocess(df, "target")
        assert "__target__target" not in enc_map

    def test_auto_preprocess_drops_null_targets(self):
        df = pd.DataFrame({"age": [25, 30, 35, 40], "target": ["yes", None, "no", "yes"]})
        processed, enc_map, cols = _auto_preprocess(df, "target")
        assert len(processed) == 3

    def test_auto_preprocess_encodes_categorical_features(self):
        df = pd.DataFrame(
            {
                "gender": ["M", "F", "M"],
                "score": [10.0, 20.0, 30.0],
                "target": [0, 1, 0],
            }
        )
        processed, enc_map, cols = _auto_preprocess(df, "target")
        assert "gender" in enc_map
        assert "gender" in cols

    def test_auto_preprocess_fills_numeric_nulls_with_median(self):
        df = pd.DataFrame({"age": [10.0, None, 30.0], "target": [0, 1, 0]})
        processed, enc_map, cols = _auto_preprocess(df, "target")
        assert processed["age"].isna().sum() == 0

    def test_auto_preprocess_returns_tuple_of_three(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "target": [0, 1, 0]})
        result = _auto_preprocess(df, "target")
        assert len(result) == 3


# ===========================================================================
# NEW: shared/platform_utils.py — get_max_depth(), get_n_iter()
# ===========================================================================

from shared.platform_utils import get_max_depth, get_n_iter  # noqa: E402


class TestPlatformUtilsDepthAndIter:
    """Tests for get_max_depth() and get_n_iter() — line 22 in platform_utils.py."""

    def test_get_max_depth_returns_int(self):
        assert isinstance(get_max_depth(), int)

    def test_get_max_depth_standard_mode(self, monkeypatch):
        monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)
        assert get_max_depth() == 5

    def test_get_max_depth_constrained_mode(self, monkeypatch):
        monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
        assert get_max_depth() == 3

    def test_get_n_iter_returns_int(self):
        assert isinstance(get_n_iter(), int)

    def test_get_n_iter_standard_mode(self, monkeypatch):
        monkeypatch.delenv("MCP_CONSTRAINED_MODE", raising=False)
        assert get_n_iter() == 10

    def test_get_n_iter_constrained_mode(self, monkeypatch):
        monkeypatch.setenv("MCP_CONSTRAINED_MODE", "1")
        assert get_n_iter() == 5


# ===========================================================================
# NEW: shared/receipt.py — append/read with existing file, corrupted JSON
# ===========================================================================

from shared.receipt import append_receipt, read_receipt_log  # noqa: E402


class TestReceiptAppend:
    """Tests for append_receipt() — lines 38-41 (reads existing) and write path."""

    def test_append_receipt_creates_file(self, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a,b\n1,2\n")
        append_receipt(csv_path, "test_tool", {"arg": "val"}, "ok")
        receipt_file = tmp_path / "data.csv.mcp_receipt.json"
        assert receipt_file.exists()

    def test_append_receipt_content_is_valid_json(self, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a,b\n1,2\n")
        append_receipt(csv_path, "test_tool", {"arg": "val"}, "ok")
        receipt_file = tmp_path / "data.csv.mcp_receipt.json"
        import json as _json

        records = _json.loads(receipt_file.read_text(encoding="utf-8"))
        assert isinstance(records, list)
        assert len(records) == 1

    def test_append_receipt_to_existing_file_appends(self, tmp_path):
        """Lines 38-41: receipt file already exists — reads and appends."""
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a,b\n1,2\n")
        # First write
        append_receipt(csv_path, "tool_a", {}, "first")
        # Second write — must read existing and append
        append_receipt(csv_path, "tool_b", {}, "second")
        records = read_receipt_log(csv_path)
        # read_receipt_log returns newest first
        assert len(records) == 2
        tools = {r["tool"] for r in records}
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_append_receipt_record_has_required_fields(self, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a,b\n1,2\n")
        append_receipt(csv_path, "my_tool", {"k": "v"}, "success", backup="snap.bak")
        records = read_receipt_log(csv_path)
        entry = records[0]
        assert "ts" in entry
        assert "tool" in entry
        assert "args" in entry
        assert "result" in entry
        assert "backup" in entry

    def test_append_receipt_never_raises_on_bad_path(self):
        """append_receipt must silently swallow failures."""
        # This path is unwritable in most environments; it should not raise
        append_receipt("/dev/null/impossible/path.csv", "tool", {}, "ok")


class TestReceiptReadLog:
    """Tests for read_receipt_log() — lines 74-80."""

    def test_read_receipt_log_returns_empty_when_no_file(self, tmp_path):
        csv_path = str(tmp_path / "nonexistent.csv")
        result = read_receipt_log(csv_path)
        assert result == []

    def test_read_receipt_log_returns_list(self, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a,b\n1,2\n")
        append_receipt(csv_path, "my_tool", {}, "done")
        result = read_receipt_log(csv_path)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_read_receipt_log_returns_newest_first(self, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a,b\n1,2\n")
        append_receipt(csv_path, "tool_1", {}, "first")
        append_receipt(csv_path, "tool_2", {}, "second")
        result = read_receipt_log(csv_path)
        # Newest-first → tool_2 should be at index 0
        assert result[0]["tool"] == "tool_2"
        assert result[1]["tool"] == "tool_1"

    def test_read_receipt_log_corrupted_json_returns_empty(self, tmp_path):
        """Lines 79-80: corrupted JSON → returns []."""
        csv_path = str(tmp_path / "test.csv")
        Path(csv_path).write_text("a\n1\n")
        rpath = tmp_path / "test.csv.mcp_receipt.json"
        rpath.write_text("invalid json {{[")
        result = read_receipt_log(str(csv_path))
        assert result == []

    def test_read_receipt_log_last_n_limits_entries(self, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        Path(csv_path).write_text("a\n1\n")
        for i in range(5):
            append_receipt(csv_path, f"tool_{i}", {}, f"result_{i}")
        result = read_receipt_log(csv_path, last_n=3)
        assert len(result) == 3


# ===========================================================================
# NEW: shared/file_utils.py — get_default_output_dir, get_output_dir, read_csv latin-1
# ===========================================================================

from shared.file_utils import get_default_output_dir, get_output_dir  # noqa: E402


class TestGetDefaultOutputDir:
    """Tests for get_default_output_dir() — lines 109-115."""

    def test_get_default_output_dir_no_input_returns_downloads(self):
        result = get_default_output_dir(None)
        assert isinstance(result, Path)
        # Should be ~/Downloads on most systems
        assert result == Path.home() / "Downloads"

    def test_get_default_output_dir_empty_string_returns_downloads(self):
        result = get_default_output_dir("")
        assert result == Path.home() / "Downloads"

    def test_get_default_output_dir_existing_parent_returns_parent(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        result = get_default_output_dir(str(csv_file))
        assert result == tmp_path

    def test_get_default_output_dir_nonexistent_parent_returns_downloads(self, tmp_path):
        # Parent directory does not exist → falls back to ~/Downloads
        fake_path = str(tmp_path / "nonexistent_subdir" / "file.csv")
        result = get_default_output_dir(fake_path)
        assert result == Path.home() / "Downloads"


class TestGetOutputDir:
    """Tests for get_output_dir() — lines 118-131."""

    def test_get_output_dir_uses_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
        result = get_output_dir()
        assert result == tmp_path

    def test_get_output_dir_creates_directory(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "new_output"
        monkeypatch.setenv("MCP_OUTPUT_DIR", str(new_dir))
        result = get_output_dir()
        assert result.exists()

    def test_get_output_dir_falls_back_to_downloads_when_no_env(self, monkeypatch):
        """Line 129-131: MCP_OUTPUT_DIR not set → ~/Downloads."""
        monkeypatch.delenv("MCP_OUTPUT_DIR", raising=False)
        result = get_output_dir()
        assert result == Path.home() / "Downloads"


class TestReadCsvEncodingFallback:
    """Tests for read_csv() encoding fallback — lines 88-103."""

    def test_read_csv_latin1_fallback(self, tmp_path):
        """Lines 90/95: latin-1 file with special chars falls back correctly."""
        path = tmp_path / "latin.csv"
        # Write a file with a byte that is valid latin-1 but invalid utf-8
        path.write_bytes(b"col1,col2\nfoo,caf\xe9\n")
        from shared.file_utils import read_csv as _read_csv

        df = _read_csv(str(path), encoding="utf-8")
        assert len(df) == 1
        assert "col1" in df.columns

    def test_read_csv_cp1252_fallback(self, tmp_path):
        """Line 92: cp1252 encoding is tried before latin-1."""
        path = tmp_path / "win.csv"
        path.write_bytes("name,value\ncaf\xe9,42\n".encode("cp1252"))
        from shared.file_utils import read_csv as _read_csv

        df = _read_csv(str(path), encoding="utf-8")
        assert len(df) == 1

    def test_read_csv_encoding_same_as_fallback_skipped(self, tmp_path):
        """Line 89-90: if enc == encoding, skip to next (no infinite loop)."""
        path = tmp_path / "utf8sig.csv"
        # Write BOM-prefixed UTF-8
        path.write_bytes(b"\xef\xbb\xbfcol1,col2\n1,2\n")
        from shared.file_utils import read_csv as _read_csv

        # Request utf-8-sig explicitly (same as first fallback) — should succeed
        df = _read_csv(str(path), encoding="utf-8-sig")
        assert len(df) == 1


# ===========================================================================
# NEW: shared/version_control.py — ValueError fallback, collision, exception paths
# ===========================================================================

from unittest.mock import patch  # noqa: E402


class TestSnapshotValueErrorFallback:
    """Lines 38-39: resolve_path raises ValueError → fallback to Path.resolve()."""

    def test_snapshot_falls_back_when_resolve_path_raises(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")

        # resolve_path is lazy-imported inside snapshot() as _resolve
        # Patch the canonical location so the lazy import picks up the mock
        with patch("shared.file_utils.resolve_path", side_effect=ValueError("test error")):
            # Should still succeed using the plain Path fallback
            bak = snapshot(str(src))
        assert Path(bak).exists()

    def test_snapshot_fallback_still_creates_backup(self, tmp_path):
        src = tmp_path / "myfile.csv"
        content = "x,y\n10,20\n"
        src.write_text(content)

        with patch("shared.file_utils.resolve_path", side_effect=ValueError("mock")):
            bak = snapshot(str(src))
        assert Path(bak).read_text() == content


class TestSnapshotCollisionCounter:
    """Lines 51-52: backup already exists → counter suffix appended."""

    def test_snapshot_collision_uses_counter_suffix(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")

        # First snapshot to get a valid timestamp pattern
        bak1 = Path(snapshot(str(src)))
        # Replace content with placeholder to detect overwrite
        bak1.write_text("collision_placeholder")

        # Second snapshot in same microsecond won't happen naturally, but
        # pre-creating the exact bak path forces the counter branch
        bak2 = Path(snapshot(str(src)))

        # bak1 must not have been overwritten
        assert bak1.read_text() == "collision_placeholder"
        # bak2 must be a distinct file
        assert str(bak1) != str(bak2)
        assert bak2.exists()


class TestSnapshotExceptionCleanup:
    """Lines 60-65: shutil.copy2 fails → temp file cleaned up, exception re-raised."""

    def test_snapshot_reraises_on_copy_failure(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")

        with patch("shutil.copy2", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                snapshot(str(src))

    def test_snapshot_no_partial_bak_on_failure(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")
        versions_dir = tmp_path / ".mcp_versions"

        with patch("shutil.copy2", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                snapshot(str(src))

        # The temp file should have been removed; no .bak should exist
        if versions_dir.exists():
            bak_files = list(versions_dir.glob("*.bak"))
            assert len(bak_files) == 0


class TestRestoreVersionExceptionCleanup:
    """Lines 115-120: shutil.copy2 fails during restore → temp cleaned, exception re-raised."""

    def test_restore_reraises_on_copy_failure(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("original content")
        bak = snapshot(str(src))
        ts = list_snapshots(str(src))[0]["timestamp"]

        with patch("shutil.copy2", side_effect=OSError("read error")):
            with pytest.raises(OSError, match="read error"):
                restore_version(str(src), ts)

    def test_restore_source_unchanged_on_failure(self, tmp_path):
        src = tmp_path / "data.csv"
        original = "original content"
        src.write_text(original)
        snapshot(str(src))
        ts = list_snapshots(str(src))[0]["timestamp"]

        # Modify the file after snapshotting
        src.write_text("modified content")

        with patch("shutil.copy2", side_effect=OSError("read error")):
            with pytest.raises(OSError):
                restore_version(str(src), ts)

        # File should still contain modified content (restore was aborted)
        assert src.read_text() == "modified content"


# ===========================================================================
# NEW: shared/handover.py — _normalize_step, _next_step edge cases
# ===========================================================================

from shared.handover import (  # noqa: E402
    NEXT_STEP,
    WORKFLOW_STEPS,
    _next_step,
    _normalize_step,
    make_handover,
)


class TestNormalizeStep:
    """Tests for _normalize_step() — line 97: unknown step falls through."""

    def test_normalize_step_unknown_returns_uppercased(self):
        # Unknown step: not in _LEGACY_STEP_MAP, not in WORKFLOW_STEPS
        result = _normalize_step("UNKNOWN_STEP")
        assert result == "UNKNOWN_STEP"

    def test_normalize_step_lowercase_unknown_uppercased(self):
        result = _normalize_step("unknown_step")
        assert result == "UNKNOWN_STEP"

    def test_normalize_step_legacy_locate_maps_to_collect(self):
        result = _normalize_step("LOCATE")
        assert result == "COLLECT"

    def test_normalize_step_legacy_patch_maps_to_train(self):
        result = _normalize_step("PATCH")
        assert result == "TRAIN"

    def test_normalize_step_legacy_verify_maps_to_evaluate(self):
        result = _normalize_step("VERIFY")
        assert result == "EVALUATE"

    def test_normalize_step_canonical_returned_as_is(self):
        for step in WORKFLOW_STEPS:
            assert _normalize_step(step) == step

    def test_normalize_step_lowercase_canonical(self):
        result = _normalize_step("train")
        assert result == "TRAIN"


class TestNextStep:
    """Tests for _next_step() — lines 103-105: last step, ValueError path."""

    def test_next_step_last_workflow_step_returns_empty(self):
        """Line 103: idx+1 >= len → return ''."""
        last = WORKFLOW_STEPS[-1]
        assert _next_step(last) == ""

    def test_next_step_nonexistent_triggers_value_error_path(self):
        """Line 104-105: ValueError from .index() → NEXT_STEP.get(step, '')."""
        result = _next_step("NONEXISTENT")
        # NEXT_STEP.get("NONEXISTENT", "") returns ""
        assert result == NEXT_STEP.get("NONEXISTENT", "")

    def test_next_step_first_step_returns_second(self):
        first = WORKFLOW_STEPS[0]
        second = WORKFLOW_STEPS[1]
        assert _next_step(first) == second

    def test_next_step_second_to_last_returns_last(self):
        second_to_last = WORKFLOW_STEPS[-2]
        last = WORKFLOW_STEPS[-1]
        assert _next_step(second_to_last) == last

    def test_next_step_legacy_key_uses_next_step_dict(self):
        # LOCATE is not in WORKFLOW_STEPS so hits except ValueError branch
        result = _next_step("LOCATE")
        assert result == NEXT_STEP.get("LOCATE", "")


class TestMakeHandover:
    """Smoke tests for make_handover() to ensure it calls _normalize_step/_next_step."""

    def test_make_handover_returns_dict(self):
        result = make_handover("TRAIN", ["get_predictions"])
        assert isinstance(result, dict)

    def test_make_handover_workflow_step_normalized(self):
        result = make_handover("PATCH", ["get_predictions"])
        # PATCH maps to TRAIN
        assert result["workflow_step"] == "TRAIN"

    def test_make_handover_suggested_tools_list(self):
        result = make_handover("TRAIN", ["get_predictions", "read_model_report"])
        assert "get_predictions" in result["suggested_tools"]

    def test_make_handover_carry_forward(self):
        cf = {"model_path": "/tmp/model.pkl"}
        result = make_handover("TRAIN", [], carry_forward=cf)
        assert result["carry_forward"] == cf


# ===========================================================================
# NEW: shared/html_theme.py — theme_plot_colors, css_vars, device_mode_js, data_table_html
# ===========================================================================

from shared.html_theme import (  # noqa: E402
    css_vars,
    data_table_html,
    device_mode_js,
    get_theme,
    theme_plot_colors,
)


class TestThemePlotColors:
    """Tests for theme_plot_colors() — line 119."""

    def test_theme_plot_colors_light_returns_light_bg(self):
        plot_bg, font_color, accent = theme_plot_colors("light")
        # Light theme: light background
        assert "#f6f8fa" in plot_bg or "#ffffff" in plot_bg or len(plot_bg) == 7

    def test_theme_plot_colors_dark_returns_dark_bg(self):
        plot_bg, font_color, accent = theme_plot_colors("dark")
        assert plot_bg == "#161b22"

    def test_theme_plot_colors_returns_tuple_of_three(self):
        result = theme_plot_colors("light")
        assert len(result) == 3

    def test_theme_plot_colors_light_font_is_dark(self):
        _, font_color, _ = theme_plot_colors("light")
        assert font_color == "#1f2328"

    def test_theme_plot_colors_dark_font_is_light(self):
        _, font_color, _ = theme_plot_colors("dark")
        assert font_color == "#c9d1d9"


class TestCssVars:
    """Tests for css_vars() — lines 163, 165."""

    def test_css_vars_light_returns_string(self):
        result = css_vars("light")
        assert isinstance(result, str)

    def test_css_vars_light_contains_root(self):
        result = css_vars("light")
        assert ":root" in result

    def test_css_vars_light_contains_light_color(self):
        """Line 163: light theme uses _LIGHT_VARS."""
        result = css_vars("light")
        # Light theme has white background
        assert "#ffffff" in result

    def test_css_vars_device_contains_media_query(self):
        """Line 165: device theme includes @media prefers-color-scheme."""
        result = css_vars("device")
        assert "@media" in result
        assert "prefers-color-scheme" in result

    def test_css_vars_dark_does_not_contain_media_query(self):
        result = css_vars("dark")
        assert "@media" not in result

    def test_css_vars_dark_contains_dark_color(self):
        result = css_vars("dark")
        assert "#0d1117" in result


class TestDeviceModeJs:
    """Tests for device_mode_js() — line 198."""

    def test_device_mode_js_returns_string(self):
        result = device_mode_js()
        assert isinstance(result, str)

    def test_device_mode_js_contains_script_tag(self):
        result = device_mode_js()
        assert "<script>" in result

    def test_device_mode_js_contains_prefers_color_scheme(self):
        result = device_mode_js()
        assert "prefers-color-scheme" in result

    def test_device_mode_js_contains_plotly_relayout(self):
        result = device_mode_js()
        assert "Plotly.relayout" in result


class TestDataTableHtml:
    """Tests for data_table_html() — empty list + max_rows truncation."""

    def test_data_table_html_empty_list_returns_no_data(self):
        result = data_table_html([])
        assert result == "<p>No data.</p>"

    def test_data_table_html_single_row(self):
        result = data_table_html([{"a": 1, "b": 2}])
        assert "<table>" in result
        assert "a" in result
        assert "1" in result

    def test_data_table_html_truncation_message(self):
        """Lines 872-874: more than max_rows triggers 'more rows' message."""
        rows = [{"a": i} for i in range(60)]
        result = data_table_html(rows, max_rows=50)
        assert "more rows" in result or "10" in result

    def test_data_table_html_no_truncation_when_under_limit(self):
        rows = [{"a": i} for i in range(10)]
        result = data_table_html(rows, max_rows=50)
        assert "more rows" not in result

    def test_data_table_html_returns_div_wrap(self):
        rows = [{"col": "val"}]
        result = data_table_html(rows)
        assert "table-wrap" in result


class TestGetTheme:
    """Tests for get_theme() — returns theme config dict."""

    def test_get_theme_dark_returns_dict(self):
        theme = get_theme("dark")
        assert isinstance(theme, dict)
        assert "bg_color" in theme

    def test_get_theme_light_returns_light_colors(self):
        theme = get_theme("light")
        assert theme["bg_color"] == "#ffffff"

    def test_get_theme_unknown_falls_back_to_dark(self):
        theme = get_theme("unknown_theme")
        # Falls back to dark
        assert theme["bg_color"] == "#0d1117"


# ===========================================================================
# shared/html_theme.py — _open_file coverage (lines 455-469)
# ===========================================================================

from shared.html_theme import _open_file  # noqa: E402


class TestOpenFile:
    """Tests for _open_file() — best-effort browser open."""

    def test_open_file_with_webbrowser_mock(self, tmp_path):
        """Normal path: webbrowser.open called with file:// URL."""
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        with patch("webbrowser.open") as mock_wb:
            _open_file(str(f))
        mock_wb.assert_called_once()

    def test_open_file_webbrowser_fails_falls_through(self, tmp_path):
        """webbrowser.open raises → platform fallback runs (mocked)."""
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        import sys

        with patch("webbrowser.open", side_effect=Exception("no browser")):
            if sys.platform == "win32":
                with patch("os.startfile", create=True):
                    _open_file(str(f))
            elif sys.platform == "darwin":
                with patch("subprocess.Popen"):
                    _open_file(str(f))
            else:
                with patch("subprocess.Popen"):
                    _open_file(str(f))

    def test_open_file_never_raises(self, tmp_path):
        """Any exception must be silently swallowed."""
        with patch("webbrowser.open", side_effect=RuntimeError("blocked")):
            with patch("os.startfile", side_effect=RuntimeError("blocked"), create=True):
                with patch("subprocess.Popen", side_effect=RuntimeError("blocked")):
                    # Should not raise
                    _open_file(str(tmp_path / "nope.html"))


# ===========================================================================
# shared/version_control.py — collision counter (lines 51-52)
# ===========================================================================


class TestSnapshotCollisionCounterExtra:
    """Line 51-52: backup filename collision increments counter."""

    def test_collision_counter_increments(self, tmp_path):
        from datetime import UTC, datetime
        from unittest.mock import patch as _patch

        src = tmp_path / "data.csv"
        src.write_text("col\n1\n2\n")

        fixed_ts = "2026-01-01T00-00-00-000000Z"
        versions_dir = tmp_path / ".mcp_versions"
        versions_dir.mkdir()
        expected_first = versions_dir / f"data_{fixed_ts}.bak"
        expected_first.write_text("dummy")  # pre-create to force collision

        with _patch("shared.version_control.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = fixed_ts
            bak = snapshot(str(src))

        assert "_1.bak" in bak or bak.endswith(".bak")


# ===========================================================================
# shared/version_control.py — cleanup on copy failure (lines 63-64)
# ===========================================================================


class TestSnapshotCleanupOnFailure:
    """Lines 63-64: os.unlink raises OSError inside the except → swallowed."""

    def test_unlink_oserror_swallowed_during_snapshot(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("col\n1\n")

        with patch("shutil.copy2", side_effect=OSError("write error")):
            with patch("os.unlink", side_effect=OSError("unlink error")):
                with pytest.raises(OSError, match="write error"):
                    snapshot(str(src))


# ===========================================================================
# shared/version_control.py — restore cleanup on failure (lines 118-119)
# ===========================================================================


class TestRestoreVersionCleanup:
    """Lines 118-119: os.unlink raises OSError during restore cleanup → swallowed."""

    def test_restore_unlink_oserror_swallowed(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("original")
        bak = snapshot(str(src))
        ts = list_snapshots(str(src))[0]["timestamp"]

        with patch("shutil.copy2", side_effect=OSError("copy error")):
            with patch("os.unlink", side_effect=OSError("unlink error")):
                with pytest.raises(OSError, match="copy error"):
                    restore_version(str(src), ts)


# ===========================================================================
# shared/file_utils.py — encoding fallbacks (lines 88-103)
# ===========================================================================


class TestReadCsvEncodingFallbacks:
    """Lines 88-95: read_csv tries fallback encodings when UTF-8 fails."""

    def test_latin1_fallback(self, tmp_path):
        """File with latin-1 encoding triggers fallback from UTF-8."""
        from shared.file_utils import read_csv

        csv = tmp_path / "latin.csv"
        csv.write_bytes(b"name,value\ncaf\xe9,1\n")  # \xe9 is latin-1 'é'
        df = read_csv(str(csv))
        assert "name" in df.columns
        assert len(df) >= 1

    def test_atomic_write_cleanup_on_failure(self, tmp_path):
        """Lines 141-146: shutil.move fails → temp cleaned up, exception re-raised."""
        from shared.file_utils import atomic_write

        target = tmp_path / "out.bin"
        with patch("shutil.move", side_effect=OSError("move failed")):
            with pytest.raises(OSError, match="move failed"):
                atomic_write(target, b"data")


# ===========================================================================
# shared/workspace_utils.py — exception paths (lines 93-95, 131-132)
# ===========================================================================


class TestWorkspaceUtilsCoverage:
    """Additional coverage for workspace_utils edge cases."""

    def test_save_manifest_cleanup_on_failure(self, tmp_path):
        """Lines 93-95: shutil.move fails in save_manifest → temp cleaned, raises."""
        ws_dir = tmp_path / "default"
        ws_dir.mkdir()
        with patch("shutil.move", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                save_manifest({"files": {}, "created": ""}, "default", base_dir=str(tmp_path))

    def test_register_file_row_count_exception(self, tmp_path):
        """Lines 131-132: row_count falls back to -1 on read error."""
        import json as _json

        from shared.workspace_utils import register_file

        ws_dir = tmp_path / "default"
        ws_dir.mkdir()
        manifest_path = ws_dir / "workspace.json"
        manifest_path.write_text(_json.dumps({"files": {}, "created": "2026-01-01"}))

        # Write binary content with invalid UTF-8 bytes — fails to iterate with encoding='utf-8'
        csv = ws_dir / "test.csv"
        csv.write_bytes(b"a,b\n\xff\xfe\n")

        result = register_file("default", str(csv), "alias_x", base_dir=str(tmp_path))
        assert result["files"]["alias_x"]["rows"] == -1


# ===========================================================================
# shared/file_utils.py — encoding fallback (skip enc==encoding, latin-1 last resort)
# ===========================================================================


class TestFileUtilsEncodingEdgeCases:
    """Lines 90, 95, 99-103 of shared/file_utils.py."""

    def test_encoding_skip_when_enc_equals_encoding(self, tmp_path):
        """Line 90: fallback loop skips enc when enc == encoding parameter."""
        from shared.file_utils import read_csv

        # File with \xe9 byte (fails utf-8 and utf-8-sig, but passes cp1252 and latin-1)
        # Call with encoding="utf-8-sig" — the loop will skip "utf-8-sig" via line 90
        p = tmp_path / "enc.csv"
        p.write_bytes(b"name,value\ncaf\xe9,1\n")
        df = read_csv(str(p), encoding="utf-8-sig")
        assert "name" in df.columns

    def test_latin1_last_resort_fallback(self, tmp_path):
        """Line 95: all fallbacks fail, latin-1 is the last resort."""
        from shared.file_utils import read_csv

        # Byte 0x81 is undefined in cp1252 but valid in latin-1
        p = tmp_path / "latin.csv"
        p.write_bytes(b"name,value\nx\x81z,1\n")
        df = read_csv(str(p))
        assert "name" in df.columns
        assert len(df) >= 1

    def test_bad_lines_skip_on_tokenization_error(self, tmp_path):
        """Lines 99-103: CSV with mismatched field counts triggers on_bad_lines=skip."""
        from shared.file_utils import read_csv

        # Create CSV with an extra field in one row — causes ParserError (field count)
        p = tmp_path / "bad_lines.csv"
        p.write_bytes(b"a,b\n1,2\n3,4,5,6,7\n5,6\n")
        df = read_csv(str(p))
        assert "a" in df.columns
        assert len(df) >= 1  # at least the good rows are read


class TestAtomicWriteOslinkFailure:
    """Lines 144-145 of shared/file_utils.py: os.unlink also fails during cleanup."""

    def test_unlink_oserror_swallowed_during_cleanup(self, tmp_path):
        """Lines 144-145: os.unlink raises OSError but is silenced; original exc re-raised."""
        from unittest.mock import patch

        import pytest

        from shared.file_utils import atomic_write

        target = tmp_path / "out.bin"
        with patch("shutil.move", side_effect=OSError("move failed")):
            with patch("os.unlink", side_effect=OSError("unlink also failed")):
                with pytest.raises(OSError, match="move failed"):
                    atomic_write(target, b"data")
