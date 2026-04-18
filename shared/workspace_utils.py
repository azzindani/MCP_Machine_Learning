"""Shared workspace utilities — universal workspace management for MCP ecosystem.

Alias syntax:
  "workspace:name/alias"  canonical (new)
  "project:name/alias"    legacy backward-compatible

Both resolve to absolute paths via workspace.json (or legacy project.json)
stored under the workspace root directory.

Environment variables:
  MCP_WORKSPACE_DIR  override workspace root (preferred)
  MCP_PROJECTS_DIR   legacy override (still honoured)

Compatible with MCP_Data_Analyst format — both servers share the same root
so aliases registered by either server are resolvable by the other.

All I/O errors are raised; callers must catch them.
No MCP imports. No stdout writes.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_WORKSPACE_ROOT_ENV = "MCP_WORKSPACE_DIR"
_LEGACY_ROOT_ENV = "MCP_PROJECTS_DIR"
_DEFAULT_WORKSPACE_DIR = Path.home() / "mcp_workspace"
_MANIFEST_FILENAME = "workspace.json"
_LEGACY_MANIFEST = "project.json"
_ALIAS_PREFIX = "workspace:"
_LEGACY_ALIAS_PREFIX = "project:"


def get_workspace_root(base_dir: str = "") -> Path:
    """Return root directory that holds all workspaces.

    Priority: base_dir arg -> MCP_WORKSPACE_DIR -> MCP_PROJECTS_DIR -> ~/mcp_workspace
    Compatible with MCP_Data_Analyst's workspace root convention.
    """
    if base_dir:
        return Path(base_dir).expanduser().resolve()
    env_dir = os.environ.get(_WORKSPACE_ROOT_ENV, "")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    legacy_dir = os.environ.get(_LEGACY_ROOT_ENV, "")
    if legacy_dir:
        return Path(legacy_dir).expanduser().resolve()
    return _DEFAULT_WORKSPACE_DIR


def get_workspace_dir(name: str, base_dir: str = "") -> Path:
    """Return directory for a named workspace."""
    return get_workspace_root(base_dir) / name


# Backward-compat aliases
get_projects_root = get_workspace_root
get_project_dir = get_workspace_dir


def load_manifest(workspace_name: str, base_dir: str = "") -> dict:
    """Load manifest for workspace_name.

    Tries workspace.json first, falls back to project.json for legacy workspaces
    created by older tool versions or MCP_Data_Analyst.
    Raises FileNotFoundError if neither exists.
    """
    ws_dir = get_workspace_dir(workspace_name, base_dir)
    for filename in (_MANIFEST_FILENAME, _LEGACY_MANIFEST):
        path = ws_dir / filename
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Workspace '{workspace_name}' not found. "
        f"Expected manifest at: {ws_dir / _MANIFEST_FILENAME}"
    )


def save_manifest(manifest: dict, workspace_name: str, base_dir: str = "") -> None:
    """Write manifest dict to workspace.json atomically."""
    ws_dir = get_workspace_dir(workspace_name, base_dir)
    manifest_path = ws_dir / _MANIFEST_FILENAME
    tmp_fd, tmp_path = tempfile.mkstemp(dir=ws_dir, suffix=".tmp")
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        shutil.move(tmp_path, manifest_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def register_file(
    workspace_name: str,
    file_path: str,
    alias: str,
    stage: str = "working",
    base_dir: str = "",
) -> dict:
    """Add a file alias to the workspace manifest. Returns updated manifest.

    ML tools call this to register preprocessed CSVs, prediction outputs, or
    any derived file, making them accessible to MCP_Data_Analyst and vice versa.

    stage: raw | working | trial | output
    """
    valid_stages = {"raw", "working", "trial", "output"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Valid: {', '.join(sorted(valid_stages))}")

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    manifest = load_manifest(workspace_name, base_dir)
    ws_dir = get_workspace_dir(workspace_name, base_dir)

    try:
        rel = path.relative_to(ws_dir)
        stored_path = str(rel)
    except ValueError:
        stored_path = str(path)

    try:
        row_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
    except Exception:
        row_count = -1

    manifest["files"][alias] = {
        "path": stored_path,
        "stage": stage,
        "rows": row_count,
        "size_bytes": path.stat().st_size,
        "registered": datetime.now(UTC).isoformat(),
    }
    manifest["updated"] = datetime.now(UTC).isoformat()
    save_manifest(manifest, workspace_name, base_dir)
    return manifest


def resolve_alias(alias_str: str, base_dir: str = "") -> Path:
    """Resolve workspace:name/alias or project:name/alias to an absolute Path.

    Compatible with MCP_Data_Analyst alias format. Both servers share the same
    workspace root so aliases registered by either server are resolvable by the
    other without any extra configuration.

    Returns Path(alias_str) unchanged if neither prefix is present.
    Raises ValueError / FileNotFoundError on lookup failures.
    """
    prefix: str | None = None
    if alias_str.startswith(_ALIAS_PREFIX):
        prefix = _ALIAS_PREFIX
    elif alias_str.startswith(_LEGACY_ALIAS_PREFIX):
        prefix = _LEGACY_ALIAS_PREFIX

    if prefix is None:
        return Path(alias_str).expanduser().resolve()

    rest = alias_str[len(prefix) :]
    if "/" not in rest:
        raise ValueError(
            f"Invalid alias format '{alias_str}'. "
            f"Expected 'workspace:name/alias' or 'project:name/alias'."
        )
    workspace_name, file_alias = rest.split("/", 1)
    manifest = load_manifest(workspace_name, base_dir)
    files = manifest.get("files", {})
    if file_alias not in files:
        available = list(files.keys())
        raise ValueError(
            f"Alias '{file_alias}' not found in workspace '{workspace_name}'. "
            f"Available: {available}"
        )
    stored_path = files[file_alias]["path"]
    ws_dir = get_workspace_dir(workspace_name, base_dir)
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = ws_dir / candidate
    return candidate.resolve()


def is_alias(path_or_alias: str) -> bool:
    """Return True if string uses workspace: or legacy project: alias syntax."""
    return path_or_alias.startswith(_ALIAS_PREFIX) or path_or_alias.startswith(_LEGACY_ALIAS_PREFIX)


def create_workspace_dirs(workspace_name: str, base_dir: str = "") -> dict:
    """Create standard workspace directory structure. Returns paths dict."""
    ws_dir = get_workspace_dir(workspace_name, base_dir)
    dirs: dict[str, Path] = {
        "root": ws_dir,
        "data_raw": ws_dir / "data" / "raw",
        "data_working": ws_dir / "data" / "working",
        "data_trials": ws_dir / "data" / "trials",
        "reports": ws_dir / "reports",
        "models": ws_dir / "models",
        "pipelines": ws_dir / "pipelines",
        "versions": ws_dir / ".versions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


# Backward-compat alias
create_project_dirs = create_workspace_dirs


__all__ = [
    "get_workspace_root",
    "get_workspace_dir",
    "get_projects_root",
    "get_project_dir",
    "load_manifest",
    "save_manifest",
    "register_file",
    "resolve_alias",
    "is_alias",
    "create_workspace_dirs",
    "create_project_dirs",
]
