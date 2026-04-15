"""Shared project workspace utilities — compatible with MCP_Data_Analyst format.

Resolves 'project:project_name/alias' paths registered by MCP_Data_Analyst
or by ML tools. Both servers share ~/mcp_projects/ as the common root
directory, so aliases registered by either server are resolvable by the other.

Alias syntax: "project:project_name/alias"
  -> resolves to absolute path via project.json manifest.

Manifest format is identical to MCP_Data_Analyst's project.json schema so
projects created by DA can be opened directly by ML tools.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECTS_ROOT_ENV = "MCP_PROJECTS_DIR"
_DEFAULT_PROJECTS_DIR = Path.home() / "mcp_projects"
_MANIFEST_FILENAME = "project.json"
_ALIAS_PREFIX = "project:"


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


def get_projects_root(base_dir: str = "") -> Path:
    """Return root directory for all projects.

    Priority: base_dir arg -> MCP_PROJECTS_DIR env -> ~/mcp_projects
    Compatible with MCP_Data_Analyst's project root convention.
    """
    if base_dir:
        return Path(base_dir).expanduser().resolve()
    env_dir = os.environ.get(_PROJECTS_ROOT_ENV, "")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return _DEFAULT_PROJECTS_DIR


def get_project_dir(name: str, base_dir: str = "") -> Path:
    """Return directory for a named project."""
    return get_projects_root(base_dir) / name


# ---------------------------------------------------------------------------
# Manifest read / write
# ---------------------------------------------------------------------------


def load_manifest(project_name: str, base_dir: str = "") -> dict:
    """Load project.json for project_name. Raises FileNotFoundError if absent."""
    manifest_path = get_project_dir(project_name, base_dir) / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Project '{project_name}' not found. Expected manifest at: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, project_name: str, base_dir: str = "") -> None:
    """Write manifest dict to project.json atomically."""
    project_dir = get_project_dir(project_name, base_dir)
    manifest_path = project_dir / _MANIFEST_FILENAME
    tmp_fd, tmp_path = tempfile.mkstemp(dir=project_dir, suffix=".tmp")
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        shutil.move(tmp_path, manifest_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def resolve_alias(alias_str: str, base_dir: str = "") -> Path:
    """Resolve 'project:project_name/alias' to an absolute Path.

    Compatible with MCP_Data_Analyst alias format. Both servers share the
    same ~/mcp_projects root, so aliases registered by either server are
    resolvable by the other.

    Raises ValueError / FileNotFoundError on lookup failures.
    """
    if not alias_str.startswith(_ALIAS_PREFIX):
        return Path(alias_str).expanduser().resolve()

    rest = alias_str[len(_ALIAS_PREFIX):]
    if "/" not in rest:
        raise ValueError(
            f"Invalid alias format '{alias_str}'. Expected 'project:project_name/alias'."
        )
    project_name, file_alias = rest.split("/", 1)
    manifest = load_manifest(project_name, base_dir)
    files = manifest.get("files", {})
    if file_alias not in files:
        available = list(files.keys())
        raise ValueError(
            f"Alias '{file_alias}' not found in project '{project_name}'. "
            f"Available: {available}"
        )
    stored_path = files[file_alias]["path"]
    project_dir = get_project_dir(project_name, base_dir)
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return candidate.resolve()


def is_alias(path_or_alias: str) -> bool:
    """Return True if string uses the project:name/alias syntax."""
    return path_or_alias.startswith(_ALIAS_PREFIX)


# ---------------------------------------------------------------------------
# File registration
# ---------------------------------------------------------------------------


def register_file(
    project_name: str,
    file_path: str,
    alias: str,
    stage: str = "working",
    base_dir: str = "",
) -> dict:
    """Add a file alias to the project manifest. Returns updated manifest.

    ML tools call this to register preprocessed CSVs, prediction outputs,
    or any derived file into a shared project, making them accessible to
    MCP_Data_Analyst tools and vice versa.

    stage: raw | working | trial | output
    """
    valid_stages = {"raw", "working", "trial", "output"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Valid: {', '.join(sorted(valid_stages))}")

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    manifest = load_manifest(project_name, base_dir)
    project_dir = get_project_dir(project_name, base_dir)

    try:
        rel = path.relative_to(project_dir)
        stored_path = str(rel)
    except ValueError:
        stored_path = str(path)

    try:
        row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
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
    save_manifest(manifest, project_name, base_dir)
    return manifest


# ---------------------------------------------------------------------------
# Project directory scaffolding
# ---------------------------------------------------------------------------


def create_project_dirs(project_name: str, base_dir: str = "") -> dict:
    """Create standard project directory structure. Returns paths dict."""
    project_dir = get_project_dir(project_name, base_dir)
    dirs = {
        "root": project_dir,
        "data_raw": project_dir / "data" / "raw",
        "data_working": project_dir / "data" / "working",
        "data_trials": project_dir / "data" / "trials",
        "reports": project_dir / "reports",
        "models": project_dir / "models",
        "pipelines": project_dir / "pipelines",
        "versions": project_dir / ".versions",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


__all__ = [
    "get_projects_root",
    "get_project_dir",
    "load_manifest",
    "save_manifest",
    "resolve_alias",
    "is_alias",
    "register_file",
    "create_project_dirs",
]
