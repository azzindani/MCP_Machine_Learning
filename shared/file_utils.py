"""Path resolution, CSV reading, and atomic write helpers.

resolve_path() supports:
  - 'project:name/alias' -> resolves via shared project_utils (DA-compatible)
  - Absolute/relative paths with null-byte and filesystem-root blocking

read_csv() provides auto-encoding detection with utf-8-sig / cp1252 / latin-1
fallbacks and on_bad_lines='skip' recovery, compatible with files produced by
MCP_Data_Analyst so encoding-detected files hand over cleanly between servers.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def resolve_path(
    file_path: str,
    allowed_extensions: tuple[str, ...] = (),
) -> Path:
    """Resolve to absolute path. Supports project:name/alias syntax.

    If file_path starts with 'project:', delegates to project_utils.resolve_alias
    to translate the alias into an absolute path (DA-compatible format).
    Both servers share the same ~/mcp_projects root so aliases registered
    by MCP_Data_Analyst are resolvable here without any extra configuration.

    Also blocks null bytes and bare filesystem roots for path traversal safety.

    Raises:
        ValueError: invalid path, null byte, filesystem root, or bad extension
        FileNotFoundError: project alias project not found
    """
    # Project alias resolution (DA-compatible)
    if file_path.startswith("project:"):
        try:
            from shared.project_utils import resolve_alias
            path = resolve_alias(file_path)
        except Exception as exc:
            raise ValueError(f"Cannot resolve project alias '{file_path}': {exc}") from exc
    else:
        raw = str(file_path)
        # Block null bytes — common in traversal payloads
        if "\x00" in raw:
            raise ValueError(f"Invalid path (null byte): {file_path}")
        path = Path(raw).resolve()
        # Block bare filesystem roots — nothing useful lives at '/' or 'C:\\'
        if path.parent == path:
            raise ValueError(f"Path resolves to filesystem root: {file_path}")

    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Extension {path.suffix!r} not allowed. Expected one of: {', '.join(allowed_extensions)}"
        )

    return path


# ---------------------------------------------------------------------------
# read_csv — encoding-aware, compatible with DA-produced files
# ---------------------------------------------------------------------------

_ENCODING_FALLBACKS = ("utf-8-sig", "cp1252", "latin-1")


def read_csv(
    file_path: str,
    encoding: str = "utf-8",
    separator: str = ",",
    max_rows: int = 0,
) -> pd.DataFrame:
    """Read CSV with automatic encoding detection and bad-line fallback.

    Tries the specified encoding first. On UnicodeDecodeError walks through
    utf-8-sig (BOM), cp1252 (Windows/Excel), then latin-1 (never fails).
    On tokenization errors (mismatched field counts) retries with
    on_bad_lines='skip' to drop malformed rows.

    Strips leading/trailing whitespace from column names.
    Compatible with files produced by MCP_Data_Analyst.
    """
    kwargs: dict = {"sep": separator, "low_memory": False}
    if max_rows > 0:
        kwargs["nrows"] = max_rows

    def _try_encs(extra: dict) -> pd.DataFrame:
        kw = {**kwargs, **extra}
        try:
            return pd.read_csv(file_path, encoding=encoding, **kw)
        except UnicodeDecodeError:
            pass
        for enc in _ENCODING_FALLBACKS:
            if enc == encoding:
                continue
            try:
                return pd.read_csv(file_path, encoding=enc, **kw)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(file_path, encoding="latin-1", **kw)

    try:
        df = _try_encs({})
    except Exception as exc:
        if "tokeniz" in str(exc).lower() or "field" in str(exc).lower():
            df = _try_encs({"on_bad_lines": "skip"})
        else:
            raise

    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------


def get_default_output_dir(input_path: str | None = None) -> Path:
    """Return default output dir: input file's parent if provided, else ~/Downloads."""
    if input_path:
        p = Path(input_path).resolve()
        if p.parent.exists():
            return p.parent
    return Path.home() / "Downloads"


def get_output_dir() -> Path:
    """Return the standard output directory for generated files.

    Checks MCP_OUTPUT_DIR env var first (used in tests to redirect to tmp_path),
    then falls back to ~/Downloads/. The directory is created automatically.
    """
    override = os.environ.get("MCP_OUTPUT_DIR")
    if override:
        out = Path(override)
        out.mkdir(parents=True, exist_ok=True)
        return out
    out = Path.home() / "Downloads"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def atomic_write(target: Path, content: bytes) -> None:
    """Write bytes to target atomically via temp file + move."""
    fd, tmp_path = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        shutil.move(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(target: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to target atomically."""
    atomic_write(target, content.encode(encoding))


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".json",
        dir=path.parent,
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
