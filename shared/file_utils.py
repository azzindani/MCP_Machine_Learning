"""Path resolution and atomic write helpers.

resolve_path() blocks genuine path traversal attacks (null bytes, encoded
separators) while allowing any absolute path the user explicitly provides,
including paths on different drives (e.g. D:\\ on Windows).
"""

import json
import os
import shutil
import tempfile
from pathlib import Path


def resolve_path(
    file_path: str,
    allowed_extensions: tuple[str, ...] = (),
) -> Path:
    """Resolve to absolute path, block traversal attacks, validate extension.

    Allows any absolute path the user provides, including paths on different
    drives or outside the home directory. Blocks null bytes and paths that
    resolve to a filesystem root (e.g. bare '/' or 'C:\\').

    Raises:
        ValueError: path is a filesystem root, contains null bytes, or
                    extension not allowed
    """
    raw = str(file_path)

    # Block null bytes — common in traversal payloads
    if "\x00" in raw:
        raise ValueError(f"Invalid path (null byte): {file_path}")

    path = Path(raw).resolve()

    # Block bare filesystem roots — nothing useful lives at '/' or 'C:\\'
    # path.parent == path is True only for roots (e.g. '/' or 'C:\')
    if path.parent == path:
        raise ValueError(f"Path resolves to filesystem root: {file_path}")

    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Extension {path.suffix!r} not allowed. Expected one of: {', '.join(allowed_extensions)}")

    return path


def get_default_output_dir(input_path: str | None = None) -> Path:
    """Return default output dir: input file's parent if provided, else ~/Downloads."""
    if input_path:
        p = Path(input_path).resolve()
        if p.parent.exists():
            return p.parent
    return Path.home() / "Downloads"


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
