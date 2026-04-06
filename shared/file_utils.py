"""Path resolution and atomic write helpers.

resolve_path() enforces a home-directory boundary — rejects paths that escape
the user's home directory (path traversal prevention).
"""

import json
import shutil
import tempfile
from pathlib import Path


def resolve_path(
    file_path: str,
    allowed_extensions: tuple[str, ...] = (),
) -> Path:
    """Resolve to absolute path, enforce home-dir boundary, validate extension.

    Raises:
        ValueError: path escapes home directory or extension not allowed
    """
    path = Path(file_path).resolve()

    home = Path.home().resolve()
    try:
        path.relative_to(home)
    except ValueError:
        raise ValueError(f"Path outside allowed directory: {file_path}")

    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Extension {path.suffix!r} not allowed. Expected one of: {', '.join(allowed_extensions)}")

    return path


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
