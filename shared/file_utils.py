"""Path resolution and atomic write helpers.

resolve_path() blocks genuine path traversal attacks (null bytes, encoded
separators) while allowing any absolute path the user explicitly provides,
including paths on different drives (e.g. D:\\ on Windows).
"""

import json
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


def get_output_dir() -> Path:
    """Return the standard output directory for generated files.

    All generated outputs (models, HTML reports, charts, exported CSVs,
    predictions) are saved directly to ~/Downloads/. The directory is
    created automatically if it does not exist.
    """
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
