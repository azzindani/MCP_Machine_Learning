"""Snapshot and restore functionality for all MCP ML servers.

Every tool that writes to disk calls snapshot() before writing.
Backups are stored in .mcp_versions/ next to the source file.

Naming: {stem}_{timestamp}{ext}.bak  where timestamp uses microsecond precision
plus a counter suffix for collision safety. Format matches MCP_File_System and
MCP_Data_Analyst so all three servers' backups coexist in the same
.mcp_versions/ directory and are cross-restorable.

The extension used to be dropped, and that made one file's history another
file's history: `report.csv` and `report.docx` in one directory both
snapshotted to `report_{ts}.bak`, and `list_snapshots` found them with
`glob(f"{stem}_*.bak")`. Against the live endpoints, restoring a CSV with no
timestamp returned the newest snapshot under that stem -- the Word document --
and reported success. The same glob also let `Ad_Data_test.csv`'s snapshots
answer a query about `Ad_Data.csv`.

Reading stays more forgiving than writing so snapshots taken before this change
are not stranded, but an extension-less legacy name is only offered when nothing
else in the directory shares the stem -- exactly where it cannot be ambiguous.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# A snapshot name is the stem, an underscore, then a UTC timestamp that always
# begins with a four-digit year. Globbing `{stem}_*` alone lets a snapshot of
# `Ad_Data_test.csv` answer a query about `Ad_Data.csv`.
_TS_GLOB = "[0-9][0-9][0-9][0-9]-*"


def _legacy_is_unambiguous(path: Path) -> bool:
    """True when no other file beside this one shares its stem."""
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        return False
    return not any(p.is_file() and p.stem == path.stem and p.suffix != path.suffix for p in siblings)


def snapshot(file_path: str) -> str:
    """Snapshot file to .mcp_versions/ atomically. Returns backup path string.

    Uses microsecond timestamps and a counter suffix to avoid collisions on
    rapid successive saves (important on Windows where datetime resolution can
    be coarser than microseconds). Atomic via temp file + shutil.move so a
    mid-copy crash cannot leave a partial .bak file.

    Backup filename format: {stem}_{ts}{ext}.bak  (matches DA and File_System).

    Raises:
        FileNotFoundError: source file does not exist
    """
    try:
        from shared.file_utils import resolve_path as _resolve

        source = _resolve(str(file_path))
    except ValueError:
        source = Path(str(file_path)).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Cannot snapshot \u2014 file not found: {source}")

    versions_dir = source.parent / ".mcp_versions"
    versions_dir.mkdir(exist_ok=True)

    # Microsecond precision; counter suffix handles same-microsecond edge case
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    # The extension is part of the name: without it this file's history is
    # indistinguishable from that of any namesake with a different extension.
    backup = versions_dir / f"{source.stem}_{ts}{source.suffix}.bak"
    counter = 1
    while backup.exists():
        backup = versions_dir / f"{source.stem}_{ts}_{counter}{source.suffix}.bak"
        counter += 1

    # Atomic write: copy to temp then rename so crashes leave no partial .bak
    fd, tmp = tempfile.mkstemp(dir=versions_dir)
    try:
        os.close(fd)
        shutil.copy2(str(source), tmp)
        shutil.move(tmp, str(backup))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return str(backup)


def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file from snapshot. Empty timestamp = list available snapshots.

    Returns:
        dict with success + snapshots list (when timestamp="")
        dict with success + restored_from path (when timestamp provided)
    """
    source = Path(file_path).resolve()
    snapshots = list_snapshots(file_path)
    progress: list[dict] = []

    if not timestamp:
        progress.append({"status": "ok", "msg": f"Found {len(snapshots)} snapshot(s)", "detail": source.name})
        return {
            "success": True,
            "op": "list_snapshots",
            "file": source.name,
            "snapshots": snapshots,
            "hint": "Pass a timestamp string to restore. Latest snapshot is first.",
            "progress": progress,
            "token_estimate": len(str(snapshots)) // 4,
        }

    match = next(
        (s for s in snapshots if timestamp in s["timestamp"] or timestamp in s["path"]),
        None,
    )
    if not match:
        available = [s["timestamp"] for s in snapshots[:5]]
        progress.append({"status": "fail", "msg": f"No snapshot matching '{timestamp}'"})
        return {
            "success": False,
            "error": f"No snapshot found matching '{timestamp}'.",
            "hint": f"Available timestamps: {', '.join(available) if available else 'none'}",
            "progress": progress,
            "token_estimate": 40,
        }

    backup_path = Path(match["path"])
    # Atomic restore via temp + rename
    fd, tmp = tempfile.mkstemp(dir=source.parent)
    try:
        os.close(fd)
        shutil.copy2(str(backup_path), tmp)
        shutil.move(tmp, str(source))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    progress.append({"status": "ok", "msg": "Restored", "detail": backup_path.name})
    return {
        "success": True,
        "op": "restore_version",
        "file": source.name,
        "restored_from": str(backup_path),
        "timestamp": match["timestamp"],
        "progress": progress,
        "token_estimate": 60,
    }


def size_kb(n_bytes: int) -> float:
    """Size in KB, rounded so a file that exists never reports as 0.0.

    round(n / 1024, 1) sends everything under 51 bytes to 0.0, and a sweep
    checking snapshots read two real 34-byte backups as "size_kb: 0.0" -- which
    is what an empty file looks like, and this number is what someone decides a
    restore on. Small files keep enough decimals to stay non-zero; only a
    genuinely empty file returns 0.0.
    """
    if n_bytes <= 0:
        return 0.0
    kb = n_bytes / 1024
    return round(kb, 1) if kb >= 0.1 else round(kb, 3)


def list_snapshots(file_path: str) -> list[dict]:
    """List available snapshots for file. Returns [{timestamp, path, size_kb}].

    Returns [] when no snapshots exist. Never raises.
    Matches this file's own `{stem}_{ts}{ext}.bak` snapshots, and the older
    extension-less `{stem}_{ts}.bak` name only where nothing else in the
    directory shares the stem, so an upgrade strands nothing and a namesake
    cannot be mistaken for a version of this file.
    """
    source = Path(file_path).resolve()
    versions_dir = source.parent / ".mcp_versions"

    if not versions_dir.exists():
        return []

    stem = source.stem
    found = set(versions_dir.glob(f"{stem}_{_TS_GLOB}{source.suffix}.bak"))
    if _legacy_is_unambiguous(source):
        found |= set(versions_dir.glob(f"{stem}_{_TS_GLOB}.bak"))
    results = []
    for bak in sorted(found, reverse=True):
        # Slice off "stem_" prefix to isolate timestamp
        ts_raw = bak.stem[len(stem) + 1 :]
        # Strip embedded extension for backward compat with old .csv.bak format
        ts = ts_raw.split(".")[0] if "." in ts_raw else ts_raw
        results.append(
            {
                "timestamp": ts,
                "path": str(bak),
                "size_kb": size_kb(bak.stat().st_size),
            }
        )

    return results
