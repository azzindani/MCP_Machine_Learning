"""Snapshot and restore functionality for all MCP ML servers.

Every tool that writes to disk calls snapshot() before writing.
Backups are stored in .mcp_versions/ next to the source file.

Naming: {stem}_{timestamp}.bak  where timestamp uses microsecond precision
plus a counter suffix for collision safety. Format is compatible with
MCP_Data_Analyst snapshots so both servers' backups coexist in the same
.mcp_versions/ directory and are cross-restorable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def snapshot(file_path: str) -> str:
    """Snapshot file to .mcp_versions/ atomically. Returns backup path string.

    Uses microsecond timestamps and a counter suffix to avoid collisions on
    rapid successive saves (important on Windows where datetime resolution can
    be coarser than microseconds). Atomic via temp file + shutil.move so a
    mid-copy crash cannot leave a partial .bak file.

    Backup filename format: {stem}_{ts}.bak  (compatible with DA format).

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
    backup = versions_dir / f"{source.stem}_{ts}.bak"
    counter = 1
    while backup.exists():
        backup = versions_dir / f"{source.stem}_{ts}_{counter}.bak"
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


def list_snapshots(file_path: str) -> list[dict]:
    """List available snapshots for file. Returns [{timestamp, path, size_kb}].

    Returns [] when no snapshots exist. Never raises.
    Handles both new format ({stem}_{ts}.bak) and legacy ({stem}_{ts}.csv.bak)
    so existing snapshots are not lost after upgrading.
    """
    source = Path(file_path).resolve()
    versions_dir = source.parent / ".mcp_versions"

    if not versions_dir.exists():
        return []

    stem = source.stem
    results = []
    for bak in sorted(versions_dir.glob(f"{stem}_*.bak"), reverse=True):
        # Slice off "stem_" prefix to isolate timestamp
        ts_raw = bak.stem[len(stem) + 1 :]
        # Strip embedded extension for backward compat with old .csv.bak format
        ts = ts_raw.split(".")[0] if "." in ts_raw else ts_raw
        results.append(
            {
                "timestamp": ts,
                "path": str(bak),
                "size_kb": round(bak.stat().st_size / 1024, 1),
            }
        )

    return results
