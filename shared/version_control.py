"""Snapshot and restore functionality for all MCP ML servers.

Every tool that writes to disk calls snapshot() before writing.
Backups are stored in .mcp_versions/ next to the source file.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path


def snapshot(file_path: str) -> str:
    """Snapshot file to .mcp_versions/. Returns backup path.

    Raises:
        FileNotFoundError: source file does not exist
    """
    source = Path(file_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Cannot snapshot — file not found: {source}")

    versions_dir = source.parent / ".mcp_versions"
    versions_dir.mkdir(exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    backup = versions_dir / f"{source.stem}_{ts}{source.suffix}.bak"
    shutil.copy2(source, backup)
    return str(backup)


def restore_version(file_path: str, timestamp: str = "") -> dict:
    """Restore file from snapshot. Empty timestamp = list available snapshots.

    Returns:
        dict with success/error + list of snapshots (when timestamp="")
        dict with success + restored_from path (when timestamp provided)
    """
    source = Path(file_path).resolve()
    snapshots = list_snapshots(file_path)

    if not timestamp:
        return {
            "success": True,
            "op": "list_snapshots",
            "file": source.name,
            "snapshots": snapshots,
            "hint": "Pass a timestamp string to restore. Latest snapshot is first.",
            "token_estimate": len(str(snapshots)) // 4,
        }

    # find matching snapshot
    match = next(
        (s for s in snapshots if timestamp in s["timestamp"] or timestamp in s["path"]),
        None,
    )
    if not match:
        available = [s["timestamp"] for s in snapshots[:5]]
        return {
            "success": False,
            "error": f"No snapshot found matching '{timestamp}'.",
            "hint": f"Available timestamps: {', '.join(available) if available else 'none'}",
            "token_estimate": 40,
        }

    backup_path = Path(match["path"])
    shutil.copy2(backup_path, source)
    return {
        "success": True,
        "op": "restore_version",
        "file": source.name,
        "restored_from": str(backup_path),
        "timestamp": match["timestamp"],
        "token_estimate": 60,
    }


def list_snapshots(file_path: str) -> list[dict]:
    """List available snapshots for file. Returns [{timestamp, path, size_kb}].

    Returns [] when no snapshots exist. Never raises.
    """
    source = Path(file_path).resolve()
    versions_dir = source.parent / ".mcp_versions"

    if not versions_dir.exists():
        return []

    stem = source.stem
    results = []
    for bak in sorted(versions_dir.glob(f"{stem}_*.bak"), reverse=True):
        # extract timestamp from filename: stem_TIMESTAMP.ext.bak
        name_no_bak = bak.stem  # e.g. customer_churn_2026-04-06T10-30-00Z.csv
        parts = name_no_bak.rsplit("_", 1)
        ts = parts[-1] if len(parts) > 1 else name_no_bak
        results.append(
            {
                "timestamp": ts,
                "path": str(bak),
                "size_kb": round(bak.stat().st_size / 1024, 1),
            }
        )

    return results
