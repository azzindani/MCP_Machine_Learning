"""Operation receipt log — tracks every write operation per file.

append_receipt() never raises — silently drops on failure.
Receipt is stored as {file}.mcp_receipt.json alongside the data file.

read_receipt_log() returns entries newest-first, consistent with
MCP_Data_Analyst's convention so mixed DA+ML audit trails on the same
file are readable without re-sorting.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _receipt_path(file_path: str) -> Path:
    p = Path(file_path).resolve()
    return p.parent / (p.name + ".mcp_receipt.json")


def append_receipt(
    file_path: str,
    tool: str,
    args: dict,
    result: str,
    backup: str = "",
) -> None:
    """Append one record to the receipt log. Never raises."""
    try:
        rpath = _receipt_path(file_path)
        records: list[dict] = []
        if rpath.exists():
            try:
                records = json.loads(rpath.read_text(encoding="utf-8"))
            except Exception:
                records = []

        records.append(
            {
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
                "tool": tool,
                "args": args,
                "result": result,
                "backup": backup,
            }
        )

        from shared.file_utils import atomic_write_text

        atomic_write_text(rpath, json.dumps(records, indent=2, default=str))
    except Exception as exc:
        logger.debug("append_receipt failed silently: %s", exc)


def read_receipt_log(file_path: str, last_n: int = 50) -> list[dict]:
    """Read receipt log. Returns entries newest-first, [] if none exists.

    Consistent with MCP_Data_Analyst's read_receipt_log() order so mixed
    DA+ML audit trails written to the same .mcp_receipt.json file are
    readable in natural newest-first order from either server.

    Args:
        last_n: max entries to return (0 = all). Default 50.
    """
    try:
        rpath = _receipt_path(file_path)
        if not rpath.exists():
            return []
        entries = json.loads(rpath.read_text(encoding="utf-8"))
        entries = list(reversed(entries))
        if last_n > 0:
            entries = entries[:last_n]
        return entries
    except Exception:
        return []
