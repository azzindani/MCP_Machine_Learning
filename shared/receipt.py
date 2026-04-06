"""Operation receipt log — tracks every write operation per file.

append_receipt() never raises — silently drops on failure.
Receipt is stored as {file}.mcp_receipt.json alongside the data file.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _receipt_path(file_path: str) -> Path:
    return Path(file_path).resolve().parent / (Path(file_path).resolve().name + ".mcp_receipt.json")


def append_receipt(
    file_path: str,
    tool: str,
    args: dict,
    result: str,
    backup: str | None,
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

        record: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool,
            "args": args,
            "result": result,
        }
        if backup:
            record["backup"] = backup

        records.append(record)
        rpath.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("append_receipt failed silently: %s", exc)


def read_receipt_log(file_path: str) -> list[dict]:
    """Read full receipt log for a file. Returns [] if no log exists."""
    try:
        rpath = _receipt_path(file_path)
        if not rpath.exists():
            return []
        return json.loads(rpath.read_text(encoding="utf-8"))
    except Exception:
        return []
