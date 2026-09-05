"""Operation receipt log — tracks every write operation per file.

append_receipt() never raises — silently drops on failure.
Receipt is stored as {file}.mcp_receipt.json alongside the data file.

read_receipt_log() returns entries newest-first, consistent with
MCP_Data_Analyst's convention so mixed DA+ML audit trails on the same
file are readable without re-sorting.

**Why this file knows about a header.**

That "mixed DA+ML audit trails on the same file" line is not a nicety, it is a
constraint: both servers append to the same `.mcp_receipt.json` beside the same
CSV, and a caller runs them in one session. When MCP_Data_Analyst grew a scope
header at index 0 -- to answer a user review that read two entries after twenty
calls and concluded eighteen operations had vanished -- this reader had no idea
it was there and returned it as an entry. One real operation read as two, and
the extra had no `tool` field. Reproduced before fixing: DA writes one receipt,
ML reads two.

So the format is now understood on both sides rather than known to one. A v1
file (a bare list, no header) still reads exactly as it was written, because
existing receipts on disk do not get to become unreadable when the format
grows.

**What the log records, and why the file says so.**

`append_receipt` is called by the tools that CHANGE a file. Reads change
nothing, so they are not in here -- which is true, defensible, and was
invisible. `RECEIPT_SCOPE` is that sentence, carried in the file and handed
back by `read_receipt` so a tool can print it instead of leaving a caller to
infer that a short log means a lost history.

Entries carry a hash of the arguments, a fingerprint of the file, and how long
the operation took: the difference between "train_classifier ran" and
"train_classifier turned exactly this file into exactly that one". Hashing is
capped, because a content hash of a 200 MB CSV costs more than the write it
describes; above the cap the file is fingerprinted by size and mtime and the
entry says which kind it is rather than pretending to a hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# What the log holds. Stated once, so a tool never has to guess. Kept identical
# to MCP_Data_Analyst's wording: the two write the same file, and a caller
# reading the scope should not be able to tell which server wrote it.
RECEIPT_SCOPE = (
    "mutations only: operations that wrote to this file. Reads, inspections, "
    "correlations and chart generation are not recorded here."
)

# Above this, a content hash costs more than the operation it describes.
_MAX_HASH_BYTES = 64 * 1024 * 1024


def _receipt_path(file_path: str) -> Path:
    p = Path(file_path).resolve()
    return p.parent / (p.name + ".mcp_receipt.json")


def _hash_args(args: dict) -> str:
    """Stable hash of the arguments, so two calls can be told apart."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        blob = repr(sorted(args.items()))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fingerprint(file_path: str | Path) -> str:
    """Identify a file's contents, or say honestly that this is not a hash.

    Returns `sha256:<16 hex>` for a file small enough to read, and
    `size-mtime:<...>` for one that is not. The prefix is the point: a caller
    comparing two fingerprints must be able to tell a content hash from a
    cheaper stand-in, because only one of them proves the bytes are the same.
    """
    p = Path(file_path)
    try:
        stat = p.stat()
    except OSError:
        return ""
    if stat.st_size > _MAX_HASH_BYTES:
        return f"size-mtime:{stat.st_size}-{int(stat.st_mtime)}"
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return f"size-mtime:{stat.st_size}-{int(stat.st_mtime)}"
    return f"sha256:{digest}"


def _split_header(loaded: Any) -> tuple[list[dict], dict | None]:
    """Separate the scope header from the entries, for either file format.

    Version 1 files are a bare list of entries and are still read exactly as
    they were written -- an existing receipt does not become unreadable because
    the format grew a header.
    """
    if isinstance(loaded, dict):
        # MCP_Microsoft_Office wrote `{"file": ..., "entries": [...]}` until the
        # formats were converged. Files in that shape still exist on disk, and
        # every reader in the fleet returned [] for them.
        entries = loaded.get("entries", [])
        return [e for e in entries if isinstance(e, dict)], None
    if not isinstance(loaded, list) or not loaded:
        return [], None
    first = loaded[0]
    if isinstance(first, dict) and "_scope" in first:
        return [e for e in loaded[1:] if isinstance(e, dict)], first
    return [e for e in loaded if isinstance(e, dict)], None


def append_receipt(
    file_path: str,
    tool: str,
    args: dict,
    result: str,
    backup: str = "",
    input_fingerprint: str = "",
    duration_ms: float | None = None,
) -> None:
    """Append one record to the receipt log. Never raises.

    `input_fingerprint` is what `fingerprint()` returned BEFORE the write; the
    output side is measured here, after it. Pass it and the entry says what the
    operation turned into what. Omit it and the entry is still valid -- one side
    of a lineage is better than none, and no call site is obliged to change.
    """
    try:
        rpath = _receipt_path(file_path)
        entries: list[Any] = []
        scope_header: dict[str, Any] | None = None
        if rpath.exists():
            try:
                loaded = json.loads(rpath.read_text(encoding="utf-8"))
            except Exception:
                loaded = []
            entries, scope_header = _split_header(loaded)

        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
            "tool": tool,
            "args": args,
            "args_hash": _hash_args(args),
            "result": result,
            "backup": backup,
        }
        if input_fingerprint:
            entry["input"] = input_fingerprint
        after = fingerprint(file_path)
        if after:
            entry["output"] = after
        if duration_ms is not None:
            entry["duration_ms"] = round(float(duration_ms), 1)
        entries.append(entry)

        from shared.file_utils import atomic_write_text

        header = scope_header or {"_scope": RECEIPT_SCOPE, "_format": 2}
        atomic_write_text(rpath, json.dumps([header, *entries], indent=2, default=str))
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
    entries, _ = read_receipt(file_path, last_n)
    return entries


def read_receipt(file_path: str, last_n: int = 50) -> tuple[list[dict], str]:
    """Entries newest first, and the scope sentence that belongs beside them.

    Two return values rather than one because the count alone is what misled a
    caller: twenty operations, two entries, and no way to learn from the log
    that eighteen of them were never eligible for it.
    """
    try:
        rpath = _receipt_path(file_path)
        if not rpath.exists():
            return [], RECEIPT_SCOPE
        loaded = json.loads(rpath.read_text(encoding="utf-8"))
    except Exception:
        return [], RECEIPT_SCOPE
    entries, header = _split_header(loaded)
    scope = str(header.get("_scope")) if header else RECEIPT_SCOPE
    entries = list(reversed(entries))
    if last_n > 0:
        entries = entries[:last_n]
    return entries, scope
