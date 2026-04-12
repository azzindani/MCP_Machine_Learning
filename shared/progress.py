"""Progress step helpers — use these instead of constructing dicts by hand.

Always use Path(x).name in msg — never full absolute paths.
Never print() — all output goes through the progress array.
"""

from pathlib import Path


def ok(msg: str, detail: str = "") -> dict:
    entry: dict = {"status": "ok", "message": msg}
    if detail:
        entry["detail"] = detail
    return entry


def fail(msg: str, detail: str = "") -> dict:
    entry: dict = {"status": "fail", "message": msg}
    if detail:
        entry["detail"] = detail
    return entry


def info(msg: str, detail: str = "") -> dict:
    entry: dict = {"status": "info", "message": msg}
    if detail:
        entry["detail"] = detail
    return entry


def warn(msg: str, detail: str = "") -> dict:
    entry: dict = {"status": "warn", "message": msg}
    if detail:
        entry["detail"] = detail
    return entry


def undo(msg: str, detail: str = "") -> dict:
    entry: dict = {"status": "undo", "message": msg}
    if detail:
        entry["detail"] = detail
    return entry


def name(path: str) -> str:
    """Return filename only — use in progress messages instead of full path."""
    return Path(path).name
