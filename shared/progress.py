"""Progress step helpers — use these instead of constructing dicts by hand.

Always use Path(x).name in msg — never full absolute paths.
Never print() — all output goes through the progress array.

Schema: every dict has keys: icon, msg, and optionally detail.
"""

from pathlib import Path


def ok(msg: str, detail: str = "") -> dict:
    entry: dict = {"icon": "✔", "msg": msg}
    if detail:
        entry["detail"] = detail
    return entry


def fail(msg: str, detail: str = "") -> dict:
    entry: dict = {"icon": "✘", "msg": msg}
    if detail:
        entry["detail"] = detail
    return entry


def info(msg: str, detail: str = "") -> dict:
    entry: dict = {"icon": "→", "msg": msg}
    if detail:
        entry["detail"] = detail
    return entry


def warn(msg: str, detail: str = "") -> dict:
    entry: dict = {"icon": "⚠", "msg": msg}
    if detail:
        entry["detail"] = detail
    return entry


def undo(msg: str, detail: str = "") -> dict:
    entry: dict = {"icon": "↩", "msg": msg}
    if detail:
        entry["detail"] = detail
    return entry


def name(path: str) -> str:
    """Return filename only — use in progress messages instead of full path."""
    return Path(path).name
