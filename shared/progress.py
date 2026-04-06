"""Progress step helpers — use these instead of constructing dicts by hand.

Always use Path(x).name in msg — never full absolute paths.
Never print() — all output goes through the progress array.
"""

from pathlib import Path


def ok(msg: str, detail: str = "") -> dict:
    return {"icon": "✔", "msg": msg, "detail": detail}


def fail(msg: str, detail: str = "") -> dict:
    return {"icon": "✘", "msg": msg, "detail": detail}


def info(msg: str, detail: str = "") -> dict:
    return {"icon": "→", "msg": msg, "detail": detail}


def warn(msg: str, detail: str = "") -> dict:
    return {"icon": "⚠", "msg": msg, "detail": detail}


def undo(msg: str, detail: str = "") -> dict:
    return {"icon": "↩", "msg": msg, "detail": detail}


def name(path: str) -> str:
    """Return filename only — use in progress messages instead of full path."""
    return Path(path).name
