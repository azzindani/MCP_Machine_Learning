#!/usr/bin/env python3
"""mcp_config_writer.py — Write MCP client configs for all supported clients.

Writes JSON config snippets for:
- LM Studio (Windows + macOS/Linux)
- Claude Desktop (Windows + macOS)
- Cursor
- Windsurf

Usage:
    python install/mcp_config_writer.py [--constrained 0|1] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

REPO_URL = "https://github.com/azzindani/MCP_Machine_Learning.git"
REPO_NAME = "MCP_Machine_Learning"

# Map server name to its subdirectory under servers/
SERVER_DIRS = {
    "ml-basic": "servers/ml_basic",
    "ml-medium": "servers/ml_medium",
    "ml-advanced": "servers/ml_advanced",
}

# --------------------------------------------------------------------------- #
# Platform helpers
# --------------------------------------------------------------------------- #


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _install_dir_win() -> str:
    return r"%USERPROFILE%\.mcp_servers\\" + REPO_NAME


def _install_dir_posix() -> str:
    home = str(Path.home())
    return f"{home}/.mcp_servers/{REPO_NAME}"


def _ps_launch_cmd(server_name: str) -> str:
    """Return PowerShell -Command string for a server entry.

    Uses a global named mutex so that when all three servers start at once,
    only one at a time runs the git-fetch + optional uv-sync block.
    Skips uv sync when .venv already exists (fast subsequent starts).
    Uses 'git checkout -B main origin/main' to avoid detached HEAD.
    Launches via 'cd tier_dir && uv run python server.py' (same pattern
    as MCP_Data_Analyst) for reliable imports.
    """
    server_dir = SERVER_DIRS[server_name].replace("/", "\\\\")
    d_expr = r"Join-Path $env:USERPROFILE '.mcp_servers\\" + REPO_NAME + "'"
    return (
        f"$d={d_expr}; "
        f"$m=[System.Threading.Mutex]::new($false,'Global\\\\MCP_ML'); "
        f"$m.WaitOne(300000)|Out-Null; "
        f"try {{ "
        f"if(!(Test-Path(Join-Path $d '.git')))"
        f"{{git clone {REPO_URL} $d -q 2>$null}}; "
        f"Set-Location $d; "
        f"git fetch origin main -q 2>$null; "
        f"git checkout -B main origin/main -q 2>$null; "
        f"if(!(Test-Path(Join-Path $d '.venv'))){{uv sync -q}} "
        f"}} finally {{$m.ReleaseMutex()}}; "
        f"Set-Location (Join-Path $d '{server_dir}'); "
        f"uv run python server.py"
    )


def _sh_launch_cmd(server_name: str) -> str:
    """Return sh -c command string for a server entry (macOS/Linux).

    Uses a mkdir-based spin lock (/tmp/mcp_ml.lock) so concurrent server
    starts don't race on git clone / uv sync.
    Skips uv sync when .venv already exists (fast subsequent starts).
    Uses 'git checkout -B main origin/main' to avoid detached HEAD.
    Launches via 'cd tier_dir && uv run python server.py' (same pattern
    as MCP_Data_Analyst) for reliable imports.
    """
    server_dir = SERVER_DIRS[server_name]
    home = str(Path.home())
    d = f"{home}/.mcp_servers/{REPO_NAME}"
    return (
        f'd="{d}"; '
        f"lf=/tmp/mcp_ml.lock; n=0; "
        f'until mkdir "$lf" 2>/dev/null; do sleep 1; n=$((n+1)); [ $n -gt 300 ] && break; done; '
        f'[ -d "$d/.git" ] || git clone {REPO_URL} "$d" -q; '
        f'cd "$d"; '
        f"git fetch origin main -q 2>/dev/null; "
        f"git checkout -B main origin/main -q 2>/dev/null; "
        f'[ -d "$d/.venv" ] || uv sync -q; '
        f'rmdir "$lf" 2>/dev/null; '
        f'cd "$d/{server_dir}"; '
        f"uv run python server.py"
    )


# --------------------------------------------------------------------------- #
# Config builders
# --------------------------------------------------------------------------- #


def _build_servers(is_win: bool, constrained: bool) -> dict:
    servers = {}
    env_val = "1" if constrained else "0"
    server_names = ["ml-basic", "ml-medium", "ml-advanced"]

    for name in server_names:
        if is_win:
            servers[name] = {
                "command": "powershell",
                "args": [
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    _ps_launch_cmd(name),
                ],
                "env": {"MCP_CONSTRAINED_MODE": env_val},
                "timeout": 600000,
            }
        else:
            servers[name] = {
                "command": "sh",
                "args": ["-c", _sh_launch_cmd(name)],
                "env": {"MCP_CONSTRAINED_MODE": env_val},
                "timeout": 600000,
            }
    return servers


def _config_paths() -> dict[str, Path | None]:
    """Return dict of client -> config path for the current platform."""
    home = Path.home()
    is_win = _is_windows()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    paths: dict[str, Path | None] = {}

    if is_win:
        paths["LM Studio"] = appdata / "LM-Studio" / "mcp-config.json"
        paths["Claude Desktop"] = appdata / "Claude" / "claude_desktop_config.json"
        paths["Cursor"] = home / ".cursor" / "mcp.json"
        paths["Windsurf"] = home / ".codeium" / "windsurf" / "mcp_config.json"
    else:
        paths["LM Studio"] = home / ".lmstudio" / "mcp-config.json"
        if platform.system() == "Darwin":
            paths["Claude Desktop"] = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        else:
            paths["Claude Desktop"] = None  # not supported on Linux
        paths["Cursor"] = home / ".cursor" / "mcp.json"
        paths["Windsurf"] = home / ".codeium" / "windsurf" / "mcp_config.json"

    return paths


def _merge_config(existing: dict, new_servers: dict) -> dict:
    """Merge new_servers into existing config, preserving other keys."""
    config = dict(existing)
    mcp = config.setdefault("mcpServers", {})
    mcp.update(new_servers)
    return config


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Write MCP client configs.")
    parser.add_argument("--constrained", type=int, default=0, choices=[0, 1], help="1 = constrained mode (<=8 GB VRAM)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written but don't write")
    args = parser.parse_args()

    is_win = _is_windows()
    servers = _build_servers(is_win, bool(args.constrained))
    config_paths = _config_paths()

    written = []
    skipped = []

    for client, cfg_path in config_paths.items():
        if cfg_path is None:
            skipped.append(f"{client}: not supported on this platform")
            continue

        existing: dict = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}

        merged = _merge_config(existing, servers)
        json_text = json.dumps(merged, indent=2)

        if args.dry_run:
            print(f"\n--- {client} ({cfg_path}) ---")
            print(json_text)
        else:
            try:
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json_text, encoding="utf-8")
                written.append(f"  {client}: {cfg_path}")
            except OSError as e:
                skipped.append(f"{client}: could not write {cfg_path} ({e})")

    if not args.dry_run:
        if written:
            print("Config written to:")
            for line in written:
                print(line)
        if skipped:
            print("\nSkipped:")
            for line in skipped:
                print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
