#!/usr/bin/env python3
"""Verify all @mcp.tool() docstrings are <= 80 characters.

Run with: python verify_tool_docstrings.py
Exits 1 if any violation is found.
"""

import ast
import sys
from pathlib import Path


def check_file(path: Path) -> list[tuple[int, str, int]]:
    """Return list of (line, docstring_text, length) for violations."""
    violations = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Check if decorated with @mcp.tool or @mcp.tool(...)
        for dec in node.decorator_list:
            is_tool = False
            if isinstance(dec, ast.Attribute) and dec.attr == "tool":
                is_tool = True
            elif isinstance(dec, ast.Call):
                func = dec.func
                if isinstance(func, ast.Attribute) and func.attr == "tool":
                    is_tool = True
            if not is_tool:
                continue
            # Get docstring
            docstring = ast.get_docstring(node)
            if docstring is None:
                violations.append((node.lineno, f"{node.name}: missing docstring", 0))
                continue
            first_line = docstring.splitlines()[0].strip()
            if len(first_line) > 80:
                violations.append((node.lineno, f"{node.name}: {first_line!r}", len(first_line)))
    return violations


def main() -> int:
    server_dirs = [
        Path("servers/ml_basic"),
        Path("servers/ml_medium"),
        Path("servers/ml_advanced"),
    ]
    all_violations = []
    for d in server_dirs:
        server_py = d / "server.py"
        if server_py.exists():
            for lineno, text, length in check_file(server_py):
                all_violations.append((server_py, lineno, text, length))

    if not all_violations:
        print("All tool docstrings are within the 80-character limit.")
        return 0

    print(f"Found {len(all_violations)} docstring violation(s):\n")
    for path, lineno, text, length in all_violations:
        if length == 0:
            print(f"  {path}:{lineno}  {text}")
        else:
            print(f"  {path}:{lineno}  length={length}  {text}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
