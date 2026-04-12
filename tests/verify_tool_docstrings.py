"""Verify all @mcp.tool() docstrings are <= 80 characters.

Run via pytest (auto-discovered) or directly:
    python tests/verify_tool_docstrings.py
"""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SERVER_DIRS = [
    _ROOT / "servers" / "ml_basic",
    _ROOT / "servers" / "ml_medium",
    _ROOT / "servers" / "ml_advanced",
]


def check_file(path: Path) -> list[tuple[int, str, int]]:
    """Return list of (line, description, length) for violations."""
    violations = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
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
            docstring = ast.get_docstring(node)
            if docstring is None:
                violations.append((node.lineno, f"{node.name}: missing docstring", 0))
                continue
            first_line = docstring.splitlines()[0].strip()
            if len(first_line) > 80:
                violations.append((node.lineno, f"{node.name}: {first_line!r}", len(first_line)))
    return violations


def collect_violations() -> list[tuple[Path, int, str, int]]:
    all_violations = []
    for d in _SERVER_DIRS:
        server_py = d / "server.py"
        if server_py.exists():
            for lineno, text, length in check_file(server_py):
                all_violations.append((server_py, lineno, text, length))
    return all_violations


def test_tool_docstrings_under_80_chars() -> None:
    """All @mcp.tool() docstrings must be <= 80 characters (STANDARDS.md §11)."""
    violations = collect_violations()
    if violations:
        lines = [f"Found {len(violations)} docstring violation(s):"]
        for path, lineno, text, length in violations:
            rel = path.relative_to(_ROOT)
            if length == 0:
                lines.append(f"  {rel}:{lineno}  {text}")
            else:
                lines.append(f"  {rel}:{lineno}  length={length}  {text}")
        raise AssertionError("\n".join(lines))


if __name__ == "__main__":
    import sys

    violations = collect_violations()
    if not violations:
        print("All tool docstrings are within the 80-character limit.")
        sys.exit(0)
    print(f"Found {len(violations)} docstring violation(s):\n")
    for path, lineno, text, length in violations:
        if length == 0:
            print(f"  {path}:{lineno}  {text}")
        else:
            print(f"  {path}:{lineno}  length={length}  {text}")
    sys.exit(1)
