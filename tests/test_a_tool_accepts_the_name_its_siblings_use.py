"""detect_outliers rejected the column-list spelling its five siblings use.

A census of every `@mcp.tool()` signature in this repo, by what the argument
means:

    a list of columns    feature_columns  5    columns  1

The outlier is ml_medium.detect_outliers, which lives in the same module as
ml_medium.anomaly_detection and does conceptually the same job. Against the
same file and the same two columns, on the live endpoint:

    anomaly_detection(file_path=..., feature_columns=["spends","clicks"])
      {"success": true, "n_anomalies": 842}

    detect_outliers(file_path=..., feature_columns=["spends","clicks"])
      2 validation errors for call[detect_outliers]
        columns          Missing required argument
        feature_columns  Unexpected keyword argument

pydantic refuses that before any engine code runs, so the tool cannot suggest
the name it wanted, and the live schemas carry no property descriptions -- the
parameter name is the whole contract.

The Office repo lost three sweep-phase attempts to this exact shape (a bare
`row` beside `table_index`, `new_sheet_name` beside `new_name`, an optional
`cols` that was not optional). The census that found those is what found this
one, so it runs here as a test too: a tool that invents a new spelling for a
concept its siblings already name fails the build.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

import pandas as pd
import pytest

from servers.ml_medium import server as _srv

# Reach it through the registry: the official SDK's @mcp.tool returns the
# plain function, so the module-level name skips every registry wrapper.
detect_outliers = _srv.mcp._tool_manager._tools["detect_outliers"].fn

ROOT = Path(__file__).parent.parent


def every_tool() -> list[tuple[str, ast.FunctionDef]]:
    out = []
    for path in sorted((ROOT / "servers").rglob("server.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                if any("mcp.tool" in ast.unparse(d) for d in node.decorator_list):
                    out.append((path.parent.name, node))
    return out


@pytest.fixture()
def csv(tmp_path: Path) -> str:
    frame = pd.DataFrame(
        {
            "spends": [1.0, 2.0, 3.0, 4.0, 5.0, 900.0],
            "clicks": [1, 2, 3, 4, 5, 800],
            "label": list("abcdef"),
        }
    )
    dst = tmp_path / "d.csv"
    frame.to_csv(dst, index=False)
    return str(dst)


class TestEitherSpellingWorks:
    def test_its_own_spelling_still_works(self, csv: str):
        r = detect_outliers(csv, columns=["spends", "clicks"])
        assert r["success"] is True, r.get("error")
        assert r["columns_checked"] == 2

    def test_the_sibling_spelling_now_works(self, csv: str):
        """The call that failed against the live endpoint."""
        r = detect_outliers(csv, feature_columns=["spends", "clicks"])
        assert r["success"] is True, r.get("error")
        assert r["columns_checked"] == 2

    def test_both_spellings_give_the_same_answer(self, csv: str):
        own = detect_outliers(csv, columns=["spends"])
        sibling = detect_outliers(csv, feature_columns=["spends"])
        assert own["results"] == sibling["results"]

    def test_the_documented_spelling_wins_if_both_are_sent(self, csv: str):
        r = detect_outliers(csv, columns=["spends"], feature_columns=["spends", "clicks"])
        assert r["columns_checked"] == 1, r

    def test_the_alias_is_recorded_in_progress(self, csv: str):
        r = detect_outliers(csv, feature_columns=["spends"])
        msgs = " ".join(str(p.get("msg", "")) for p in r["progress"])
        assert "alias" in msgs.lower(), r["progress"]

    def test_the_documented_spelling_is_not_announced(self, csv: str):
        r = detect_outliers(csv, columns=["spends"])
        msgs = " ".join(str(p.get("msg", "")) for p in r["progress"])
        assert "alias" not in msgs.lower()

    def test_the_method_argument_still_applies(self, csv: str):
        r = detect_outliers(csv, feature_columns=["spends"], method="std")
        assert r["success"] is True and r["method"] == "std"


class TestNeitherSpellingIsNamed:
    def test_it_fails_rather_than_guessing(self, csv: str):
        r = detect_outliers(csv)
        assert r["success"] is False

    def test_the_error_names_the_documented_spelling(self, csv: str):
        assert "columns" in detect_outliers(csv)["error"]

    def test_the_hint_names_both_and_shows_the_shape(self, csv: str):
        hint = detect_outliers(csv)["hint"]
        assert "feature_columns" in hint and "columns=[" in hint

    def test_an_empty_list_is_the_same_as_absent(self, csv: str):
        assert detect_outliers(csv, columns=[], feature_columns=[])["success"] is False


class TestTheCensusHasNoNewOutliers:
    CONCEPTS = {
        "a list of columns": {"feature_columns", "columns", "column_names", "cols"},
        "a single column": {"target_column", "column_name", "column", "label_column"},
    }

    def census(self, names: set[str]) -> dict[str, list[str]]:
        counts: dict[str, list[str]] = collections.defaultdict(list)
        for server, fn in every_tool():
            for arg in fn.args.args:
                if arg.arg in names:
                    counts[arg.arg].append(f"{server}.{fn.name}")
        return counts

    def test_every_minority_column_list_spelling_is_an_accepted_alias(self):
        counts = self.census(self.CONCEPTS["a list of columns"])
        majority = max(counts, key=lambda k: len(counts[k]))
        assert majority == "feature_columns", counts
        offenders = []
        for spelling, tools in counts.items():
            if spelling == majority:
                continue
            for qualified in tools:
                server, tool = qualified.split(".", 1)
                fn = next(f for s, f in every_tool() if s == server and f.name == tool)
                if majority not in {a.arg for a in fn.args.args}:
                    offenders.append(f"{qualified} takes {spelling} but not {majority}")
        assert not offenders, offenders

    def test_detect_outliers_is_the_only_one_needing_the_alias(self):
        counts = self.census(self.CONCEPTS["a list of columns"])
        assert counts.get("columns") == ["ml_medium.detect_outliers"], counts

    def test_the_alias_is_still_wired(self):
        fn = next(f for s, f in every_tool() if s == "ml_medium" and f.name == "detect_outliers")
        assert {"columns", "feature_columns"} <= {a.arg for a in fn.args.args}

    def test_every_tool_docstring_still_fits(self):
        for _server, fn in every_tool():
            doc = ast.get_docstring(fn) or ""
            assert len(doc) <= 80, (fn.name, len(doc), doc)
