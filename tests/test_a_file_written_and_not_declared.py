"""Two tools wrote a CSV and told the structured surface they had made nothing.

    run_clustering(..., output_path="clusters.csv")
      -> output_path: ".../clusters.csv"
         context.artifacts: []

The file is there, `output_path` names it, and `context.artifacts` -- the field
a client walks to find what a call produced -- is empty. So the prose said one
thing and the structure said another, in the same response.

The sweep called the bookkeeping "unreliable across this server's tools" and
named three: run_clustering, train_with_cv, compare_models. Counted rather than
sampled, 22 of the fleet's 35 make_context calls already declare their
artifacts, and of the thirteen that do not, eleven are read-only tools where an
empty list is exactly right. Two were wrong: run_clustering, and
anomaly_detection -- which the report never mentioned. train_with_cv and
compare_models both declare theirs and always did.

That is why this is a census and not two edits. The check below walks every
make_context call in the repo, works out whether its tool writes a file, and
requires the two to agree -- so the next tool that writes something cannot be
added without saying so, and a read-only tool is not made to invent one.

Also fixed here, from the same phase and the same cause -- a true thing the
response declined to mention: `output_path="cv_lr.json"` wrote `cv_lr.pkl`
without a word, and the sweep reasonably concluded output_path was ignored. The
correction is right (every loader expects .pkl, and list_models finds models by
that glob); the silence is what made it look like a defect. The chart tools in
the sibling repo have warned about exactly this since an earlier round.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from servers.ml_medium import engine as mm

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "tenure", "monthly_charges"]


@pytest.fixture()
def data(tmp_path: Path, classification_simple: Path) -> Path:
    src = tmp_path / "data.csv"
    shutil.copy(classification_simple, src)
    return src


def artifacts_of(response: dict) -> list[dict]:
    return (response.get("context") or {}).get("artifacts") or []


class TestAWrittenFileIsDeclared:
    def test_run_clustering_names_its_csv(self, data: Path, tmp_path: Path) -> None:
        out = tmp_path / "clusters.csv"
        r = mm.run_clustering(
            str(data), feature_columns=FEATURES, algorithm="kmeans", n_clusters=3, output_path=str(out)
        )
        assert r["success"] is True, r.get("error")
        assert [a["path"] for a in artifacts_of(r)] == [str(out)], artifacts_of(r)

    def test_anomaly_detection_names_its_csv(self, data: Path, tmp_path: Path) -> None:
        out = tmp_path / "anomalies.csv"
        r = mm.anomaly_detection(str(data), feature_columns=FEATURES, output_path=str(out))
        assert r["success"] is True, r.get("error")
        assert [a["path"] for a in artifacts_of(r)] == [str(out)], artifacts_of(r)

    def test_the_declared_file_exists(self, data: Path, tmp_path: Path) -> None:
        out = tmp_path / "clusters.csv"
        r = mm.run_clustering(
            str(data), feature_columns=FEATURES, algorithm="kmeans", n_clusters=3, output_path=str(out)
        )
        assert Path(artifacts_of(r)[0]["path"]).is_file()

    def test_nothing_is_declared_when_nothing_is_written(self, data: Path) -> None:
        """An artifact list must not be filled in with an empty string."""
        r = mm.run_clustering(str(data), feature_columns=FEATURES, algorithm="kmeans", n_clusters=3)
        assert r["success"] is True, r.get("error")
        assert r.get("output_path") == ""
        assert artifacts_of(r) == []

    def test_output_path_and_artifacts_agree(self, data: Path, tmp_path: Path) -> None:
        """The two halves of the response that disagreed."""
        out = tmp_path / "clusters.csv"
        r = mm.run_clustering(
            str(data), feature_columns=FEATURES, algorithm="kmeans", n_clusters=3, output_path=str(out)
        )
        assert [a["path"] for a in artifacts_of(r)] == [r["output_path"]]


class TestTheExtensionCorrectionIsSpoken:
    def test_a_json_output_path_still_writes_a_pkl(self, data: Path, tmp_path: Path) -> None:
        r = mm.train_with_cv(
            str(data),
            target_column="churned",
            model="lr",
            task="classification",
            n_splits=3,
            output_path=str(tmp_path / "cv.json"),
        )
        assert r["success"] is True, r.get("error")
        assert Path(r["model_path"]).suffix == ".pkl"

    def test_and_says_that_it_did(self, data: Path, tmp_path: Path) -> None:
        r = mm.train_with_cv(
            str(data),
            target_column="churned",
            model="lr",
            task="classification",
            n_splits=3,
            output_path=str(tmp_path / "cv.json"),
        )
        notes = [p for p in r["progress"] if "extension" in str(p.get("msg", "")).lower()]
        assert notes, r["progress"]
        assert "cv.json" in notes[0]["detail"] and "cv.pkl" in notes[0]["detail"], notes

    def test_a_pkl_path_is_not_commented_on(self, data: Path, tmp_path: Path) -> None:
        """The note must fire on a change, not on every call."""
        r = mm.train_with_cv(
            str(data),
            target_column="churned",
            model="lr",
            task="classification",
            n_splits=3,
            output_path=str(tmp_path / "cv.pkl"),
        )
        assert not [p for p in r["progress"] if "extension" in str(p.get("msg", "")).lower()]


class TestEveryWriterDeclares:
    """The census. Two edits would have left the next one free to skip it."""

    def test_a_tool_that_writes_a_file_passes_artifacts(self) -> None:
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            if ".venv" in path.parts or "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError, UnicodeDecodeError:
                continue
            funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "make_context"):
                    continue
                declares = len(node.args) >= 3 or any(k.arg == "artifacts" for k in node.keywords)
                if declares:
                    continue
                owner = max((f for f in funcs if f.lineno <= node.lineno), default=None, key=lambda f: f.lineno)
                body = ast.unparse(owner) if owner is not None else ""
                if '"output_path"' in body or "'output_path'" in body:
                    name = node.args[0].value if node.args else "?"
                    offenders.append(f"{name} at {path.relative_to(ROOT)}:{node.lineno}")
        assert not offenders, "writes a file and declares no artifact: " + "; ".join(offenders)

    def test_the_census_reads_something(self) -> None:
        """A scan that finds no make_context calls would pass for free."""
        seen = 0
        for path in sorted(ROOT.rglob("*.py")):
            if ".venv" in path.parts or "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError, UnicodeDecodeError:
                continue
            seen += sum(
                1 for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "make_context"
            )
        assert seen >= 30, seen
