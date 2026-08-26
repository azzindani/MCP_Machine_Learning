"""Every trainer wrote two files and named one.

    train_classifier(...) -> model_path: ".../classifier_rf.pkl"

and beside it, unmentioned, `classifier_rf.manifest.json`: the model type, the
target, the fifteen feature columns, the metrics, the sklearn version. A caller
who copied the path they were given took the pickle and left the report behind
-- and `read_model_report` reads the manifest, so the report went with it.

`export_model` on the advanced server has always returned `manifest_path`. The
three trainers were the ones that did not, which is the asymmetry that makes
these findable: same fleet, same two files, one tool telling you and three not.

Underneath it was the reason the fix took two goes. `_save_model` existed
TWICE -- `ml_basic/_basic_helpers.py` and `ml_advanced/_adv_helpers.py` -- and
the trainers import the basic one. Patching the advanced copy changed nothing
and the response came back with the string "None" in it, which is how the
second copy announced itself.

The two had drifted, each holding something the other lacked: the basic copy
ran `path.parent.mkdir(parents=True)` and the advanced copy did not, so the
same call was robust from one server and raised FileNotFoundError from the
other. One implementation now, in `shared/model_output.py` beside the path
rules these tools already share, returning the manifest path so a caller can
be told about both files.

Found in a round-15 sweep report, in a parenthesis: "(+ sibling
classifier_rf.manifest.json the reply did NOT name)".
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from servers.ml_advanced import engine as adv
from servers.ml_basic import engine as mb
from servers.ml_medium import engine as mm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def data(tmp_path: Path, classification_simple: Path) -> Path:
    src = tmp_path / "data.csv"
    shutil.copy(classification_simple, src)
    return src


TRAINERS = {
    "train_classifier": lambda src, out: mb.train_classifier(
        str(src), target_column="churned", model="dtc", output_path=str(out)
    ),
    "train_regressor": lambda src, out: mb.train_regressor(
        str(src), target_column="monthly_charges", model="lir", output_path=str(out)
    ),
    "train_with_cv": lambda src, out: mm.train_with_cv(
        str(src), target_column="monthly_charges", model="lir", task="regression", n_splits=3, output_path=str(out)
    ),
}


@pytest.mark.parametrize("trainer", sorted(TRAINERS))
class TestBothFilesAreNamed:
    def test_the_manifest_path_is_reported(self, trainer: str, data: Path, tmp_path: Path) -> None:
        r = TRAINERS[trainer](data, tmp_path / f"{trainer}.pkl")
        assert r["success"] is True, r.get("error")
        assert r.get("manifest_path"), r
        assert r["manifest_path"] != "None", "the second copy of _save_model returned nothing"

    def test_the_reported_manifest_exists(self, trainer: str, data: Path, tmp_path: Path) -> None:
        r = TRAINERS[trainer](data, tmp_path / f"{trainer}.pkl")
        assert Path(r["manifest_path"]).is_file(), r["manifest_path"]

    def test_it_sits_beside_the_model(self, trainer: str, data: Path, tmp_path: Path) -> None:
        r = TRAINERS[trainer](data, tmp_path / f"{trainer}.pkl")
        assert Path(r["manifest_path"]).parent == Path(r["model_path"]).parent

    def test_the_manifest_describes_the_model(self, trainer: str, data: Path, tmp_path: Path) -> None:
        """A path is only worth reporting if what is at the end of it is real."""
        r = TRAINERS[trainer](data, tmp_path / f"{trainer}.pkl")
        meta = json.loads(Path(r["manifest_path"]).read_text(encoding="utf-8"))
        assert meta.get("feature_columns"), meta.keys()
        assert meta.get("target_column") or meta.get("target"), meta.keys()


class TestTheModelIsStillSaved:
    """A reporting fix must not disturb the thing being reported on."""

    def test_the_pickle_loads_and_predicts(self, data: Path, tmp_path: Path) -> None:
        out = tmp_path / "c.pkl"
        r = mb.train_classifier(str(data), target_column="churned", model="dtc", output_path=str(out))
        assert r["success"] is True, r.get("error")
        got = mb.get_predictions(str(out), str(data), max_rows=5)
        assert got["success"] is True, got.get("error")
        assert got["predictions"]

    def test_a_missing_parent_directory_is_created(self, data: Path, tmp_path: Path) -> None:
        """The half the two copies disagreed about.

        ml_basic's made the directory; ml_advanced's did not, so the same call
        was robust from one server and raised FileNotFoundError from the other.
        """
        nested = tmp_path / "does" / "not" / "exist" / "m.pkl"
        r = mb.train_classifier(str(data), target_column="churned", model="dtc", output_path=str(nested))
        assert r["success"] is True, r.get("error")
        assert nested.is_file()
        assert Path(r["manifest_path"]).is_file()


class TestThereIsOneSaver:
    """The cause: patching one of two copies changed nothing."""

    def test_only_shared_defines_the_body(self) -> None:
        writers = []
        for path in ROOT.rglob("*.py"):
            if ".venv" in path.parts or "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name not in {"save_model", "_save_model"}:
                    continue
                # A delegating wrapper is fine; a second implementation is not.
                body = ast.unparse(node)
                if "atomic_write_json" in body:
                    # as_posix(), not str(): on Windows a relative path prints
                    # with backslashes and the assertion below is written with a
                    # forward slash, so CI went red on windows-latest alone --
                    # 'shared/model_output.py' in 'shared\\model_output.py:83'.
                    writers.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
        assert len(writers) == 1, f"more than one place writes the manifest: {writers}"
        assert "shared/model_output.py" in writers[0], writers

    def test_export_model_still_reports_its_manifest(self, data: Path, tmp_path: Path) -> None:
        """The one tool that always did. It must not regress with the rest."""
        src = tmp_path / "m.pkl"
        mb.train_classifier(str(data), target_column="churned", model="dtc", output_path=str(src))
        r = adv.export_model(str(src), str(tmp_path / "exported.pkl"))
        assert r["success"] is True, r.get("error")
        assert Path(r["manifest_path"]).is_file(), r
