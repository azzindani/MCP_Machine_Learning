"""A retried training run must not leave a second copy of the model.

The saved model's name carries a wall-clock timestamp --
`{stem}_{model}_{UTC}.pkl` -- and neither trainer took an output path, so an
identical call always landed somewhere new. Round 11's sweep called each tool
twice with byte-identical arguments and measured it against the live endpoint:

    train_classifier  Ad_Data_rf_2026-08-23T15-01-28Z.pkl    53,812,898 B
    train_classifier  Ad_Data_rf_2026-08-23T15-02-55Z.pkl    53,812,898 B

Same metrics to the digit (accuracy 0.7672, identical 4x4 confusion matrix --
the seeding works), and the sweep unpickled both to confirm they differ only in
the embedded `training_date`. So a client whose training call timed out and
re-sent it pays 53.8 MB per retry for a model it already had, with no argument
anywhere to say where the file should go.

Four sibling tools -- split_dataset, run_clustering, batch_predict,
export_model -- already take an output path. The two writing the largest
artifacts took none.

The timestamped default stays, so nothing that relies on it breaks; the new
argument is appended last, because putting a parameter where an existing one sat
silently rebinds positional callers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.ml_basic import engine  # noqa: E402


@pytest.fixture
def csv(tmp_path: Path) -> str:
    rows = ["feature,other,target"]
    for i in range(120):
        rows.append(f"{i},{i % 7},{'a' if i % 2 else 'b'}")
    p = tmp_path / "data.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def numeric_csv(tmp_path: Path) -> str:
    rows = ["feature,other,target"]
    for i in range(120):
        rows.append(f"{i},{i % 7},{i * 1.5}")
    p = tmp_path / "num.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def models_under(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*.pkl") if ".mcp_versions" not in p.parts)


class TestTheClassifierHonoursAnOutputPath:
    def test_it_writes_where_it_was_told(self, csv, tmp_path):
        out = tmp_path / "models" / "mine.pkl"
        r = engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        assert r["success"] is True, r.get("error")
        assert out.exists(), r

    def test_a_retry_replaces_rather_than_adds(self, csv, tmp_path):
        out = tmp_path / "models" / "mine.pkl"
        engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        assert models_under(tmp_path / "models") == [out]

    def test_the_replaced_model_is_recoverable(self, csv, tmp_path):
        out = tmp_path / "models" / "mine.pkl"
        engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        second = engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        assert second.get("backup"), "the overwritten model was not snapshotted"

    def test_a_missing_extension_is_corrected(self, csv, tmp_path):
        out = tmp_path / "models" / "noext"
        r = engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        assert r["success"] is True, r.get("error")
        assert out.with_suffix(".pkl").exists()

    def test_the_reported_path_is_the_one_written(self, csv, tmp_path):
        out = tmp_path / "models" / "mine.pkl"
        r = engine.train_classifier(csv, "target", "dtc", output_path=str(out))
        assert Path(r["model_path"]) == out, r["model_path"]


class TestTheRegressorHonoursAnOutputPath:
    def test_it_writes_where_it_was_told(self, numeric_csv, tmp_path):
        out = tmp_path / "models" / "reg.pkl"
        r = engine.train_regressor(numeric_csv, "target", "lir", output_path=str(out))
        assert r["success"] is True, r.get("error")
        assert out.exists(), r

    def test_a_retry_replaces_rather_than_adds(self, numeric_csv, tmp_path):
        out = tmp_path / "models" / "reg.pkl"
        engine.train_regressor(numeric_csv, "target", "lir", output_path=str(out))
        engine.train_regressor(numeric_csv, "target", "lir", output_path=str(out))
        assert models_under(tmp_path / "models") == [out]


class TestTheDefaultIsUnchanged:
    def test_no_output_path_still_uses_the_timestamped_name(self, csv, tmp_path):
        r = engine.train_classifier(csv, "target", "dtc")
        assert r["success"] is True, r.get("error")
        written = Path(r["model_path"])
        assert written.name.startswith("data_dtc_"), written.name
        assert written.suffix == ".pkl"

    def test_output_path_is_the_last_parameter(self):
        # Appending it keeps every existing parameter in position; putting a new
        # name where an old one sat silently rebinds positional callers, which
        # has shipped once already in a sibling repo.
        import inspect

        for fn in (engine.train_classifier, engine.train_regressor):
            assert list(inspect.signature(fn).parameters)[-1] == "output_path"

    def test_the_wrapper_offers_it_too(self):
        import inspect

        sys.path.insert(0, str(ROOT / "servers" / "ml_basic"))
        from servers.ml_basic import server

        for name in ("train_classifier", "train_regressor"):
            fn = getattr(server, name)
            fn = getattr(fn, "fn", fn)
            params = list(inspect.signature(fn).parameters)
            assert params[-1] == "output_path", params
